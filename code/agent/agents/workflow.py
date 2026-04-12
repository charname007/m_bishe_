"""
LangGraph工作流定义 - 使用StateGraph构建知识图谱抽取工作流
P1改进：添加 RetryPolicy 支持自动重试
"""
import asyncio
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, cast

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from loguru import logger

from .state import CorpusState, KGState, StepEnum, PhaseEnum, DEFAULT_MAX_RETRIES
from .nodes import (
    create_filter_node,          # P5新增：Filter 筛选节点
    create_normalize_node,       # P6新增：Normalize 归一化节点
    create_ner_node,
    create_re_node,
    create_eval_1_node,
    create_eval_2_node,
    create_eval_simplified_node,  # P2改进：简化评估节点
    create_label_node,
    create_coordinator_node,
    create_aggregator_node,
    create_self_check_ner_node,   # 新增：Self-Check-NER 节点
    create_self_check_re_node,    # 新增：Self-Check-RE 节点
)


# ===== 配置 =====
# P2改进：使用 ExtractionConfig 替代硬编码的 WorkflowConfig
from .config import ExtractionConfig, DEFAULT_CONFIG


def _validate_corpus_text(text: str, config: ExtractionConfig = DEFAULT_CONFIG) -> str:
    """
    验证并清理语料文本

    Args:
        text: 原始文本
        config: 配置实例

    Returns:
        清理后的文本

    Raises:
        ValueError: 文本无效
    """
    if not text or not isinstance(text, str):
        raise ValueError("语料文本不能为空")

    # 去除首尾空白
    text = text.strip()

    # 检查长度
    if len(text) < config.min_text_length:
        raise ValueError(f"语料文本长度不足（最小 {config.min_text_length} 字符）")

    if len(text) > config.max_text_length:
        logger.warning(f"语料文本过长（{len(text)} 字符），将被截断")
        text = text[:config.max_text_length]

    # 移除危险字符（防止注入攻击）
    # 保留中文、英文、数字、标点符号
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    return text


def _validate_corpus_id(corpus_id: Any) -> str:
    """
    验证语料ID

    Args:
        corpus_id: 原始ID

    Returns:
        验证后的ID字符串
    """
    if corpus_id is None:
        return f"auto_{uuid.uuid4().hex[:8]}"

    corpus_id = str(corpus_id).strip()
    if not corpus_id:
        return f"auto_{uuid.uuid4().hex[:8]}"

    # 限制ID长度
    if len(corpus_id) > 100:
        corpus_id = corpus_id[:100]

    return corpus_id


def _get_database_config() -> Dict[str, Any]:
    """
    获取数据库配置

    Returns:
        包含所有数据库连接参数的字典

    Raises:
        ValueError: 必需的环境变量未设置
    """
    # Neo4j密码是必需的
    neo4j_password = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS") or os.getenv("NEO4J_PWD")
    if not neo4j_password:
        raise ValueError("Neo4j密码未设置，请配置环境变量 NEO4J_PASSWORD")

    # PostgreSQL密码是必需的
    pg_password = os.getenv("PG_PASSWORD")
    if not pg_password:
        raise ValueError("PostgreSQL密码未设置，请配置环境变量 PG_PASSWORD")

    return {
        # Neo4j配置（兼容多种环境变量命名）
        "neo4j_uri": (
            os.getenv("NEO4J_URI") or
            os.getenv("NEO4J_URL") or
            "bolt://localhost:7687"
        ),
        "neo4j_user": (
            os.getenv("NEO4J_USER") or
            os.getenv("NEO4J_USERNAME") or
            "neo4j"
        ),
        "neo4j_password": neo4j_password,
        # PostgreSQL配置
        "pg_host": os.getenv("PG_HOST", "localhost"),
        "pg_port": int(os.getenv("PG_PORT", "5432")),
        "pg_database": os.getenv("PG_DATABASE", "kg"),
        "pg_user": os.getenv("PG_USER", "postgres"),
        "pg_password": pg_password,
    }


# ===== 条件路由函数（模块级，便于测试） =====

def route_after_filter(state: CorpusState) -> str:
    """
    Filter 后路由（P5新增）

    决策逻辑：
    - is_valid=true → 继续到 NER
    - is_valid=false → 直接跳到 END

    Args:
        state: 当前语料状态

    Returns:
        下一个节点名称或END
    """
    filter_result = state.get("filter_result", {})

    # 如果筛选失败（error），默认继续处理（保守策略）
    if state.get("error") and not filter_result:
        logger.warning(f"[Filter-Route] 筛选失败但有错误，继续到 NER")
        return "ner"

    # 判断是否有效
    is_valid = filter_result.get("is_valid", True)  # 默认继续处理
    confidence = filter_result.get("confidence", "medium")

    if is_valid:
        logger.info(f"[Filter-Route] 文本有效，继续到 NER，置信度: {confidence}")
        return "ner"
    else:
        skip_reason = filter_result.get("skip_reason", "未指定原因")
        logger.info(f"[Filter-Route] 文本无效，跳过处理，原因: {skip_reason}")
        return END


def route_after_filter_to_normalize(state: CorpusState) -> str:
    """
    Filter 后路由（同时启用 Normalize 时使用）

    有效文本 → Normalize
    无效文本 → END

    Args:
        state: 当前语料状态

    Returns:
        下一个节点名称: "normalize" 或 END
    """
    filter_result = state.get("filter_result", {})

    if state.get("error") and not filter_result:
        logger.warning(f"[Filter-Route] 筛选失败但有错误，继续到 Normalize")
        return "normalize"

    is_valid = filter_result.get("is_valid", True)

    if is_valid:
        logger.info(f"[Filter-Route] 文本有效，继续到 Normalize")
        return "normalize"
    else:
        skip_reason = filter_result.get("skip_reason", "未指定原因")
        logger.info(f"[Filter-Route] 文本无效，跳过处理，原因: {skip_reason}")
        return END


def route_after_normalize(state: CorpusState) -> str:
    """
    Normalize 后路由（P6新增）

    Normalize 节点总是输出有效结果（失败时使用原文），所以直接路由到 NER。

    Args:
        state: 当前语料状态

    Returns:
        下一个节点名称
    """
    # Normalize 失败时使用原文继续，不阻塞流程
    normalize_result = state.get("normalize_result", {})
    confidence = normalize_result.get("confidence", "medium")
    has_changes = normalize_result.get("has_changes", False)

    if state.get("error") and not normalize_result:
        logger.warning(f"[Normalize-Route] 归一化失败但有错误，使用原文继续")
    else:
        logger.info(f"[Normalize-Route] 归一化完成，置信度: {confidence}, 有改动: {has_changes}")

    return "ner"


def route_after_ner(state: CorpusState) -> str:
    """
    NER后路由（普通模式）：失败则END，成功则继续RE

    Args:
        state: 当前语料状态

    Returns:
        下一个节点名称或END
    """
    if state.get("error") or state.get("current_step") == StepEnum.DONE:
        return END
    return "re"  # 普通模式：NER → RE


def route_after_ner_for_self_check(state: CorpusState) -> str:
    """
    NER后路由（Self-Check模式）：失败则END，成功则继续Self-Check-NER

    Args:
        state: 当前语料状态

    Returns:
        下一个节点名称或END
    """
    if state.get("error") or state.get("current_step") == StepEnum.DONE:
        return END
    return "self_check_ner"  # Self-Check模式：NER → Self-Check-NER


def route_after_self_check_ner(state: CorpusState) -> str:
    """
    Self-Check-NER 后的路由决策

    决策逻辑：
    1. 检查重试次数是否达到上限
    2. 判断实体遗漏是否严重
    3. 决定回退到 NER 重抽还是继续到 RE

    Args:
        state: 当前语料状态

    Returns:
        下一个节点名称: "ner" / "re" / END
    """
    # 1. 检查是否有错误
    if state.get("error"):
        logger.warning(f"[Self-Check-NER-Route] 有错误，跳转到 END")
        return END

    # 2. 获取重试次数和上限
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    # 3. 获取 Self-Check-NER 结果
    ner_result = state.get("self_check_ner_result", {})
    ner_confidence = ner_result.get("overall_confidence", "medium")
    missing_entities = ner_result.get("missing_entities", [])

    # 4. 检查是否达到最大重试次数
    if retry_count >= max_retries:
        logger.warning(
            f"[Self-Check-NER-Route] 达到最大重试次数 {retry_count}/{max_retries}，"
            f"强制通过，置信度: {ner_confidence}"
        )
        return "re"  # 强制继续到 RE

    # 5. 判断是否需要重试 NER
    # NER 问题：遗漏实体过多（>阈值）
    # P4改进：使用配置阈值而非硬编码
    ner_low_threshold = DEFAULT_CONFIG.self_check_ner_low_threshold
    if ner_confidence == "low" and len(missing_entities) > ner_low_threshold:
        missing_names = [e.get("name", str(e)) for e in missing_entities[:3]]
        logger.info(f"[Self-Check-NER-Route] 触发 NER 重抽，遗漏实体: {missing_names}")
        return "ner"  # 回退到 NER

    # Self-Check-NER 明确建议重抽
    if ner_result.get("retry_suggested", False):
        logger.info(f"[Self-Check-NER-Route] Self-Check-NER 建议重抽")
        return "ner"

    # 6. 通过，继续到 RE
    logger.info(f"[Self-Check-NER-Route] 通过，置信度: {ner_confidence}，继续到 RE")
    return "re"


def route_after_self_check_re(state: CorpusState) -> str:
    """
    Self-Check-RE 后的路由决策（反思循环）

    决策逻辑：
    1. 检查重试次数是否达到上限
    2. 判断是 NER 问题还是 RE 问题
    3. 决定回退到哪个节点或继续到 Eval

    Args:
        state: 当前语料状态

    Returns:
        下一个节点名称: "ner" / "re" / "eval" / END
    """
    # 1. 检查是否有错误
    if state.get("error"):
        logger.warning(f"[Self-Check-RE-Route] 有错误，跳转到 END")
        return END

    # 2. 获取重试次数和上限
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    # 3. 获取 Self-Check-RE 结果
    re_result = state.get("self_check_re_result", {})
    re_confidence = re_result.get("overall_confidence", "medium")
    rejected_triples = re_result.get("rejected_triples", [])

    # 4. 检查是否达到最大重试次数
    if retry_count >= max_retries:
        logger.warning(
            f"[Self-Check-RE-Route] 达到最大重试次数 {retry_count}/{max_retries}，"
            f"强制通过，置信度: {re_confidence}"
        )
        return "eval"  # 强制继续到 Eval

    # 5. 判断是否需要重试
    need_retry = False
    retry_target = None
    retry_reason = ""

    # P4改进：使用配置阈值而非硬编码
    re_low_threshold = DEFAULT_CONFIG.self_check_re_low_threshold

    # RE 问题：幻觉三元组过多
    if re_confidence == "low" and len(rejected_triples) > re_low_threshold:
        rejected_heads = [t.get("head", "") for t in rejected_triples[:3]]
        need_retry = True
        retry_target = "re"
        retry_reason = f"幻觉三元组过多: {rejected_heads}"
        logger.info(f"[Self-Check-RE-Route] 触发 RE 重抽: {retry_reason}")

    # Self-Check-RE 明确建议重抽 NER（实体问题导致三元组问题）
    elif re_result.get("retry_suggested", False) and re_result.get("retry_target") == "ner":
        need_retry = True
        retry_target = "ner"
        retry_reason = re_result.get("retry_reason", "Self-Check 建议")
        logger.info(f"[Self-Check-RE-Route] Self-Check 建议回退到 NER: {retry_reason}")

    # Self-Check-RE 明确建议重抽 RE
    elif re_result.get("retry_suggested", False) and re_result.get("retry_target") == "re":
        need_retry = True
        retry_target = "re"
        retry_reason = re_result.get("retry_reason", "Self-Check 建议")
        logger.info(f"[Self-Check-RE-Route] Self-Check 建议回退到 RE: {retry_reason}")

    # 6. 返回路由结果
    if need_retry:
        return retry_target  # "ner" 或 "re"
    else:
        logger.info(f"[Self-Check-RE-Route] 通过，置信度: {re_confidence}，继续到 Eval")
        return "eval"


# 保留旧函数名兼容性
route_after_self_check = route_after_self_check_re


# ===== 单条语料工作流 =====

# P1改进：定义 LLM 调用节点的 RetryPolicy
def _should_retry_llm(error: Exception) -> bool:
    """判断是否应该重试 LLM 调用"""
    # 连接错误、超时、API 限流等临时故障应重试
    retryable_types = (ConnectionError, TimeoutError)
    retryable_messages = ["rate limit", "timeout", "connection", "503", "429"]

    if isinstance(error, retryable_types):
        return True

    error_msg = str(error).lower()
    return any(msg in error_msg for msg in retryable_messages)


LLM_RETRY_POLICY = RetryPolicy(
    initial_interval=1.0,      # 初始等待 1 秒
    backoff_factor=2.0,        # 每次翻倍
    max_interval=30.0,         # 最大等待 30 秒
    max_attempts=3,            # 最多重试 3 次
    jitter=True,               # 添加随机抖动防止雪崩
    retry_on=_should_retry_llm
)


def build_corpus_workflow(
    llm: Any,
    use_simplified_eval: bool = True,
    enable_self_check: bool = False,
    enable_filter: bool = False,
    enable_normalize: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> CompiledStateGraph:
    """
    构建单条语料处理工作流

    流程模式：
    - 基础模式: START → NER → RE → Eval → Label → END
    - Filter模式: START → Filter → [路由] → NER → RE → Eval → Label → END
    - Normalize模式: START → Normalize → NER → RE → Eval → Label → END
    - Filter+Normalize: START → Filter → [路由] → Normalize → NER → RE → Eval → Label → END
    - Self-Check模式: START → NER → Self-Check-NER → [反思] → RE → Self-Check-RE → [反思] → Eval → Label → END
    - 完整模式: START → Filter → [路由] → Normalize → NER → Self-Check-NER → [反思] → RE → Self-Check-RE → [反思] → Eval → Label → END

    P1改进：为 LLM 调用节点添加 RetryPolicy，自动处理临时故障
    P2改进：支持简化评估模式，减少 LLM 调用成本
    P4改进：支持 Self-Check + 反思循环，提升抽取质量
    P5改进：支持 Filter 筛选节点，提前过滤无效文本
    P6改进：支持 Normalize 归一化节点，消解指代和归一化别名

    Args:
        llm: LangChain LLM 实例
        use_simplified_eval: 是否使用简化评估（单次评估+规则校验），默认True
        enable_self_check: 是否启用 Self-Check + 反思循环，默认False
        enable_filter: 是否启用 Filter 筛选节点，默认False
        enable_normalize: 是否启用 Normalize 归一化节点，默认False
        max_retries: 反思循环最大重试次数，默认3
    """

    # 创建节点函数
    ner_node = create_ner_node(llm)
    re_node = create_re_node(llm)
    label_node = create_label_node(llm)

    # 创建StateGraph
    builder = StateGraph(CorpusState)

    # 添加基础节点
    builder.add_node("ner", ner_node, retry_policy=LLM_RETRY_POLICY)
    builder.add_node("re", re_node, retry_policy=LLM_RETRY_POLICY)
    builder.add_node("label", label_node, retry_policy=LLM_RETRY_POLICY)

    # P5+P6改进：Filter + Normalize 节点组合
    # 流程优先级：Filter 先筛选，Normalize 后归一化
    if enable_filter and enable_normalize:
        # 同时启用：Filter → [路由] → Normalize → NER 或 END
        filter_node = create_filter_node(llm)
        normalize_node = create_normalize_node(llm)
        builder.add_node("filter", filter_node, retry_policy=LLM_RETRY_POLICY)
        builder.add_node("normalize", normalize_node, retry_policy=LLM_RETRY_POLICY)

        # START → Filter → [路由] → Normalize 或 END
        builder.add_edge(START, "filter")
        builder.add_conditional_edges("filter", route_after_filter_to_normalize)
        # Normalize 总是继续到 NER
        builder.add_edge("normalize", "ner")

        logger.info(f"[Workflow] 启用 Filter + Normalize 筛选归一化模式")

    elif enable_filter:
        # 只启用 Filter: START → Filter → [路由] → NER 或 END
        filter_node = create_filter_node(llm)
        builder.add_node("filter", filter_node, retry_policy=LLM_RETRY_POLICY)

        builder.add_edge(START, "filter")
        builder.add_conditional_edges("filter", route_after_filter)

        logger.info(f"[Workflow] 启用 Filter 筛选节点")

    elif enable_normalize:
        # 只启用 Normalize: START → Normalize → NER
        normalize_node = create_normalize_node(llm)
        builder.add_node("normalize", normalize_node, retry_policy=LLM_RETRY_POLICY)

        builder.add_edge(START, "normalize")
        builder.add_edge("normalize", "ner")

        logger.info(f"[Workflow] 启用 Normalize 归一化节点")

    else:
        # 无 Filter 无 Normalize 时，直接从 START 到 NER
        builder.add_edge(START, "ner")

    # P4改进：Self-Check + 反思循环模式
    if enable_self_check:
        # Self-Check 节点
        self_check_ner_node = create_self_check_ner_node(llm)
        self_check_re_node = create_self_check_re_node(llm)
        eval_node = create_eval_simplified_node(llm)

        builder.add_node("self_check_ner", self_check_ner_node, retry_policy=LLM_RETRY_POLICY)
        builder.add_node("self_check_re", self_check_re_node, retry_policy=LLM_RETRY_POLICY)
        builder.add_node("eval", eval_node, retry_policy=LLM_RETRY_POLICY)

        # NER 失败时直接 END，成功时进入 Self-Check-NER
        builder.add_conditional_edges("ner", route_after_ner_for_self_check)

        # Self-Check-NER 后路由：回退 NER 或继续 RE
        builder.add_conditional_edges(
            "self_check_ner",
            route_after_self_check_ner,
            {
                "ner": "ner",     # 回退到 NER 重抽
                "re": "re",       # 通过，继续到 RE
            }
        )

        # RE → Self-Check-RE
        builder.add_edge("re", "self_check_re")

        # Self-Check-RE 后路由：回退 NER/RE 或继续 Eval
        builder.add_conditional_edges(
            "self_check_re",
            route_after_self_check_re,
            {
                "ner": "ner",     # 回退到 NER（实体问题导致）
                "re": "re",       # 回退到 RE（三元组问题）
                "eval": "eval",   # 通过，继续到 Eval
            }
        )

        # Eval → Label → END
        builder.add_edge("eval", "label")
        builder.add_edge("label", END)

        logger.info(f"[Workflow] 启用 Self-Check + 反思循环模式，最大重试次数: {max_retries}")

    # P2改进：根据配置选择评估模式（无 Self-Check）
    elif use_simplified_eval:
        # 简化模式：单次评估 + 规则校验
        eval_node = create_eval_simplified_node(llm)
        builder.add_node("eval", eval_node, retry_policy=LLM_RETRY_POLICY)

        builder.add_conditional_edges("ner", route_after_ner)
        builder.add_edge("re", "eval")
        builder.add_edge("eval", "label")
        builder.add_edge("label", END)

    else:
        # 原模式：两轮评估
        eval_1_node = create_eval_1_node(llm)
        eval_2_node = create_eval_2_node(llm)
        builder.add_node("eval_1", eval_1_node, retry_policy=LLM_RETRY_POLICY)
        builder.add_node("eval_2", eval_2_node, retry_policy=LLM_RETRY_POLICY)

        builder.add_conditional_edges("ner", route_after_ner)
        builder.add_edge("re", "eval_1")
        builder.add_edge("eval_1", "eval_2")
        builder.add_edge("eval_2", "label")
        builder.add_edge("label", END)

    # 编译并返回
    return builder.compile(checkpointer=InMemorySaver())


# ===== 分布式工作流 =====

def build_distributed_workflow(
    llm: Any,
    config: Optional[ExtractionConfig] = None
) -> CompiledStateGraph:
    """
    构建分布式知识图谱构建工作流

    流程: START → Coordinator → Workers(并行) → Aggregator → Finalizer → END

    改进：
    - P0：预编译 corpus_workflow，避免重复编译开销
    - P2：使用 ExtractionConfig 支持配置化

    Args:
        llm: LangChain LLM 实例
        config: ExtractionConfig 配置实例，默认使用 DEFAULT_CONFIG
    """
    config = config or DEFAULT_CONFIG

    # 创建节点函数
    coordinator_node = create_coordinator_node(config.corpus_per_worker, config.max_workers)
    aggregator_node = create_aggregator_node(config.similarity_threshold)

    # P0改进：预编译单条语料 workflow，避免在 workers_node 中重复编译
    # P2改进：使用配置决定是否使用简化评估
    # P4改进：使用配置决定是否启用 Self-Check
    # P5改进：使用配置决定是否启用 Filter
    corpus_workflow = build_corpus_workflow(
        llm,
        use_simplified_eval=config.use_simplified_eval,
        enable_self_check=config.enable_self_check,
        enable_filter=config.enable_filter,
        enable_normalize=config.enable_normalize,
        max_retries=config.self_check_max_retries
    )

    # Worker处理函数
    async def workers_node(state: KGState) -> Dict:
        """并行执行所有Worker - 按分片并行处理"""
        async def process_corpus(corpus: Dict) -> Dict:
            """处理单条语料"""
            try:
                # 验证输入 - P2改进：传入配置
                corpus_id = _validate_corpus_id(corpus.get("id"))
                raw_text = _validate_corpus_text(corpus.get("text", ""), config)

                initial_state: CorpusState = {
                    "corpus_id": corpus_id,
                    "raw_text": raw_text,
                    # P5改进：Filter 筛选初始状态
                    "filter_result": {},
                    # P6改进：Normalize 归一化初始状态
                    "normalize_result": {},
                    "normalized_text": "",
                    "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
                    "triples": [],
                    "eval_scores": [],
                    "eval_passed": False,
                    "corrected_triples": [],
                    "entity_attrs": {},
                    "relation_attrs": {},
                    # P4改进：Self-Check + 反思循环初始状态
                    "self_check_ner_result": {},
                    "self_check_re_result": {},
                    "final_entities": [],
                    "final_triples": [],
                    "verification_confidence": "medium",
                    "retry_count": 0,
                    "max_retries": config.self_check_max_retries,
                    "retry_reason": "",
                    "problem_entities": [],
                    "problem_triples": [],
                    "needs_review": False,
                    "current_step": StepEnum.NER,
                    "error": None,
                }
                # 为每条语料生成唯一的 thread_id，避免并发状态串扰
                thread_config = {"configurable": {"thread_id": f"corpus_{corpus_id}_{uuid.uuid4().hex[:8]}"}}
                result = await corpus_workflow.ainvoke(initial_state, thread_config)  # type: ignore
                return result
            except ValueError as e:
                # 输入验证错误
                logger.warning(f"语料验证失败: {e}")
                return {
                    "corpus_id": _validate_corpus_id(corpus.get("id")),
                    "error": f"输入验证失败: {e}",
                }
            except Exception as e:
                logger.error(f"处理语料失败: {e}")
                return {
                    "corpus_id": _validate_corpus_id(corpus.get("id")),
                    "error": str(e),
                }

        async def process_partition(worker_id: str, corpus_list: List[Dict]) -> Dict:
            """处理单个分片（Worker级别）"""
            start_time = time.time()
            tasks = [process_corpus(corpus) for corpus in corpus_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 分离成功和失败的结果
            success_results = []
            errors = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(str(result))
                    logger.error(f"[{worker_id}] 处理语料失败: {result}")
                else:
                    success_results.append(result)

            processing_time = time.time() - start_time
            logger.info(f"[{worker_id}] 完成: {len(success_results)}/{len(corpus_list)} 条语料, 耗时 {processing_time:.2f}s")

            return {
                "worker_id": worker_id,
                "corpus_ids": [r.get("corpus_id", "unknown") for r in success_results],
                "results": success_results,
                "processing_time": processing_time,
                "error": "; ".join(errors) if errors else None,
            }

        # 按分片并行处理，每个分片一个 Worker
        start_time = time.time()
        partition_tasks = [
            process_partition(worker_id, corpus_list)
            for worker_id, corpus_list in state["corpus_partitions"].items()
        ]
        worker_results = await asyncio.gather(*partition_tasks, return_exceptions=True)

        # 处理结果
        final_worker_results = []
        failed_workers = []
        for i, result in enumerate(worker_results):
            if isinstance(result, Exception):
                worker_ids = list(state["corpus_partitions"].keys())
                failed_workers.append(worker_ids[i])
                logger.error(f"Worker失败: {result}")
            else:
                final_worker_results.append(result)

        total_processing_time = time.time() - start_time
        logger.info(f"[Workers] 全部完成: {len(final_worker_results)} 个Worker成功, {len(failed_workers)} 个失败, 总耗时 {total_processing_time:.2f}s")

        return {
            "worker_results": final_worker_results,
            "failed_workers": failed_workers,
            "current_phase": PhaseEnum.REDUCE,
        }

    # 最终化节点
    async def finalizer_node(state: KGState) -> Dict:
        """FINALIZE阶段 - 输出到数据库"""
        from ..kg.neo4j_client import Neo4jClient
        from ..kg.postgres_client import PostgresClient

        neo4j_stats = {"merged_entities": 0, "merged_relations": 0}
        postgres_stats = {"inserted": 0}

        try:
            # 获取数据库配置
            db_config = _get_database_config()

            # 写入 Neo4j
            with Neo4jClient(
                db_config["neo4j_uri"],
                db_config["neo4j_user"],
                db_config["neo4j_password"]
            ) as neo4j:
                # 创建索引
                neo4j.create_indexes()

                # 批量合并实体
                if state["aggregated_entities"]:
                    entity_stats = neo4j.batch_merge_entities(state["aggregated_entities"])
                    neo4j_stats["merged_entities"] = entity_stats.get("merged", 0)

                # 批量合并关系
                if state["aggregated_triples"]:
                    relation_stats = neo4j.batch_merge_relations(state["aggregated_triples"])
                    neo4j_stats["merged_relations"] = relation_stats.get("merged", 0)

            # 写入 PostgreSQL
            with PostgresClient(
                db_config["pg_host"],
                db_config["pg_port"],
                db_config["pg_database"],
                db_config["pg_user"],
                db_config["pg_password"]
            ) as pg:
                # 创建表结构
                pg.create_tables()

                # 插入批次记录
                pg.insert_batch(state["batch_id"], state["total_count"], state["worker_count"])

                # 插入实体
                if state["aggregated_entities"]:
                    entity_count = pg.insert_entities(state["batch_id"], state["aggregated_entities"])
                    postgres_stats["entities"] = entity_count

                # 插入三元组
                if state["aggregated_triples"]:
                    triple_count = pg.insert_triples(state["batch_id"], state["aggregated_triples"])
                    postgres_stats["triples"] = triple_count

                # 插入语料来源（保留证据链）- 过滤掉失败语料
                all_corpus_states = []
                for worker_result in state["worker_results"]:
                    for corpus_state in worker_result.get("results", []):
                        # 只收集成功处理的语料（无error且有raw_text）
                        if not corpus_state.get("error") and corpus_state.get("raw_text"):
                            all_corpus_states.append(corpus_state)
                if all_corpus_states:
                    corpus_count = pg.insert_corpus_sources(state["batch_id"], all_corpus_states)
                    postgres_stats["corpus_sources"] = corpus_count

                # 更新批次状态
                pg.update_batch_status(state["batch_id"], "completed")

            logger.info(f"[Finalizer] Neo4j: {neo4j_stats}, PostgreSQL: {postgres_stats}")

        except Exception as e:
            logger.error(f"[Finalizer] 数据库写入失败: {e}")
            # 即使失败也继续，返回已处理的结果

        return {
            "neo4j_stats": neo4j_stats,
            "postgres_stats": postgres_stats,
            "current_phase": PhaseEnum.FINALIZE,
            "end_time": time.time(),
        }

    # 创建StateGraph
    builder = StateGraph(KGState)

    # 添加节点
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("workers", workers_node)
    builder.add_node("aggregator", aggregator_node)
    builder.add_node("finalizer", finalizer_node)

    # 定义边
    builder.add_edge(START, "coordinator")
    builder.add_edge("coordinator", "workers")
    builder.add_edge("workers", "aggregator")
    builder.add_edge("aggregator", "finalizer")
    builder.add_edge("finalizer", END)

    # 编译并返回
    return builder.compile(checkpointer=InMemorySaver())


# ===== 便捷函数 =====

async def process_corpus(llm: Any, corpus: Dict, config: Optional[ExtractionConfig] = None) -> CorpusState:
    """
    处理单条语料的便捷函数

    P4改进：支持 Self-Check + 反思循环配置

    Args:
        llm: LangChain LLM 实例
        corpus: 语料字典 {"id": "...", "text": "..."}
        config: ExtractionConfig 配置实例，默认使用 DEFAULT_CONFIG
    """
    config = config or DEFAULT_CONFIG
    workflow = build_corpus_workflow(
        llm,
        use_simplified_eval=config.use_simplified_eval,
        enable_self_check=config.enable_self_check,
        enable_filter=config.enable_filter,
        max_retries=config.self_check_max_retries
    )

    corpus_id = _validate_corpus_id(corpus.get("id"))
    raw_text = _validate_corpus_text(corpus.get("text", ""), config)

    initial_state: CorpusState = {
        "corpus_id": corpus_id,
        "raw_text": raw_text,
        # P5改进：Filter 筛选初始状态
        "filter_result": {},
        # P6改进：Normalize 归一化初始状态
        "normalize_result": {},
        "normalized_text": "",
        "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
        "triples": [],
        "eval_scores": [],
        "eval_passed": False,
        "corrected_triples": [],
        "entity_attrs": {},
        "relation_attrs": {},
        # P4改进：Self-Check + 反思循环初始状态
        "self_check_ner_result": {},
        "self_check_re_result": {},
        "final_entities": [],
        "final_triples": [],
        "verification_confidence": "medium",
        "retry_count": 0,
        "max_retries": config.self_check_max_retries,
        "retry_reason": "",
        "problem_entities": [],
        "problem_triples": [],
        "needs_review": False,
        "current_step": StepEnum.NER,
        "error": None,
    }

    # 使用唯一 thread_id 避免状态串扰
    thread_config = {"configurable": {"thread_id": f"corpus_{corpus_id}_{uuid.uuid4().hex[:8]}"}}
    result = await workflow.ainvoke(initial_state, thread_config)
    return cast(CorpusState, result)


async def process_corpus_streaming(
    llm: Any,
    corpus: Dict,
    callback: Optional[Any] = None,
    config: Optional[ExtractionConfig] = None
):
    """
    P3改进：流式处理单条语料，支持实时进度回调

    Args:
        llm: LangChain LLM 实例
        corpus: 语料字典 {"id": "...", "text": "..."}
        callback: 进度回调函数，接收事件字典
        config: ExtractionConfig 配置实例

    Yields:
        事件字典，包含 step, corpus_id, status 等字段
    """
    config = config or DEFAULT_CONFIG
    # P4改进：传递完整的配置参数，包括 enable_self_check
    workflow = build_corpus_workflow(
        llm,
        use_simplified_eval=config.use_simplified_eval,
        enable_self_check=config.enable_self_check,
        enable_filter=config.enable_filter,
        max_retries=config.self_check_max_retries
    )

    corpus_id = _validate_corpus_id(corpus.get("id"))
    raw_text = _validate_corpus_text(corpus.get("text", ""), config)

    initial_state: CorpusState = {
        "corpus_id": corpus_id,
        "raw_text": raw_text,
        # P5改进：Filter 筛选初始状态
        "filter_result": {},
        # P6改进：Normalize 归一化初始状态
        "normalize_result": {},
        "normalized_text": "",
        "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
        "triples": [],
        "eval_scores": [],
        "eval_passed": False,
        "corrected_triples": [],
        "entity_attrs": {},
        "relation_attrs": {},
        # P4改进：Self-Check + 反思循环初始状态
        "self_check_ner_result": {},
        "self_check_re_result": {},
        "final_entities": [],
        "final_triples": [],
        "verification_confidence": "medium",
        "retry_count": 0,
        "max_retries": config.self_check_max_retries,
        "retry_reason": "",
        "problem_entities": [],
        "problem_triples": [],
        "needs_review": False,
        "current_step": StepEnum.NER,
        "error": None,
    }

    thread_config = {"configurable": {"thread_id": f"corpus_{corpus_id}_{uuid.uuid4().hex[:8]}"}}

    # 使用 stream_mode="custom" 获取 StreamWriter 写入的自定义事件
    async for event_data in workflow.astream(initial_state, thread_config, stream_mode="custom"):
        # 调用回调函数
        if callback:
            try:
                callback(event_data)
            except Exception as e:
                logger.warning(f"回调函数执行失败: {e}")

        yield event_data

    # 获取最终结果
    final_state = await workflow.aget_state(thread_config)
    yield {
        "step": "final",
        "corpus_id": corpus_id,
        "status": "completed",
        "result": final_state.values
    }


async def process_batch_streaming(
    llm: Any,
    corpus_list: List[Dict],
    callback: Optional[Any] = None,
    config: Optional[ExtractionConfig] = None
):
    """
    P3改进：流式批量处理语料

    Args:
        llm: LangChain LLM 实例
        corpus_list: 语料列表
        callback: 进度回调函数
        config: ExtractionConfig 配置实例

    Yields:
        事件字典
    """
    config = config or DEFAULT_CONFIG

    for i, corpus in enumerate(corpus_list):
        yield {
            "step": "batch",
            "status": "processing",
            "current_index": i,
            "total_count": len(corpus_list),
            "corpus_id": corpus.get("id", f"unknown_{i}")
        }

        async for event in process_corpus_streaming(llm, corpus, callback, config):
            yield event

    yield {
        "step": "batch",
        "status": "completed",
        "total_count": len(corpus_list)
    }


async def process_batch(
    llm: Any,
    corpus_list: List[Dict],
    config: Optional[ExtractionConfig] = None
) -> KGState:
    """
    批量处理语料的便捷函数

    P2改进：使用 ExtractionConfig 支持配置化

    Args:
        llm: LangChain LLM 实例
        corpus_list: 语料列表
        config: ExtractionConfig 配置实例
    """
    config = config or DEFAULT_CONFIG
    workflow = build_distributed_workflow(llm, config)

    initial_state: KGState = {
        "batch_id": f"batch_{int(time.time())}",
        "corpus_list": corpus_list,
        "total_count": len(corpus_list),
        "worker_count": 0,
        "corpus_partitions": {},
        "worker_results": [],
        "aggregated_entities": [],
        "aggregated_triples": [],
        "entity_aliases": {},
        "neo4j_stats": {},
        "postgres_stats": {},
        "current_phase": PhaseEnum.INIT,
        "active_workers": [],
        "failed_workers": [],
        "start_time": time.time(),
        "end_time": None,
    }

    # 使用唯一 thread_id 避免状态串扰
    thread_config = {"configurable": {"thread_id": f"batch_{uuid.uuid4().hex}"}}
    result = await workflow.ainvoke(initial_state, thread_config)
    return cast(KGState, result)