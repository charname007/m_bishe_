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
    create_qa_scaffold_node,     # P8新增：QA Scaffold 脚手架节点
    create_ner_node,
    create_re_node,
    create_eval_1_node,
    create_eval_2_node,
    create_eval_simplified_node,  # P2改进：简化评估节点
    create_label_node,
    create_coordinator_node,
    create_aggregator_node,
    create_self_check_ner_node,   # Self-Check-NER 节点
    create_self_check_re_node,    # Self-Check-RE 节点
    # P9新增：联合抽取和所有Self-Check节点
    create_joint_ner_re_node,
    create_self_check_joint_node,
    create_self_check_qa_node,
    create_self_check_eval_node,
    create_self_check_label_node,
    # P9新增：Filter/Normalize二次检查节点（可选）
    create_self_check_filter_node,
    create_self_check_normalize_node,
    # P10新增：QA导师节点
    create_qa_mentor_node,
    create_qa_approval_node,
    create_revision_joint_node,
    # P11新增：实体对齐节点
    create_entity_alignment_node,
    # P13新增：优化版节点（RISEN/CARE/TIDD-EC框架）
    create_joint_ner_re_node_v3,
    create_filter_node_v3,
    create_self_check_joint_node_v3,
    create_re_node_v3,
    create_label_node_v3,
    get_node_creators,  # 版本切换辅助函数
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


def route_after_filter_to_joint(state: CorpusState) -> str:
    """
    Filter 后路由（联合抽取模式专用，P9新增）

    决策逻辑：
    - is_valid=true → 继续到 Joint_NER_RE
    - is_valid=false → 直接跳到 END

    Args:
        state: 当前语料状态

    Returns:
        下一个节点名称或END
    """
    filter_result = state.get("filter_result", {})

    if state.get("error") and not filter_result:
        logger.warning(f"[Filter-Route-Joint] 筛选失败但有错误，继续到联合抽取")
        return "joint_ner_re"

    is_valid = filter_result.get("is_valid", True)
    confidence = filter_result.get("confidence", "medium")

    if is_valid:
        logger.info(f"[Filter-Route-Joint] 文本有效，继续到联合抽取，置信度: {confidence}")
        return "joint_ner_re"
    else:
        skip_reason = filter_result.get("skip_reason", "未指定原因")
        logger.info(f"[Filter-Route-Joint] 文本无效，跳过处理，原因: {skip_reason}")
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


def route_after_qa_scaffold(state: CorpusState) -> str:
    """
    QA Scaffold 后路由（P8新增）

    根据 QA Scaffold 的输出决定下一步：
    - should_skip_detailed_extraction=True: 跳过后续处理，直接 END
    - 否则: 继续 NER

    Args:
        state: 当前语料状态

    Returns:
        下一个节点名称 ("ner" 或 END)
    """
    qa_result = state.get("qa_scaffold_result", {})
    should_skip = qa_result.get("should_skip_detailed_extraction", False)
    confidence = qa_result.get("overall_confidence", "medium")

    if should_skip:
        logger.info(f"[QA_Scaffold-Route] 建议跳过详细抽取，直接结束")
        return END
    else:
        logger.info(f"[QA_Scaffold-Route] 继续到 NER，置信度: {confidence}")
        return "ner"


def route_after_qa_scaffold_for_joint(state: CorpusState) -> str:
    """
    QA Scaffold 后路由（联合抽取模式专用，P9新增）

    根据 QA Scaffold 的输出决定下一步：
    - should_skip_detailed_extraction=True: 跳过后续处理，直接 END
    - 否则: 继续 Joint_NER_RE

    Args:
        state: 当前语料状态

    Returns:
        下一个节点名称 ("joint_ner_re" 或 END)
    """
    qa_result = state.get("qa_scaffold_result", {})
    should_skip = qa_result.get("should_skip_detailed_extraction", False)
    confidence = qa_result.get("overall_confidence", "medium")

    if should_skip:
        logger.info(f"[QA_Scaffold-Route-Joint] 建议跳过详细抽取，直接结束")
        return END
    else:
        logger.info(f"[QA_Scaffold-Route-Joint] 继续到联合抽取，置信度: {confidence}")
        return "joint_ner_re"


def route_after_filter_to_qa_scaffold(state: CorpusState) -> str:
    """
    Filter 后路由到 QA Scaffold（P8新增）

    Filter 有效时走 QA Scaffold，无效时直接 END。

    Args:
        state: 当前语料状态

    Returns:
        下一个节点名称 ("qa_scaffold" 或 END)
    """
    filter_result = state.get("filter_result", {})
    is_valid = filter_result.get("is_valid", True)

    if is_valid:
        logger.info(f"[Filter-Route] 文本有效，继续到 QA Scaffold")
        return "qa_scaffold"
    else:
        skip_reason = filter_result.get("skip_reason", "无地理信息")
        logger.info(f"[Filter-Route] 文本无效，跳过处理: {skip_reason}")
        return END


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


# ===== P9新增：联合抽取路由函数 =====

def create_config_init_node(enable_normalize: bool, enable_qa_scaffold: bool):
    """创建配置初始化节点（P9新增）

    在流程开始时设置配置标记字段，供路由函数判断后续节点是否启用。

    Args:
        enable_normalize: 是否启用 Normalize 节点
        enable_qa_scaffold: 是否启用 QA Scaffold 节点

    Returns:
        配置初始化节点函数
    """
    async def config_init_node(state: CorpusState) -> Dict:
        """设置配置标记字段"""
        logger.info(f"[Config-Init] 设置配置标记: normalize={enable_normalize}, qa_scaffold={enable_qa_scaffold}")
        return {
            "_config_enable_normalize": enable_normalize,
            "_config_enable_qa_scaffold": enable_qa_scaffold,
        }

    return config_init_node


def route_after_joint_extraction(state: CorpusState) -> str:
    """Joint_NER_RE 后路由"""
    if state.get("error"):
        logger.warning(f"[Joint-Route] 有错误，跳转到 Eval")
        return "eval"
    return "self_check_joint"


def route_after_self_check_joint(state: CorpusState) -> str:
    """Self-Check-Joint 后路由 - Reflexion驱动的重试"""

    if state.get("error"):
        logger.warning(f"[Self-Check-Joint-Route] 有错误，跳转到 END")
        return END

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    check_result = state.get("self_check_joint_result", {})
    retry_suggested = check_result.get("retry_suggested", False)
    confidence = check_result.get("overall_confidence", "medium")

    # 达到最大重试次数，强制通过
    if retry_count >= max_retries:
        logger.warning(f"[Self-Check-Joint-Route] 达到最大重试 {retry_count}/{max_retries}")
        return "eval"

    # Reflexion建议重试且置信度低
    if retry_suggested and confidence == "low":
        reflection = state.get("reflection_text", "")
        logger.info(f"[Self-Check-Joint-Route] 触发重试，反思: {reflection[:100]}...")
        return "joint_ner_re"  # 回退到联合抽取

    # 通过
    logger.info(f"[Self-Check-Joint-Route] 通过，置信度: {confidence}")
    return "eval"


def route_after_self_check_qa(state: CorpusState) -> str:
    """Self-Check-QA 后路由"""

    if state.get("error"):
        logger.warning(f"[Self-Check-QA-Route] 有错误，继续到联合抽取")
        return "joint_ner_re"

    check_result = state.get("self_check_qa_result", {})
    retry_suggested = check_result.get("retry_suggested", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    # 达到最大重试次数，强制通过
    if retry_count >= max_retries:
        logger.warning(f"[Self-Check-QA-Route] 达到最大重试 {retry_count}/{max_retries}")
        return "joint_ner_re"

    if retry_suggested:
        logger.info(f"[Self-Check-QA-Route] 建议重新生成QA")
        return "qa_scaffold"

    logger.info(f"[Self-Check-QA-Route] 通过，继续到联合抽取")
    return "joint_ner_re"


def route_after_self_check_eval(state: CorpusState) -> str:
    """Self-Check-Eval 后路由"""

    if state.get("error"):
        logger.warning(f"[Self-Check-Eval-Route] 有错误，继续到Label")
        return "label"

    check_result = state.get("self_check_eval_result", {})
    retry_suggested = check_result.get("retry_suggested", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    if retry_count >= max_retries:
        logger.warning(f"[Self-Check-Eval-Route] 达到最大重试")
        return "label"

    if retry_suggested:
        logger.info(f"[Self-Check-Eval-Route] 建议重新评估")
        return "eval"

    logger.info(f"[Self-Check-Eval-Route] 通过，继续到Label")
    return "label"


def route_after_self_check_label(state: CorpusState) -> str:
    """Self-Check-Label 后路由"""

    if state.get("error"):
        logger.warning(f"[Self-Check-Label-Route] 有错误，结束")
        return END

    check_result = state.get("self_check_label_result", {})
    retry_suggested = check_result.get("retry_suggested", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    if retry_count >= max_retries:
        logger.warning(f"[Self-Check-Label-Route] 达到最大重试")
        return END

    if retry_suggested:
        logger.info(f"[Self-Check-Label-Route] 建议重新标注")
        return "label"

    logger.info(f"[Self-Check-Label-Route] 通过，结束")
    return END


def route_after_self_check_filter(state: CorpusState) -> str:
    """Self-Check-Filter 后路由（P9新增，可选）

    根据配置标记判断下一步：
    - 如果文本有效且启用了 Normalize → normalize
    - 如果文本有效且未启用 Normalize 但启用了 QA Scaffold → qa_scaffold
    - 如果文本有效且两者都未启用 → joint_ner_re
    - 如果文本无效 → END

    Args:
        state: 当前语料状态

    Returns:
        下一个节点名称
    """
    # 从配置标记字段获取信息
    enable_normalize = state.get("_config_enable_normalize", False)
    enable_qa_scaffold = state.get("_config_enable_qa_scaffold", False)

    if state.get("error"):
        # 根据配置决定下一个节点
        if enable_normalize:
            logger.warning(f"[Self-Check-Filter-Route] 有错误，继续到Normalize")
            return "normalize"
        elif enable_qa_scaffold:
            logger.warning(f"[Self-Check-Filter-Route] 有错误，继续到QA Scaffold")
            return "qa_scaffold"
        else:
            logger.warning(f"[Self-Check-Filter-Route] 有错误，继续到联合抽取")
            return "joint_ner_re"

    check_result = state.get("self_check_filter_result", {})
    retry_suggested = check_result.get("retry_suggested", False)
    verified_is_valid = check_result.get("verified_is_valid", True)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    # 达到最大重试次数，强制通过
    if retry_count >= max_retries:
        logger.warning(f"[Self-Check-Filter-Route] 达到最大重试 {retry_count}/{max_retries}")
        if not verified_is_valid:
            return END
        # 根据配置决定下一个节点
        if enable_normalize:
            return "normalize"
        elif enable_qa_scaffold:
            return "qa_scaffold"
        return "joint_ner_re"

    # 如果建议重试
    if retry_suggested:
        logger.info(f"[Self-Check-Filter-Route] 建议重新筛选")
        return "filter"

    # 根据校验后的判定决定下一步
    if not verified_is_valid:
        logger.info(f"[Self-Check-Filter-Route] 文本无效，结束")
        return END

    # 文本有效，根据配置决定下一个节点
    if enable_normalize:
        logger.info(f"[Self-Check-Filter-Route] 文本有效，继续到Normalize")
        return "normalize"

    if enable_qa_scaffold:
        logger.info(f"[Self-Check-Filter-Route] 文本有效，继续到QA Scaffold")
        return "qa_scaffold"

    logger.info(f"[Self-Check-Filter-Route] 文本有效，继续到联合抽取")
    return "joint_ner_re"


def route_after_self_check_normalize(state: CorpusState) -> str:
    """Self-Check-Normalize 后路由（P9新增，可选）

    根据配置标记判断下一步：
    - 如果启用了 QA Scaffold → qa_scaffold
    - 否则 → joint_ner_re（联合抽取模式）

    Args:
        state: 当前语料状态

    Returns:
        下一个节点名称
    """
    # 从配置标记字段获取信息
    enable_qa_scaffold = state.get("_config_enable_qa_scaffold", False)

    if state.get("error"):
        if enable_qa_scaffold:
            logger.warning(f"[Self-Check-Normalize-Route] 有错误，继续到QA Scaffold")
            return "qa_scaffold"
        else:
            logger.warning(f"[Self-Check-Normalize-Route] 有错误，继续到联合抽取")
            return "joint_ner_re"

    check_result = state.get("self_check_normalize_result", {})
    retry_suggested = check_result.get("retry_suggested", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)

    # 达到最大重试次数，强制通过
    if retry_count >= max_retries:
        logger.warning(f"[Self-Check-Normalize-Route] 达到最大重试 {retry_count}/{max_retries}")
        if enable_qa_scaffold:
            return "qa_scaffold"
        else:
            return "joint_ner_re"

    if retry_suggested:
        logger.info(f"[Self-Check-Normalize-Route] 建议重新归一化")
        return "normalize"

    # 根据配置决定下一个节点
    if enable_qa_scaffold:
        logger.info(f"[Self-Check-Normalize-Route] 通过，继续到QA Scaffold")
        return "qa_scaffold"
    else:
        logger.info(f"[Self-Check-Normalize-Route] 通过，继续到联合抽取")
        return "joint_ner_re"


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
    enable_qa_scaffold: bool = False,
    use_joint_extraction: bool = True,  # P9新增：默认使用联合抽取
    enable_full_self_check: bool = False,  # P9新增：启用所有二次检查
    enable_self_check_filter: bool = False,  # P9新增：Filter二次检查（可选）
    enable_self_check_normalize: bool = False,  # P9新增：Normalize二次检查（可选）
    enable_entity_alignment: bool = False,  # P11新增：实体对齐
    config: ExtractionConfig = None,  # P11新增：配置对象（用于实体对齐）
    max_retries: int = DEFAULT_MAX_RETRIES,
    prompt_version: str = "v2",  # P13新增：提示词版本切换
) -> CompiledStateGraph:
    """
    构建单条语料处理工作流

    流程模式：
    - 基础模式: START → NER → RE → Eval → Label → END
    - Filter模式: START → Filter → [路由] → NER → RE → Eval → Label → END
    - Normalize模式: START → Normalize → NER → RE → Eval → Label → END
    - Filter+Normalize: START → Filter → [路由] → Normalize → NER → RE → Eval → Label → END
    - Filter+Normalize+QA: START → Filter → [路由] → Normalize → QA_Scaffold → NER → RE → Eval → Label → END
    - Self-Check模式: START → NER → Self-Check-NER → [反思] → RE → Self-Check-RE → [反思] → Eval → Label → END
    - 完整模式: START → Filter → [路由] → Normalize → QA_Scaffold → NER → Self-Check-NER → [反思] → RE → Self-Check-RE → [反思] → Eval → Label → END

    P9新增模式（联合抽取 + 二次检查）：
    - 联合抽取模式: START → Filter → Normalize → QA_Scaffold → Self-Check-QA → Joint_NER_RE → Self-Check-Joint(Reflexion) → Eval → Self-Check-Eval → Label → Self-Check-Label → END
    - 含Filter/Normalize二次检查: START → Filter → Self-Check-Filter → Normalize → Self-Check-Normalize → QA_Scaffold → ...

    P11新增模式（实体对齐）：
    - 实体对齐模式: ... → Label → Entity_Alignment → END

    P13新增（提示词版本切换）：
    - prompt_version="v2": 原版提示词（约4000 Token）
    - prompt_version="v3": RISEN优化版（约1500 Token，节省60%成本）

    P1改进：为 LLM 调用节点添加 RetryPolicy，自动处理临时故障
    P2改进：支持简化评估模式，减少 LLM 调用成本
    P4改进：支持 Self-Check + 反思循环，提升抽取质量
    P5改进：支持 Filter 筛选节点，提前过滤无效文本
    P6改进：支持 Normalize 归一化节点，消解指代和归一化别名
    P8改进：支持 QA Scaffold 节点，5W1H问答构建语义脚手架
    P9改进：支持联合抽取 + Reflexion机制 + 所有节点二次检查
    P11改进：支持实体对齐节点，将抽取实体与数据库已有实体匹配
    P13改进：支持提示词版本切换，优化Token消耗

    Args:
        llm: LangChain LLM 实例
        use_simplified_eval: 是否使用简化评估（单次评估+规则校验），默认True
        enable_self_check: 是否启用 Self-Check + 反思循环（流水线模式），默认False
        enable_filter: 是否启用 Filter 筛选节点，默认False
        enable_normalize: 是否启用 Normalize 归一化节点，默认False
        enable_qa_scaffold: 是否启用 QA Scaffold 脚手架节点，默认False
        use_joint_extraction: 是否使用联合抽取模式（默认True，False则使用流水线NER+RE）
        enable_full_self_check: 是否启用所有节点二次检查（QA、Joint、Eval、Label），默认False
        enable_entity_alignment: 是否启用实体对齐节点，默认False
        config: ExtractionConfig配置对象（用于实体对齐节点的参数）
        max_retries: 反思循环最大重试次数，默认3
    """

    # 使用默认配置如果未提供
    if config is None:
        config = DEFAULT_CONFIG

    # P13新增：从配置中获取提示词版本（如果未显式指定）
    if prompt_version == "v2" and hasattr(config, 'prompt_version'):
        prompt_version = config.prompt_version

    # P13新增：根据提示词版本选择节点创建函数
    logger.info(f"[Workflow] 提示词版本: {prompt_version}")

    if prompt_version == "v3":
        # 使用优化版节点（RISEN/CARE/TIDD-EC框架）
        node_creators = get_node_creators("v3")
        joint_ner_re_node = node_creators["joint_ner_re"](llm)
        self_check_joint_node = node_creators["self_check_joint"](llm)
        filter_node_v3 = node_creators["filter"](llm)
        re_node_v3 = node_creators["re"](llm)
        label_node_v3 = node_creators["label"](llm)
        logger.info("[Workflow] 使用优化版提示词（Token节省约60%）")
    else:
        # 使用原版节点
        node_creators = get_node_creators("v2")
        joint_ner_re_node = node_creators["joint_ner_re"](llm)
        self_check_joint_node = node_creators["self_check_joint"](llm)
        filter_node_v3 = create_filter_node(llm)  # Filter节点保持原版
        re_node_v3 = node_creators["re"](llm)
        label_node_v3 = node_creators["label"](llm)
        logger.info("[Workflow] 使用原版提示词")

    # 创建节点函数（通用节点保持不变）
    ner_node = create_ner_node(llm)
    re_node = create_re_node(llm)
    label_node = create_label_node(llm)

    # 创建StateGraph
    builder = StateGraph(CorpusState)

    # P9新增：联合抽取模式
    if use_joint_extraction:
        # P13改进：使用版本化节点
        builder.add_node("joint_ner_re", joint_ner_re_node, retry_policy=LLM_RETRY_POLICY)

        # 二次检查节点（如果启用）
        if enable_full_self_check:
            self_check_qa_node = create_self_check_qa_node(llm)
            # P13改进：使用版本化Self-Check-Joint节点
            builder.add_node("self_check_qa", self_check_qa_node, retry_policy=LLM_RETRY_POLICY)
            builder.add_node("self_check_joint", self_check_joint_node, retry_policy=LLM_RETRY_POLICY)

            self_check_eval_node = create_self_check_eval_node(llm)
            self_check_label_node = create_self_check_label_node(llm)
            builder.add_node("self_check_eval", self_check_eval_node, retry_policy=LLM_RETRY_POLICY)
            builder.add_node("self_check_label", self_check_label_node, retry_policy=LLM_RETRY_POLICY)

        # P9新增：Filter/Normalize二次检查节点（可选）
        if enable_self_check_filter:
            self_check_filter_node = create_self_check_filter_node(llm)
            builder.add_node("self_check_filter", self_check_filter_node, retry_policy=LLM_RETRY_POLICY)

        if enable_self_check_normalize:
            self_check_normalize_node = create_self_check_normalize_node(llm)
            builder.add_node("self_check_normalize", self_check_normalize_node, retry_policy=LLM_RETRY_POLICY)

        # 评估和标注节点
        eval_node = create_eval_simplified_node(llm)
        builder.add_node("eval", eval_node, retry_policy=LLM_RETRY_POLICY)
        # P13改进：使用版本化Label节点
        builder.add_node("label", label_node_v3 if prompt_version == "v3" else label_node, retry_policy=LLM_RETRY_POLICY)

        # 前置节点处理（Filter → Self-Check-Filter → Normalize → Self-Check-Normalize → QA_Scaffold）
        # P5+P6+P8改进：前置节点组合
        # P9改进：如果启用了 Self-Check-Filter 或 Self-Check-Normalize，需要添加配置初始化节点

        # 创建配置初始化节点（如果需要）
        need_config_init = enable_self_check_filter or enable_self_check_normalize
        if need_config_init:
            config_init_node = create_config_init_node(enable_normalize, enable_qa_scaffold)
            builder.add_node("config_init", config_init_node)
            logger.info(f"[Workflow] 添加配置初始化节点: normalize={enable_normalize}, qa_scaffold={enable_qa_scaffold}")

        if enable_filter and enable_normalize:
            # P13改进：使用版本化Filter节点
            builder.add_node("filter", filter_node_v3 if prompt_version == "v3" else create_filter_node(llm), retry_policy=LLM_RETRY_POLICY)
            normalize_node = create_normalize_node(llm)
            builder.add_node("normalize", normalize_node, retry_policy=LLM_RETRY_POLICY)

            # Filter → Self-Check-Filter → Normalize
            if enable_self_check_filter:
                # 需要配置初始化节点来设置标记字段
                builder.add_edge(START, "config_init")
                builder.add_edge("config_init", "filter")
                builder.add_edge("filter", "self_check_filter")

                # 动态构建映射，只包含实际存在的节点
                self_check_filter_targets = {"filter": "filter", "normalize": "normalize", END: END}
                if enable_qa_scaffold:
                    self_check_filter_targets["qa_scaffold"] = "qa_scaffold"
                else:
                    self_check_filter_targets["joint_ner_re"] = "joint_ner_re"

                builder.add_conditional_edges(
                    "self_check_filter",
                    route_after_self_check_filter,
                    self_check_filter_targets
                )
            else:
                if need_config_init:
                    builder.add_edge(START, "config_init")
                    builder.add_edge("config_init", "filter")
                else:
                    builder.add_edge(START, "filter")
                builder.add_conditional_edges("filter", route_after_filter_to_normalize)

            # Normalize → Self-Check-Normalize → QA_Scaffold
            if enable_self_check_normalize:
                builder.add_edge("normalize", "self_check_normalize")

                # 动态构建映射，只包含实际存在的节点
                self_check_normalize_targets = {"normalize": "normalize"}
                if enable_qa_scaffold:
                    self_check_normalize_targets["qa_scaffold"] = "qa_scaffold"
                else:
                    self_check_normalize_targets["joint_ner_re"] = "joint_ner_re"

                builder.add_conditional_edges(
                    "self_check_normalize",
                    route_after_self_check_normalize,
                    self_check_normalize_targets
                )
            else:
                if enable_qa_scaffold:
                    builder.add_edge("normalize", "qa_scaffold")
                else:
                    builder.add_edge("normalize", "joint_ner_re")

            if enable_qa_scaffold:
                qa_scaffold_node = create_qa_scaffold_node(llm)
                builder.add_node("qa_scaffold", qa_scaffold_node, retry_policy=LLM_RETRY_POLICY)

                if enable_full_self_check:
                    # QA_Scaffold → Self-Check-QA → Joint_NER_RE
                    builder.add_edge("qa_scaffold", "self_check_qa")
                    builder.add_conditional_edges(
                        "self_check_qa",
                        route_after_self_check_qa,
                        {"qa_scaffold": "qa_scaffold", "joint_ner_re": "joint_ner_re"}
                    )
                else:
                    builder.add_conditional_edges("qa_scaffold", route_after_qa_scaffold_for_joint)
                logger.info(f"[Workflow] 联合抽取: Filter + Normalize + QA Scaffold + 二次检查")
            else:
                # 无QA Scaffold，直接到联合抽取
                logger.info(f"[Workflow] 联合抽取: Filter + Normalize")

        elif enable_filter:
            # P13改进：使用版本化Filter节点
            builder.add_node("filter", filter_node_v3 if prompt_version == "v3" else create_filter_node(llm), retry_policy=LLM_RETRY_POLICY)

            if enable_self_check_filter:
                # 需要配置初始化节点来设置标记字段（normalize=False, qa_scaffold由配置决定）
                builder.add_edge(START, "config_init")
                builder.add_edge("config_init", "filter")
                builder.add_edge("filter", "self_check_filter")

                # 动态构建映射，只包含实际存在的节点
                self_check_filter_targets = {"filter": "filter", END: END}
                if enable_qa_scaffold:
                    self_check_filter_targets["qa_scaffold"] = "qa_scaffold"
                else:
                    self_check_filter_targets["joint_ner_re"] = "joint_ner_re"

                builder.add_conditional_edges(
                    "self_check_filter",
                    route_after_self_check_filter,
                    self_check_filter_targets
                )
            else:
                builder.add_edge(START, "filter")

                if enable_qa_scaffold:
                    builder.add_conditional_edges("filter", route_after_filter_to_qa_scaffold)
                else:
                    builder.add_conditional_edges("filter", route_after_filter_to_joint)

            if enable_qa_scaffold:
                qa_scaffold_node = create_qa_scaffold_node(llm)
                builder.add_node("qa_scaffold", qa_scaffold_node, retry_policy=LLM_RETRY_POLICY)

                if enable_full_self_check:
                    builder.add_edge("qa_scaffold", "self_check_qa")
                    builder.add_conditional_edges(
                        "self_check_qa",
                        route_after_self_check_qa,
                        {"qa_scaffold": "qa_scaffold", "joint_ner_re": "joint_ner_re"}
                    )
                else:
                    builder.add_conditional_edges("qa_scaffold", route_after_qa_scaffold_for_joint)

            logger.info(f"[Workflow] 联合抽取: Filter")

        elif enable_normalize:
            normalize_node = create_normalize_node(llm)
            builder.add_node("normalize", normalize_node, retry_policy=LLM_RETRY_POLICY)

            if enable_self_check_normalize:
                # 需要配置初始化节点来设置标记字段（normalize=True, qa_scaffold由配置决定）
                builder.add_edge(START, "config_init")
                builder.add_edge("config_init", "normalize")
                builder.add_edge("normalize", "self_check_normalize")

                # 动态构建映射，只包含实际存在的节点
                self_check_normalize_targets = {"normalize": "normalize"}
                if enable_qa_scaffold:
                    self_check_normalize_targets["qa_scaffold"] = "qa_scaffold"
                else:
                    self_check_normalize_targets["joint_ner_re"] = "joint_ner_re"

                builder.add_conditional_edges(
                    "self_check_normalize",
                    route_after_self_check_normalize,
                    self_check_normalize_targets
                )
            else:
                builder.add_edge(START, "normalize")

                if enable_qa_scaffold:
                    builder.add_edge("normalize", "qa_scaffold")
                else:
                    builder.add_edge("normalize", "joint_ner_re")

            if enable_qa_scaffold:
                qa_scaffold_node = create_qa_scaffold_node(llm)
                builder.add_node("qa_scaffold", qa_scaffold_node, retry_policy=LLM_RETRY_POLICY)

                if enable_full_self_check:
                    builder.add_edge("qa_scaffold", "self_check_qa")
                    builder.add_conditional_edges(
                        "self_check_qa",
                        route_after_self_check_qa,
                        {"qa_scaffold": "qa_scaffold", "joint_ner_re": "joint_ner_re"}
                    )
                else:
                    builder.add_conditional_edges("qa_scaffold", route_after_qa_scaffold_for_joint)

            logger.info(f"[Workflow] 联合抽取: Normalize")

        elif enable_qa_scaffold:
            qa_scaffold_node = create_qa_scaffold_node(llm)
            builder.add_node("qa_scaffold", qa_scaffold_node, retry_policy=LLM_RETRY_POLICY)
            builder.add_edge(START, "qa_scaffold")

            if enable_full_self_check:
                builder.add_edge("qa_scaffold", "self_check_qa")
                builder.add_conditional_edges(
                    "self_check_qa",
                    route_after_self_check_qa,
                    {"qa_scaffold": "qa_scaffold", "joint_ner_re": "joint_ner_re"}
                )
            else:
                builder.add_conditional_edges("qa_scaffold", route_after_qa_scaffold_for_joint)

            logger.info(f"[Workflow] 联合抽取: QA Scaffold")

        else:
            # 无前置节点，直接从START到联合抽取
            builder.add_edge(START, "joint_ner_re")
            logger.info(f"[Workflow] 联合抽取: 基础模式")

        # 联合抽取后的流程
        if enable_full_self_check:
            # Joint_NER_RE → Self-Check-Joint(Reflexion) → Eval → Self-Check-Eval → Label → Self-Check-Label → END
            builder.add_conditional_edges(
                "joint_ner_re",
                route_after_joint_extraction,
                {"self_check_joint": "self_check_joint", "eval": "eval"}
            )
            builder.add_conditional_edges(
                "self_check_joint",
                route_after_self_check_joint,
                {"joint_ner_re": "joint_ner_re", "eval": "eval"}
            )
            builder.add_edge("eval", "self_check_eval")
            builder.add_conditional_edges(
                "self_check_eval",
                route_after_self_check_eval,
                {"eval": "eval", "label": "label"}
            )
            builder.add_edge("label", "self_check_label")

            # P11新增：实体对齐节点（在self_check_label之后）
            if enable_entity_alignment:
                entity_alignment_node = create_entity_alignment_node(llm, config)
                builder.add_node("entity_alignment", entity_alignment_node)
                builder.add_conditional_edges(
                    "self_check_label",
                    route_after_self_check_label,
                    {"label": "label", "entity_alignment": "entity_alignment"}
                )
                builder.add_edge("entity_alignment", END)
                logger.info(f"[Workflow] 联合抽取 + Reflexion + 全二次检查 + 实体对齐启用")
            else:
                builder.add_conditional_edges(
                    "self_check_label",
                    route_after_self_check_label,
                    {"label": "label", END: END}
                )
                logger.info(f"[Workflow] 联合抽取 + Reflexion + 全二次检查启用")
        else:
            # 简化流程: Joint_NER_RE → Eval → Label → [Entity_Alignment] → END
            builder.add_edge("joint_ner_re", "eval")
            builder.add_edge("eval", "label")

            # P11新增：实体对齐节点（在label之后）
            if enable_entity_alignment:
                entity_alignment_node = create_entity_alignment_node(llm, config)
                builder.add_node("entity_alignment", entity_alignment_node)
                builder.add_edge("label", "entity_alignment")
                builder.add_edge("entity_alignment", END)
                logger.info(f"[Workflow] 联合抽取模式 + 实体对齐启用")
            else:
                builder.add_edge("label", END)
                logger.info(f"[Workflow] 联合抽取模式启用（无二次检查）")

        return builder.compile(checkpointer=InMemorySaver())

    # ===== 流水线模式（保留原有逻辑） =====
    # 添加基础节点
    builder.add_node("ner", ner_node, retry_policy=LLM_RETRY_POLICY)
    builder.add_node("re", re_node, retry_policy=LLM_RETRY_POLICY)
    builder.add_node("label", label_node, retry_policy=LLM_RETRY_POLICY)

    # P8新增：QA Scaffold 节点创建（如果启用）
    qa_scaffold_node = None
    if enable_qa_scaffold:
        qa_scaffold_node = create_qa_scaffold_node(llm)
        builder.add_node("qa_scaffold", qa_scaffold_node, retry_policy=LLM_RETRY_POLICY)

    # P5+P6+P8改进：Filter + Normalize + QA Scaffold 节点组合
    # 流程优先级：Filter 先筛选 → Normalize 归一化 → QA Scaffold 构建脚手架 → NER
    if enable_filter and enable_normalize:
        filter_node = create_filter_node(llm)
        normalize_node = create_normalize_node(llm)
        builder.add_node("filter", filter_node, retry_policy=LLM_RETRY_POLICY)
        builder.add_node("normalize", normalize_node, retry_policy=LLM_RETRY_POLICY)

        builder.add_edge(START, "filter")
        builder.add_conditional_edges("filter", route_after_filter_to_normalize)

        if enable_qa_scaffold:
            # Filter → Normalize → QA_Scaffold → NER
            builder.add_edge("normalize", "qa_scaffold")
            builder.add_conditional_edges("qa_scaffold", route_after_qa_scaffold)
            logger.info(f"[Workflow] 启用 Filter + Normalize + QA Scaffold 模式")
        else:
            # Filter → Normalize → NER
            builder.add_edge("normalize", "ner")
            logger.info(f"[Workflow] 启用 Filter + Normalize 模式")

    elif enable_filter:
        filter_node = create_filter_node(llm)
        builder.add_node("filter", filter_node, retry_policy=LLM_RETRY_POLICY)
        builder.add_edge(START, "filter")

        if enable_qa_scaffold:
            # Filter → QA_Scaffold → NER
            builder.add_conditional_edges("filter", route_after_filter_to_qa_scaffold)
            builder.add_conditional_edges("qa_scaffold", route_after_qa_scaffold)
            logger.info(f"[Workflow] 启用 Filter + QA Scaffold 模式")
        else:
            # Filter → NER
            builder.add_conditional_edges("filter", route_after_filter)
            logger.info(f"[Workflow] 启用 Filter 筛选节点")

    elif enable_normalize:
        normalize_node = create_normalize_node(llm)
        builder.add_node("normalize", normalize_node, retry_policy=LLM_RETRY_POLICY)
        builder.add_edge(START, "normalize")

        if enable_qa_scaffold:
            # Normalize → QA_Scaffold → NER
            builder.add_edge("normalize", "qa_scaffold")
            builder.add_conditional_edges("qa_scaffold", route_after_qa_scaffold)
            logger.info(f"[Workflow] 启用 Normalize + QA Scaffold 模式")
        else:
            # Normalize → NER
            builder.add_edge("normalize", "ner")
            logger.info(f"[Workflow] 启用 Normalize 归一化节点")

    elif enable_qa_scaffold:
        # QA Scaffold alone: START → QA_Scaffold → NER
        builder.add_edge(START, "qa_scaffold")
        builder.add_conditional_edges("qa_scaffold", route_after_qa_scaffold)
        logger.info(f"[Workflow] 启用 QA Scaffold 节点")

    else:
        # 无前置节点时，直接从 START 到 NER
        builder.add_edge(START, "ner")

    # P4+P8改进：Self-Check + 反思循环模式（可与 QA Scaffold 组合）
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
                END: END,         # 错误时结束
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
                END: END,         # 错误时结束
            }
        )

        # Eval → Label → END
        builder.add_edge("eval", "label")
        builder.add_edge("label", END)

        if enable_qa_scaffold:
            logger.info(f"[Workflow] 启用 QA Scaffold + Self-Check + 反思循环模式")
        else:
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

    # P7改进：并发控制 - 防止API限流（从配置读取）
    max_concurrent = config.max_concurrent_corpus

    # Worker处理函数
    async def workers_node(state: KGState) -> Dict:
        """并行执行所有Worker - 按分片并行处理，带并发控制

        P10改进：支持批量LLM调用模式
        - 如果 enable_batch_llm=True，每个Worker内部使用批量处理
        - 每次LLM调用处理 batch_llm_size 条语料
        - 失败时自动退化为单条处理（fallback）
        """
        # 创建并发控制信号量
        semaphore = asyncio.Semaphore(max_concurrent)

        # P10新增：读取批量处理配置
        batch_llm_size = config.batch_llm_size
        enable_batch_llm = config.enable_batch_llm
        batch_llm_fallback = config.batch_llm_fallback

        if enable_batch_llm:
            logger.info(f"[Workers] 批量LLM模式: 每次处理 {batch_llm_size} 条语料")
        else:
            logger.info(f"[Workers] 单条处理模式: 最大 {max_concurrent} 条语料同时处理")

        async def process_corpus(corpus: Dict) -> Dict:
            """处理单条语料（原有逻辑，用于fallback）"""
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
                    # P8改进：QA Scaffold 脚手架初始状态
                    "qa_scaffold_result": {},
                    "semantic_summary": "",
                    "qa_entity_hints": [],
                    "qa_relation_hints": [],
                    "qa_context_dependencies": [],
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
                    "retry_suggested": False,
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

        # P10新增：批量处理函数
        async def process_corpus_batch(corpus_list: List[Dict]) -> List[Dict]:
            """
            批量处理语料（一次LLM调用处理多条）

            Args:
                corpus_list: 语料列表（batch_llm_size条）

            Returns:
                处理结果列表
            """
            from .nodes import (
                create_batch_joint_extraction_node,
                create_batch_self_check_node,
                process_corpus_batch_with_llm,
            )

            try:
                # 使用批量处理
                batch_result = await process_corpus_batch_with_llm(
                    llm, corpus_list, config,
                )

                results = []

                # 转换批量结果为单条结果格式
                for corpus_id, data in batch_result["batch_results"].items():
                    # 构建兼容的结果格式
                    result = {
                        "corpus_id": corpus_id,
                        "entities": data.get("entities", {"道路": [], "POI": [], "建筑物": [], "街区": []}),
                        "triples": data.get("triples", []),
                        "corrected_triples": data.get("triples", []),
                        "eval_passed": True,
                        "entity_attrs": {},
                        "relation_attrs": {},
                        "verification_confidence": data.get("confidence", "medium"),
                        "batch_processed": True,  # 标记为批量处理
                        "cross_corpus_aliases": batch_result["cross_corpus_aliases"],
                    }
                    results.append(result)

                # 处理fallback的语料（单条处理）
                if batch_llm_fallback and batch_result["needs_single_processing"]:
                    fallback_corpus = batch_result["fallback_corpus_list"]
                    logger.info(f"[Batch] Fallback处理 {len(fallback_corpus)} 条语料")
                    for corpus in fallback_corpus:
                        single_result = await process_corpus(corpus)
                        results.append(single_result)

                return results

            except Exception as e:
                logger.error(f"批量处理失败: {e}")

                # 如果启用fallback，退化为单条处理
                if batch_llm_fallback:
                    logger.warning(f"[Batch] 退化为单条处理")
                    results = []
                    for corpus in corpus_list:
                        single_result = await process_corpus(corpus)
                        results.append(single_result)
                    return results
                else:
                    # 返回错误结果
                    return [
                        {
                            "corpus_id": corpus.get("id", "unknown"),
                            "error": str(e),
                        }
                        for corpus in corpus_list
                    ]

        async def process_partition(worker_id: str, corpus_list: List[Dict]) -> Dict:
            """处理单个分片（Worker级别）- P10改进：支持批量LLM调用"""
            start_time = time.time()

            results = []

            if enable_batch_llm and len(corpus_list) > 0:
                # P10改进：批量处理模式
                # 将语料分成多个batch_llm_size的批次
                for i in range(0, len(corpus_list), batch_llm_size):
                    batch_corpus = corpus_list[i:i + batch_llm_size]

                    async with semaphore:  # 获取"许可证"
                        batch_results = await process_corpus_batch(batch_corpus)
                        results.extend(batch_results)

                        logger.debug(f"[{worker_id}] Batch {i//batch_llm_size + 1}: {len(batch_results)} 条结果")
            else:
                # 原有模式：单条并行处理
                async def process_corpus_with_limit(corpus: Dict) -> Dict:
                    """带并发限制的语料处理"""
                    async with semaphore:  # 获取"许可证"，超过限制则排队等待
                        return await process_corpus(corpus)

                tasks = [process_corpus_with_limit(corpus) for corpus in corpus_list]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            # 分离成功和失败的结果
            success_results = []
            errors = []
            for result in results:
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
        neo4j_stats = {"merged_entities": 0, "merged_relations": 0}
        postgres_stats = {"inserted": 0}
        error_message = None  # 错误标记

        try:
            # 尝试导入数据库客户端（支持直接运行和模块运行）
            try:
                # 模块运行模式
                from kg.neo4j_client import Neo4jClient
                from kg.postgres_client import PostgresClient
            except ImportError:
                # 直接运行模式，尝试相对导入
                try:
                    from ...kg.neo4j_client import Neo4jClient
                    from ...kg.postgres_client import PostgresClient
                except ImportError:
                    logger.warning("[Finalizer] 数据库模块未找到，跳过数据库写入")
                    return {
                        "neo4j_stats": neo4j_stats,
                        "postgres_stats": postgres_stats,
                        "error": "数据库模块未找到",
                        "current_phase": PhaseEnum.FINALIZE,
                        "end_time": time.time(),
                    }

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

                # 记录 Neo4j 同步状态
                pg.update_neo4j_sync_status(state["batch_id"], neo4j_stats["merged_entities"] > 0 or neo4j_stats["merged_relations"] > 0)

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

                # 更新批次状态为成功
                pg.update_batch_status(state["batch_id"], "completed")

            logger.info(f"[Finalizer] 数据库写入成功 - Neo4j: {neo4j_stats}, PostgreSQL: {postgres_stats}")

        except Exception as e:
            error_message = str(e)
            logger.error(f"[Finalizer] 数据库写入失败: {e}")
            # 尝试更新批次状态为失败并记录错误信息
            try:
                db_config = _get_database_config()
                with PostgresClient(
                    db_config["pg_host"],
                    db_config["pg_port"],
                    db_config["pg_database"],
                    db_config["pg_user"],
                    db_config["pg_password"]
                ) as pg:
                    pg.update_batch_status_with_error(state["batch_id"], "failed", error_message)
                    # 检查 Neo4j 是否已同步但 PostgreSQL 失败（需要补偿）
                    batch_status = pg.get_batch_status(state["batch_id"])
                    if batch_status and batch_status.get("neo4j_sync"):
                        logger.warning(f"[Finalizer] Neo4j已同步但PostgreSQL失败，需要人工检查数据一致性")
                    logger.info(f"[Finalizer] 批次状态已更新为 failed，错误: {error_message[:100]}")
            except Exception as inner_e:
                logger.error(f"[Finalizer] 无法更新批次失败状态: {inner_e}")

        return {
            "neo4j_stats": neo4j_stats,
            "postgres_stats": postgres_stats,
            "error": error_message,  # 返回错误信息（None表示成功）
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
        # P8改进：QA Scaffold 脚手架初始状态
        "qa_scaffold_result": {},
        "semantic_summary": "",
        "qa_entity_hints": [],
        "qa_relation_hints": [],
        "qa_context_dependencies": [],
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
        # P8改进：QA Scaffold 脚手架初始状态
        "qa_scaffold_result": {},
        "semantic_summary": "",
        "qa_entity_hints": [],
        "qa_relation_hints": [],
        "qa_context_dependencies": [],
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
        "error": None,  # 初始无错误
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


# ===== P7新增：分批次处理入口 =====

async def process_corpus_in_batches(
    llm: Any,
    config: Optional[ExtractionConfig] = None,
    batch_size: Optional[int] = None,
    total_limit: Optional[int] = None,
    table_name: str = "xiaohongshu_notes",
    text_column: str = "desc_cleaned",
    id_column: str = "note_id",
    where_clause: Optional[str] = None,
    dry_run: bool = False
) -> Dict:
    """
    分批次从数据库读取语料并处理

    每次读取 batch_size 条语料，处理完成后继续下一批，
    直到全部处理完成或达到 total_limit。

    Args:
        llm: LangChain LLM 实例
        config: ExtractionConfig 配置实例
        batch_size: 每批次处理数量（默认从config读取，默认100）
        total_limit: 总处理数量限制（可选，用于测试）
        table_name: 数据表名
        text_column: 文本列名
        id_column: ID列名
        where_clause: 可选WHERE条件
        dry_run: 是否只测试不写入数据库

    Returns:
        处理统计 {"total_processed": int, "total_entities": int, "total_triples": int, "batches": list}
    """
    config = config or DEFAULT_CONFIG
    batch_size = batch_size or config.batch_size  # 使用配置默认值
    db_config = _get_database_config()

    try:
        from kg.postgres_client import PostgresClient
    except ImportError:
        from ...kg.postgres_client import PostgresClient

    stats = {
        "total_processed": 0,
        "total_entities": 0,
        "total_triples": 0,
        "batches": [],
        "errors": []
    }

    offset = 0
    batch_num = 0

    # 连接数据库
    with PostgresClient(
        db_config["pg_host"],
        db_config["pg_port"],
        db_config["pg_database"],
        db_config["pg_user"],
        db_config["pg_password"]
    ) as pg:
        # 获取总数
        total_count = pg.count_corpus_for_extraction(table_name, text_column, where_clause)
        if total_limit:
            total_count = min(total_count, total_limit)

        logger.info(f"[分批处理] 总语料数: {total_count}, 每批: {batch_size}")

        while offset < total_count:
            batch_num += 1
            current_batch_size = min(batch_size, total_count - offset)

            logger.info(f"[批次 {batch_num}] 读取语料 offset={offset}, limit={current_batch_size}")

            # 读取一批语料
            corpus_list = pg.fetch_corpus_for_extraction(
                table_name=table_name,
                text_column=text_column,
                id_column=id_column,
                limit=current_batch_size,
                offset=offset,
                where_clause=where_clause
            )

            if not corpus_list:
                logger.info(f"[批次 {batch_num}] 无有效语料，跳过")
                offset += current_batch_size
                continue

            # 处理这一批
            try:
                if dry_run:
                    # 测试模式：不写入数据库，只跑工作流
                    result = await process_batch(llm, corpus_list, config)
                else:
                    # 正常模式：完整处理
                    result = await process_batch(llm, corpus_list, config)

                batch_entities = len(result.get("aggregated_entities", []))
                batch_triples = len(result.get("aggregated_triples", []))

                stats["total_processed"] += len(corpus_list)
                stats["total_entities"] += batch_entities
                stats["total_triples"] += batch_triples
                stats["batches"].append({
                    "batch_num": batch_num,
                    "corpus_count": len(corpus_list),
                    "entities": batch_entities,
                    "triples": batch_triples
                })

                logger.info(
                    f"[批次 {batch_num}] 完成: {len(corpus_list)} 条语料, "
                    f"{batch_entities} 实体, {batch_triples} 三元组"
                )

            except Exception as e:
                logger.error(f"[批次 {batch_num}] 处理失败: {e}")
                stats["errors"].append({"batch_num": batch_num, "error": str(e)})

            # 下一批
            offset += current_batch_size

            # 显示进度
            progress = min(offset / total_count * 100, 100)
            logger.info(f"[进度] {progress:.1f}% ({offset}/{total_count})")

    logger.info(
        f"[完成] 总处理: {stats['total_processed']} 条, "
        f"总实体: {stats['total_entities']}, "
        f"总三元组: {stats['total_triples']}"
    )

    return stats


# ===== P10新增：QA导师工作流 =====

def route_after_qa_mentor(state: CorpusState) -> str:
    """QA导师后路由"""
    if state.get("error"):
        return "joint_ner_re"

    qa_result = state.get("qa_scaffold_result", {})
    should_skip = qa_result.get("should_skip_detailed_extraction", False)

    if should_skip:
        logger.info(f"[QA_Mentor-Route] 建议跳过详细抽取")
        return END
    else:
        logger.info(f"[QA_Mentor-Route] 继续到联合抽取")
        return "joint_ner_re"


def route_after_qa_approval(state: CorpusState) -> str:
    """QA审批后路由 - 决定是否进入修改循环"""

    if state.get("error"):
        return END

    revision_cycle_count = state.get("revision_cycle_count", 0)
    max_revision_cycles = state.get("max_revision_cycles", 3)

    approval_result = state.get("qa_approval_result", {})
    overall_status = approval_result.get("overall_status", "approved")
    retry_suggested = approval_result.get("retry_suggested", False)
    retry_target_nodes = approval_result.get("retry_target_nodes", [])

    # 达到最大修改轮次，强制结束
    if revision_cycle_count >= max_revision_cycles:
        logger.warning(f"[QA_Approval-Route] 达到最大修改轮次 {revision_cycle_count}/{max_revision_cycles}")
        return END

    # 审批通过，结束
    if overall_status == "approved" and not retry_suggested:
        logger.info(f"[QA_Approval-Route] 审批通过")
        return END

    # 需要修改
    if retry_suggested and retry_target_nodes:
        target_node = retry_target_nodes[0]  # 取第一个需要修改的节点
        logger.info(f"[QA_Approval-Route] 需要修改: {target_node}")

        if target_node == "joint_ner_re":
            return "revision_joint"
        elif target_node == "eval":
            return "eval"  # 重新评估
        elif target_node == "label":
            return "label"  # 重新标注

    # 默认结束
    return END


def route_after_revision_joint(state: CorpusState) -> str:
    """修改联合抽取后路由"""
    if state.get("error"):
        return "eval"

    # 修改后需要重新评估
    logger.info(f"[Revision_Joint-Route] 继续到评估")
    return "eval"


def build_qa_mentor_workflow(
    qa_llm: Any,
    worker_llm: Any,
    config: ExtractionConfig
) -> CompiledStateGraph:
    """
    构建QA导师模式工作流

    流程：
    START → Filter → Normalize → QA_Mentor → Joint_NER_RE → Eval → Label → QA_Approval
            ↑                                                    ↓
            └──────────── Revision Loop (if needed) ─────────────┘

    Args:
        qa_llm: QA导师使用的LLM（如DeepSeek Reasoner）
        worker_llm: 后续节点使用的LLM（如DeepSeek Chat）
        config: 配置实例

    Returns:
        CompiledStateGraph
    """
    builder = StateGraph(CorpusState)

    # 创建节点
    qa_mentor_node = create_qa_mentor_node(qa_llm, config)
    qa_approval_node = create_qa_approval_node(qa_llm, config)
    joint_ner_re_node = create_joint_ner_re_node(worker_llm)
    eval_node = create_eval_simplified_node(worker_llm)
    label_node = create_label_node(worker_llm)
    revision_joint_node = create_revision_joint_node(worker_llm)

    # 添加节点
    builder.add_node("qa_mentor", qa_mentor_node, retry_policy=LLM_RETRY_POLICY)
    builder.add_node("joint_ner_re", joint_ner_re_node, retry_policy=LLM_RETRY_POLICY)
    builder.add_node("eval", eval_node, retry_policy=LLM_RETRY_POLICY)
    builder.add_node("label", label_node, retry_policy=LLM_RETRY_POLICY)
    builder.add_node("qa_approval", qa_approval_node, retry_policy=LLM_RETRY_POLICY)
    builder.add_node("revision_joint", revision_joint_node, retry_policy=LLM_RETRY_POLICY)

    # 前置节点（可选）- 支持同时启用Filter和Normalize
    if config.enable_filter:
        filter_node = create_filter_node(worker_llm)
        builder.add_node("filter", filter_node, retry_policy=LLM_RETRY_POLICY)
        builder.add_edge(START, "filter")
        # 根据是否启用normalize选择路由
        if config.enable_normalize:
            builder.add_conditional_edges("filter", route_after_filter_to_normalize)
        else:
            builder.add_conditional_edges("filter", route_after_filter_to_joint)

    if config.enable_normalize:
        normalize_node = create_normalize_node(worker_llm)
        builder.add_node("normalize", normalize_node, retry_policy=LLM_RETRY_POLICY)
        # 如果没有filter，normalize是起点；如果有filter，filter会路由到normalize
        if not config.enable_filter:
            builder.add_edge(START, "normalize")
        builder.add_edge("normalize", "qa_mentor")

    # 如果没有前置节点，直接从START到qa_mentor
    if not config.enable_filter and not config.enable_normalize:
        builder.add_edge(START, "qa_mentor")

    # QA导师 → 联合抽取
    builder.add_conditional_edges(
        "qa_mentor",
        route_after_qa_mentor,
        {"joint_ner_re": "joint_ner_re", END: END}
    )

    # 联合抽取 → 评估
    builder.add_edge("joint_ner_re", "eval")

    # 评估 → 标注
    builder.add_edge("eval", "label")

    # 标注 → QA审批
    builder.add_edge("label", "qa_approval")

    # QA审批 → 修改循环或结束
    builder.add_conditional_edges(
        "qa_approval",
        route_after_qa_approval,
        {
            "revision_joint": "revision_joint",
            "eval": "eval",
            "label": "label",
            END: END,
        }
    )

    # 修改联合抽取 → 评估
    builder.add_conditional_edges(
        "revision_joint",
        route_after_revision_joint,
        {"eval": "eval"}
    )

    logger.info("[Workflow] QA导师模式工作流构建完成")

    return builder.compile(checkpointer=InMemorySaver())


async def process_corpus_with_qa_mentor(
    qa_llm: Any,
    worker_llm: Any,
    corpus: Dict,
    config: ExtractionConfig
) -> CorpusState:
    """
    使用QA导师模式处理单条语料

    Args:
        qa_llm: QA导师LLM
        worker_llm: 工作节点LLM
        corpus: 语料字典 {"id": "...", "text": "..."}
        config: 配置实例

    Returns:
        处理结果
    """
    workflow = build_qa_mentor_workflow(qa_llm, worker_llm, config)

    corpus_id = _validate_corpus_id(corpus.get("id"))
    raw_text = _validate_corpus_text(corpus.get("text", ""), config)

    initial_state: CorpusState = {
        "corpus_id": corpus_id,
        "raw_text": raw_text,
        "_config_enable_normalize": config.enable_normalize,
        "_config_enable_qa_scaffold": config.enable_qa_scaffold,
        "filter_result": {},
        "normalize_result": {},
        "normalized_text": "",
        "qa_scaffold_result": {},
        "semantic_summary": "",
        "qa_entity_hints": [],
        "qa_relation_hints": [],
        "qa_context_dependencies": [],
        "mentor_guidance": {},
        "qa_approval_result": {},
        "integrated_semantic_summary": "",
        "revision_feedbacks": [],
        "revision_cycle_count": 0,
        "max_revision_cycles": config.max_revision_cycles,
        "pending_approval_nodes": [],
        "reasoning_trace": "",
        "joint_extraction_result": {},
        "extraction_strategy": "",
        "self_check_filter_result": {},
        "self_check_normalize_result": {},
        "self_check_qa_result": {},
        "self_check_joint_result": {},
        "self_check_eval_result": {},
        "self_check_label_result": {},
        "reflection_text": "",
        "improvement_strategy": "",
        "reflection_history": [],
        "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
        "triples": [],
        "eval_scores": [],
        "eval_passed": False,
        "corrected_triples": [],
        "self_check_ner_result": {},
        "self_check_re_result": {},
        "final_entities": [],
        "final_triples": [],
        "verification_confidence": "medium",
        "retry_count": 0,
        "max_retries": DEFAULT_MAX_RETRIES,
        "retry_reason": "",
        "retry_suggested": False,
        "problem_entities": [],
        "problem_triples": [],
        "needs_review": False,
        "entity_attrs": {},
        "relation_attrs": {},
        "current_step": StepEnum.QA_MENTOR,
        "error": None,
    }

    thread_config = {"configurable": {"thread_id": f"qa_mentor_{corpus_id}_{uuid.uuid4().hex[:8]}"}}
    result = await workflow.ainvoke(initial_state, thread_config)
    return cast(CorpusState, result)