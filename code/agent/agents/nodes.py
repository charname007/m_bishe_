"""
LangGraph节点函数 - 四步骤工作流节点
使用 LangChain PydanticOutputParser 进行结构化输出（兼容 DeepSeek API）
P2改进：简化评估节点，单次评估+规则校验
P3改进：支持 StreamWriter 流式输出
P5改进：添加 Filter 筛选节点
P6改进：添加 Normalize 归一化节点，NER/RE/Eval 节点优先使用归一化文本
"""

import asyncio
import json
import math
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Any, Optional, Union

from langchain_core.output_parsers import PydanticOutputParser
from langgraph.types import StreamWriter

from loguru import logger

from .state import (
    CorpusState,
    KGState,
    PhaseEnum,
    StepEnum,
    DEFAULT_MAX_RETRIES,
    RELATION_TYPES,
    DEFAULT_ENTITY_DICT,
    create_default_corpus_state,
)
from .config import ExtractionConfig


# ===== P15新增：枚举值提取工具函数 =====


def extract_enum_value(enum_or_value: Any) -> Any:
    """
    从 Enum 或原始值中提取实际值

    Args:
        enum_or_value: 可能是 Enum 实例或原始值

    Returns:
        提取的值（如果是 Enum 则返回 .value，否则返回原值）

    Example:
        >>> from .schemas import RelationTypeEnum
        >>> extract_enum_value(RelationTypeEnum.LOCATED)
        '位于'
        >>> extract_enum_value('位于')
        '位于'
    """
    if hasattr(enum_or_value, "value"):
        return enum_or_value.value
    return enum_or_value


def extract_enum_values_from_list(items: List[Any]) -> List[Any]:
    """
    从列表中批量提取枚举值

    Args:
        items: 可能包含 Enum 实例的列表

    Returns:
        提取值后的列表
    """
    return [extract_enum_value(item) for item in items]


# ===== P16新增：JSON预处理函数 - 处理LLM输出的未转义引号 =====


def escape_unescaped_quotes_in_json_strings(text: str) -> str:
    """
    转义JSON字符串值内部的未转义引号

    解决LLM在JSON字符串值中使用未转义引号导致的JSON解析失败问题。
    例如："text with "quote" inside" -> "text with \"quote\" inside"

    Args:
        text: LLM返回的JSON文本（可能包含markdown代码块）

    Returns:
        修复后的JSON文本，字符串值内部的引号已转义

    状态机逻辑:
    - 追踪字符串边界（使用英文双引号作为边界）
    - 字符串内部遇到引号时，判断是内容引号还是结束引号：
      - 内容引号：后面跟着文本字符（字母、数字、中文等） -> 需要转义
      - 结束引号：后面跟着逗号、冒号、括号、空白符 -> 不转义
    """
    # 先移除markdown代码块标记（如果有）
    if "```" in text:
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```", "", text)

    text = text.strip()

    result = []
    in_string = False
    i = 0

    while i < len(text):
        char = text[i]

        if char == '"':
            if not in_string:
                # 开始一个JSON字符串 - 这是结构引号
                in_string = True
                result.append(char)
                i += 1
            else:
                # 在字符串内部 - 判断这是内容引号还是结束引号
                if i + 1 >= len(text):
                    # 文本结束 - 这是结束引号
                    in_string = False
                    result.append(char)
                    i += 1
                else:
                    next_char = text[i + 1]

                    # 检查是否看起来像结束引号
                    # 结束引号后面通常是：逗号、冒号、方括号、花括号、换行、空白
                    if next_char in [",", ":", "]", "}", "\n", "\r", " ", "\t"]:
                        # 这是结束引号
                        in_string = False
                        result.append(char)
                        i += 1
                    else:
                        # 这看起来是字符串内部的内容引号
                        # 检查是否已转义
                        if i > 0 and text[i - 1] == "\\":
                            # 已转义
                            result.append(char)
                            i += 1
                        else:
                            # 需要转义
                            result.append("\\")
                            result.append(char)
                            i += 1
        elif char == "\\" and i + 1 < len(text):
            # 处理转义序列
            next_char = text[i + 1]
            if next_char == '"':
                # 已转义的引号
                result.append(char)
                result.append(next_char)
                i += 2
            elif next_char in ["n", "r", "t", "\\", "/"]:
                # 其他转义序列
                result.append(char)
                result.append(next_char)
                i += 2
            else:
                result.append(char)
                i += 1
        elif char in ["\n", "\r"] and in_string:
            # JSON字符串内部的换行符需要处理
            # JSON规范不允许字符串内有未转义的换行符
            result.append(" ")
            i += 1
        else:
            result.append(char)
            i += 1

    return "".join(result)


def safe_parse_json_with_quote_fix(parser: PydanticOutputParser, text: str) -> Any:
    """
    安全的JSON解析函数，自动处理LLM输出中的常见问题

    P16新增：增加枚举值修复和字段缺失填充

    Args:
        parser: PydanticOutputParser实例
        text: LLM返回的文本

    Returns:
        解析后的Pydantic模型实例
    """
    # 先尝试正常解析
    try:
        return parser.parse(text)
    except Exception as e:
        logger.debug(f"[JSON-Preprocess] Initial parse failed: {e}, applying fixes...")

        # Step 1: 引号转义修复
        fixed_text = escape_unescaped_quotes_in_json_strings(text)

        try:
            return parser.parse(fixed_text)
        except Exception as e2:
            logger.debug(f"[JSON-Preprocess] Quote fix failed, trying content fix...")

            # Step 2: 内容修复（枚举值映射 + 字段缺失填充）
            fixed_text = fix_llm_json_content(fixed_text)

            try:
                return parser.parse(fixed_text)
            except Exception as e3:
                logger.warning(f"[JSON-Preprocess] All fixes failed: {e3}")
                raise e3


def parse_batch_extraction_lenient(text: str) -> Dict[str, Any]:
    """
    批量抽取宽松解析：
    1) 尽量解析出JSON
    2) 对 results/full_entities/triples 做容错清洗
    3) 返回可被后续流程消费的字典
    """
    # 去掉代码块包裹
    cleaned = re.sub(r"```(?:json)?\s*", "", text or "")
    cleaned = re.sub(r"\s*```", "", cleaned).strip()

    try:
        payload = json.loads(cleaned)
    except Exception:
        # 再走一次通用修复
        payload = json.loads(fix_llm_json_content(cleaned))

    if not isinstance(payload, dict):
        return {"results": [], "cross_corpus_aliases": [], "cross_corpus_relations": []}

    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []

    normalized_results = []
    for item in results:
        if not isinstance(item, dict):
            continue

        corpus_id = str(item.get("corpus_id", "")).strip()
        if not corpus_id:
            continue

        entities = item.get("entities", {})
        if not isinstance(entities, dict):
            entities = {}

        # 清洗full_entities（尽量保留）
        full_entities = item.get("full_entities", [])
        if not isinstance(full_entities, list):
            full_entities = []
        cleaned_entities = []
        for e in full_entities:
            if not isinstance(e, dict):
                continue
            name = str(e.get("name", "")).strip()
            if not name:
                continue
            e_type = e.get("type", "POI")
            evidence = e.get("evidence") or name
            cleaned_entities.append({**e, "name": name, "type": e_type, "evidence": evidence})

        triples = item.get("triples", [])
        if not isinstance(triples, list):
            triples = []

        confidence = item.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        normalized_results.append(
            {
                "corpus_id": corpus_id,
                "entities": entities,
                "full_entities": cleaned_entities,
                "triples": triples,
                "confidence": confidence,
                "has_geo_info": bool(item.get("has_geo_info", True)),
                "skip_reason": item.get("skip_reason"),
            }
        )

    aliases = payload.get("cross_corpus_aliases", [])
    if not isinstance(aliases, list):
        aliases = []
    relations = payload.get("cross_corpus_relations", [])
    if not isinstance(relations, list):
        relations = []

    return {
        "results": normalized_results,
        "cross_corpus_aliases": aliases,
        "cross_corpus_relations": relations,
        "overall_confidence": payload.get("overall_confidence", "medium"),
    }


# P16新增：无效枚举值映射表
INVALID_ENUM_MAPPING = {
    # 事件类别修复：休闲活动类 → 人文事件
    "休闲活动": "人文事件",
    "娱乐活动": "人文事件",
    "体育活动": "人文事件",
    "文化活动": "人文事件",
    # 功能类型修复（如有）
    "娱乐": "休闲",
    "运动": "休闲",
}


def fix_llm_json_content(text: str) -> str:
    """
    修复LLM输出的JSON内容中的常见问题

    P16新增：处理两类问题
    1. 无效枚举值：映射到有效值
    2. 字段缺失：填充默认值（仅针对三元组的 evidence 和 confidence）

    Args:
        text: LLM返回的JSON文本

    Returns:
        修复后的JSON文本
    """
    import json
    import re

    # 修复无效枚举值（直接字符串替换）
    for invalid, valid in INVALID_ENUM_MAPPING.items():
        # 匹配 JSON 中的枚举值字段
        # 例如: "事件类别": "休闲活动" → "事件类别": "人文事件"
        pattern = rf'"[^"]*类别[^"]*"\s*:\s*"{invalid}"'
        replacement = f'"事件类别": "{valid}"'
        text = re.sub(pattern, replacement, text)

    # 修复三元组缺失字段
    # 匹配不完整的三元组: {"head": "...", "relation": "...", "tail": "..."}
    # 添加缺失的 evidence 和 confidence
    def fix_triple(match):
        triple_str = match.group(0)
        try:
            triple = json.loads(triple_str)
            # 检查缺失字段
            if "evidence" not in triple:
                triple["evidence"] = triple.get("head", "") + "具有功能"
            if "confidence" not in triple:
                triple["confidence"] = "medium"
            if "attributes" not in triple:
                triple["attributes"] = {}
            return json.dumps(triple, ensure_ascii=False)
        except:
            return triple_str

    # 匹配不完整的三元组（没有 evidence/confidence 的）
    # 注意：需要匹配那些只有 head/relation/tail 但缺少其他字段的
    triple_pattern = (
        r'\{"head":\s*"[^"]+",\s*"relation":\s*"[^"]+",\s*"tail":\s*"[^"]+"\}'
    )
    text = re.sub(triple_pattern, fix_triple, text)

    logger.info(f"[JSON-ContentFix] Applied enum mapping and field filling")
    return text


# ===== P6改进：辅助函数 - 获取处理文本 =====
# P15修复：统一使用 node_template.py 中的公开版本，避免重复定义
from .node_template import get_text_for_processing

from .schemas import (
    FilterResult,  # P5新增
    NormalizeResult,  # P6新增
    QAScaffoldResult,  # P8新增
    EntityRecognitionResult,
    RelationExtractionResult,
    EvalResultFirst,
    EvalResultSecond,
    EvalResultSimplified,
    LabelResult,
    SelfCheckNERResult,
    SelfCheckREResult,
    # P9新增：联合抽取和所有Self-Check模型
    JointEntity,
    JointTriple,
    JointExtractionResult,
    SelfCheckJointResult,
    SelfCheckQAResult,
    SelfCheckEvalResult,
    SelfCheckLabelResult,
    # P12新增：Self-Check增强版模型
    SelfCheckJointResultV2,
    # v3.4新增：关系属性映射常量
    RELATION_ATTRS_MAP,
    BatchMentorQueryResponse,
)
from .prompts import (
    FILTER_PROMPT,  # P5新增
    NORMALIZE_PROMPT,  # P6新增
    QA_SCAFFOLD_PROMPT,  # P8新增
    NER_PROMPT,
    RE_PROMPT,
    EVAL_PROMPT_1,
    EVAL_PROMPT_2,
    EVAL_PROMPT_SIMPLIFIED,
    LABEL_PROMPT,
    SELF_CHECK_NER_PROMPT,
    SELF_CHECK_RE_PROMPT,
    format_entities,
    format_triples,
    format_verified_entities,
    format_retry_hint,
    format_entity_hints,
    format_relation_hints,
    format_context_dependencies,  # P8新增
    # P9新增：联合抽取和所有Self-Check提示词
    JOINT_NER_RE_PROMPT,
    JOINT_NER_RE_PROMPT_V2,  # P12新增：改进版提示词
    SELF_CHECK_JOINT_PROMPT,
    SELF_CHECK_JOINT_PROMPT_V2,  # P12新增：改进版提示词
    SELF_CHECK_QA_PROMPT,
    SELF_CHECK_EVAL_PROMPT,
    SELF_CHECK_LABEL_PROMPT,
    format_joint_entities,
    format_joint_triples,
    format_qa_pairs_for_check,
    format_eval_scores_for_check,
    format_reflection_history,
    # P10新增：QA导师提示词和格式化函数
    QA_MENTOR_PROMPT,
    QA_APPROVAL_PROMPT,
    REVISION_JOINT_PROMPT,
    format_mentor_guidance,
    format_feedbacks_for_revision,
    format_feedback_summary,
    format_joint_for_approval,
    format_eval_for_approval,
    format_label_for_approval,
    format_revision_feedbacks,
    format_reflection_for_approval,
    # P14新增：导师查询提示词
    MENTOR_QUERY_PROMPT,
    BATCH_MENTOR_QUERY_PROMPT,
    # P11新增：实体对齐提示词和格式化函数
    ENTITY_ALIGNMENT_PROMPT,
    BATCH_ENTITY_ALIGNMENT_DECISION_PROMPT,
    format_alignment_candidates,
    format_alignment_result_for_output,
    # P12新增：四维度评分格式化函数
    format_dimension_scores,
    format_improvement_strategy,
    # P13新增：优化版提示词（RISEN/CARE/TIDD-EC框架）
    JOINT_NER_RE_PROMPT_V3,
    FILTER_PROMPT_V2,
    RE_PROMPT_V2,
    LABEL_PROMPT_V2,
    SELF_CHECK_JOINT_PROMPT_V3,
    assemble_optimized_joint_prompt,
)


# ===== Filter 筛选节点（P5新增） =====


def create_filter_node(llm: Any):
    """
    创建 Filter 筛选节点

    职责：
    1. 快速判断文本是否包含有价值的地理信息
    2. 筛选无效文本以节省后续处理成本
    3. 输出筛选结果和置信度
    4. 判断是否为武汉地区（P9改进）
    """
    parser = PydanticOutputParser(pydantic_object=FilterResult)

    async def filter_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Step 0: 文本筛选"""
        corpus_id = state["corpus_id"]
        logger.info(f"[Filter] 筛选语料: {corpus_id}")

        # 发送进度事件
        writer(
            {
                "step": "filter",
                "corpus_id": corpus_id,
                "status": "started",
                "message": "开始文本筛选",
            }
        )

        try:
            # 调用 LLM 进行筛选判断（使用 OutputParser）
            prompt_text = FILTER_PROMPT.invoke({"raw_text": state["raw_text"]})
            # 添加格式化指令
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: FilterResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            logger.info(
                f"[Filter] 结果: is_valid={result.is_valid}, "
                f"confidence={result.confidence}, "
                f"skip_reason={result.skip_reason}, "
                f"is_non_wuhan_region={result.is_non_wuhan_region}, "
                f"region_hint={result.region_hint}"
            )

            # 发送完成事件（包含新字段）
            writer(
                {
                    "step": "filter",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "is_valid": result.is_valid,
                    "confidence": result.confidence,
                    "skip_reason": result.skip_reason,
                    "has_geo_entity": result.has_geo_entity,
                    "has_spatial_relation": result.has_spatial_relation,
                    "geo_entity_hint": result.geo_entity_hint,
                    "is_non_wuhan_region": result.is_non_wuhan_region,
                    "region_hint": result.region_hint,
                }
            )

            # 根据筛选结果决定下一步
            if result.is_valid:
                next_step = StepEnum.NER
            else:
                next_step = StepEnum.DONE  # 无效文本直接结束

            return {
                "filter_result": result.model_dump(),
                "current_step": next_step,
            }

        except Exception as e:
            logger.error(f"[Filter] 失败: {e}")
            writer(
                {
                    "step": "filter",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            # 筛选失败时默认继续处理（保守策略）
            return {
                "filter_result": {
                    "is_valid": True,
                    "confidence": "low",
                    "skip_reason": None,
                    "has_geo_entity": False,
                    "has_spatial_relation": False,
                    "geo_entity_hint": None,
                    "is_non_wuhan_region": False,  # 无法确定时默认放行
                    "region_hint": "未知",
                },
                "error": str(e),
                "current_step": StepEnum.NER,
            }

    return filter_node


# ===== Normalize 归一化节点（P6新增） =====


def create_normalize_node(llm: Any):
    """
    创建 Normalize 归一化节点

    职责：
    1. 消解省略主语和模糊指代
    2. 归一化别名简称（如"武大"→"武汉大学"）
    3. 将口语化文本改写为标准句式
    4. 严格保留原文语义，不添加新信息
    """
    parser = PydanticOutputParser(pydantic_object=NormalizeResult)

    async def normalize_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Step 0.5: 文本归一化"""
        corpus_id = state["corpus_id"]
        logger.info(f"[Normalize] 归一化语料: {corpus_id}")

        # 发送进度事件
        writer(
            {
                "step": "normalize",
                "corpus_id": corpus_id,
                "status": "started",
                "message": "开始文本归一化",
            }
        )

        try:
            # 使用原始文本进行归一化（使用 OutputParser）
            raw_text = state["raw_text"]
            prompt_text = NORMALIZE_PROMPT.invoke({"raw_text": raw_text})
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: NormalizeResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            logger.info(
                f"[Normalize] 结果: has_changes={result.has_changes}, "
                f"confidence={result.confidence}, "
                f"normalizations={len(result.normalizations)}条"
            )

            # 发送完成事件
            writer(
                {
                    "step": "normalize",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "normalized_text": result.normalized_text,
                    "normalizations_count": len(result.normalizations),
                    "confidence": result.confidence,
                    "has_changes": result.has_changes,
                }
            )

            # 输出归一化结果
            # normalized_text 供后续 NER/RE 节点使用
            return {
                "normalize_result": result.model_dump(),
                "normalized_text": result.normalized_text,
                "current_step": StepEnum.NER,
            }

        except Exception as e:
            logger.error(f"[Normalize] 失败: {e}")
            writer(
                {
                    "step": "normalize",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            # 归一化失败时使用原始文本继续处理（保守策略）
            return {
                "normalize_result": {
                    "normalized_text": state["raw_text"],
                    "normalizations": [],
                    "confidence": "low",
                    "preserved_semantics": True,
                    "has_changes": False,
                },
                "normalized_text": state["raw_text"],  # 使用原文
                "error": str(e),
                "current_step": StepEnum.NER,
            }

    return normalize_node


# ===== Step 0.7: QA Scaffold 节点（P8新增） =====


def create_qa_scaffold_node(llm: Any):
    """创建QA脚手架节点 - 5W1H问答扩展构建语义脚手架"""
    from .schemas import QAScaffoldResult

    parser = PydanticOutputParser(pydantic_object=QAScaffoldResult)

    async def qa_scaffold_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Step 0.7: 5W1H问答扩展，构建语义脚手架"""
        corpus_id = state["corpus_id"]

        # 使用归一化后的文本（如果有的话）
        text_for_processing = get_text_for_processing(state)

        logger.info(f"[QA_Scaffold] 处理语料: {corpus_id}")

        # 发送进度事件
        writer(
            {
                "step": "qa_scaffold",
                "corpus_id": corpus_id,
                "status": "started",
                "message": "开始构建语义脚手架",
            }
        )

        try:
            # 调用LLM生成QA脚手架
            prompt_text = QA_SCAFFOLD_PROMPT.invoke(
                {"normalized_text": text_for_processing}
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: QAScaffoldResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            logger.info(
                f"[QA_Scaffold] 完成: {len(result.qa_pairs)} 个问答对, "
                f"{len(result.entity_hints)} 个实体提示, "
                f"置信度={result.overall_confidence}"
            )

            # 发送完成事件
            writer(
                {
                    "step": "qa_scaffold",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "qa_count": len(result.qa_pairs),
                    "entity_hints": result.entity_hints,
                    "relation_hints": result.relation_hints,
                    "confidence": result.overall_confidence,
                    "semantic_summary": result.semantic_summary,
                }
            )

            # 根据结果决定下一步
            if result.should_skip_detailed_extraction:
                # 简单文本，跳过后续处理
                logger.info(f"[QA_Scaffold] 建议跳过详细抽取: {corpus_id}")
                return {
                    "qa_scaffold_result": result.model_dump(),
                    "semantic_summary": result.semantic_summary,
                    "current_step": StepEnum.DONE,
                }
            else:
                # 复杂文本，继续到 NER
                return {
                    "qa_scaffold_result": result.model_dump(),
                    "semantic_summary": result.semantic_summary,
                    "qa_entity_hints": result.entity_hints,
                    "qa_relation_hints": result.relation_hints,
                    "qa_context_dependencies": result.context_dependencies,
                    "current_step": StepEnum.NER,
                }

        except Exception as e:
            logger.error(f"[QA_Scaffold] 处理失败: {e}")
            # 保守策略：失败时继续处理
            writer(
                {
                    "step": "qa_scaffold",
                    "corpus_id": corpus_id,
                    "status": "failed",
                    "error": str(e),
                }
            )
            return {
                "qa_scaffold_result": {},
                "semantic_summary": "",
                "qa_entity_hints": [],
                "qa_relation_hints": [],
                "qa_context_dependencies": [],
                "current_step": StepEnum.NER,  # 失败时继续到NER
            }

    return qa_scaffold_node


# ===== 单条语料处理节点（四步骤工作流） =====


def create_ner_node(llm: Any):
    """创建NER节点"""
    parser = PydanticOutputParser(pydantic_object=EntityRecognitionResult)

    async def ner_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Step 1: 命名实体识别"""
        corpus_id = state["corpus_id"]
        logger.info(f"[NER] 处理语料: {corpus_id}")

        # P3改进：发送进度事件
        writer(
            {
                "step": "ner",
                "corpus_id": corpus_id,
                "status": "started",
                "message": "开始命名实体识别",
            }
        )

        try:
            # P6改进：优先使用归一化文本
            text_for_processing = get_text_for_processing(state)
            logger.debug(f"[NER] 使用文本: {text_for_processing[:50]}...")

            # P8改进：获取 QA Scaffold 上下文
            qa_entity_hints = state.get("qa_entity_hints", [])
            qa_context_dependencies = state.get("qa_context_dependencies", [])

            # 使用 OutputParser 进行结构化输出
            prompt_text = NER_PROMPT.invoke(
                {
                    "raw_text": text_for_processing,
                    "entity_hints": format_entity_hints(qa_entity_hints),
                    "context_dependencies": format_context_dependencies(
                        qa_context_dependencies
                    ),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: EntityRecognitionResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            entity_count = (
                len(result.道路)
                + len(result.POI)
                + len(result.建筑物)
                + len(result.街区)
            )
            logger.debug(f"[NER] 结果: {result}")

            # P3改进：发送完成事件
            writer(
                {
                    "step": "ner",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "entity_count": entity_count,
                    "entities": {
                        "道路": result.道路,
                        "POI": result.POI,
                        "建筑物": result.建筑物,
                        "街区": result.街区,
                    },
                }
            )

            return {
                "entities": {
                    "道路": result.道路,
                    "POI": result.POI,
                    "建筑物": result.建筑物,
                    "街区": result.街区,
                },
                "current_step": StepEnum.RE,
            }
        except Exception as e:
            logger.error(f"[NER] 失败: {e}")
            # P3改进：发送错误事件
            writer(
                {
                    "step": "ner",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            return {
                "entities": DEFAULT_ENTITY_DICT.copy(),  # v3.4修复：使用6种实体类型
                "error": str(e),
                "current_step": StepEnum.DONE,  # 出错时直接结束
            }

    return ner_node


def create_re_node(llm: Any):
    """创建RE节点（v2.2改进：支持attributes）"""
    parser = PydanticOutputParser(pydantic_object=RelationExtractionResult)

    async def re_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Step 2: 关系抽取"""
        corpus_id = state["corpus_id"]
        logger.info(f"[RE] 处理语料: {corpus_id}")

        # P3改进：发送进度事件
        writer(
            {
                "step": "re",
                "corpus_id": corpus_id,
                "status": "started",
                "message": "开始关系抽取",
            }
        )

        # 检查是否有实体
        total_entities = sum(len(v) for v in state["entities"].values())
        if total_entities == 0:
            logger.debug(f"[RE] 无实体，跳过")
            writer(
                {
                    "step": "re",
                    "corpus_id": corpus_id,
                    "status": "skipped",
                    "reason": "无实体",
                }
            )
            return {"current_step": StepEnum.EVAL, "triples": []}

        try:
            # P6改进：优先使用归一化文本
            text_for_processing = get_text_for_processing(state)

            # P8改进：获取 QA Scaffold 上下文
            qa_relation_hints = state.get("qa_relation_hints", [])
            qa_context_dependencies = state.get("qa_context_dependencies", [])

            # 使用 OutputParser 进行结构化输出
            prompt_text = RE_PROMPT.invoke(
                {
                    "raw_text": text_for_processing,
                    "entities": format_entities(state["entities"]),
                    "relation_hints": format_relation_hints(qa_relation_hints),
                    "context_dependencies": format_context_dependencies(
                        qa_context_dependencies
                    ),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: RelationExtractionResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # v2.2改进：提取三元组及属性（Enum转字符串+强类型属性转字典）
            triples = [
                {
                    "head": t.head,
                    "relation": extract_enum_value(t.relation),  # P15改进：使用工具函数
                    "tail": t.tail,
                    "evidence": t.evidence or "",
                    "attributes": t.attributes.model_dump(exclude_none=True)
                    if t.attributes
                    else {},  # TripleAttributes转字典
                }
                for t in result.triples
            ]

            logger.debug(f"[RE] 结果: {len(triples)}个三元组")

            # P3改进：发送完成事件
            writer(
                {
                    "step": "re",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "triple_count": len(triples),
                    "triples": triples,
                }
            )

            return {"triples": triples, "current_step": StepEnum.EVAL}
        except Exception as e:
            logger.error(f"[RE] 失败: {e}")
            # P3改进：发送错误事件
            writer(
                {
                    "step": "re",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            return {"triples": [], "error": str(e), "current_step": StepEnum.EVAL}

    return re_node


def create_eval_1_node(llm: Any):
    """创建第一次评估节点"""
    parser = PydanticOutputParser(pydantic_object=EvalResultFirst)

    async def eval_1_node(state: CorpusState) -> Dict:
        """Step 3a: 第一次评估"""
        logger.info(f"[Eval1] 处理语料: {state['corpus_id']}")

        if not state["triples"]:
            logger.debug(f"[Eval1] 无三元组，跳过")
            return {"eval_scores": [], "current_step": StepEnum.LABEL}

        try:
            # 使用 OutputParser 进行结构化输出
            prompt_text = EVAL_PROMPT_1.invoke(
                {
                    "triples": state["triples"],
                    "raw_text": state["raw_text"],
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: EvalResultFirst = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            scores = [
                {
                    "triple": {
                        "head": s.triple.head if s.triple else "",
                        "relation": s.triple.relation if s.triple else "",
                        "tail": s.triple.tail if s.triple else "",
                    },
                    "SEM": s.SEM or 3,
                    "FAC": s.FAC or 3,
                    "CON": s.CON or 3,
                }
                for s in result.scores
                if s.triple  # P20: 跳过 None triple
            ]

            logger.debug(f"[Eval1] 结果: {len(scores)}个评分")

            return {"eval_scores": scores, "current_step": StepEnum.EVAL}
        except Exception as e:
            logger.error(f"[Eval1] 失败: {e}")
            return {"eval_scores": [], "error": str(e), "current_step": StepEnum.EVAL}

    return eval_1_node


def create_eval_2_node(llm: Any):
    """创建第二次评估节点（自检）"""
    parser = PydanticOutputParser(pydantic_object=EvalResultSecond)

    async def eval_2_node(state: CorpusState) -> Dict:
        """Step 3b: 第二次评估（自检）"""
        logger.info(f"[Eval2] 处理语料: {state['corpus_id']}")

        # 无三元组时，视为跳过评估（而非失败）
        if not state["triples"]:
            logger.debug(f"[Eval2] 无三元组，跳过评估")
            return {
                "corrected_triples": [],
                "eval_passed": True,  # 无三元组视为通过（没有需要评估的内容）
                "current_step": StepEnum.LABEL,
            }

        if not state["eval_scores"]:
            logger.debug(f"[Eval2] 无评分，使用原始三元组")
            return {
                "corrected_triples": state["triples"],
                "eval_passed": False,
                "current_step": StepEnum.LABEL,
            }

        try:
            # 使用 OutputParser 进行结构化输出
            prompt_text = EVAL_PROMPT_2.invoke(
                {
                    "previous_scores": state["eval_scores"],
                    "raw_text": state["raw_text"],
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: EvalResultSecond = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # 更新评分
            final_scores = (
                [
                    {
                        "triple": {
                            "head": s.triple.head if s.triple else "",
                            "relation": s.triple.relation if s.triple else "",
                            "tail": s.triple.tail if s.triple else "",
                        },
                        "SEM": s.SEM or 3,
                        "FAC": s.FAC or 3,
                        "CON": s.CON or 3,
                    }
                    for s in result.final_scores
                    if s.triple  # P20: 跳过 None triple
                ]
                if result.final_scores
                else state["eval_scores"]
            )

            # 创建评分查找字典
            score_map = {}
            for score_item in final_scores:
                triple_key = (
                    score_item["triple"]["head"],
                    score_item["triple"]["relation"],
                    score_item["triple"]["tail"],
                )
                score_map[triple_key] = {
                    "sem_score": score_item["SEM"],
                    "fac_score": score_item["FAC"],
                    "con_score": score_item["CON"],
                }

            # 应用修正
            correction_mapping = {}
            if result.need_correction and result.corrections:
                corrected_triples, correction_mapping = apply_corrections(
                    state["triples"], result.corrections
                )
            else:
                corrected_triples = state["triples"]

            # 将评分写入三元组
            passed_threshold = 3.5
            for triple in corrected_triples:
                triple_key = (triple["head"], triple["relation"], triple["tail"])
                scores_for_triple = score_map.get(triple_key)

                # 如果新三元组没有直接评分，尝试从原始三元组继承
                if not scores_for_triple and triple_key in correction_mapping:
                    original_key = correction_mapping[triple_key]
                    scores_for_triple = score_map.get(original_key, {})

                if not scores_for_triple:
                    scores_for_triple = {}

                triple["sem_score"] = scores_for_triple.get("sem_score", 0)
                triple["fac_score"] = scores_for_triple.get("fac_score", 0)
                triple["con_score"] = scores_for_triple.get("con_score", 0)
                # 计算该三元组的平均评分并设置 passed_eval
                avg_triple_score = (
                    triple["sem_score"] + triple["fac_score"] + triple["con_score"]
                ) / 3
                triple["passed_eval"] = (
                    avg_triple_score >= passed_threshold
                    if avg_triple_score > 0
                    else False
                )

            # 计算平均评分判断是否通过
            avg_score = (
                sum(s["SEM"] + s["FAC"] + s["CON"] for s in final_scores)
                / (len(final_scores) * 3)
                if final_scores
                else 0
            )

            logger.debug(
                f"[Eval2] 平均评分: {avg_score}, 需修正: {result.need_correction}"
            )

            return {
                "eval_scores": final_scores,
                "corrected_triples": corrected_triples,
                "eval_passed": avg_score >= 3.5,
                "current_step": StepEnum.LABEL,
            }
        except Exception as e:
            logger.error(f"[Eval2] 失败: {e}")
            return {
                "corrected_triples": state["triples"],
                "eval_passed": False,
                "error": str(e),
                "retry_count": retry_count + 1,
                "current_step": StepEnum.LABEL,
            }

    return eval_2_node


# P2改进：规则校验函数（不依赖 LLM）
def rule_based_validation(
    triples: List[Dict], entities: Dict[str, List[str]]
) -> List[Dict]:
    """
    规则校验三元组（不依赖 LLM）

    校验规则：
    1. 实体存在性：头实体和尾实体必须在已识别实体中
    2. 关系类型有效性：关系必须在预定义类型中
    3. 基本逻辑检查：某些关系类型有约束（如"连接"需要两个道路实体）
    """
    # v3.2精简版：使用完整的7个关系类型列表
    # 关系类型：位于/包含/相对方位/具有功能/优于/相似/劣于/发生事件
    VALID_RELATIONS = RELATION_TYPES  # 从 state.py 导入

    all_entities = []
    for entity_list in entities.values():
        all_entities.extend(entity_list)

    validated_triples = []
    for triple in triples:
        head = triple.get("head", "")
        tail = triple.get("tail", "")
        relation = triple.get("relation", "")

        # 规则1：实体存在性检查（宽松匹配，允许别名）
        head_valid = (
            any(head in e or e in head for e in all_entities) if head else False
        )
        tail_valid = (
            any(tail in e or e in tail for e in all_entities) if tail else False
        )

        # 规则2：关系类型有效性
        relation_valid = relation in VALID_RELATIONS

        # 规则3：关系逻辑检查（可选，可扩展）
        # 例如："连接"关系通常连接两个道路或地点
        logic_valid = True  # 默认通过，后续可扩展

        # 记录校验结果
        triple["_rule_valid"] = (
            head_valid and tail_valid and relation_valid and logic_valid
        )
        triple["_rule_issues"] = []
        if not head_valid:
            triple["_rule_issues"].append(f"头实体'{head}'未在NER结果中")
        if not tail_valid:
            triple["_rule_issues"].append(f"尾实体'{tail}'未在NER结果中")
        if not relation_valid:
            triple["_rule_issues"].append(f"关系'{relation}'不在预定义类型中")

        validated_triples.append(triple)

    return validated_triples


def sanitize_triples_for_pipeline(
    triples: List[Any], context: str = "unknown"
) -> List[Dict[str, Any]]:
    """清洗三元组，过滤缺失 head/relation/tail 的脏数据。"""
    if not triples:
        return []

    sanitized: List[Dict[str, Any]] = []
    dropped = 0

    for t in triples:
        if t is None:
            dropped += 1
            continue

        if hasattr(t, "model_dump"):
            t = t.model_dump(mode="json")

        if not isinstance(t, dict):
            dropped += 1
            continue

        head = t.get("head")
        relation = t.get("relation")
        tail = t.get("tail")

        if hasattr(relation, "value"):
            relation = relation.value

        head = str(head).strip() if head is not None else ""
        relation = str(relation).strip() if relation is not None else ""
        tail = str(tail).strip() if tail is not None else ""

        if not head or not relation or not tail:
            dropped += 1
            continue

        attrs = t.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}

        sanitized.append(
            {
                **t,
                "head": head,
                "relation": relation,
                "tail": tail,
                "evidence": t.get("evidence", ""),
                "attributes": attrs,
            }
        )

    if dropped:
        logger.warning(f"[Triple-Sanitize] {context}: 丢弃无效三元组 {dropped} 条")

    return sanitized


def create_eval_simplified_node(
    llm: Any, eval_threshold: float = 3.5, enable_query: bool = False
):
    """
    P2改进：创建简化的单次评估节点
    P3改进：支持 StreamWriter 流式输出
    P14改进：支持向导师发起查询（enable_query=True时）

    合原来两轮评估为单次评估 + 规则校验，减少 LLM 调用成本
    """
    parser = PydanticOutputParser(pydantic_object=EvalResultSimplified)

    async def eval_simplified_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Step 3: 简化评估（单次LLM调用 + 规则校验）"""
        corpus_id = state["corpus_id"]
        logger.info(f"[Eval] 处理语料: {corpus_id}")
        retry_count = state.get("retry_count", 0)

        # P3改进：发送进度事件
        writer(
            {
                "step": "eval",
                "corpus_id": corpus_id,
                "status": "started",
                "message": "开始三元组评估",
            }
        )

        if not state["triples"]:
            logger.debug(f"[Eval] 无三元组，跳过评估")
            writer(
                {
                    "step": "eval",
                    "corpus_id": corpus_id,
                    "status": "skipped",
                    "reason": "无三元组",
                }
            )
            return {
                "eval_scores": [],
                "corrected_triples": [],
                "eval_passed": True,
                "needs_mentor_help": False,  # P14新增
                "current_step": StepEnum.LABEL,
            }

        try:
            # P6改进：优先使用归一化文本
            text_for_processing = get_text_for_processing(state)

            # P8改进：获取 QA Scaffold 上下文
            semantic_summary = state.get("semantic_summary", "")
            qa_context_dependencies = state.get("qa_context_dependencies", [])

            # P14新增：如果有导师回答，更新语义摘要
            mentor_response = state.get("mentor_response")
            if mentor_response:
                integrated_summary = mentor_response.get("clarification", "")
                if integrated_summary:
                    semantic_summary = (
                        f"{semantic_summary}\n导师澄清: {integrated_summary}"
                    )
                logger.info(f"[Eval] 使用导师更新的语义理解")

            # 格式化三元组用于提示词
            triples_text = format_triples(state["triples"])

            # 使用 OutputParser 进行结构化输出
            prompt_text = EVAL_PROMPT_SIMPLIFIED.invoke(
                {
                    "triples": triples_text,
                    "raw_text": text_for_processing,
                    "semantic_summary": semantic_summary or "(无语义摘要)",
                    "context_dependencies": format_context_dependencies(
                        qa_context_dependencies
                    ),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: EvalResultSimplified = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # 处理评分
            scores = []
            for s in (result.scores or []):
                if not s or not getattr(s, "triple", None):
                    continue
                triple_obj = s.triple
                head = getattr(triple_obj, "head", "")
                relation = getattr(triple_obj, "relation", "")
                tail = getattr(triple_obj, "tail", "")
                if not head or not relation or not tail:
                    continue
                relation = extract_enum_value(relation)
                scores.append(
                    {
                        "triple": {
                            "head": head,
                            "relation": relation,
                            "tail": tail,
                        },
                        "SEM": s.SEM if s.SEM is not None else 3,
                        "FAC": s.FAC if s.FAC is not None else 3,
                        "CON": s.CON if s.CON is not None else 3,
                    }
                )

            # 应用 LLM 修正（如果有）
            if result.need_correction and result.corrections:
                corrected_triples = apply_llm_corrections(
                    state["triples"], result.corrections
                )
            else:
                corrected_triples = list(state["triples"])

            corrected_triples = sanitize_triples_for_pipeline(
                corrected_triples, context=f"eval:{corpus_id}"
            )

            # P15调试：检查 attributes 是否保留
            corpus_id_debug = state.get("corpus_id", "unknown")
            for t in corrected_triples[:3]:
                rel = t.get("relation")
                attrs = t.get("attributes", {})
                logger.debug(
                    f"[Eval-Debug] corpus={corpus_id_debug}: <{t.get('head')}, {rel} (type={type(rel).__name__}), {t.get('tail')}> attributes={attrs}"
                )

            # P2改进：规则校验（不依赖 LLM）
            corrected_triples = rule_based_validation(
                corrected_triples, state["entities"]
            )

            # 将评分写入三元组
            score_map = {}
            for s in scores:
                key = (
                    s["triple"]["head"],
                    s["triple"]["relation"],
                    s["triple"]["tail"],
                )
                score_map[key] = s

            for triple in corrected_triples:
                key = (triple["head"], triple["relation"], triple["tail"])
                score_data = score_map.get(key, {})
                triple["sem_score"] = score_data.get("SEM", 3)
                triple["fac_score"] = score_data.get("FAC", 3)
                triple["con_score"] = score_data.get("CON", 3)

                # 综合评分和规则校验结果
                avg_score = (
                    triple["sem_score"] + triple["fac_score"] + triple["con_score"]
                ) / 3
                rule_valid = triple.get("_rule_valid", True)
                triple["passed_eval"] = avg_score >= eval_threshold and rule_valid

            # 计算整体通过率
            passed_count = sum(
                1 for t in corrected_triples if t.get("passed_eval", False)
            )
            overall_passed = passed_count > 0 if corrected_triples else True

            logger.debug(
                f"[Eval] 评分完成: {len(scores)}个三元组, {passed_count}个通过, 规则校验已应用"
            )

            # P3改进：发送完成事件
            writer(
                {
                    "step": "eval",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "triple_count": len(corrected_triples),
                    "passed_count": passed_count,
                    "eval_passed": overall_passed,
                }
            )

            # P14新增：困惑检测（如果启用了查询功能）
            if enable_query:
                from .prompts import detect_eval_confusion

                query_count = state.get("query_count", 0)
                max_queries = state.get("max_queries", 2)

                if query_count < max_queries:
                    eval_result = {
                        "eval_passed": overall_passed,
                        "corrected_triples": corrected_triples,
                    }
                    confusion = detect_eval_confusion(eval_result, dict(state))
                    confidence_gate = str(
                        state.get("mentor_query_min_confidence", "medium")
                    ).lower()
                    should_ask = False
                    if confusion:
                        cc = str(confusion.get("current_confidence", "medium")).lower()
                        should_ask = (
                            cc == "low"
                            if confidence_gate == "low"
                            else cc in ("low", "medium")
                        )
                    if should_ask:
                        logger.info(
                            f"[Eval] 检测到困惑，请求导师帮助: {confusion['query_type']}"
                        )
                        writer(
                            {
                                "step": "eval",
                                "corpus_id": corpus_id,
                                "status": "needs_mentor_help",
                                "query_type": confusion["query_type"],
                                "query_content": confusion["query_content"],
                            }
                        )

                        return {
                            "eval_scores": scores,
                            "corrected_triples": corrected_triples,
                            "eval_passed": overall_passed,
                            "retry_count": retry_count + (0 if overall_passed else 1),
                            "mentor_query": confusion,
                            "query_source_node": "eval",
                            "needs_mentor_help": True,
                            "query_count": query_count,
                            "current_step": StepEnum.QA_MENTOR,  # 回退到导师
                        }

            return {
                "eval_scores": scores,
                "corrected_triples": corrected_triples,
                "eval_passed": overall_passed,
                "needs_mentor_help": False,  # P14新增
                "current_step": StepEnum.LABEL,
            }

        except Exception as e:
            logger.error(f"[Eval] 失败: {e}")
            # P3改进：发送错误事件
            writer(
                {
                    "step": "eval",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            # 失败时仍应用规则校验
            fallback_triples = rule_based_validation(
                state["triples"], state["entities"]
            )
            fallback_triples = sanitize_triples_for_pipeline(
                fallback_triples, context=f"eval_fallback:{corpus_id}"
            )
            return {
                "eval_scores": [],
                "corrected_triples": fallback_triples,
                "eval_passed": False,
                "error": str(e),
                "retry_count": retry_count + 1,
                "needs_mentor_help": False,
                "current_step": StepEnum.LABEL,
            }

    return eval_simplified_node


def apply_llm_corrections(
    original_triples: List[Dict], corrections: List[Any]
) -> List[Dict]:
    """应用 LLM 返回的修正

    P15修复：保留原 triple 的 attributes 字段（防止 relation_attrs 丢失）
    """
    corrected = list(original_triples)

    for correction in corrections:
        if not correction or not getattr(correction, "original", None):
            continue
        original_key = (
            correction.original.head,
            correction.original.relation,
            correction.original.tail,
        )

        # 查找原 triple 以获取其 attributes
        original_attrs = {}
        original_evidence = ""
        for triple in original_triples:
            if (triple["head"], triple["relation"], triple["tail"]) == original_key:
                original_attrs = triple.get("attributes", {})
                original_evidence = triple.get("evidence", "")
                break

        corrected_obj = getattr(correction, "corrected", None)
        # P20场景：corrected=None 表示删除该三元组
        if corrected_obj is None:
            corrected = [
                triple
                for triple in corrected
                if (triple.get("head"), triple.get("relation"), triple.get("tail"))
                != original_key
            ]
            continue

        new_triple = {
            "head": corrected_obj.head,
            "relation": extract_enum_value(corrected_obj.relation),
            "tail": corrected_obj.tail,
            "evidence": original_evidence,  # 保留原 evidence
            "attributes": original_attrs,  # P15修复：保留原 attributes
        }

        for i, triple in enumerate(corrected):
            if (triple["head"], triple["relation"], triple["tail"]) == original_key:
                corrected[i] = new_triple
                break

    return corrected


def create_label_node(llm: Any, enable_query: bool = False):
    """创建属性标注节点（v2.2改进：扩展实体属性）

    P14改进：支持向导师发起查询（enable_query=True时）
    """
    parser = PydanticOutputParser(pydantic_object=LabelResult)

    async def label_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Step 4: 属性标注"""
        corpus_id = state["corpus_id"]
        logger.info(f"[Label] 处理语料: {corpus_id}")

        # P3改进：发送进度事件
        writer(
            {
                "step": "label",
                "corpus_id": corpus_id,
                "status": "started",
                "message": "开始属性标注",
            }
        )

        # 收集所有实体名称
        all_entities = []
        for entity_list in state["entities"].values():
            all_entities.extend(entity_list)

        if not all_entities:
            logger.debug(f"[Label] 无实体，跳过")
            writer(
                {
                    "step": "label",
                    "corpus_id": corpus_id,
                    "status": "skipped",
                    "reason": "无实体",
                }
            )
            return {
                "needs_mentor_help": False,  # P14新增
                "current_step": StepEnum.DONE,
            }

        try:
            # v2.2改进：获取原始文本用于提取情感标签、体验评价
            text_for_processing = get_text_for_processing(state)
            corrected_triples = sanitize_triples_for_pipeline(
                state.get("corrected_triples", []), context=f"label:{corpus_id}"
            )

            # P8改进：获取 QA Scaffold 上下文
            semantic_summary = state.get("semantic_summary", "")
            qa_entity_hints = state.get("qa_entity_hints", [])
            qa_relation_hints = state.get("qa_relation_hints", [])

            # P14新增：如果有导师回答，更新提示
            mentor_response = state.get("mentor_response")
            if mentor_response:
                # 使用导师更新的提示
                updated_entity_hints = mentor_response.get("updated_entity_hints")
                if updated_entity_hints:
                    qa_entity_hints = updated_entity_hints
                clarification = mentor_response.get("clarification", "")
                if clarification:
                    semantic_summary = f"{semantic_summary}\n导师澄清: {clarification}"
                logger.info(f"[Label] 使用导师更新的提示")

            # 使用 OutputParser 进行结构化输出
            prompt_text = LABEL_PROMPT.invoke(
                {
                    "entities": all_entities,
                    "relations": format_triples(corrected_triples),
                    "raw_text": text_for_processing,
                    "semantic_summary": semantic_summary or "(无语义摘要)",
                    "entity_hints": format_entity_hints(qa_entity_hints),
                    "relation_hints": format_relation_hints(qa_relation_hints),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: LabelResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # v3.2精简版：仅提取schema定义的属性（P20改进：添加空值检查）
            entity_attrs = {}
            for name, attrs in result.entities.items():
                if attrs is None:
                    entity_attrs[name] = {}
                    continue
                entity_attrs[name] = {
                    "类别": attrs.类别,
                    "细分": attrs.细分,
                    "特征标签": attrs.特征标签 or [],
                    "推荐指数": attrs.推荐指数,
                    "情感倾向": attrs.情感倾向,
                }

            # v3.4精简版：根据关系类型选择性提取属性（P20改进：添加空值检查）
            # 使用 RELATION_ATTRS_MAP 动态过滤，避免硬编码
            relation_attrs = {}
            for key, attrs in result.relations.items():
                if attrs is None:
                    relation_attrs[key] = {}
                    continue
                normalized_key = normalize_relation_key(key)
                if normalized_key:
                    # 提取关系类型
                    relation_type = (
                        normalized_key.split(", ")[1].strip()
                        if ", " in normalized_key
                        else ""
                    )

                    # 动态过滤：根据RELATION_ATTRS_MAP获取允许的属性列表
                    allowed_attrs = RELATION_ATTRS_MAP.get(relation_type, [])
                    # 使用model_dump获取所有非空属性
                    attrs_dict = attrs.model_dump(exclude_none=True)
                    # 仅保留允许的属性
                    filtered_attrs = {
                        k: v for k, v in attrs_dict.items() if k in allowed_attrs
                    }

                    relation_attrs[normalized_key] = filtered_attrs
                else:
                    # 无法解析时保留原始 key
                    relation_attrs[key] = {}

            logger.debug(
                f"[Label] 完成: {len(entity_attrs)}个实体, {len(relation_attrs)}个关系"
            )

            # P3改进：发送完成事件
            writer(
                {
                    "step": "label",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "entity_count": len(entity_attrs),
                    "relation_count": len(relation_attrs),
                }
            )

            # P14新增：困惑检测（如果启用了查询功能）
            if enable_query:
                from .prompts import detect_label_confusion

                query_count = state.get("query_count", 0)
                max_queries = state.get("max_queries", 2)

                if query_count < max_queries:
                    label_result = {
                        "entity_attrs": entity_attrs,
                        "relation_attrs": relation_attrs,
                    }
                    confusion = detect_label_confusion(label_result, dict(state))
                    confidence_gate = str(
                        state.get("mentor_query_min_confidence", "medium")
                    ).lower()
                    should_ask = False
                    if confusion:
                        cc = str(confusion.get("current_confidence", "medium")).lower()
                        should_ask = (
                            cc == "low"
                            if confidence_gate == "low"
                            else cc in ("low", "medium")
                        )
                    if should_ask:
                        logger.info(
                            f"[Label] 检测到困惑，请求导师帮助: {confusion['query_type']}"
                        )
                        writer(
                            {
                                "step": "label",
                                "corpus_id": corpus_id,
                                "status": "needs_mentor_help",
                                "query_type": confusion["query_type"],
                                "query_content": confusion["query_content"],
                            }
                        )

                        return {
                            "entity_attrs": entity_attrs,
                            "relation_attrs": relation_attrs,
                            "mentor_query": confusion,
                            "query_source_node": "label",
                            "needs_mentor_help": True,
                            "query_count": query_count,
                            "current_step": StepEnum.QA_MENTOR,  # 回退到导师
                        }

            return {
                "entity_attrs": entity_attrs,
                "relation_attrs": relation_attrs,
                "needs_mentor_help": False,  # P14新增
                "current_step": StepEnum.DONE,
            }
        except Exception as e:
            logger.error(f"[Label] 失败: {e}")
            # P3改进：发送错误事件
            writer(
                {
                    "step": "label",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            return {
                "entity_attrs": {},
                "relation_attrs": {},
                "error": str(e),
                "needs_mentor_help": False,
                "current_step": StepEnum.DONE,
            }

    return label_node


# ===== 辅助函数 =====


def normalize_relation_key(key: str) -> Optional[str]:
    """
    规范化关系属性 key 格式

    支持的输入格式:
    - "<武汉大学, 位于, 珞喻路>"
    - "武汉大学, 位于, 珞喻路"
    - "武汉大学,位于,珞喻路"

    返回标准格式: "<武汉大学, 位于, 珞喻路>"
    """
    if not key:
        return None

    # 尝试匹配格式: <A, 关系, B> 或 A, 关系, B
    # 使用正则提取三个部分
    # 格式1: <A, 关系, B>
    match = re.match(r"^<\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*>$", key)
    if match:
        head, relation, tail = match.groups()
        return f"<{head.strip()}, {relation.strip()}, {tail.strip()}>"

    # 格式2: A, 关系, B (没有尖括号)
    parts = [p.strip() for p in key.split(",")]
    if len(parts) == 3:
        return f"<{parts[0]}, {parts[1]}, {parts[2]}>"

    # 无法解析，返回 None
    return None


def apply_corrections(original_triples: List[Dict], corrections: List[Any]) -> tuple:
    """
    应用三元组修正

    P15修复：保留原 triple 的 attributes 字段（防止 relation_attrs 丢失）

    Returns:
        (corrected_triples, correction_mapping)
        correction_mapping: {new_triple_key: original_triple_key} 用于继承评分
    """
    corrected = list(original_triples)
    correction_mapping = {}

    for correction in corrections:
        original = correction.original
        original_key = (original.head, original.relation, original.tail)

        # 查找原 triple 以获取其 attributes
        original_attrs = {}
        original_evidence = ""
        for triple in original_triples:
            if (
                triple["head"] == original.head
                and triple["relation"] == original.relation
                and triple["tail"] == original.tail
            ):
                original_attrs = triple.get("attributes", {})
                original_evidence = triple.get("evidence", "")
                break

        new_triple = {
            "head": correction.corrected.head,
            "relation": correction.corrected.relation,
            "tail": correction.corrected.tail,
            "evidence": original_evidence,  # 保留原 evidence
            "attributes": original_attrs,  # P15修复：保留原 attributes
        }
        new_key = (new_triple["head"], new_triple["relation"], new_triple["tail"])

        # 记录修正映射，用于继承评分
        correction_mapping[new_key] = original_key

        # 找到并替换原始三元组
        for i, triple in enumerate(corrected):
            if (
                triple["head"] == original.head
                and triple["relation"] == original.relation
                and triple["tail"] == original.tail
            ):
                corrected[i] = new_triple
                break

    return corrected, correction_mapping


# ===== 分布式处理节点 =====


def create_coordinator_node(corpus_per_worker: int = 10, max_workers: int = 10):
    """创建调度器节点"""

    def coordinator_node(state: KGState) -> Dict:
        """MAP阶段入口 - 计算Worker数量并分配语料"""
        corpus_list = state["corpus_list"]
        corpus_count = len(corpus_list)

        # 计算需要的Worker数量
        worker_count = min(max_workers, math.ceil(corpus_count / corpus_per_worker))

        # 计算每个Worker实际处理的语料数量（均匀分配）
        actual_corpus_per_worker = math.ceil(corpus_count / worker_count)

        # 分片语料
        partitions = {}
        active_workers = []
        for i in range(worker_count):
            start_idx = i * actual_corpus_per_worker
            end_idx = min((i + 1) * actual_corpus_per_worker, corpus_count)
            worker_id = f"worker_{i}"
            partitions[worker_id] = corpus_list[start_idx:end_idx]
            active_workers.append(worker_id)

        logger.info(
            f"[Coordinator] 创建 {worker_count} 个Worker, 每个处理 {actual_corpus_per_worker} 条语料, 共 {corpus_count} 条"
        )

        return {
            "worker_count": worker_count,
            "corpus_partitions": partitions,
            "active_workers": active_workers,
            "current_phase": PhaseEnum.MAP,
        }

    return coordinator_node


def create_aggregator_node(similarity_threshold: float = 0.85):
    """创建聚合器节点"""

    def _has_usable_result(corpus_state: Dict) -> bool:
        """判断语料是否已有可聚合结果（用于容忍历史 error 残留）。"""
        entities = corpus_state.get("entities", {}) or {}
        has_entities = any(bool(names) for names in entities.values())
        has_triples = bool(corpus_state.get("corrected_triples", []))
        return has_entities or has_triples

    def aggregator_node(state: KGState) -> Dict:
        """REDUCE阶段 - 合并Worker结果"""
        logger.info("[Aggregator] 开始聚合Worker结果")

        all_entities = []
        all_triples = []

        # 收集所有Worker的结果
        for worker_result in state["worker_results"]:
            for corpus_state in worker_result["results"]:
                # 仅在“有错误且没有可用产出”时跳过，避免历史error残留导致误丢弃。
                if corpus_state.get("error") and not _has_usable_result(corpus_state):
                    logger.warning(
                        f"[Aggregator] 跳过错误语料: {corpus_state.get('corpus_id')}"
                    )
                    continue

                corpus_id = corpus_state.get("corpus_id", "unknown")

                # 收集实体
                entities = corpus_state.get("entities", {})
                for entity_type, names in entities.items():
                    for name in names:
                        all_entities.append(
                            {
                                "name": name,
                                "type": entity_type,
                                "corpus_id": corpus_id,
                                "attrs": corpus_state.get("entity_attrs", {}).get(
                                    name, {}
                                ),
                            }
                        )

                # 收集三元组，并写入relation_attrs
                relation_attrs = corpus_state.get("relation_attrs", {})
                corrected_triples = corpus_state.get("corrected_triples", [])

                # v3.4精简版：关系类型属性映射（8个关系类型）
                # 使用从 schemas.py 导入的 RELATION_ATTRS_MAP 常量
                # Schema v3.4定义：
                # - 位于、包含：无关系属性
                # - 相对方位：距离值、方向值（删除联动推荐）
                # - 具有功能：时段、适合人群、具有限制、情感倾向、功能描述
                # - 优于/相似/劣于：维度、维度描述
                # - 发生事件：无关系属性（属性在事件实体上）

                for triple in corrected_triples:
                    # P15修复：如果 triple 是 Pydantic 对象（JointTriple），先转换为字典
                    if hasattr(triple, "model_dump"):
                        triple = triple.model_dump(mode="json")
                    if not isinstance(triple, dict):
                        logger.warning(f"[Aggregator] 跳过非字典三元组: corpus={corpus_id}")
                        continue
                    # P21修复：支持 subject/object 字段映射到 head/tail
                    if "head" not in triple or not triple.get("head"):
                        triple["head"] = triple.get("subject", "")
                    if "tail" not in triple or not triple.get("tail"):
                        triple["tail"] = triple.get("object", "")
                    if not triple.get("head") or not triple.get("relation") or not triple.get("tail"):
                        logger.warning(f"[Aggregator] 跳过缺失字段三元组: corpus={corpus_id}, triple={triple}")
                        continue
                    triple["_corpus_id"] = corpus_id
                    # 查找关系属性（使用标准格式）
                    triple_key = (
                        f"<{triple['head']}, {triple['relation']}, {triple['tail']}>"
                    )

                    # P15调试：打印 triple 的 attributes 字段状态
                    triple_attrs = triple.get("attributes", {})
                    triple_relation_attrs = triple.get("relation_attrs", {})
                    # 如果关系类型有属性映射但 triple 没有 attributes，记录警告
                    relation_raw = triple.get("relation", "")
                    relation_str = (
                        extract_enum_value(relation_raw)
                        if hasattr(relation_raw, "value")
                        else str(relation_raw)
                    )
                    expected_fields = RELATION_ATTRS_MAP.get(relation_str, [])
                    if expected_fields and not triple_attrs:
                        logger.warning(
                            f"[Aggregator-Missing] {triple_key}: relation={relation_str}, expected_fields={expected_fields}, but attributes={triple_attrs}"
                        )
                    elif triple_attrs:
                        logger.debug(
                            f"[Aggregator-Found] {triple_key}: relation={relation_str}, attributes={triple_attrs}"
                        )

                    # 尝试多种 key 格式查找（优先从 Label node 输出查找）
                    attrs = (
                        relation_attrs.get(triple_key)
                        or relation_attrs.get(
                            f"{triple['head']}, {triple['relation']}, {triple['tail']}"
                        )
                        or relation_attrs.get(
                            f"<{triple['head']},{triple['relation']},{triple['tail']}>"
                        )
                    )

                    # P15修复：如果 Label node 没有输出 relation_attrs，检查 triple 本身是否已有
                    # RE node 输出的 attributes 字段也包含关系属性
                    if not attrs:
                        attrs = triple.get("relation_attrs") or triple.get("attributes")

                    if attrs:
                        # v3.2精简版：relation_type 直接使用7种标准关系类型
                        triple["relation_type"] = triple.get("relation", "")
                        # 根据关系类型选择对应的属性集（Schema v3.2）
                        relation_raw = triple.get("relation", "")
                        # P15修复：确保 relation 是字符串（RELATION_ATTRS_MAP 的 key 是字符串）
                        relation = (
                            extract_enum_value(relation_raw)
                            if hasattr(relation_raw, "value")
                            else str(relation_raw)
                        )
                        attr_fields = RELATION_ATTRS_MAP.get(relation, [])
                        # 提取该关系类型的有效属性
                        triple["relation_attrs"] = {
                            field: attrs.get(field)
                            for field in attr_fields
                            if attrs.get(field) is not None
                        }
                    all_triples.append(triple)

        # 实体去重
        unique_entities, aliases = deduplicate_entities(
            all_entities, similarity_threshold
        )

        # 三元组去重
        unique_triples = deduplicate_triples(all_triples)

        logger.info(
            f"[Aggregator] 完成: {len(unique_entities)}个实体, {len(unique_triples)}个三元组"
        )

        return {
            "aggregated_entities": unique_entities,
            "aggregated_triples": unique_triples,
            "entity_aliases": aliases,
            "current_phase": PhaseEnum.FINALIZE,
        }

    return aggregator_node


def deduplicate_entities(entities: List[Dict], threshold: float) -> tuple:
    """
    实体去重，发现别名

    P1改进：使用多层 blocking 策略优化相似度比较：
    1. 第一层：按实体类型分组（同类型才比较）
    2. 第二层：同类型内按首字符分组
    3. 预构建长度索引，优化跨 block 简称检查

    时间复杂度从 O(n²) 降至 O(n*k)，k为平均block大小
    """
    unique_entities = []
    aliases = {}
    processed = set()

    # P1改进：按实体类型分组（不同类型实体无需比较相似度）
    type_blocks: Dict[str, List[Dict]] = defaultdict(list)
    for e in entities:
        if e.get("name"):
            type_blocks[e.get("type", "unknown")].append(e)

    # 预构建长度索引：按长度分组，用于简称检查优化
    def build_length_index(type_entities: List[Dict]) -> Dict[int, List[str]]:
        """为某类型的实体构建长度索引"""
        index: Dict[int, List[str]] = defaultdict(list)
        for e in type_entities:
            name = e["name"]
            if name:
                index[len(name)].append(name)
        return index

    def find_similar_in_block(name: str, block_names: List[str]) -> List[str]:
        """在单个block内查找相似实体"""
        similar = []
        for other in block_names:
            if other != name and other not in processed:
                if is_similar(name, other, threshold):
                    similar.append(other)
        return similar

    def find_abbreviation_candidates(
        name: str, name_len: int, length_index: Dict[int, List[str]]
    ) -> List[str]:
        """查找可能的简称别名（跨 block，但同类型）"""
        candidates = []
        min_ratio = 0.4

        # 较短名称查找包含它的较长名称
        min_longer_len = int(name_len / min_ratio) + 1
        for length in sorted(length_index.keys()):
            if length > name_len and length <= min_longer_len:
                for other in length_index[length]:
                    if other != name and other not in processed:
                        if name in other:
                            candidates.append(other)

        # 较长名称查找包含在其中的较短名称
        max_shorter_len = int(name_len * min_ratio)
        for length in sorted(length_index.keys()):
            if length < name_len and length >= max_shorter_len:
                for other in length_index[length]:
                    if other != name and other not in processed:
                        if other in name:
                            candidates.append(other)

        return candidates

    # 按类型分块处理（第一层 blocking）
    for entity_type, type_entities in type_blocks.items():
        # 构建该类型的名称集合和映射
        type_names = list({e["name"] for e in type_entities})
        name_to_entities = defaultdict(list)
        for e in type_entities:
            name_to_entities[e["name"]].append(e)

        # 构建该类型的长度索引
        type_length_index = build_length_index(type_entities)

        # 第二层 blocking：按首字符分组
        char_blocks: Dict[str, List[str]] = defaultdict(list)
        for name in type_names:
            if name:
                char_blocks[name[0].lower()].append(name)

        # 在同类型内按字符block处理
        for char_key, block_names in char_blocks.items():
            for name in block_names:
                if name in processed:
                    continue

                name_len = len(name)

                # 在同block内查找相似实体
                similar_names = [name] + find_similar_in_block(name, block_names)

                # 跨 char_block 简称检查（但仍需同类型）
                abbreviation_candidates = find_abbreviation_candidates(
                    name, name_len, type_length_index
                )
                for other in abbreviation_candidates:
                    if other not in similar_names:
                        similar_names.append(other)

                for n in similar_names:
                    processed.add(n)

                # 选择最长的名称作为标准名
                standard_name = max(similar_names, key=len)

                # 收集所有出现信息
                occurrences = []
                for n in similar_names:
                    occurrences.extend(name_to_entities.get(n, []))

                entity_attrs = {}
                for occ in occurrences:
                    if occ.get("attrs"):
                        entity_attrs.update(occ["attrs"])

                # 记录别名
                other_names = [n for n in similar_names if n != standard_name]
                if other_names:
                    aliases[standard_name] = other_names

                unique_entities.append(
                    {
                        "name": standard_name,
                        "type": entity_type,
                        "category": entity_attrs.get("细分", ""),
                        "aliases": other_names,
                        "occurrence_count": len(occurrences),
                        "corpus_ids": list(set(o["corpus_id"] for o in occurrences)),
                        "attrs": entity_attrs,
                    }
                )

    return unique_entities, aliases


def is_similar(name1: str, name2: str, threshold: float) -> bool:
    """判断两个名称是否相似"""
    if name1 == name2:
        return True

    # 长度差异检查
    len1, len2 = len(name1), len(name2)
    if abs(len1 - len2) > 2:
        len_ratio = min(len1, len2) / max(len1, len2)
        if len_ratio < 0.5:
            return False

    # 编辑距离相似度
    similarity = SequenceMatcher(None, name1, name2).ratio()
    if similarity >= threshold:
        return True

    # 简称别名检查
    if len1 != len2:
        shorter, longer = (name1, name2) if len1 < len2 else (name2, name1)
        if shorter in longer and len(shorter) >= len(longer) * 0.4:
            return True

    return False


def deduplicate_triples(triples: List[Dict]) -> List[Dict]:
    """三元组去重"""
    seen = set()
    unique_triples = []

    for triple in triples:
        key = (triple["head"], triple["relation"], triple["tail"])
        if key not in seen:
            seen.add(key)
            unique_triples.append(
                {
                    "head": triple["head"],
                    "relation": triple["relation"],
                    "tail": triple["tail"],
                    "evidence": triple.get("evidence", ""),
                    "corpus_ids": [triple.get("_corpus_id", "")],
                    "sem_score": triple.get("sem_score", 0),
                    "fac_score": triple.get("fac_score", 0),
                    "con_score": triple.get("con_score", 0),
                    "passed_eval": triple.get(
                        "passed_eval", False
                    ),  # 默认 False 更安全
                    "relation_type": triple.get("relation_type", ""),
                    "relation_subtype": triple.get("relation_subtype", ""),
                    "relation_attrs": triple.get(
                        "relation_attrs", {}
                    ),  # P15修复：保留关系属性
                }
            )
        else:
            # 更新corpus_ids
            for t in unique_triples:
                if (t["head"], t["relation"], t["tail"]) == key:
                    if triple.get("_corpus_id") not in t["corpus_ids"]:
                        t["corpus_ids"].append(triple.get("_corpus_id"))
                    break

    return unique_triples


# ===== Self-Check 节点（二次对话验证）=====


def create_self_check_ner_node(llm: Any):
    """
    创建 Self-Check-NER 节点

    职责：
    1. 检查遗漏实体
    2. 识别别名/简称，建议归一化
    3. 过滤无关实体
    4. 给出置信度评估
    """
    parser = PydanticOutputParser(pydantic_object=SelfCheckNERResult)

    async def self_check_ner_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Step 3.5a: 实体校验"""
        corpus_id = state["corpus_id"]
        retry_count = state.get("retry_count", 0)
        logger.info(f"[Self-Check-NER] 校验语料: {corpus_id}, 重试次数: {retry_count}")

        writer(
            {
                "step": "self_check_ner",
                "corpus_id": corpus_id,
                "status": "started",
                "message": "开始实体校验",
                "retry_count": retry_count,
            }
        )

        try:
            # P6改进：优先使用归一化文本
            text_for_processing = get_text_for_processing(state)

            # P8改进：获取 QA Scaffold 上下文
            qa_entity_hints = state.get("qa_entity_hints", [])
            qa_context_dependencies = state.get("qa_context_dependencies", [])
            semantic_summary = state.get("semantic_summary", "")

            # 构建重试提示（如有）
            problem_entities = state.get("problem_entities", [])
            retry_hint = format_retry_hint(problem_entities, [])

            # 使用 OutputParser 进行结构化输出
            prompt_text = SELF_CHECK_NER_PROMPT.invoke(
                {
                    "raw_text": text_for_processing,
                    "entities": format_entities(state["entities"]),
                    "retry_hint": retry_hint,
                    "semantic_summary": semantic_summary or "(无语义摘要)",
                    "context_dependencies": format_context_dependencies(
                        qa_context_dependencies
                    ),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckNERResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # 应用归一化，生成 final_entities
            final_entities = _apply_entity_normalizations(state["entities"], result)

            # 提取问题实体（供重试参考）
            missing_names = [e.name for e in result.missing_entities]

            writer(
                {
                    "step": "self_check_ner",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "verified_count": len(result.verified_entities),
                    "missing_count": len(result.missing_entities),
                    "normalization_count": len(result.entity_normalizations),
                    "confidence": result.overall_confidence,
                }
            )

            return {
                "self_check_ner_result": result.model_dump(),
                "final_entities": final_entities,
                "problem_entities": missing_names,
                # P8修复：移除Self-Check-NER中的计数器增加，由路由函数统一处理
                "current_step": StepEnum.RE,  # Self-Check-NER 在 NER 和 RE 之间
            }

        except Exception as e:
            logger.error(f"[Self-Check-NER] 失败: {e}")
            writer(
                {
                    "step": "self_check_ner",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            return {
                "self_check_ner_result": {},
                "error": str(e),
                "current_step": StepEnum.RE,  # 即使失败也继续到 RE
            }

    return self_check_ner_node


def create_self_check_re_node(llm: Any):
    """
    创建 Self-Check-RE 节点

    职责：
    1. 幻觉检测
    2. 关系验证
    3. 证据匹配
    4. 判断是否需要重抽
    """
    parser = PydanticOutputParser(pydantic_object=SelfCheckREResult)

    async def self_check_re_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Step 3.5b: 三元组校验"""
        corpus_id = state["corpus_id"]
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)
        logger.info(
            f"[Self-Check-RE] 校验语料: {corpus_id}, 重试次数: {retry_count}/{max_retries}"
        )

        writer(
            {
                "step": "self_check_re",
                "corpus_id": corpus_id,
                "status": "started",
                "message": "开始三元组校验",
                "retry_count": retry_count,
            }
        )

        try:
            # 构建重试提示（如有）
            problem_triples = state.get("problem_triples", [])
            retry_hint = format_retry_hint([], problem_triples)

            # P8改进：获取 QA Scaffold 上下文
            semantic_summary = state.get("semantic_summary", "")
            qa_context_dependencies = state.get("qa_context_dependencies", [])

            # 获取已校验实体
            verified_entities = state.get("final_entities", [])
            ner_result = state.get("self_check_ner_result", {})
            if ner_result and "verified_entities" in ner_result:
                verified_entities = ner_result["verified_entities"]

            # P6改进：优先使用归一化文本
            text_for_processing = get_text_for_processing(state)

            # 使用 OutputParser 进行结构化输出
            prompt_text = SELF_CHECK_RE_PROMPT.invoke(
                {
                    "raw_text": text_for_processing,
                    "triples": format_triples(state.get("triples", [])),  # RE 输出
                    "verified_entities": format_verified_entities(verified_entities),
                    "retry_hint": retry_hint,
                    "semantic_summary": semantic_summary or "(无语义摘要)",
                    "context_dependencies": format_context_dependencies(
                        qa_context_dependencies
                    ),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckREResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # 应用修正，生成 corrected_triples（供 Eval 使用）
            corrected_triples = _apply_triple_corrections_for_self_check(
                state.get("triples", []),
                result,  # RE 输出
            )

            # 提取问题三元组（供重试参考）
            problem_triples_list = [
                {
                    "head": t.head,
                    "relation": t.relation,
                    "tail": t.tail,
                    "reason": t.reason,
                }
                for t in result.rejected_triples
            ]

            # 计算整体置信度
            overall_confidence = _calculate_overall_confidence(
                state.get("self_check_ner_result", {}), result.model_dump()
            )

            # 判断是否需要重试
            retry_count = state.get("retry_count", 0) + 1  # 每次进入 Self-Check 计数
            needs_retry = _should_trigger_retry(
                result, retry_count, max_retries, state.get("self_check_ner_result", {})
            )

            writer(
                {
                    "step": "self_check_re",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "verified_count": len(result.verified_triples),
                    "rejected_count": len(result.rejected_triples),
                    "corrected_count": len(result.corrected_triples),
                    "confidence": overall_confidence,
                    "needs_retry": needs_retry,
                }
            )

            return {
                "self_check_re_result": result.model_dump(),
                "corrected_triples": corrected_triples,  # 输出给 Eval 使用
                "final_triples": corrected_triples,  # 同时保留 final_triples 兼容
                "problem_triples": problem_triples_list,
                "verification_confidence": overall_confidence,
                "retry_count": retry_count,
                "needs_review": overall_confidence == "low"
                or retry_count >= max_retries,
                "current_step": StepEnum.EVAL,  # 默认值，实际由路由决定
            }

        except Exception as e:
            logger.error(f"[Self-Check-RE] 失败: {e}")
            writer(
                {
                    "step": "self_check_re",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            return {
                "self_check_re_result": {},
                "corrected_triples": state.get("triples", []),  # 失败时保留原 triples
                "error": str(e),
                "retry_count": min(
                    state.get("retry_count", 0) + 1, max_retries
                ),  # 确保不超过上限
                "retry_suggested": True,  # 异常时建议重试
                "current_step": StepEnum.EVAL,
            }

    return self_check_re_node


# ===== Self-Check 辅助函数 =====


def _apply_entity_normalizations(
    original_entities: Dict[str, List[str]], result: SelfCheckNERResult
) -> List[Dict]:
    """应用实体归一化，生成 final_entities"""
    final_entities = []

    # 构建归一化映射
    normalization_map = {}
    for norm in result.entity_normalizations:
        normalization_map[norm.raw] = norm.canonical

    # 应用归一化到校验通过的实体
    for ve in result.verified_entities:
        entity = {
            "name": ve.name,
            "type": ve.type,
            "confidence": ve.confidence,
            "aliases": ve.aliases,
            "evidence": ve.evidence,
        }
        final_entities.append(entity)

    return final_entities


def _apply_triple_corrections_for_self_check(
    original_triples: List[Dict], result: SelfCheckREResult
) -> List[Dict]:
    """应用三元组修正

    P15修复：保留原 triple 的 attributes 字段（防止 relation_attrs 丢失）
    """
    final_triples = []

    # 构建原 triple 的 attributes 映射（用于继承）
    original_attrs_map = {}
    for triple in original_triples:
        key = (triple["head"], triple["relation"], triple["tail"])
        original_attrs_map[key] = triple.get("attributes", {})

    # 保留验证通过的三元组
    for vt in result.verified_triples:
        key = (vt.head, vt.relation, vt.tail)
        original_attrs = original_attrs_map.get(key, {})

        triple = {
            "head": vt.head,
            "relation": vt.relation,
            "tail": vt.tail,
            "confidence": vt.confidence,
            "evidence_valid": vt.evidence_valid,
            "evidence_match": vt.evidence_match,
            "passed_eval": True,
            "attributes": original_attrs,  # P15修复：保留原 attributes
        }
        final_triples.append(triple)

    # 应用修正的三元组
    for tc in result.corrected_triples:
        if tc.action == "delete":
            continue  # 删除操作：不添加到最终结果

        # 查找原 triple 的 attributes
        original_key = (tc.original_head, tc.original_relation, tc.original_tail)
        original_attrs = original_attrs_map.get(original_key, {})

        corrected_triple = {
            "head": tc.corrected_head or tc.original_head,
            "relation": tc.corrected_relation or tc.original_relation,
            "tail": tc.corrected_tail or tc.original_tail,
            "confidence": "medium",  # 修正后的置信度默认为 medium
            "correction_reason": tc.reason,
            "passed_eval": True,
            "attributes": original_attrs,  # P15修复：继承原 attributes
        }
        final_triples.append(corrected_triple)

    return final_triples


def _calculate_overall_confidence(ner_result: Dict, re_result: Dict) -> str:
    """计算整体置信度"""
    ner_conf = ner_result.get("overall_confidence", "medium")
    re_conf = re_result.get("overall_confidence", "medium")

    # 置信度等级映射
    conf_level = {"high": 3, "medium": 2, "low": 1}

    avg_level = (conf_level.get(ner_conf, 2) + conf_level.get(re_conf, 2)) / 2

    if avg_level >= 2.5:
        return "high"
    elif avg_level >= 1.5:
        return "medium"
    else:
        return "low"


def _should_trigger_retry(
    re_result: SelfCheckREResult, retry_count: int, max_retries: int, ner_result: Dict
) -> bool:
    """
    判断是否需要触发重抽

    条件：
    1. 重试次数未达上限
    2. Self-Check-RE 明确建议重抽
    3. 或者 NER 有较多遗漏 + RE 有较多幻觉
    """
    if retry_count >= max_retries:
        return False

    # Self-Check-RE 明确建议重抽
    if re_result.retry_suggested:
        return True

    # NER 遗漏较多（>3个）且置信度低
    ner_conf = ner_result.get("overall_confidence", "medium")
    missing_count = len(ner_result.get("missing_entities", []))
    if ner_conf == "low" and missing_count > 3:
        return True

    # RE 幻觉较多（>3个）且置信度低
    re_conf = re_result.overall_confidence
    rejected_count = len(re_result.rejected_triples)
    if re_conf == "low" and rejected_count > 3:
        return True

    return False


# ===== P9新增：联合抽取节点 =====


def create_joint_ner_re_node(llm: Any, enable_query: bool = False):
    """
    创建联合抽取节点

    职责：
    1. 一次LLM推理同时抽取实体和关系
    2. 避免NER→RE流水线的错误传播
    3. 全局理解文本，输出一致性更好的结果

    P14改进：支持向导师发起查询（enable_query=True时）
    """
    parser = PydanticOutputParser(pydantic_object=JointExtractionResult)

    async def joint_ner_re_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Joint NER + RE: 一次推理同时抽取实体和关系"""
        corpus_id = state["corpus_id"]
        logger.info(f"[Joint_NER_RE] 处理语料: {corpus_id}")

        writer(
            {
                "step": "joint_ner_re",
                "corpus_id": corpus_id,
                "status": "started",
                "message": "开始联合抽取",
            }
        )

        try:
            # 使用归一化文本
            text_for_processing = get_text_for_processing(state)

            # 获取 QA Scaffold 上下文
            qa_entity_hints = state.get("qa_entity_hints", [])
            qa_relation_hints = state.get("qa_relation_hints", [])
            qa_context_dependencies = state.get("qa_context_dependencies", [])

            # 获取导师指导（QA Mentor模式）
            mentor_guidance = state.get("mentor_guidance", {})

            # P14新增：如果有导师回答，更新提示
            mentor_response = state.get("mentor_response")
            if mentor_response:
                # 使用导师更新的提示
                updated_entity_hints = mentor_response.get("updated_entity_hints")
                updated_relation_hints = mentor_response.get("updated_relation_hints")
                if updated_entity_hints:
                    qa_entity_hints = updated_entity_hints
                if updated_relation_hints:
                    qa_relation_hints = updated_relation_hints
                # 更新导师指导
                updated_guidance = mentor_response.get("updated_guidance")
                if updated_guidance:
                    mentor_guidance = updated_guidance
                logger.info(f"[Joint_NER_RE] 使用导师更新的提示")

            # 调用 LLM
            # P12改进：使用改进版提示词（含反向验证+反面示例）
            prompt_text = JOINT_NER_RE_PROMPT_V2.invoke(
                {
                    "raw_text": text_for_processing,
                    "entity_hints": format_entity_hints(qa_entity_hints),
                    "relation_hints": format_relation_hints(qa_relation_hints),
                    "context_dependencies": format_context_dependencies(
                        qa_context_dependencies
                    ),
                    "mentor_guidance": format_mentor_guidance(mentor_guidance),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: JointExtractionResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # 转换为现有格式（兼容后续节点）
            # v3.4扩展版：实体类型扩展为6种（新增功能、事件）
            entities_dict = {
                "道路": [],
                "POI": [],
                "建筑物": [],
                "街区": [],
                "功能": [],
                "事件": [],
            }
            # P15修复：创建功能实体和事件实体详细列表
            function_entities_list = []
            event_entities_list = []

            for e in result.entities:
                entity_type = extract_enum_value(e.type)  # P15改进：使用工具函数
                if entity_type in entities_dict:
                    entities_dict[entity_type].append(e.name)

                # P15修复：功能实体和事件实体需要保存完整属性
                if entity_type == "功能" and e.function_attrs:
                    function_entities_list.append(
                        {
                            "name": e.name,
                            "evidence": e.evidence,
                            "function_attrs": e.function_attrs.model_dump(
                                exclude_none=True
                            ),
                        }
                    )
                elif entity_type == "事件" and e.event_attrs:
                    event_entities_list.append(
                        {
                            "name": e.name,
                            "evidence": e.evidence,
                            "event_attrs": e.event_attrs.model_dump(exclude_none=True),
                        }
                    )

            triples_list = [
                {
                    "head": t.head,
                    "relation": extract_enum_value(t.relation),  # P15改进：使用工具函数
                    "tail": t.tail,
                    "evidence": t.evidence,
                    "confidence": extract_enum_value(
                        t.confidence
                    ),  # P15改进：使用工具函数
                    "attributes": t.attributes.model_dump(
                        exclude_none=True, mode="json"
                    )
                    if t.attributes
                    else {},  # P15修复：mode='json' 转换 Enum 为字符串
                }
                for t in result.triples
            ]

            # P15调试：确认 triples_list 的 relation 类型
            for t_item in triples_list[:3]:
                rel = t_item.get("relation")
                logger.debug(
                    f"[Joint_NER_RE-Debug] corpus={corpus_id}: <{t_item.get('head')}, {rel} (type={type(rel).__name__}), {t_item.get('tail')}>"
                )

            logger.info(
                f"[Joint_NER_RE] 完成: {len(result.entities)}个实体, "
                f"{len(result.triples)}个三元组, 置信度={result.overall_confidence}"
            )

            writer(
                {
                    "step": "joint_ner_re",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "entity_count": len(result.entities),
                    "triple_count": len(result.triples),
                    "confidence": result.overall_confidence,
                }
            )

            # P14新增：困惑检测（如果启用了查询功能）
            if enable_query:
                from .prompts import detect_extraction_confusion

                query_count = state.get("query_count", 0)
                max_queries = state.get("max_queries", 2)

                # 检测困惑（限制查询次数防止无限循环）
                if query_count < max_queries:
                    confusion = detect_extraction_confusion(
                        result.model_dump(), dict(state)
                    )
                    confidence_gate = str(
                        state.get("mentor_query_min_confidence", "medium")
                    ).lower()
                    should_ask = False
                    if confusion:
                        cc = str(confusion.get("current_confidence", "medium")).lower()
                        should_ask = (
                            cc == "low"
                            if confidence_gate == "low"
                            else cc in ("low", "medium")
                        )
                    if should_ask:
                        logger.info(
                            f"[Joint_NER_RE] 检测到困惑，请求导师帮助: {confusion['query_type']}"
                        )
                        writer(
                            {
                                "step": "joint_ner_re",
                                "corpus_id": corpus_id,
                                "status": "needs_mentor_help",
                                "query_type": confusion["query_type"],
                                "query_content": confusion["query_content"],
                            }
                        )

                        return {
                            "entities": entities_dict,
                            "triples": triples_list,
                            "joint_extraction_result": result.model_dump(),
                            "extraction_strategy": "joint",
                            "function_entities": function_entities_list,  # P15修复：新增
                            "event_entities": event_entities_list,  # P15修复：新增
                            "mentor_query": confusion,
                            "query_source_node": "joint_ner_re",
                            "needs_mentor_help": True,
                            "query_count": query_count,
                            "current_step": StepEnum.QA_MENTOR,  # 回退到导师
                        }

            # 正常返回
            return {
                "entities": entities_dict,
                "triples": triples_list,
                "joint_extraction_result": result.model_dump(),
                "extraction_strategy": "joint",
                "function_entities": function_entities_list,  # P15修复：新增
                "event_entities": event_entities_list,  # P15修复：新增
                "needs_mentor_help": False,  # P14新增：标记不需要帮助
                "current_step": StepEnum.SELF_CHECK_JOINT,
            }

        except Exception as e:
            logger.error(f"[Joint_NER_RE] 失败: {e}")
            writer(
                {
                    "step": "joint_ner_re",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            return {
                "entities": {
                    "道路": [],
                    "POI": [],
                    "建筑物": [],
                    "街区": [],
                    "功能": [],
                    "事件": [],
                },
                "triples": [],
                "error": str(e),
                "needs_mentor_help": False,
                "current_step": StepEnum.EVAL,  # 失败时跳过校验，直接评估
            }

    return joint_ner_re_node


# ===== P9新增：Self-Check-Joint节点（含Reflexion） =====


def create_self_check_joint_node(llm: Any):
    """
    创建联合抽取校验节点（含Reflexion）

    职责：
    1. 校验联合抽取结果
    2. 生成自然语言反思建议
    3. 决定是否需要重抽
    """
    parser = PydanticOutputParser(pydantic_object=SelfCheckJointResult)

    async def self_check_joint_node(state: CorpusState, writer: StreamWriter) -> Dict:
        corpus_id = state["corpus_id"]
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)
        logger.info(
            f"[Self-Check-Joint] 校验语料: {corpus_id}, 重试: {retry_count}/{max_retries}"
        )

        writer(
            {
                "step": "self_check_joint",
                "corpus_id": corpus_id,
                "status": "started",
                "retry_count": retry_count,
            }
        )

        try:
            text = get_text_for_processing(state)

            # 获取反思历史（用于迭代改进）
            reflection_history = state.get("reflection_history", [])
            previous_reflection = format_reflection_history(reflection_history)

            # P12改进：使用增强版提示词（四维度校验+结构化反思）
            prompt_text = SELF_CHECK_JOINT_PROMPT_V2.invoke(
                {
                    "raw_text": text,
                    "entities": format_joint_entities(
                        state.get("joint_extraction_result", {}).get("entities", [])
                    ),
                    "triples": format_joint_triples(state.get("triples", [])),
                    "semantic_summary": state.get("semantic_summary", ""),
                    "context_dependencies": format_context_dependencies(
                        state.get("qa_context_dependencies", [])
                    ),
                    "previous_reflection": previous_reflection,
                    "improvement_attempts": state.get("improvement_strategy", ""),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckJointResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # 记录反思历史
            reflection_history.append(result.reflection_text)

            logger.info(
                f"[Self-Check-Joint] 完成: confidence={result.overall_confidence}, "
                f"retry_suggested={result.retry_suggested}"
            )
            logger.info(f"[Self-Check-Joint] 反思: {result.reflection_text[:100]}...")

            writer(
                {
                    "step": "self_check_joint",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "confidence": result.overall_confidence,
                    "reflection": result.reflection_text[:200],
                    "retry_suggested": result.retry_suggested,
                }
            )

            return {
                "self_check_joint_result": result.model_dump(),
                "reflection_text": result.reflection_text,
                "improvement_strategy": result.improvement_strategy,
                "reflection_history": reflection_history,
                "retry_count": retry_count
                + (1 if result.retry_suggested else 0),  # P17修复：只在建议重试时增加
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "current_step": StepEnum.EVAL,
            }

        except Exception as e:
            logger.error(f"[Self-Check-Joint] 失败: {e}")
            writer(
                {
                    "step": "self_check_joint",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            return {
                "self_check_joint_result": {},
                "error": str(e),
                "retry_count": min(retry_count + 1, max_retries),  # 确保不超过上限
                "retry_suggested": True,  # 异常时建议重试
                "current_step": StepEnum.EVAL,
            }

    return self_check_joint_node


# ===== P9新增：Self-Check-QA节点 =====


def create_self_check_qa_node(llm: Any):
    """
    创建QA脚手架校验节点

    职责：
    1. 校验QA问答质量
    2. 检查实体/关系覆盖度
    3. 决定是否重新生成QA
    """
    parser = PydanticOutputParser(pydantic_object=SelfCheckQAResult)

    async def self_check_qa_node(state: CorpusState, writer: StreamWriter) -> Dict:
        corpus_id = state["corpus_id"]
        retry_count = state.get("retry_count", 0)
        logger.info(f"[Self-Check-QA] 校验语料: {corpus_id}, 重试: {retry_count}")

        writer(
            {
                "step": "self_check_qa",
                "corpus_id": corpus_id,
                "status": "started",
                "retry_count": retry_count,
            }
        )

        try:
            text = get_text_for_processing(state)

            qa_result = state.get("qa_scaffold_result", {})

            prompt_text = SELF_CHECK_QA_PROMPT.invoke(
                {
                    "raw_text": text,
                    "qa_pairs": format_qa_pairs_for_check(
                        qa_result.get("qa_pairs", [])
                    ),
                    "entity_hints": format_entity_hints(
                        qa_result.get("entity_hints", [])
                    ),
                    "relation_hints": format_relation_hints(
                        qa_result.get("relation_hints", [])
                    ),
                    "semantic_summary": qa_result.get("semantic_summary", ""),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckQAResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            logger.info(
                f"[Self-Check-QA] 完成: entity_coverage={result.entity_coverage}, "
                f"relation_coverage={result.relation_coverage}, retry={result.retry_suggested}"
            )

            writer(
                {
                    "step": "self_check_qa",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "entity_coverage": result.entity_coverage,
                    "relation_coverage": result.relation_coverage,
                    "retry_suggested": result.retry_suggested,
                }
            )

            return {
                "self_check_qa_result": result.model_dump(),
                "retry_count": retry_count
                + (1 if result.retry_suggested else 0),  # P17修复：只在建议重试时增加
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "current_step": StepEnum.JOINT_NER_RE,
            }

        except Exception as e:
            logger.error(f"[Self-Check-QA] 失败: {e}")
            writer(
                {
                    "step": "self_check_qa",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            # 失败时继续到联合抽取
            return {
                "self_check_qa_result": {},
                "error": str(e),
                "current_step": StepEnum.JOINT_NER_RE,
            }

    return self_check_qa_node


# ===== P9新增：Self-Check-Eval节点 =====


def create_self_check_eval_node(llm: Any):
    """
    创建评估结果校验节点

    职责：
    1. 校验评分合理性
    2. 检查修正效果
    3. 决定是否重新评估
    """
    parser = PydanticOutputParser(pydantic_object=SelfCheckEvalResult)

    async def self_check_eval_node(state: CorpusState, writer: StreamWriter) -> Dict:
        corpus_id = state["corpus_id"]
        retry_count = state.get("retry_count", 0)
        logger.info(f"[Self-Check-Eval] 校验语料: {corpus_id}, 重试: {retry_count}")

        writer(
            {
                "step": "self_check_eval",
                "corpus_id": corpus_id,
                "status": "started",
                "retry_count": retry_count,
            }
        )

        try:
            text = get_text_for_processing(state)

            prompt_text = SELF_CHECK_EVAL_PROMPT.invoke(
                {
                    "raw_text": text,
                    "eval_scores": format_eval_scores_for_check(
                        state.get("eval_scores", [])
                    ),
                    "corrected_triples": format_triples(
                        state.get("corrected_triples", [])
                    ),
                    "eval_passed": state.get("eval_passed", False),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckEvalResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            logger.info(
                f"[Self-Check-Eval] 完成: score_consistency={result.score_consistency}, "
                f"retry={result.retry_suggested}"
            )

            writer(
                {
                    "step": "self_check_eval",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "score_consistency": result.score_consistency,
                    "retry_suggested": result.retry_suggested,
                }
            )

            return {
                "self_check_eval_result": result.model_dump(),
                "retry_count": retry_count
                + (1 if result.retry_suggested else 0),  # P17修复：只在建议重试时增加
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "current_step": StepEnum.LABEL,
            }

        except Exception as e:
            logger.error(f"[Self-Check-Eval] 失败: {e}")
            writer(
                {
                    "step": "self_check_eval",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            return {
                "self_check_eval_result": {},
                "error": str(e),
                "current_step": StepEnum.LABEL,
            }

    return self_check_eval_node


# ===== P9新增：Self-Check-Label节点 =====


def create_self_check_label_node(llm: Any):
    """
    创建标注结果校验节点

    职责：
    1. 校验实体/关系属性
    2. 检查属性完整性
    3. 决定是否重新标注
    """
    parser = PydanticOutputParser(pydantic_object=SelfCheckLabelResult)

    async def self_check_label_node(state: CorpusState, writer: StreamWriter) -> Dict:
        corpus_id = state["corpus_id"]
        retry_count = state.get("retry_count", 0)
        logger.info(f"[Self-Check-Label] 校验语料: {corpus_id}, 重试: {retry_count}")

        writer(
            {
                "step": "self_check_label",
                "corpus_id": corpus_id,
                "status": "started",
                "retry_count": retry_count,
            }
        )

        try:
            text = get_text_for_processing(state)

            entity_attrs = state.get("entity_attrs", {})
            relation_attrs = state.get("relation_attrs", {})

            # 格式化属性用于提示词
            entity_attrs_str = (
                "\n".join(
                    [f"- {name}: {attrs}" for name, attrs in entity_attrs.items()]
                )
                if entity_attrs
                else "(无实体属性)"
            )

            relation_attrs_str = (
                "\n".join(
                    [f"- {key}: {attrs}" for key, attrs in relation_attrs.items()]
                )
                if relation_attrs
                else "(无关系属性)"
            )

            prompt_text = SELF_CHECK_LABEL_PROMPT.invoke(
                {
                    "raw_text": text,
                    "entity_attrs": entity_attrs_str,
                    "relation_attrs": relation_attrs_str,
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckLabelResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            logger.info(
                f"[Self-Check-Label] 完成: attr_completeness={result.attr_completeness}, "
                f"retry={result.retry_suggested}"
            )

            writer(
                {
                    "step": "self_check_label",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "attr_completeness": result.attr_completeness,
                    "retry_suggested": result.retry_suggested,
                }
            )

            # 如果校验通过，更新属性为校验后的版本
            if result.verified_entity_attrs:
                entity_attrs = result.verified_entity_attrs
            if result.verified_relation_attrs:
                relation_attrs = result.verified_relation_attrs

            return {
                "self_check_label_result": result.model_dump(),
                "entity_attrs": entity_attrs,
                "relation_attrs": relation_attrs,
                "retry_count": retry_count
                + (1 if result.retry_suggested else 0),  # P17修复：只在建议重试时增加
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "current_step": StepEnum.DONE,
            }

        except Exception as e:
            logger.error(f"[Self-Check-Label] 失败: {e}")
            writer(
                {
                    "step": "self_check_label",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            return {
                "self_check_label_result": {},
                "error": str(e),
                "current_step": StepEnum.DONE,
            }

    return self_check_label_node


# ===== P9新增：Self-Check-Filter节点（可选） =====


def create_self_check_filter_node(llm: Any):
    """
    创建筛选校验节点（可选，默认不启用）

    职责：
    1. 校验Filter筛选判定的合理性
    2. 检测误筛（有效文本被判定为无效）
    3. 检测误判（无效文本被判定为有效）
    4. 生成反思建议
    """
    from .schemas import SelfCheckFilterResult
    from .prompts import SELF_CHECK_FILTER_PROMPT

    parser = PydanticOutputParser(pydantic_object=SelfCheckFilterResult)

    async def self_check_filter_node(state: CorpusState, writer: StreamWriter) -> Dict:
        corpus_id = state["corpus_id"]
        retry_count = state.get("retry_count", 0)
        logger.info(f"[Self-Check-Filter] 校验语料: {corpus_id}, 重试: {retry_count}")

        writer(
            {
                "step": "self_check_filter",
                "corpus_id": corpus_id,
                "status": "started",
                "retry_count": retry_count,
            }
        )

        try:
            text = state.get("raw_text", "")
            filter_result = state.get("filter_result", {})

            prompt_text = SELF_CHECK_FILTER_PROMPT.invoke(
                {
                    "raw_text": text,
                    "is_valid": filter_result.get("is_valid", True),
                    "confidence": filter_result.get("confidence", "medium"),
                    "skip_reason": filter_result.get("skip_reason", ""),
                    "has_geo_entity": filter_result.get("has_geo_entity", False),
                    "has_spatial_relation": filter_result.get(
                        "has_spatial_relation", False
                    ),
                    "geo_entity_hint": filter_result.get("geo_entity_hint", ""),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckFilterResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            logger.info(
                f"[Self-Check-Filter] 完成: verified_is_valid={result.verified_is_valid}, "
                f"false_negative={result.false_negative_detected}, retry={result.retry_suggested}"
            )

            writer(
                {
                    "step": "self_check_filter",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "verified_is_valid": result.verified_is_valid,
                    "false_negative_detected": result.false_negative_detected,
                    "retry_suggested": result.retry_suggested,
                }
            )

            # 如果检测到误筛，更新filter_result
            updated_filter_result = {}
            if result.false_negative_detected:
                updated_filter_result = {
                    "is_valid": True,
                    "confidence": result.verified_confidence,
                    "has_geo_entity": True,
                    "geo_entity_hint": ", ".join(result.geo_entity_missed[:3]),
                }
            elif result.false_positive_detected:
                updated_filter_result = {
                    "is_valid": False,
                    "confidence": result.verified_confidence,
                    "skip_reason": result.invalid_reason,
                }

            return {
                "self_check_filter_result": result.model_dump(),
                "filter_result": updated_filter_result
                if updated_filter_result
                else state.get("filter_result", {}),
                "retry_count": retry_count
                + (1 if result.retry_suggested else 0),  # P17修复：只在建议重试时增加
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "current_step": StepEnum.NORMALIZE
                if result.verified_is_valid
                else StepEnum.DONE,
            }

        except Exception as e:
            logger.error(f"[Self-Check-Filter] 失败: {e}")
            writer(
                {
                    "step": "self_check_filter",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            return {
                "self_check_filter_result": {},
                "error": str(e),
                "current_step": StepEnum.NORMALIZE,  # 失败时继续处理
            }

    return self_check_filter_node


# ===== P9新增：Self-Check-Normalize节点（可选） =====


def create_self_check_normalize_node(llm: Any):
    """
    创建归一化校验节点（可选，默认不启用）

    职责：
    1. 校验归一化质量
    2. 检查语义保留
    3. 检测信息添加/丢失问题
    4. 生成反思建议
    """
    from .schemas import SelfCheckNormalizeResult
    from .prompts import SELF_CHECK_NORMALIZE_PROMPT, format_normalizations_for_check

    parser = PydanticOutputParser(pydantic_object=SelfCheckNormalizeResult)

    async def self_check_normalize_node(
        state: CorpusState, writer: StreamWriter
    ) -> Dict:
        corpus_id = state["corpus_id"]
        retry_count = state.get("retry_count", 0)
        logger.info(
            f"[Self-Check-Normalize] 校验语料: {corpus_id}, 重试: {retry_count}"
        )

        writer(
            {
                "step": "self_check_normalize",
                "corpus_id": corpus_id,
                "status": "started",
                "retry_count": retry_count,
            }
        )

        try:
            raw_text = state.get("raw_text", "")
            normalize_result = state.get("normalize_result", {})

            prompt_text = SELF_CHECK_NORMALIZE_PROMPT.invoke(
                {
                    "raw_text": raw_text,
                    "normalized_text": normalize_result.get("normalized_text", ""),
                    "confidence": normalize_result.get("confidence", "medium"),
                    "has_changes": normalize_result.get("has_changes", False),
                    "preserved_semantics": normalize_result.get(
                        "preserved_semantics", True
                    ),
                    "normalizations": format_normalizations_for_check(
                        normalize_result.get("normalizations", [])
                    ),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckNormalizeResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            logger.info(
                f"[Self-Check-Normalize] 完成: semantics_preserved={result.semantics_preserved}, "
                f"info_added={result.info_added}, retry={result.retry_suggested}"
            )

            writer(
                {
                    "step": "self_check_normalize",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "semantics_preserved": result.semantics_preserved,
                    "retry_suggested": result.retry_suggested,
                }
            )

            # 如果语义丢失或添加了信息，使用原文
            updated_normalized_text = result.verified_normalized_text
            if result.info_added or result.info_lost:
                logger.warning(f"[Self-Check-Normalize] 检测到语义问题，使用原文")
                updated_normalized_text = raw_text

            return {
                "self_check_normalize_result": result.model_dump(),
                "normalized_text": updated_normalized_text,
                "retry_count": retry_count
                + (1 if result.retry_suggested else 0),  # P17修复：只在建议重试时增加
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "current_step": StepEnum.QA_SCAFFOLD,
            }

        except Exception as e:
            logger.error(f"[Self-Check-Normalize] 失败: {e}")
            writer(
                {
                    "step": "self_check_normalize",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            return {
                "self_check_normalize_result": {},
                "error": str(e),
                "current_step": StepEnum.QA_SCAFFOLD,  # 失败时继续处理
            }

    return self_check_normalize_node


# ===== P10新增：批量LLM调用节点 =====


def create_batch_joint_extraction_node(llm: Any, batch_llm_size: int = 5):
    """
    创建批量联合抽取节点

    职责：
    1. 一次LLM调用处理batch_llm_size条语料
    2. 同时抽取实体和三元组
    3. 发现跨语料别名关系
    4. 返回批量结果，失败时标记需要fallback
    """
    from .schemas import BatchExtractionResult, BatchCorpusResult
    from .prompts import BATCH_JOINT_PROMPT, format_batch_corpus

    parser = PydanticOutputParser(pydantic_object=BatchExtractionResult)

    async def batch_joint_extraction_node(
        corpus_list: List[Dict], writer: StreamWriter
    ) -> Dict:
        """
        批量联合抽取

        Args:
            corpus_list: 语料列表 [{"id": ..., "text": ...}, ...]
            writer: StreamWriter for progress events

        Returns:
            {
                "batch_results": {corpus_id: {entities, triples, confidence}},
                "cross_corpus_aliases": [...],
                "needs_fallback": False,
            }
        """
        batch_size = len(corpus_list)
        logger.info(f"[Batch_Joint] 处理 {batch_size} 条语料")

        writer(
            {
                "step": "batch_joint",
                "status": "started",
                "batch_size": batch_size,
            }
        )

        if batch_size == 0:
            return {
                "batch_results": {},
                "cross_corpus_aliases": [],
                "needs_fallback": False,
            }

        max_retries = 2  # 同阶段重试优先，减少直接fallback
        retry_delay = 2.0  # 初始延迟秒数

        for retry in range(max_retries):
            try:
                # 构建批量输入
                corpus_list_str = format_batch_corpus(corpus_list)

                # 调用LLM
                prompt_text = BATCH_JOINT_PROMPT.invoke(
                    {
                        "batch_size": batch_size,
                        "corpus_list": corpus_list_str,
                    }
                )
                full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"

                response = await llm.ainvoke(full_prompt)
                try:
                    result: BatchExtractionResult = safe_parse_json_with_quote_fix(
                        parser, response.content
                    )
                    parsed_results = result.results
                    cross_aliases = result.cross_corpus_aliases
                    cross_relations = result.cross_corpus_relations
                    overall_confidence = result.overall_confidence
                    model_dump_result = result.model_dump()
                except Exception as parse_error:
                    logger.warning(
                        f"[Batch_Joint] 严格解析失败，尝试宽松解析: {parse_error}"
                    )
                    lenient = parse_batch_extraction_lenient(response.content)
                    parsed_results = lenient.get("results", [])
                    cross_aliases = lenient.get("cross_corpus_aliases", [])
                    cross_relations = lenient.get("cross_corpus_relations", [])
                    overall_confidence = lenient.get("overall_confidence", "medium")
                    model_dump_result = lenient
                    if not parsed_results:
                        raise parse_error

                # 转换为字典格式（P15修复：将JointTriple转换为字典，避免后续.get()错误）
                batch_results = {}
                for r in parsed_results:
                    if isinstance(r, dict):
                        corpus_id = r.get("corpus_id")
                        triples_src = r.get("triples", [])
                        confidence = r.get("confidence", "medium")
                        has_geo_info = r.get("has_geo_info", True)
                        skip_reason = r.get("skip_reason")
                        entities_src = r.get("entities", DEFAULT_ENTITY_DICT.copy())
                    else:
                        corpus_id = r.corpus_id
                        triples_src = r.triples
                        confidence = r.confidence
                        has_geo_info = r.has_geo_info
                        skip_reason = r.skip_reason
                        entities_src = r.entities

                    # 将 JointTriple 对象转换为字典列表
                    triples_list = [
                        {
                            "head": t.get("head") if isinstance(t, dict) else t.head,
                            "relation": extract_enum_value(
                                t.get("relation") if isinstance(t, dict) else t.relation
                            ),
                            "tail": t.get("tail") if isinstance(t, dict) else t.tail,
                            "evidence": t.get("evidence", "")
                            if isinstance(t, dict)
                            else t.evidence,
                            "confidence": extract_enum_value(
                                t.get("confidence", "medium")
                                if isinstance(t, dict)
                                else t.confidence
                            ),
                            "attributes": (
                                (t.get("attributes") or {})
                                if isinstance(t, dict)
                                else (
                                    t.attributes.model_dump(
                                        exclude_none=True, mode="json"
                                    )
                                    if t.attributes
                                    else {}
                                )
                            ),
                        }
                        for t in (triples_src or [])
                    ]
                    if not corpus_id:
                        continue
                    batch_results[corpus_id] = {
                        "entities": entities_src,
                        "triples": triples_list,  # 使用转换后的字典列表
                        "confidence": confidence,
                        "has_geo_info": has_geo_info,
                        "skip_reason": skip_reason,
                    }

                logger.info(
                    f"[Batch_Joint] 完成: {len(batch_results)}条语料, "
                    f"跨语料别名: {len(cross_aliases)}个, "
                    f"置信度: {overall_confidence}"
                )

                writer(
                    {
                        "step": "batch_joint",
                        "status": "completed",
                        "batch_size": len(batch_results),
                        "cross_corpus_aliases_count": len(cross_aliases),
                        "confidence": overall_confidence,
                    }
                )

                return {
                    "batch_results": batch_results,
                    "cross_corpus_aliases": cross_aliases,
                    "cross_corpus_relations": cross_relations,
                    "batch_extraction_result": model_dump_result,
                    "needs_fallback": False,
                }

            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"[Batch_Joint] 重试 {retry + 2}/{max_retries}: {e}")
                    await asyncio.sleep(retry_delay * (retry + 1))  # 指数退避
                    continue
                else:
                    logger.error(f"[Batch_Joint] 最终失败: {e}")
                    writer(
                        {
                            "step": "batch_joint",
                            "status": "error",
                            "error": str(e),
                            "batch_size": batch_size,
                        }
                    )

                    # 标记需要fallback为单条处理
                    return {
                        "batch_results": {},
                        "cross_corpus_aliases": [],
                        "needs_fallback": True,
                        "fallback_reason": str(e),
                    }

    return batch_joint_extraction_node


def create_batch_self_check_node(llm: Any):
    """
    创建批量校验修正节点（P18改进：修正型校验）

    职责：
    1. 校验批量抽取结果的质量
    2. 发现问题直接修正（而非拒绝）
    3. 仅对无法修正的严重问题才拒绝
    """
    from .schemas import BatchSelfCheckResult
    from .prompts import (
        BATCH_SELF_CHECK_PROMPT,
        format_batch_results_for_check,
        format_cross_corpus_aliases,
    )

    parser = PydanticOutputParser(pydantic_object=BatchSelfCheckResult)

    async def batch_self_check_node(
        batch_results: Dict, cross_corpus_aliases: List[Dict], writer: StreamWriter
    ) -> Dict:
        """
        批量校验修正（P18改进）

        Args:
            batch_results: {corpus_id: {entities, triples, confidence}}
            cross_corpus_aliases: 跨语料别名列表
            writer: StreamWriter

        Returns:
            {
                "verified_results": [...],  # 包含修正后的结果
                "rejected_results": [...],  # 仅严重问题
                "verified_aliases": [...],
                "correction_count": int,    # P18新增：修正数量
                ...
            }
        """
        logger.info(f"[Batch_Self_Check] 校验 {len(batch_results)} 条语料结果")

        writer(
            {
                "step": "batch_self_check",
                "status": "started",
                "batch_size": len(batch_results),
            }
        )

        if not batch_results:
            return {
                "verified_results": [],
                "rejected_results": [],
                "verified_aliases": [],
                "correction_count": 0,
                "retry_suggested": False,
                "fallback_to_single": False,
            }

        try:
            # 格式化输入
            results_list = [
                {"corpus_id": cid, **data} for cid, data in batch_results.items()
            ]
            results_str = format_batch_results_for_check(results_list)
            aliases_str = format_cross_corpus_aliases(cross_corpus_aliases)

            # 调用LLM
            prompt_text = BATCH_SELF_CHECK_PROMPT.invoke(
                {
                    "batch_results": results_str,
                    "cross_corpus_aliases": aliases_str,
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"

            response = await llm.ainvoke(full_prompt)
            result: BatchSelfCheckResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # P18改进：合并修正结果
            final_verified_results = []
            correction_count = 0

            for r in result.verified_results:
                r_dict = r.model_dump(mode="json")

                # 合并实体：verified + corrected - rejected
                final_entities = {
                    **r_dict.get("verified_entities", {}),
                    **r_dict.get("corrected_entities", {}),
                }
                for entity in r_dict.get("rejected_entities", []):
                    # 从各类型中移除
                    for entity_type in final_entities:
                        if entity in final_entities[entity_type]:
                            final_entities[entity_type].remove(entity)

                # 合并三元组：verified + corrected
                final_triples = (
                    r_dict.get("verified_triples", []) +
                    r_dict.get("corrected_triples", [])
                )

                # 计算修正数量
                corrections = r_dict.get("correction_records", [])
                correction_count += len(corrections)

                # 构建最终结果
                final_result = {
                    "corpus_id": r_dict["corpus_id"],
                    "entities": final_entities,
                    "full_entities": r_dict.get("verified_full_entities", []) + r_dict.get("corrected_full_entities", []),
                    "triples": final_triples,
                    "confidence": r_dict.get("confidence", "medium"),
                    "has_geo_info": r_dict.get("has_geo_info", True),
                    "correction_records": corrections,
                }
                final_verified_results.append(final_result)

            logger.info(
                f"[Batch_Self_Check] 完成: 修正 {correction_count} 处, "
                f"通过 {len(final_verified_results)} 条, "
                f"拒绝 {len(result.rejected_results)} 条（严重问题）"
            )

            writer(
                {
                    "step": "batch_self_check",
                    "status": "completed",
                    "verified_count": len(final_verified_results),
                    "correction_count": correction_count,
                    "rejected_count": len(result.rejected_results),
                    "retry_suggested": result.retry_suggested,
                    "fallback_to_single": result.fallback_to_single,
                }
            )

            return {
                "verified_results": final_verified_results,
                "rejected_results": result.rejected_results,  # 仅严重问题
                "verified_aliases": result.verified_aliases,
                "rejected_aliases": result.rejected_aliases,
                "batch_self_check_result": result.model_dump(mode="json"),
                "correction_count": correction_count,
                "retry_suggested": result.retry_suggested,
                "fallback_to_single": result.fallback_to_single,
            }

        except Exception as e:
            logger.error(f"[Batch_Self_Check] 失败: {e}")
            writer(
                {
                    "step": "batch_self_check",
                    "status": "error",
                    "error": str(e),
                }
            )

            # 校验失败时，保守策略：全部通过但标记低置信度
            return {
                "verified_results": [
                    {"corpus_id": cid, **data, "confidence": "low"}
                    for cid, data in batch_results.items()
                ],
                "rejected_results": [],
                "verified_aliases": cross_corpus_aliases,
                "correction_count": 0,
                "retry_suggested": False,
                "fallback_to_single": False,
                "error": str(e),
            }

    return batch_self_check_node


# ===== P15新增：批量Self-Check-QA节点 =====


def create_batch_self_check_qa_node(llm: Any):
    """
    创建批量QA脚手架校验修正节点（P18改进：修正型校验）

    职责：
    1. 校验批量QA脚手架结果的质量
    2. 发现问题直接修正或补充
    3. 仅对严重问题才拒绝

    Args:
        llm: LLM实例

    Returns:
        batch_self_check_qa_node 函数
    """
    from .schemas import BatchSelfCheckQAResult
    from .prompts import (
        BATCH_SELF_CHECK_QA_PROMPT,
        format_batch_qa_results_for_check,
        format_corpus_texts_for_check,
    )

    parser = PydanticOutputParser(pydantic_object=BatchSelfCheckQAResult)

    async def batch_self_check_qa_node(
        batch_qa_results: Dict, corpus_texts: Dict, writer: StreamWriter
    ) -> Dict:
        """
        批量QA脚手架校验修正（P18改进）

        Args:
            batch_qa_results: {corpus_id: {qa_pairs, entity_hints, relation_hints, confidence}}
            corpus_texts: {corpus_id: text} 原始文本
            writer: StreamWriter

        Returns:
            {
                "verified_results": [{corpus_id, final_qa_pairs, correction_count}],
                "rejected_results": [{corpus_id, reason}],  # 仅严重问题
                ...
            }
        """
        batch_size = len(batch_qa_results)
        logger.info(f"[Batch_Self_Check_QA] 校验 {batch_size} 条语料QA结果")

        writer(
            {
                "step": "batch_self_check_qa",
                "status": "started",
                "batch_size": batch_size,
            }
        )

        if not batch_qa_results:
            logger.info("[Batch_Self_Check_QA] 无QA结果，返回空")
            return {
                "verified_results": [],
                "rejected_results": [],
                "overall_confidence": "medium",
                "correction_count": 0,
                "retry_suggested": False,
            }

        try:
            # 格式化输入
            qa_results_str = format_batch_qa_results_for_check(
                batch_qa_results, corpus_texts
            )
            texts_str = format_corpus_texts_for_check(corpus_texts)

            # 调用LLM
            prompt_text = BATCH_SELF_CHECK_QA_PROMPT.invoke(
                {
                    "batch_qa_results": qa_results_str,
                    "corpus_texts": texts_str,
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"

            response = await llm.ainvoke(full_prompt)
            result: BatchSelfCheckQAResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # P18改进：合并修正结果
            final_verified_results = []
            correction_count = 0

            for r in result.verified_results:
                r_dict = r.model_dump(mode="json")

                # 合并QA问答对：verified + corrected + added
                final_qa_pairs = (
                    r_dict.get("verified_qa_pairs", []) +
                    r_dict.get("corrected_qa_pairs", []) +
                    r_dict.get("added_qa_pairs", [])
                )

                # 计算修正数量
                corrections = (
                    len(r_dict.get("corrected_qa_pairs", [])) +
                    len(r_dict.get("added_qa_pairs", []))
                )
                correction_count += corrections

                final_result = {
                    "corpus_id": r_dict["corpus_id"],
                    "qa_pairs": final_qa_pairs,  # 最终QA问答对
                    "verified_qa_pairs": r_dict.get("verified_qa_pairs", []),
                    "corrected_qa_pairs": r_dict.get("corrected_qa_pairs", []),
                    "added_qa_pairs": r_dict.get("added_qa_pairs", []),
                    "entity_coverage": r_dict.get("entity_coverage", "medium"),
                    "relation_coverage": r_dict.get("relation_coverage", "medium"),
                    "confidence": r_dict.get("confidence", "medium"),
                }
                final_verified_results.append(final_result)

            logger.info(
                f"[Batch_Self_Check_QA] 完成: 修正/补充 {correction_count} 个问答对, "
                f"通过 {len(final_verified_results)} 条, "
                f"拒绝 {len(result.rejected_results)} 条（严重问题）"
            )

            writer(
                {
                    "step": "batch_self_check_qa",
                    "status": "completed",
                    "verified_count": len(final_verified_results),
                    "correction_count": correction_count,
                    "rejected_count": len(result.rejected_results),
                    "overall_confidence": result.overall_confidence,
                    "retry_suggested": result.retry_suggested,
                }
            )

            return {
                "verified_results": final_verified_results,
                "rejected_results": result.rejected_results,  # 仅严重问题
                "batch_self_check_qa_result": result.model_dump(mode="json"),
                "overall_confidence": result.overall_confidence,
                "correction_count": correction_count,
                "retry_suggested": result.retry_suggested,
            }

        except Exception as e:
            logger.error(f"[Batch_Self_Check_QA] 失败: {e}")
            writer(
                {
                    "step": "batch_self_check_qa",
                    "status": "error",
                    "error": str(e),
                }
            )

            # 校验失败时，保守策略：全部通过但标记低置信度
            return {
                "verified_results": [
                    {
                        "corpus_id": cid,
                        "qa_pairs": qa_data.get("qa_pairs", []),
                        "verified_qa_pairs": qa_data.get("qa_pairs", []),
                        "corrected_qa_pairs": [],
                        "added_qa_pairs": [],
                        "entity_coverage": "low",
                        "relation_coverage": "low",
                        "confidence": "low",
                    }
                    for cid, qa_data in batch_qa_results.items()
                ],
                "rejected_results": [],
                "overall_confidence": "low",
                "correction_count": 0,
                "retry_suggested": False,
                "error": str(e),
            }

    return batch_self_check_qa_node


# ===== P15新增：批量Eval节点 =====


def create_batch_eval_node(llm: Any, eval_threshold: float = 3.5):
    """
    创建批量评估节点

    职责：
    1. 对批量三元组进行评分（SEM、FAC、CON）
    2. 判断是否需要修正
    3. 决定每条语料的评估是否通过

    Args:
        llm: LLM实例
        eval_threshold: 评估通过阈值（默认3.5）

    Returns:
        batch_eval_node 函数
    """
    from .schemas import BatchEvalResult
    from .prompts import (
        BATCH_EVAL_PROMPT,
        format_batch_triples_for_eval,
        format_corpus_texts_for_check,
    )

    parser = PydanticOutputParser(pydantic_object=BatchEvalResult)

    async def batch_eval_node(
        batch_extraction_results: Dict, corpus_texts: Dict, writer: StreamWriter
    ) -> Dict:
        """
        批量评估

        Args:
            batch_extraction_results: {corpus_id: {entities, triples, confidence}}
            corpus_texts: {corpus_id: text} 原始文本
            writer: StreamWriter

        Returns:
            {
                "batch_eval_results": {corpus_id: {scores, eval_passed, corrected_triples}},
                "overall_confidence": "high/medium/low",
            }
        """
        batch_size = len(batch_extraction_results)
        logger.info(f"[Batch_Eval] 评估 {batch_size} 条语料")

        writer(
            {
                "step": "batch_eval",
                "status": "started",
                "batch_size": batch_size,
            }
        )

        if not batch_extraction_results:
            logger.info("[Batch_Eval] 无结果，返回空")
            return {
                "batch_eval_results": {},
                "overall_confidence": "medium",
            }

        # 检查是否有语料无三元组，直接标记为通过
        no_triple_corpus = []
        has_triple_corpus = {}
        for corpus_id, data in batch_extraction_results.items():
            if not data.get("triples"):
                no_triple_corpus.append(corpus_id)
            else:
                has_triple_corpus[corpus_id] = data

        # 无三元组的语料直接通过
        no_triple_results = {
            corpus_id: {
                "scores": [],
                "eval_passed": True,
                "corrected_triples": [],
                "confidence": "low",
            }
            for corpus_id in no_triple_corpus
        }

        if not has_triple_corpus:
            logger.info("[Batch_Eval] 所有语料无三元组")
            writer(
                {
                    "step": "batch_eval",
                    "status": "completed",
                    "batch_size": batch_size,
                    "all_no_triples": True,
                }
            )
            return {
                "batch_eval_results": no_triple_results,
                "overall_confidence": "low",
            }

        try:
            # 格式化输入
            triples_str = format_batch_triples_for_eval(has_triple_corpus)
            texts_str = format_corpus_texts_for_check(
                {cid: corpus_texts.get(cid, "") for cid in has_triple_corpus}
            )

            # 调用LLM
            prompt_text = BATCH_EVAL_PROMPT.invoke(
                {
                    "batch_triples": triples_str,
                    "corpus_texts": texts_str,
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"

            response = await llm.ainvoke(full_prompt)
            result: BatchEvalResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # 转换为字典格式
            eval_results_dict = {}
            for r in result.batch_eval_results:
                eval_results_dict[r.corpus_id] = {
                    "scores": r.scores,
                    "eval_passed": r.eval_passed,
                    "corrected_triples": r.corrected_triples,
                    "confidence": r.confidence,
                }

            # 合并无三元组的结果
            all_results = {**no_triple_results, **eval_results_dict}

            logger.info(
                f"[Batch_Eval] 完成: {len(all_results)} 条语料, "
                f"整体置信度: {result.overall_confidence}"
            )

            writer(
                {
                    "step": "batch_eval",
                    "status": "completed",
                    "batch_size": len(all_results),
                    "overall_confidence": result.overall_confidence,
                }
            )

            return {
                "batch_eval_results": all_results,
                "batch_eval_result": result.model_dump(mode="json"),
                "overall_confidence": result.overall_confidence,
            }

        except Exception as e:
            logger.error(f"[Batch_Eval] 失败: {e}")
            writer(
                {
                    "step": "batch_eval",
                    "status": "error",
                    "error": str(e),
                }
            )

            # 评估失败时，保守策略：全部通过但标记低置信度
            fallback_results = {
                corpus_id: {
                    "scores": [],
                    "eval_passed": True,
                    "corrected_triples": [],
                    "confidence": "low",
                }
                for corpus_id in batch_extraction_results
            }
            return {
                "batch_eval_results": fallback_results,
                "overall_confidence": "low",
                "error": str(e),
            }

    return batch_eval_node


# ===== P15新增：批量Self-Check-Eval节点 =====


def create_batch_self_check_eval_node(llm: Any):
    """
    创建批量评估校验修正节点（P18改进：修正型校验）

    职责：
    1. 校验批量评估结果的质量
    2. 发现问题直接修正三元组
    3. 仅对严重问题才拒绝

    Args:
        llm: LLM实例

    Returns:
        batch_self_check_eval_node 函数
    """
    from .schemas import BatchSelfCheckEvalResult
    from .prompts import (
        BATCH_SELF_CHECK_EVAL_PROMPT,
        format_corpus_texts_for_check,
    )

    parser = PydanticOutputParser(pydantic_object=BatchSelfCheckEvalResult)

    async def batch_self_check_eval_node(
        batch_eval_results: Dict, corpus_texts: Dict, writer: StreamWriter
    ) -> Dict:
        """
        批量评估校验修正（P18改进）

        Args:
            batch_eval_results: {corpus_id: {scores, eval_passed, corrected_triples}}
            corpus_texts: {corpus_id: text} 原始文本
            writer: StreamWriter

        Returns:
            {
                "verified_results": [{corpus_id, final_triples, correction_records}],
                "rejected_results": [{corpus_id, reason}],  # 仅严重问题
                ...
            }
        """
        batch_size = len(batch_eval_results)
        logger.info(f"[Batch_Self_Check_Eval] 校验 {batch_size} 条语料评估结果")

        writer(
            {
                "step": "batch_self_check_eval",
                "status": "started",
                "batch_size": batch_size,
            }
        )

        if not batch_eval_results:
            logger.info("[Batch_Self_Check_Eval] 无评估结果，返回空")
            return {
                "verified_results": [],
                "rejected_results": [],
                "overall_confidence": "medium",
                "correction_count": 0,
                "retry_suggested": False,
            }

        try:
            # 格式化输入
            eval_str = ""
            for corpus_id, data in batch_eval_results.items():
                scores = data.get("scores", [])
                eval_passed = data.get("eval_passed", True)
                confidence = data.get("confidence", "medium")
                corrected_triples = data.get("corrected_triples", [])

                score_str = ""
                if scores:
                    score_str = ", ".join(
                        [
                            f"<{s.get('triple', {}).get('head', '')},...> SEM:{s.get('SEM', 0)}"
                            for s in scores[:3]
                        ]
                    )

                triples_str = ""
                if corrected_triples:
                    triples_str = ", ".join(
                        [
                            f"<{t.get('head', '')}, {t.get('relation', '')}, {t.get('tail', '')}>"
                            for t in corrected_triples[:5]
                        ]
                    )

                eval_str += f"- [{corpus_id}] 通过:{eval_passed} 置信度:{confidence}\n  评分: {score_str}\n  三元组: {triples_str}\n"

            texts_str = format_corpus_texts_for_check(corpus_texts)

            # 调用LLM
            prompt_text = BATCH_SELF_CHECK_EVAL_PROMPT.invoke(
                {
                    "batch_eval_results": eval_str,
                    "corpus_texts": texts_str,
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"

            response = await llm.ainvoke(full_prompt)
            result: BatchSelfCheckEvalResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # P18改进：合并修正结果
            final_verified_results = []
            correction_count = 0

            for r in result.verified_results:
                r_dict = r.model_dump(mode="json")

                # 合并三元组：verified + corrected
                final_triples = (
                    r_dict.get("verified_triples", []) +
                    r_dict.get("corrected_triples", [])
                )

                # 计算修正数量
                corrections = r_dict.get("correction_records", [])
                correction_count += len(corrections)

                final_result = {
                    "corpus_id": r_dict["corpus_id"],
                    "verified_triples": final_triples,  # 最终三元组
                    "corrected_triples": r_dict.get("corrected_triples", []),
                    "rejected_triples": r_dict.get("rejected_triples", []),
                    "correction_records": corrections,
                    "score_consistency": r_dict.get("score_consistency", "medium"),
                    "confidence": r_dict.get("confidence", "medium"),
                }
                final_verified_results.append(final_result)

            logger.info(
                f"[Batch_Self_Check_Eval] 完成: 修正 {correction_count} 处, "
                f"通过 {len(final_verified_results)} 条, "
                f"拒绝 {len(result.rejected_results)} 条（严重问题）"
            )

            writer(
                {
                    "step": "batch_self_check_eval",
                    "status": "completed",
                    "verified_count": len(final_verified_results),
                    "correction_count": correction_count,
                    "rejected_count": len(result.rejected_results),
                    "overall_confidence": result.overall_confidence,
                    "retry_suggested": result.retry_suggested,
                }
            )

            return {
                "verified_results": final_verified_results,
                "rejected_results": result.rejected_results,  # 仅严重问题
                "batch_self_check_eval_result": result.model_dump(mode="json"),
                "overall_confidence": result.overall_confidence,
                "correction_count": correction_count,
                "retry_suggested": result.retry_suggested,
            }

        except Exception as e:
            logger.error(f"[Batch_Self_Check_Eval] 失败: {e}")
            writer(
                {
                    "step": "batch_self_check_eval",
                    "status": "error",
                    "error": str(e),
                }
            )

            # 校验失败时，保守策略：全部通过但标记低置信度
            return {
                "verified_results": [
                    {
                        "corpus_id": cid,
                        "verified_triples": data.get("corrected_triples", []),
                        "corrected_triples": [],
                        "rejected_triples": [],
                        "correction_records": [],
                        "score_consistency": "low",
                        "confidence": "low",
                    }
                    for cid, data in batch_eval_results.items()
                ],
                "rejected_results": [],
                "overall_confidence": "low",
                "correction_count": 0,
                "retry_suggested": False,
                "error": str(e),
            }

    return batch_self_check_eval_node


# ===== P15新增：批量Label节点 =====


def create_batch_label_node(llm: Any):
    """
    创建批量属性标注节点

    职责：
    1. 为批量实体添加类型和属性标注
    2. 为批量关系添加属性标注
    3. 确保属性有原文依据

    Args:
        llm: LLM实例

    Returns:
        batch_label_node 函数
    """
    from .schemas import BatchLabelResult
    from .prompts import (
        BATCH_LABEL_PROMPT,
        format_corpus_texts_for_check,
    )

    parser = PydanticOutputParser(pydantic_object=BatchLabelResult)

    async def batch_label_node(
        batch_eval_results: Dict, corpus_texts: Dict, writer: StreamWriter
    ) -> Dict:
        """
        批量属性标注

        Args:
            batch_eval_results: {corpus_id: {entities, triples, corrected_triples}}
            corpus_texts: {corpus_id: text} 原始文本
            writer: StreamWriter

        Returns:
            {
                "batch_label_results": {corpus_id: {entity_attrs, relation_attrs}},
                "overall_confidence": "high/medium/low",
            }
        """
        batch_size = len(batch_eval_results)
        logger.info(f"[Batch_Label] 标注 {batch_size} 条语料")

        writer(
            {
                "step": "batch_label",
                "status": "started",
                "batch_size": batch_size,
            }
        )

        if not batch_eval_results:
            logger.info("[Batch_Label] 无结果，返回空")
            return {
                "batch_label_results": {},
                "overall_confidence": "medium",
            }

        # 检查是否有语料无实体，直接标记为空属性
        no_entity_corpus = []
        has_entity_corpus = {}
        for corpus_id, data in batch_eval_results.items():
            entities = data.get("entities", {})
            if not entities or not any(entities.values()):
                no_entity_corpus.append(corpus_id)
            else:
                has_entity_corpus[corpus_id] = data

        # 无实体的语料直接返回空属性
        no_entity_results = {
            corpus_id: {
                "entity_attrs": {},
                "relation_attrs": {},
                "confidence": "low",
            }
            for corpus_id in no_entity_corpus
        }

        if not has_entity_corpus:
            logger.info("[Batch_Label] 所有语料无实体")
            writer(
                {
                    "step": "batch_label",
                    "status": "completed",
                    "batch_size": batch_size,
                    "all_no_entities": True,
                }
            )
            return {
                "batch_label_results": no_entity_results,
                "overall_confidence": "low",
            }

        try:
            # 格式化输入
            entities_triples_str = ""
            for corpus_id, data in has_entity_corpus.items():
                entities = data.get("entities", {})
                triples = data.get("corrected_triples", data.get("triples", []))
                confidence = data.get("confidence", "medium")
                mentor_note = str(data.get("mentor_note", "") or "").strip()

                # 格式化实体
                entity_str = ""
                for etype, names in entities.items():
                    if names:
                        entity_str += f"{etype}: {', '.join(names[:3])}; "

                # 格式化三元组
                triple_str = ""
                if triples:
                    triple_str = ", ".join(
                        [
                            f"<{t.get('head', '')}, {t.get('relation', '')}, {t.get('tail', '')}>"
                            for t in triples[:3]
                        ]
                    )

                entities_triples_str += (
                    f"- [{corpus_id}] 置信度:{confidence}\n"
                    f"  实体: {entity_str}\n"
                    f"  三元组: {triple_str}\n"
                    f"  导师提示: {mentor_note}\n"
                )

            texts_str = format_corpus_texts_for_check(
                {cid: corpus_texts.get(cid, "") for cid in has_entity_corpus}
            )

            # 调用LLM
            prompt_text = BATCH_LABEL_PROMPT.invoke(
                {
                    "batch_entities_triples": entities_triples_str,
                    "corpus_texts": texts_str,
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"

            response = await llm.ainvoke(full_prompt)
            result: BatchLabelResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # 转换为字典格式
            label_results_dict = {}
            for r in result.batch_label_results:
                label_results_dict[r.corpus_id] = {
                    "entity_attrs": r.entity_attrs,
                    "relation_attrs": r.relation_attrs,
                    "confidence": r.confidence,
                }

            # 合并无实体的结果
            all_results = {**no_entity_results, **label_results_dict}

            logger.info(
                f"[Batch_Label] 完成: {len(all_results)} 条语料, "
                f"整体置信度: {result.overall_confidence}"
            )

            writer(
                {
                    "step": "batch_label",
                    "status": "completed",
                    "batch_size": len(all_results),
                    "overall_confidence": result.overall_confidence,
                }
            )

            return {
                "batch_label_results": all_results,
                "batch_label_result": result.model_dump(mode="json"),
                "overall_confidence": result.overall_confidence,
            }

        except Exception as e:
            logger.error(f"[Batch_Label] 失败: {e}")
            writer(
                {
                    "step": "batch_label",
                    "status": "error",
                    "error": str(e),
                }
            )

            # 标注失败时，保守策略：返回空属性
            fallback_results = {
                corpus_id: {
                    "entity_attrs": {},
                    "relation_attrs": {},
                    "confidence": "low",
                }
                for corpus_id in batch_eval_results
            }
            return {
                "batch_label_results": fallback_results,
                "overall_confidence": "low",
                "error": str(e),
            }

    return batch_label_node


# ===== P15新增：批量Self-Check-Label节点 =====


def create_batch_self_check_label_node(llm: Any):
    """创建批量标注校验修正节点（P18改进：修正型校验）

    职责：
    1. 校验批量标注结果的质量
    2. 发现问题直接修正属性
    3. 仅对严重问题才拒绝
    """
    from .schemas import BatchSelfCheckLabelResult
    from .prompts import BATCH_SELF_CHECK_LABEL_PROMPT, format_corpus_texts_for_check

    parser = PydanticOutputParser(pydantic_object=BatchSelfCheckLabelResult)

    async def batch_self_check_label_node(
        batch_label_results: Dict, corpus_texts: Dict, writer: StreamWriter
    ) -> Dict:
        """批量标注校验修正（P18改进）"""
        batch_size = len(batch_label_results)
        logger.info(f"[Batch_Self_Check_Label] 校验 {batch_size} 条语料")

        writer(
            {
                "step": "batch_self_check_label",
                "status": "started",
                "batch_size": batch_size,
            }
        )

        if not batch_label_results:
            return {
                "verified_results": [],
                "rejected_results": [],
                "overall_confidence": "medium",
                "correction_count": 0,
                "retry_suggested": False,
            }

        try:
            # 格式化输入
            label_str = ""
            for corpus_id, data in batch_label_results.items():
                entity_attrs = data.get("entity_attrs", {})
                relation_attrs = data.get("relation_attrs", {})
                label_str += f"- [{corpus_id}] 实体属性: {len(entity_attrs)}个, 关系属性: {len(relation_attrs)}个\n"

            texts_str = format_corpus_texts_for_check(corpus_texts)

            prompt_text = BATCH_SELF_CHECK_LABEL_PROMPT.invoke(
                {"batch_label_results": label_str, "corpus_texts": texts_str}
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"

            response = await llm.ainvoke(full_prompt)
            result: BatchSelfCheckLabelResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # P18改进：合并修正结果
            final_verified_results = []
            correction_count = 0

            for r in result.verified_results:
                r_dict = r.model_dump(mode="json")

                # 合并属性：verified + corrected - rejected
                final_entity_attrs = {
                    **r_dict.get("verified_entity_attrs", {}),
                    **r_dict.get("corrected_entity_attrs", {}),
                }
                for key in r_dict.get("rejected_entity_attrs", []):
                    final_entity_attrs.pop(key, None)

                final_relation_attrs = {
                    **r_dict.get("verified_relation_attrs", {}),
                    **r_dict.get("corrected_relation_attrs", {}),
                }
                for key in r_dict.get("rejected_relation_attrs", []):
                    final_relation_attrs.pop(key, None)

                # 计算修正数量
                corrections = (
                    len(r_dict.get("corrected_entity_attrs", {})) +
                    len(r_dict.get("corrected_relation_attrs", {}))
                )
                correction_count += corrections

                final_result = {
                    "corpus_id": r_dict["corpus_id"],
                    "entity_attrs": final_entity_attrs,  # 最终实体属性
                    "relation_attrs": final_relation_attrs,  # 最终关系属性
                    "verified_entity_attrs": r_dict.get("verified_entity_attrs", {}),
                    "verified_relation_attrs": r_dict.get("verified_relation_attrs", {}),
                    "corrected_entity_attrs": r_dict.get("corrected_entity_attrs", {}),
                    "corrected_relation_attrs": r_dict.get("corrected_relation_attrs", {}),
                    "rejected_entity_attrs": r_dict.get("rejected_entity_attrs", []),
                    "rejected_relation_attrs": r_dict.get("rejected_relation_attrs", []),
                    "attr_completeness": r_dict.get("attr_completeness", "medium"),
                    "confidence": r_dict.get("confidence", "medium"),
                }
                final_verified_results.append(final_result)

            logger.info(
                f"[Batch_Self_Check_Label] 完成: 修正 {correction_count} 处属性, "
                f"通过 {len(final_verified_results)} 条, "
                f"拒绝 {len(result.rejected_results)} 条（严重问题）"
            )

            writer(
                {
                    "step": "batch_self_check_label",
                    "status": "completed",
                    "verified_count": len(final_verified_results),
                    "correction_count": correction_count,
                    "rejected_count": len(result.rejected_results),
                }
            )

            return {
                "verified_results": final_verified_results,
                "rejected_results": result.rejected_results,  # 仅严重问题
                "overall_confidence": result.overall_confidence,
                "correction_count": correction_count,
                "retry_suggested": result.retry_suggested,
            }

        except Exception as e:
            logger.error(f"[Batch_Self_Check_Label] 失败: {e}")
            return {
                "verified_results": [
                    {
                        "corpus_id": cid,
                        "entity_attrs": {},
                        "verified_entity_attrs": {},
                        "verified_relation_attrs": {},
                        "corrected_entity_attrs": {},
                        "corrected_relation_attrs": {},
                        "rejected_entity_attrs": [],
                        "rejected_relation_attrs": [],
                        "attr_completeness": "low",
                        "confidence": "low",
                    }
                    for cid in batch_label_results
                ],
                "rejected_results": [],
                "overall_confidence": "low",
                "correction_count": 0,
                "retry_suggested": False,
                "error": str(e),
            }

    return batch_self_check_label_node


# ===== P15新增：批量Entity_Alignment节点 =====


def create_batch_entity_alignment_node(llm: Any):
    """创建批量实体对齐节点"""
    from .schemas import (
        BatchEntityAlignmentDecisionResult,
    )
    from .prompts import BATCH_ENTITY_ALIGNMENT_DECISION_PROMPT

    batch_alignment_parser = PydanticOutputParser(
        pydantic_object=BatchEntityAlignmentDecisionResult
    )

    _embedding_model_cache = None
    _db_cache = None

    def _get_embedding_model():
        """懒加载embedding模型，避免重复初始化"""
        nonlocal _embedding_model_cache
        if _embedding_model_cache is None:
            from sentence_transformers import SentenceTransformer
            from settings import settings

            config = settings.get_extraction_config()
            model_name = config.alignment_embedding_model
            logger.info(f"[Batch_Entity_Alignment] 加载embedding模型: {model_name}")
            _embedding_model_cache = SentenceTransformer(model_name)
        return _embedding_model_cache

    def _load_db_embeddings(pg_client):
        """预加载数据库实体embedding到内存缓存"""
        import numpy as np

        with pg_client.conn.cursor() as cur:
            cur.execute(
                """
                SELECT entity_id, name, type, longitude, latitude, embedding
                FROM geo_entity_names
                WHERE embedding IS NOT NULL
            """
            )
            geo_rows = cur.fetchall()

        geo_entities = []
        geo_embeddings = []
        for row in geo_rows:
            entity_id, name, type_, lon, lat, emb_str = row
            if not emb_str:
                continue
            if isinstance(emb_str, str):
                emb_list = json.loads(emb_str)
            else:
                emb_list = emb_str
            geo_entities.append(
                {
                    "entity_id": entity_id,
                    "name": name,
                    "type": type_ or "",
                    "longitude": lon,
                    "latitude": lat,
                    "source": "geo_entity_names",
                }
            )
            geo_embeddings.append(emb_list)

        with pg_client.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, entity_id, name, type, longitude, latitude, address, embedding
                FROM amap_poi_wgs84
                WHERE embedding IS NOT NULL
            """
            )
            amap_rows = cur.fetchall()

        amap_entities = []
        amap_embeddings = []
        for row in amap_rows:
            amap_id, original_id, name, type_, lon, lat, address, emb_str = row
            if not emb_str:
                continue
            if isinstance(emb_str, str):
                emb_list = json.loads(emb_str)
            else:
                emb_list = emb_str
            amap_entities.append(
                {
                    "id": amap_id,
                    "original_id": original_id,
                    "name": name,
                    "type": type_ or "",
                    "longitude": lon,
                    "latitude": lat,
                    "address": address,
                    "source": "amap_poi_wgs84",
                }
            )
            amap_embeddings.append(emb_list)

        geo_embeddings_np = np.array(geo_embeddings) if geo_embeddings else np.array([])
        amap_embeddings_np = np.array(amap_embeddings) if amap_embeddings else np.array([])
        logger.info(
            f"[Batch_Entity_Alignment] 预加载embedding: geo={len(geo_entities)}, amap={len(amap_entities)}"
        )
        return geo_entities, amap_entities, geo_embeddings_np, amap_embeddings_np

    def _batch_similarity_search(query_embeddings_np, db_embeddings_np, db_entities, top_k):
        """内存批量cosine similarity检索"""
        import numpy as np

        if len(db_embeddings_np) == 0 or len(query_embeddings_np) == 0:
            return [[] for _ in range(len(query_embeddings_np))]

        query_norms = np.linalg.norm(query_embeddings_np, axis=1, keepdims=True)
        db_norms = np.linalg.norm(db_embeddings_np, axis=1, keepdims=True)
        query_normalized = query_embeddings_np / (query_norms + 1e-10)
        db_normalized = db_embeddings_np / (db_norms + 1e-10)
        similarity_matrix = np.dot(query_normalized, db_normalized.T)

        candidates_per_query = []
        for i in range(len(query_embeddings_np)):
            similarities = similarity_matrix[i]
            if len(similarities) >= top_k:
                top_indices = np.argsort(similarities)[-top_k:][::-1]
            else:
                top_indices = np.argsort(similarities)[::-1]

            candidates = []
            for idx in top_indices:
                candidate = db_entities[idx].copy()
                candidate["similarity"] = float(similarities[idx])
                candidates.append(candidate)
            candidates_per_query.append(candidates)
        return candidates_per_query

    async def batch_entity_alignment_node(
        batch_label_results: Dict,
        corpus_texts: Dict,
        existing_entities: List[str],
        writer: StreamWriter,
    ) -> Dict:
        """批量实体对齐"""
        batch_size = len(batch_label_results)
        logger.info(f"[Batch_Entity_Alignment] 对齐 {batch_size} 条语料")

        writer(
            {
                "step": "batch_entity_alignment",
                "status": "started",
                "batch_size": batch_size,
            }
        )

        if not batch_label_results:
            return {"aligned_results": {}, "overall_confidence": "medium"}

        try:
            from settings import settings
            from kg.postgres_client import PostgresClient

            config = settings.get_extraction_config()

            # 收集全batch实体（去重），并建立 corpus -> entity 映射
            corpus_entity_attrs = {}
            all_entity_names = []
            seen_entity_names = set()
            for corpus_id, data in batch_label_results.items():
                attrs = data.get("entity_attrs", {}) or {}
                corpus_entity_attrs[str(corpus_id)] = attrs
                for name in attrs.keys():
                    if name and name not in seen_entity_names:
                        seen_entity_names.add(name)
                        all_entity_names.append(name)

            if not all_entity_names:
                aligned_dict = {
                    str(cid): {
                        "aligned_entity_attrs": {},
                        "new_entities": [],
                        "confidence": "high",
                    }
                    for cid in batch_label_results.keys()
                }
                return {"aligned_results": aligned_dict, "overall_confidence": "high"}

            # 载入数据库embedding缓存
            pg_config = settings.get_postgres_config()
            with PostgresClient(**pg_config) as pg_client:
                nonlocal _db_cache
                if _db_cache is None:
                    _db_cache = _load_db_embeddings(pg_client)
                geo_entities, amap_entities, geo_embeddings_np, amap_embeddings_np = _db_cache

            # 计算query embedding并检索候选
            model = _get_embedding_model()
            entity_embeddings = model.encode(
                all_entity_names, show_progress_bar=False, convert_to_numpy=True
            )

            top_k = config.alignment_top_k
            high_threshold = config.alignment_high_confidence_threshold
            low_threshold = config.alignment_similarity_threshold
            use_llm = config.alignment_use_llm_decision

            geo_candidates_per_query = _batch_similarity_search(
                entity_embeddings, geo_embeddings_np, geo_entities, top_k
            )
            amap_candidates_per_query = _batch_similarity_search(
                entity_embeddings, amap_embeddings_np, amap_entities, top_k
            )

            # 每个实体全局决策（同名实体在不同corpus保持一致）
            decision_by_entity = {}
            medium_conf_items = []

            for i, entity_name in enumerate(all_entity_names):
                candidates = geo_candidates_per_query[i] + amap_candidates_per_query[i]
                for c in candidates:
                    if c.get("source") == "amap_poi_wgs84":
                        amap_id = c.get("id")
                        c["db_entity_id"] = (
                            f"poi_{amap_id}" if amap_id is not None else c.get("entity_id", "")
                        )
                        c["db_original_id"] = c.get("original_id")
                        c["db_name"] = c.get("name", "")
                        c["db_type"] = c.get("type", "")
                    else:
                        c["db_entity_id"] = c.get("entity_id")
                        c["db_name"] = c.get("name", "")
                        c["db_type"] = c.get("type", "")

                candidates.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
                candidates = candidates[:top_k]

                best = candidates[0] if candidates else None
                best_sim = best.get("similarity", 0.0) if best else 0.0

                if best and best_sim >= high_threshold:
                    decision_by_entity[entity_name] = {
                        "status": "aligned",
                        "best_candidate": best,
                        "confidence": "high",
                        "reason": f"高置信度匹配({best_sim:.3f})",
                    }
                elif (not best) or best_sim < low_threshold:
                    decision_by_entity[entity_name] = {
                        "status": "new_entity",
                        "best_candidate": None,
                        "confidence": "medium",
                        "reason": f"低置信度/无候选({best_sim:.3f})",
                    }
                elif use_llm and candidates:
                    medium_conf_items.append(
                        {
                            "extracted_name": entity_name,
                            "candidates": candidates,
                        }
                    )
                else:
                    decision_by_entity[entity_name] = {
                        "status": "new_entity",
                        "best_candidate": None,
                        "confidence": "medium",
                        "reason": f"中置信度但未启用LLM({best_sim:.3f})",
                    }

            # 中置信度实体一次LLM决策（N -> 1）
            if medium_conf_items:
                try:
                    items_for_prompt = []
                    for item in medium_conf_items:
                        cands = []
                        for c in item["candidates"]:
                            cands.append(
                                {
                                    "db_name": c.get("db_name", ""),
                                    "db_type": c.get("db_type", ""),
                                    "similarity": c.get("similarity", 0.0),
                                    "longitude": c.get("longitude"),
                                    "latitude": c.get("latitude"),
                                    "source": c.get("source", ""),
                                    "db_entity_id": c.get("db_entity_id", ""),
                                    "db_original_id": c.get("db_original_id", ""),
                                }
                            )
                        items_for_prompt.append(
                            {
                                "extracted_name": item["extracted_name"],
                                "extracted_type": "",
                                "candidates": cands,
                            }
                        )

                    prompt_text = BATCH_ENTITY_ALIGNMENT_DECISION_PROMPT.invoke(
                        {
                            "raw_text": json.dumps(corpus_texts, ensure_ascii=False),
                            "items_json": json.dumps(items_for_prompt, ensure_ascii=False),
                        }
                    )
                    full_prompt = (
                        f"{prompt_text.messages[1].content}\n\n"
                        f"{batch_alignment_parser.get_format_instructions()}"
                    )
                    response = await llm.ainvoke(full_prompt)
                    parsed_result: BatchEntityAlignmentDecisionResult = (
                        safe_parse_json_with_quote_fix(
                            batch_alignment_parser, response.content
                        )
                    )

                    decision_map = {
                        d.extracted_name: d for d in (parsed_result.decisions or [])
                    }
                    valid_statuses = {"aligned", "new_entity", "skip"}

                    for item in medium_conf_items:
                        entity_name = item["extracted_name"]
                        candidates = item["candidates"]
                        d = decision_map.get(entity_name)
                        if not d:
                            decision_by_entity[entity_name] = {
                                "status": "new_entity",
                                "best_candidate": None,
                                "confidence": "low",
                                "reason": "LLM未返回该实体决策",
                            }
                            continue

                        status = (
                            d.alignment_status
                            if d.alignment_status in valid_statuses
                            else "new_entity"
                        )
                        idx = int(d.best_match_index or -1)
                        if status == "aligned" and 0 <= idx < len(candidates):
                            decision_by_entity[entity_name] = {
                                "status": "aligned",
                                "best_candidate": candidates[idx],
                                "confidence": "medium",
                                "reason": d.llm_decision or "LLM中置信度对齐",
                            }
                        else:
                            decision_by_entity[entity_name] = {
                                "status": "new_entity" if status != "skip" else "skip",
                                "best_candidate": None,
                                "confidence": "medium",
                                "reason": d.llm_decision or "LLM判定非对齐",
                            }

                    logger.info(
                        f"[Batch_Entity_Alignment] 中置信度实体批量LLM判断完成: "
                        f"{len(medium_conf_items)} 个实体，调用 1 次"
                    )
                except Exception as llm_err:
                    logger.warning(
                        f"[Batch_Entity_Alignment] 中置信度实体LLM判断失败: {llm_err}，降级为新实体"
                    )
                    for item in medium_conf_items:
                        entity_name = item["extracted_name"]
                        decision_by_entity[entity_name] = {
                            "status": "new_entity",
                            "best_candidate": None,
                            "confidence": "low",
                            "reason": "LLM异常降级",
                        }

            # 组装每条语料的输出
            aligned_dict = {}
            alignment_detail_logs = []
            conf_weight = {"high": 2, "medium": 1, "low": 0}
            overall_sum = 0
            overall_cnt = 0

            for corpus_id, entity_attrs in corpus_entity_attrs.items():
                aligned_entity_attrs = {}
                new_entities = []
                status_counter = {"aligned": 0, "new_entity": 0, "skip": 0}

                for entity_name, attrs in entity_attrs.items():
                    d = decision_by_entity.get(entity_name, {})
                    status = d.get("status", "new_entity")
                    status_counter[status] = status_counter.get(status, 0) + 1
                    if status == "aligned":
                        aligned_entity_attrs[entity_name] = attrs
                    elif status == "new_entity":
                        new_entities.append(entity_name)

                    c = d.get("confidence", "medium")
                    overall_sum += conf_weight.get(str(c), 1)
                    overall_cnt += 1

                if overall_cnt == 0:
                    corpus_conf = "high"
                else:
                    ratio = overall_sum / max(overall_cnt * 2, 1)
                    corpus_conf = "high" if ratio >= 0.7 else "medium" if ratio >= 0.4 else "low"

                aligned_dict[corpus_id] = {
                    "aligned_entity_attrs": aligned_entity_attrs,
                    "new_entities": new_entities,
                    "confidence": corpus_conf,
                }

                alignment_detail_logs.append(
                    {
                        "corpus_id": corpus_id,
                        "aligned_entity_count": len(aligned_entity_attrs),
                        "new_entity_count": len(new_entities),
                        "status_breakdown": status_counter,
                        "confidence": corpus_conf,
                        "aligned_samples": list(aligned_entity_attrs.keys())[:5],
                        "new_samples": new_entities[:5],
                    }
                )

            for item in alignment_detail_logs:
                logger.info(
                    f"[Batch_Entity_Alignment][Corpus {item['corpus_id']}] "
                    f"{json.dumps(item, ensure_ascii=False)}"
                )

            if overall_cnt == 0:
                overall_confidence = "high"
            else:
                ratio = overall_sum / max(overall_cnt * 2, 1)
                overall_confidence = (
                    "high" if ratio >= 0.7 else "medium" if ratio >= 0.4 else "low"
                )

            logger.info(f"[Batch_Entity_Alignment] 完成: {len(aligned_dict)} 条")

            writer(
                {
                    "step": "batch_entity_alignment",
                    "status": "completed",
                    "batch_size": len(aligned_dict),
                }
            )

            return {
                "aligned_results": aligned_dict,
                "overall_confidence": overall_confidence,
            }

        except Exception as e:
            logger.error(f"[Batch_Entity_Alignment] 失败: {e}")
            return {
                "aligned_results": {
                    cid: {
                        "aligned_entity_attrs": {},
                        "new_entities": [],
                        "confidence": "low",
                    }
                    for cid in batch_label_results
                },
                "overall_confidence": "low",
                "error": str(e),
            }

    return batch_entity_alignment_node


# ===== P10新增：批量处理入口函数 =====


async def process_corpus_batch_with_llm(
    llm: Any,
    corpus_list: List[Dict],
    config: ExtractionConfig,
    batch_joint_node: Any = None,
    batch_self_check_node: Any = None,
    batch_eval_node: Any = None,  # P17����
    batch_label_node: Any = None,  # P17����
    batch_self_check_eval_node: Any = None,  # P17����
    batch_self_check_label_node: Any = None,  # P17����
    
    batch_entity_alignment_node: Any = None,  # P21 added
    qa_llm: Any = None,
) -> Dict:
    """
    �����������ϣ�һ��LLM���ô���batch_llm_size����

    P17�Ľ�������������У��ڵ���
    Batch_Joint �� Batch_Self_Check �� Batch_Eval �� Batch_Self_Check_Eval �� Batch_Label �� Batch_Self_Check_Label

    Args:
        llm: LLMʵ��
        corpus_list: �����б� [{"id": ..., "text": ...}, ...]
        config: ExtractionConfig
        batch_joint_node: �������ϳ�ȡ�ڵ㣨��ѡ��
        batch_self_check_node: ����У��ڵ㣨��ѡ��
        batch_eval_node: ���������ڵ㣨��ѡ��P17������
        batch_label_node: ������ע�ڵ㣨��ѡ��P17������
        batch_self_check_eval_node: ��������У��ڵ㣨��ѡ��P17������
        batch_self_check_label_node: ������עУ��ڵ㣨��ѡ��P17������
        qa_llm: ��ѡQA��ʦģ�ͣ�δ�ṩʱʹ��llm

    Returns:
        {
            "batch_results": {corpus_id: {entities, triples, eval_passed, entity_attrs, relation_attrs}},
            "cross_corpus_aliases": [...],
            "fallback_corpus_list": [...],
        }
    """
    from .prompts import (
        detect_eval_confusion,
        detect_extraction_confusion,
        detect_label_confusion,
    )

    batch_llm_size = config.batch_llm_size
    enable_batch_llm = config.enable_batch_llm
    batch_llm_fallback = config.batch_llm_fallback
    batch_stage_retry_attempts = max(
        int(getattr(config, "batch_stage_retry_attempts", 1) or 0), 0
    )
    batch_skip_on_repeated_failure = bool(
        getattr(config, "batch_skip_on_repeated_failure", False)
    )
    eval_threshold = config.eval_threshold
    enable_qa_mentor = config.enable_qa_mentor
    mentor_query_min_confidence = str(
        getattr(config, "mentor_query_min_confidence", "low") or "low"
    ).lower()
    mentor_extraction_low_item_threshold = max(
        int(getattr(config, "mentor_extraction_low_item_threshold", 2) or 2), 1
    )
    mentor_eval_reject_ratio_threshold = max(
        0.0,
        min(
            1.0,
            float(getattr(config, "mentor_eval_reject_ratio_threshold", 0.8) or 0.8),
        ),
    )
    mentor_label_missing_ratio_threshold = max(
        0.0,
        min(
            1.0,
            float(
                getattr(config, "mentor_label_missing_ratio_threshold", 0.8) or 0.8
            ),
        ),
    )
    mentor_label_min_missing_attrs = max(
        int(getattr(config, "mentor_label_min_missing_attrs", 3) or 3), 1
    )

    if not enable_batch_llm:
        return {
            "batch_results": {},
            "cross_corpus_aliases": [],
            "needs_single_processing": True,
        }

    if batch_joint_node is None:
        batch_joint_node = create_batch_joint_extraction_node(llm, batch_llm_size)
    if batch_self_check_node is None:
        batch_self_check_node = create_batch_self_check_node(llm)
    if batch_eval_node is None:
        batch_eval_node = create_batch_eval_node(llm, eval_threshold)
    if batch_label_node is None:
        batch_label_node = create_batch_label_node(llm)
    if batch_self_check_eval_node is None:
        batch_self_check_eval_node = create_batch_self_check_eval_node(llm)
    if batch_self_check_label_node is None:
        batch_self_check_label_node = create_batch_self_check_label_node(llm)

    qa_model = qa_llm or llm
    mentor_node = create_qa_mentor_node(qa_model, config) if enable_qa_mentor else None
    single_joint_node = create_joint_ner_re_node(llm, enable_query=False)
    single_eval_node = (
        create_eval_simplified_node(
            llm, eval_threshold=eval_threshold, enable_query=False
        )
        if enable_qa_mentor
        else None
    )
    single_label_node = (
        create_label_node(llm, enable_query=False) if enable_qa_mentor else None
    )

    def dummy_writer(event):
        return None

    def _dict_id_set(data: Any) -> set:
        if not isinstance(data, dict):
            return set()
        return {str(k) for k in data.keys()}

    def _list_corpus_id_set(items: Any) -> set:
        if not isinstance(items, list):
            return set()
        ids = set()
        for item in items:
            if isinstance(item, dict) and item.get("corpus_id") is not None:
                ids.add(str(item.get("corpus_id")))
        return ids

    def _has_missing_coverage(stage_name: str, expected_ids: set, actual_ids: set) -> bool:
        if not expected_ids:
            return False
        missing_ids = sorted(expected_ids - actual_ids)
        if missing_ids:
            logger.warning(
                f"[{stage_name}] 结果覆盖不完整: expected={len(expected_ids)}, "
                f"actual={len(actual_ids)}, missing={missing_ids[:5]}"
            )
            return True
        return False

    def _flatten_entities(entities: Dict) -> List[Dict]:
        entity_list = []
        for entity_type, names in entities.items():
            for name in names:
                entity_list.append(
                    {
                        "name": name,
                        "type": entity_type,
                        "confidence": "medium",
                    }
                )
        return entity_list

    def _should_request_mentor(confusion: Optional[Dict]) -> bool:
        if not confusion:
            return False
        confidence = str(confusion.get("current_confidence", "medium")).lower()
        if mentor_query_min_confidence == "low":
            return confidence == "low"
        return confidence in ("low", "medium")

    def _build_base_state(corpus: Dict) -> Dict:
        base_state = create_default_corpus_state(
            corpus_id=str(corpus.get("id", "unknown")),
            raw_text=corpus.get("text", ""),
            max_retries=config.self_check_max_retries,
            enable_normalize=config.enable_normalize,
            enable_qa_scaffold=config.enable_qa_scaffold,
            enable_entity_alignment=config.enable_entity_alignment,
            max_revision_cycles=config.max_revision_cycles,
        )
        base_state["mentor_query_min_confidence"] = mentor_query_min_confidence

        if corpus.get("normalized_text"):
            base_state["normalized_text"] = corpus.get("normalized_text", "")
        if corpus.get("entity_hints"):
            base_state["qa_entity_hints"] = corpus.get("entity_hints", [])
        if corpus.get("relation_hints"):
            base_state["qa_relation_hints"] = corpus.get("relation_hints", [])
        if corpus.get("context_dependencies"):
            base_state["qa_context_dependencies"] = corpus.get(
                "context_dependencies", []
            )
        if corpus.get("semantic_summary"):
            base_state["semantic_summary"] = corpus.get("semantic_summary", "")

        return base_state

    async def _run_batch_stage_with_retry(
        stage_name: str,
        runner,
        is_failed,
    ):
        """统一批量阶段重试包装器。"""
        last_result = None
        for attempt in range(batch_stage_retry_attempts + 1):
            last_result = await runner()
            failed = bool(is_failed(last_result))
            if not failed:
                if attempt > 0:
                    logger.info(
                        f"[{stage_name}] 批量重试成功: 第{attempt + 1}次尝试"
                    )
                return last_result

            err_msg = ""
            if isinstance(last_result, dict):
                err_msg = str(last_result.get("error", "unknown"))
            if attempt < batch_stage_retry_attempts:
                logger.warning(
                    f"[{stage_name}] 批量失败，准备重试 "
                    f"{attempt + 1}/{batch_stage_retry_attempts}: {err_msg}"
                )
            else:
                logger.error(
                    f"[{stage_name}] 批量重试耗尽({batch_stage_retry_attempts + 1}次): {err_msg}"
                )
        return last_result

    async def _mentor_batch_answer_queries(
        source_node: str,
        query_payloads: List[Dict],
        previous_guidance: Optional[Dict] = None,
    ) -> Dict[str, Dict]:
        """批量调用导师回答多个查询，返回 {corpus_id: mentor_response_dict}。"""
        if not enable_qa_mentor or not query_payloads:
            return {}

        parser = PydanticOutputParser(pydantic_object=BatchMentorQueryResponse)
        guidance_text = format_mentor_guidance(previous_guidance or {})

        query_items = []
        raw_text_map = {}
        for item in query_payloads:
            corpus_id = str(item.get("corpus_id", ""))
            source_corpus = item.get("source_corpus", {}) or {}
            confusion = item.get("confusion", {}) or {}

            query_items.append(
                {
                    "corpus_id": corpus_id,
                    "query_type": confusion.get("query_type", "unknown"),
                    "query_content": confusion.get("query_content", ""),
                    "involved_entities": confusion.get("involved_entities", []),
                    "involved_relations": confusion.get("involved_relations", []),
                    "current_confidence": confusion.get("current_confidence", "medium"),
                    "context": item.get("context", ""),
                }
            )
            raw_text_map[corpus_id] = source_corpus.get("text", "")

        try:
            prompt_text = BATCH_MENTOR_QUERY_PROMPT.invoke(
                {
                    "source_node": source_node,
                    "query_items": json.dumps(query_items, ensure_ascii=False),
                    "raw_text_map": json.dumps(raw_text_map, ensure_ascii=False),
                    "previous_guidance": guidance_text,
                }
            )
            full_prompt = (
                f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            )
            response = await qa_model.ainvoke(full_prompt)
            result: BatchMentorQueryResponse = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            response_map: Dict[str, Dict] = {}
            for r in result.results:
                response_map[str(r.corpus_id)] = {
                    "answer": r.answer,
                    "clarification": r.clarification,
                    "recommendation": r.recommendation,
                    "updated_entity_hints": r.updated_entity_hints,
                    "updated_relation_hints": r.updated_relation_hints,
                    "response_confidence": r.response_confidence,
                    "suggests_revision": r.suggests_revision,
                    "return_to_node": r.return_to_node or source_node,
                }

            logger.info(
                f"[Batch-Mentor] {source_node} 批量回答完成: "
                f"{len(response_map)}/{len(query_payloads)} 条, "
                f"置信度={result.overall_confidence}"
            )
            return response_map
        except Exception as e:
            logger.warning(f"[Batch-Mentor] {source_node} 批量回答失败: {e}")
            return {}

    async def _mentor_rerun_current_node(
        corpus: Dict,
        source_node: str,
        confusion: Dict,
        node_state: Dict,
        mentor_response: Optional[Dict] = None,
    ) -> Optional[Dict]:
        if (
            not enable_qa_mentor
            or mentor_node is None
            or single_joint_node is None
            or single_eval_node is None
            or single_label_node is None
        ):
            return None

        try:
            state = _build_base_state(corpus)
            state.update(node_state)
            if mentor_response:
                # 直接使用批量导师响应，避免重复单条导师调用
                state["mentor_response"] = mentor_response
                state["needs_mentor_help"] = False
                if mentor_response.get("updated_entity_hints"):
                    state["qa_entity_hints"] = mentor_response.get(
                        "updated_entity_hints", []
                    )
                if mentor_response.get("updated_relation_hints"):
                    state["qa_relation_hints"] = mentor_response.get(
                        "updated_relation_hints", []
                    )
            else:
                state["mentor_query"] = confusion
                state["query_source_node"] = source_node
                state["needs_mentor_help"] = True

                mentor_delta = await mentor_node(state, dummy_writer)
                if isinstance(mentor_delta, dict):
                    state.update(mentor_delta)

            if source_node == "joint_ner_re":
                rerun_delta = await single_joint_node(state, dummy_writer)
            elif source_node == "eval":
                rerun_delta = await single_eval_node(state, dummy_writer)
            elif source_node == "label":
                rerun_delta = await single_label_node(state, dummy_writer)
            else:
                return None

            if not isinstance(rerun_delta, dict) or rerun_delta.get("error"):
                return None
            return rerun_delta
        except Exception as e:
            logger.warning(
                f"[Batch-Mentor-Inline] {source_node} 导师处理失败: {e}"
            )
            return None

    all_batch_results = {}
    all_cross_corpus_aliases = []
    fallback_corpus_list = []
    fallback_corpus_ids = set()
    corpus_texts = {corpus["id"]: corpus.get("text", "") for corpus in corpus_list}

    def _mark_corpus_skipped(corpus_id: Any, error_msg: str, skip_reason: str):
        existing = all_batch_results.get(corpus_id, {})
        all_batch_results[corpus_id] = {
            "entities": existing.get("entities", DEFAULT_ENTITY_DICT.copy()),
            "triples": existing.get("triples", []),
            "eval_passed": existing.get("eval_passed", False),
            "eval_scores": existing.get("eval_scores", []),
            "entity_attrs": existing.get("entity_attrs", {}),
            "relation_attrs": existing.get("relation_attrs", {}),
            "confidence": existing.get("confidence", "error"),
            "error": error_msg,
            "skip_reason": skip_reason,
        }

    def _mark_many_skipped(corpus_ids: List[Any], error_msg: str, skip_reason: str):
        for cid in corpus_ids:
            _mark_corpus_skipped(cid, error_msg, skip_reason)

    for i in range(0, len(corpus_list), batch_llm_size):
        batch_corpus = corpus_list[i : i + batch_llm_size]
        batch_num = i // batch_llm_size + 1
        batch_corpus_map = {str(c.get("id")): c for c in batch_corpus}

        logger.info(f"[Batch {batch_num}] 处理 {len(batch_corpus)} 条语料")
        expected_batch_ids = {str(c.get("id")) for c in batch_corpus}

        extraction_result = await _run_batch_stage_with_retry(
            f"Batch {batch_num}/Batch_Joint",
            lambda: batch_joint_node(batch_corpus, dummy_writer),
            lambda r: (not isinstance(r, dict))
            or r.get("needs_fallback", False)
            or _has_missing_coverage(
                f"Batch {batch_num}/Batch_Joint",
                expected_batch_ids,
                _dict_id_set(r.get("batch_results", {})),
            ),
        )

        extraction_result = extraction_result or {
            "batch_results": {},
            "cross_corpus_aliases": [],
            "needs_fallback": True,
            "fallback_reason": "Batch_Joint 未返回结果",
        }

        if extraction_result.get("needs_fallback"):
            if batch_skip_on_repeated_failure:
                fallback_reason = extraction_result.get(
                    "fallback_reason", "批处理抽取失败（重试耗尽）"
                )
                logger.warning(
                    f"[Batch {batch_num}] 抽取重试失败，直接跳过本批 {len(batch_corpus)} 条语料"
                )
                for corpus in batch_corpus:
                    cid = corpus.get("id")
                    all_batch_results[cid] = {
                        "entities": DEFAULT_ENTITY_DICT.copy(),
                        "triples": [],
                        "confidence": "error",
                        "error": fallback_reason,
                        "skip_reason": "batch_joint_failed_after_retries",
                    }
            elif batch_llm_fallback:
                logger.warning(f"[Batch {batch_num}] 抽取失败，加入 fallback 单条处理")
                for corpus in batch_corpus:
                    cid = str(corpus.get("id"))
                    if cid not in fallback_corpus_ids:
                        fallback_corpus_ids.add(cid)
                        fallback_corpus_list.append(corpus)
            else:
                for corpus in batch_corpus:
                    all_batch_results[corpus["id"]] = {
                        "entities": DEFAULT_ENTITY_DICT.copy(),
                        "triples": [],
                        "confidence": "error",
                        "error": extraction_result.get(
                            "fallback_reason", "批处理抽取失败"
                        ),
                    }
            continue

        batch_results = extraction_result["batch_results"]
        cross_corpus_aliases = extraction_result["cross_corpus_aliases"]

        if config.enable_self_check or config.enable_full_self_check:
            check_result = await _run_batch_stage_with_retry(
                f"Batch {batch_num}/Batch_Self_Check",
                lambda: batch_self_check_node(
                    batch_results, cross_corpus_aliases, dummy_writer
                ),
                lambda r: (not isinstance(r, dict))
                or bool(r.get("error"))
                or (not r.get("verified_results") and bool(batch_results))
                or _has_missing_coverage(
                    f"Batch {batch_num}/Batch_Self_Check",
                    _dict_id_set(batch_results),
                    _list_corpus_id_set(r.get("verified_results", []))
                    | _list_corpus_id_set(r.get("rejected_results", [])),
                ),
            )
            check_result = check_result or {
                "verified_results": [],
                "rejected_results": [],
                "verified_aliases": [],
                "error": "Batch_Self_Check 未返回有效结果",
            }
            if batch_skip_on_repeated_failure and check_result.get("error"):
                err = str(check_result.get("error"))
                _mark_many_skipped(
                    list(batch_results.keys()), err, "batch_self_check_failed_after_retries"
                )
                logger.warning(
                    f"[Batch {batch_num}] Batch_Self_Check 反复失败，按策略直接跳过本批"
                )
                continue
            verified_results = check_result["verified_results"]

            # P18修复：rejected_results 中的语料都需要 fallback，不受 fallback_to_single 条件限制
            # fallback_to_single 只是 LLM 对整体质量的建议，不是 rejected 语料处理的条件
            if not batch_skip_on_repeated_failure and batch_llm_fallback:
                for r in check_result["rejected_results"]:
                    corpus_id = r.get("corpus_id")
                    for corpus in batch_corpus:
                        if corpus["id"] == corpus_id:
                            cid = str(corpus.get("id"))
                            if cid not in fallback_corpus_ids:
                                fallback_corpus_ids.add(cid)
                                fallback_corpus_list.append(corpus)
                            logger.warning(
                                f"[Batch_Self_Check] 语料 {corpus_id} 自检拒绝，加入 fallback"
                            )
                            break
            elif batch_skip_on_repeated_failure:
                for r in check_result["rejected_results"]:
                    corpus_id = r.get("corpus_id")
                    if corpus_id is None:
                        continue
                    all_batch_results[corpus_id] = {
                        "entities": DEFAULT_ENTITY_DICT.copy(),
                        "triples": [],
                        "eval_passed": False,
                        "eval_scores": [],
                        "entity_attrs": {},
                        "relation_attrs": {},
                        "confidence": "error",
                        "error": "batch_self_check_rejected",
                        "skip_reason": "batch_self_check_rejected",
                    }
                    logger.warning(
                        f"[Batch_Self_Check] 语料 {corpus_id} 自检拒绝，按策略直接跳过"
                    )

            all_cross_corpus_aliases.extend(check_result["verified_aliases"])
        else:
            verified_results = [
                {
                    "corpus_id": corpus_id,
                    "entities": data.get("entities", DEFAULT_ENTITY_DICT.copy()),
                    "triples": data.get("triples", []),
                    "confidence": data.get("confidence", "medium"),
                }
                for corpus_id, data in batch_results.items()
            ]
            all_cross_corpus_aliases.extend(cross_corpus_aliases)

        verified_results_dict = {}
        joint_confusion_items: List[Dict] = []
        for r in verified_results:
            corpus_id = r.get("corpus_id")
            entities = r.get("entities", DEFAULT_ENTITY_DICT.copy())
            triples = r.get("triples", [])
            confidence = r.get("confidence", "medium")

            current_joint = {
                "corpus_id": corpus_id,
                "entities": entities,
                "triples": triples,
                "confidence": confidence,
            }
            verified_results_dict[corpus_id] = current_joint

            source_corpus = batch_corpus_map.get(str(corpus_id))
            if not (enable_qa_mentor and source_corpus):
                continue

            joint_confusion = detect_extraction_confusion(
                {
                    "entities": _flatten_entities(entities),
                    "triples": triples,
                    "overall_confidence": confidence,
                },
                {
                    "raw_text": source_corpus.get("text", ""),
                    "entities": entities,
                    "triples": triples,
                },
                low_item_threshold=mentor_extraction_low_item_threshold,
            )
            if _should_request_mentor(joint_confusion):
                logger.info(
                    f"[Batch-Mentor-Inline] 语料 {corpus_id} 在 joint_ner_re 困惑，加入批量导师请求"
                )
                joint_confusion_items.append(
                    {
                        "corpus_id": str(corpus_id),
                        "source_corpus": source_corpus,
                        "confusion": joint_confusion,
                        "context": (
                            f"confidence={confidence}; "
                            f"entity_count={len(_flatten_entities(entities))}; "
                            f"triple_count={len(triples)}"
                        ),
                    }
                )

        mentor_joint_response_map: Dict[str, Dict] = {}
        if joint_confusion_items:
            mentor_joint_response_map = await _mentor_batch_answer_queries(
                source_node="joint_ner_re",
                query_payloads=joint_confusion_items,
            )

        joint_rerun_count = 0
        if mentor_joint_response_map:
            rerun_joint_corpus = []
            for corpus_id, mentor_resp in mentor_joint_response_map.items():
                source_corpus = batch_corpus_map.get(str(corpus_id))
                if not source_corpus:
                    continue
                rerun_joint_corpus.append(
                    {
                        "id": str(corpus_id),
                        "text": source_corpus.get("text", ""),
                        "entity_hints": mentor_resp.get("updated_entity_hints", []),
                        "relation_hints": mentor_resp.get("updated_relation_hints", []),
                        "semantic_summary": mentor_resp.get("clarification", ""),
                    }
                )

            if rerun_joint_corpus:
                joint_rerun_count = len(rerun_joint_corpus)
                rerun_joint_result = await batch_joint_node(rerun_joint_corpus, dummy_writer)
                rerun_joint_map = rerun_joint_result.get("batch_results", {}) or {}
                for corpus_id, rerun_joint in rerun_joint_map.items():
                    old_joint = verified_results_dict.get(corpus_id, {})
                    old_conf = old_joint.get("confidence", "medium")
                    verified_results_dict[corpus_id] = {
                        "corpus_id": corpus_id,
                        "entities": rerun_joint.get("entities", DEFAULT_ENTITY_DICT.copy()),
                        "triples": rerun_joint.get("triples", []),
                        "confidence": rerun_joint.get("confidence", old_conf),
                    }
        logger.info(
            f"[Batch-Mentor-Stats] stage=joint_ner_re confusion={len(joint_confusion_items)} "
            f"mentor_answered={len(mentor_joint_response_map)} rerun_batch={joint_rerun_count}"
        )

        verified_results = list(verified_results_dict.values())

        if verified_results:
            eval_input = {
                r["corpus_id"]: {
                    "entities": r.get("entities", DEFAULT_ENTITY_DICT.copy()),
                    "triples": r.get("triples", []),
                }
                for r in verified_results
            }

            eval_result = await _run_batch_stage_with_retry(
                f"Batch {batch_num}/Batch_Eval",
                lambda: batch_eval_node(eval_input, corpus_texts, dummy_writer),
                lambda r: (not isinstance(r, dict))
                or bool(r.get("error"))
                or (not r.get("batch_eval_results") and bool(eval_input))
                or _has_missing_coverage(
                    f"Batch {batch_num}/Batch_Eval",
                    _dict_id_set(eval_input),
                    _dict_id_set(r.get("batch_eval_results", {})),
                ),
            )
            eval_result = eval_result or {"batch_eval_results": {}, "error": "Batch_Eval 未返回有效结果"}
            if batch_skip_on_repeated_failure and eval_result.get("error"):
                err = str(eval_result.get("error"))
                _mark_many_skipped(
                    list(eval_input.keys()), err, "batch_eval_failed_after_retries"
                )
                logger.warning(
                    f"[Batch {batch_num}] Batch_Eval 反复失败，按策略直接跳过本批"
                )
                continue

            if config.enable_full_self_check:
                eval_check_result = await _run_batch_stage_with_retry(
                    f"Batch {batch_num}/Batch_Self_Check_Eval",
                    lambda: batch_self_check_eval_node(
                        eval_result["batch_eval_results"], corpus_texts, dummy_writer
                    ),
                    lambda r: (not isinstance(r, dict))
                    or bool(r.get("error"))
                    or (
                        not r.get("verified_results")
                        and bool(eval_result.get("batch_eval_results"))
                    )
                    or _has_missing_coverage(
                        f"Batch {batch_num}/Batch_Self_Check_Eval",
                        _dict_id_set(eval_result.get("batch_eval_results", {})),
                        _list_corpus_id_set(r.get("verified_results", []))
                        | _list_corpus_id_set(r.get("rejected_results", [])),
                    ),
                )
                eval_check_result = eval_check_result or {
                    "verified_results": [],
                    "rejected_results": [],
                    "error": "Batch_Self_Check_Eval 未返回有效结果",
                }
                if batch_skip_on_repeated_failure and eval_check_result.get("error"):
                    err = str(eval_check_result.get("error"))
                    _mark_many_skipped(
                        list(eval_result.get("batch_eval_results", {}).keys()),
                        err,
                        "batch_self_check_eval_failed_after_retries",
                    )
                    logger.warning(
                        f"[Batch {batch_num}] Batch_Self_Check_Eval 反复失败，按策略直接跳过本批"
                    )
                    continue

                for r in eval_check_result["rejected_results"]:
                    corpus_id = r.get("corpus_id")
                    if not batch_skip_on_repeated_failure and batch_llm_fallback:
                        for corpus in batch_corpus:
                            if corpus["id"] == corpus_id:
                                cid = str(corpus.get("id"))
                                if cid not in fallback_corpus_ids:
                                    fallback_corpus_ids.add(cid)
                                    fallback_corpus_list.append(corpus)
                                logger.warning(
                                    f"[Batch_Self_Check_Eval] 语料 {corpus_id} 自检拒绝，加入 fallback"
                                )
                                break
                    elif batch_skip_on_repeated_failure:
                        all_batch_results[corpus_id] = {
                            "entities": verified_results_dict.get(corpus_id, {}).get(
                                "entities", DEFAULT_ENTITY_DICT.copy()
                            ),
                            "triples": [],
                            "eval_passed": False,
                            "eval_scores": [],
                            "entity_attrs": {},
                            "relation_attrs": {},
                            "confidence": "error",
                            "error": "batch_self_check_eval_rejected",
                            "skip_reason": "batch_self_check_eval_rejected",
                        }
                        logger.warning(
                            f"[Batch_Self_Check_Eval] 语料 {corpus_id} 自检拒绝，按策略直接跳过"
                        )

                verified_eval_results = {}
                for r in eval_check_result["verified_results"]:
                    corpus_id = r["corpus_id"]
                    verified_eval_results[corpus_id] = {
                        "corrected_triples": r.get("verified_triples", []),
                        "scores": eval_result["batch_eval_results"]
                        .get(corpus_id, {})
                        .get("scores", []),
                        "eval_passed": True,
                        "confidence": r.get("confidence", "medium"),
                    }
            else:
                verified_eval_results = eval_result["batch_eval_results"]

            updated_eval_results = {}
            max_eval_reextract_rounds = max(config.self_check_max_retries, 1)

            # P19改进：先批量收集 eval 困惑，再一次性请求导师回答
            eval_confusion_items: List[Dict] = []
            for corpus_id, eval_data in verified_eval_results.items():
                source_corpus = batch_corpus_map.get(str(corpus_id))
                if not (enable_qa_mentor and source_corpus):
                    continue

                joint_data = verified_results_dict.get(corpus_id, {})
                eval_confusion = detect_eval_confusion(
                    {
                        "eval_passed": eval_data.get("eval_passed", False),
                        "corrected_triples": eval_data.get("corrected_triples", []),
                    },
                    {
                        "raw_text": source_corpus.get("text", ""),
                        "entities": joint_data.get(
                            "entities", DEFAULT_ENTITY_DICT.copy()
                        ),
                        "triples": joint_data.get("triples", []),
                    },
                    reject_ratio_threshold=mentor_eval_reject_ratio_threshold,
                )

                if _should_request_mentor(eval_confusion):
                    logger.info(
                        f"[Batch-Mentor-Inline] 语料 {corpus_id} 在 eval 困惑，加入批量导师请求"
                    )
                    eval_confusion_items.append(
                        {
                            "corpus_id": str(corpus_id),
                            "source_corpus": source_corpus,
                            "confusion": eval_confusion,
                            "context": (
                                f"eval_passed={eval_data.get('eval_passed', False)}; "
                                f"triple_count={len(eval_data.get('corrected_triples', []))}"
                            ),
                        }
                    )

            mentor_eval_response_map: Dict[str, Dict] = {}
            if eval_confusion_items:
                mentor_eval_response_map = await _mentor_batch_answer_queries(
                    source_node="eval",
                    query_payloads=eval_confusion_items,
                )

            updated_eval_results = dict(verified_eval_results)
            eval_rerun_count = 0
            if mentor_eval_response_map:
                rerun_eval_input = {}
                for corpus_id, mentor_resp in mentor_eval_response_map.items():
                    joint_data = verified_results_dict.get(corpus_id, {})
                    if not joint_data:
                        continue
                    rerun_eval_input[corpus_id] = {
                        "entities": joint_data.get("entities", DEFAULT_ENTITY_DICT.copy()),
                        "triples": joint_data.get("triples", []),
                        "mentor_note": mentor_resp.get("answer", ""),
                    }
                if rerun_eval_input:
                    eval_rerun_count = len(rerun_eval_input)
                    rerun_eval_batch = await batch_eval_node(
                        rerun_eval_input, corpus_texts, dummy_writer
                    )
                    rerun_eval_map = rerun_eval_batch.get("batch_eval_results", {}) or {}
                    for corpus_id, rerun_eval in rerun_eval_map.items():
                        old_eval = verified_eval_results.get(corpus_id, {})
                        updated_eval_results[corpus_id] = {
                            "corrected_triples": rerun_eval.get(
                                "corrected_triples",
                                old_eval.get("corrected_triples", []),
                            ),
                            "scores": rerun_eval.get("scores", old_eval.get("scores", [])),
                            "eval_passed": rerun_eval.get(
                                "eval_passed", old_eval.get("eval_passed", False)
                            ),
                            "confidence": old_eval.get("confidence", "medium"),
                        }
            logger.info(
                f"[Batch-Mentor-Stats] stage=eval confusion={len(eval_confusion_items)} "
                f"mentor_answered={len(mentor_eval_response_map)} rerun_batch={eval_rerun_count}"
            )

            for corpus_id, eval_data in updated_eval_results.items():
                triples = eval_data.get("corrected_triples", [])
                eval_passed = eval_data.get("eval_passed", False)

                if not eval_passed and batch_skip_on_repeated_failure:
                    all_batch_results[corpus_id] = {
                        "entities": verified_results_dict.get(corpus_id, {}).get(
                            "entities", DEFAULT_ENTITY_DICT.copy()
                        ),
                        "triples": triples,
                        "eval_passed": False,
                        "eval_scores": eval_data.get("scores", []),
                        "confidence": eval_data.get("confidence", "medium"),
                        "reextract_needed": False,
                        "error": "batch_eval_not_passed_after_retries",
                        "skip_reason": "batch_eval_not_passed",
                    }
                    logger.warning(
                        f"[Batch_Eval] 语料 {corpus_id} 未通过评估，按策略直接跳过"
                    )
                    continue

                if not eval_passed and batch_llm_fallback:
                    source_corpus = batch_corpus_map.get(str(corpus_id))
                    joint_data = verified_results_dict.get(corpus_id, {})

                    if source_corpus:
                        current_entities = joint_data.get(
                            "entities", DEFAULT_ENTITY_DICT.copy()
                        )
                        current_triples = joint_data.get("triples", [])
                        current_confidence = joint_data.get("confidence", "medium")
                        current_passed = False
                        current_scores = eval_data.get("scores", [])

                        for _ in range(max_eval_reextract_rounds):
                            rerun_state = {
                                "entities": current_entities,
                                "triples": current_triples,
                                "joint_extraction_result": {
                                    "entities": _flatten_entities(current_entities),
                                    "triples": current_triples,
                                    "overall_confidence": current_confidence,
                                },
                            }

                            rerun_joint = None
                            if enable_qa_mentor:
                                rerun_joint = await _mentor_rerun_current_node(
                                    corpus=source_corpus,
                                    source_node="joint_ner_re",
                                    confusion={
                                        "query_type": "eval_reextract",
                                        "query_content": "Eval 未通过，局部重抽并重评估",
                                        "involved_entities": [],
                                        "involved_relations": [],
                                        "current_confidence": "low",
                                    },
                                    node_state=rerun_state,
                                )
                            if rerun_joint is None:
                                fallback_state = _build_base_state(source_corpus)
                                fallback_state.update(rerun_state)
                                rerun_joint = await single_joint_node(
                                    fallback_state, dummy_writer
                                )

                            if not rerun_joint:
                                continue

                            current_entities = rerun_joint.get(
                                "entities", DEFAULT_ENTITY_DICT.copy()
                            )
                            current_triples = rerun_joint.get("triples", [])
                            current_confidence = rerun_joint.get(
                                "joint_extraction_result", {}
                            ).get("overall_confidence", current_confidence)

                            reeval_input = {
                                corpus_id: {
                                    "entities": current_entities,
                                    "triples": current_triples,
                                }
                            }
                            reeval_result = await batch_eval_node(
                                reeval_input, corpus_texts, dummy_writer
                            )
                            reeval_data = reeval_result.get(
                                "batch_eval_results", {}
                            ).get(corpus_id, {})

                            current_triples = reeval_data.get(
                                "corrected_triples", current_triples
                            )
                            current_scores = reeval_data.get("scores", [])
                            current_passed = reeval_data.get("eval_passed", False)

                            if current_passed:
                                break

                        verified_results_dict[corpus_id] = {
                            "corpus_id": corpus_id,
                            "entities": current_entities,
                            "triples": current_triples,
                            "confidence": current_confidence,
                        }

                        all_batch_results[corpus_id] = {
                            "entities": current_entities,
                            "triples": current_triples,
                            "eval_passed": current_passed,
                            "eval_scores": current_scores,
                            "confidence": current_confidence,
                            "reextract_needed": not current_passed,
                        }
                    else:
                        all_batch_results[corpus_id] = {
                            "entities": verified_results_dict.get(corpus_id, {}).get(
                                "entities", DEFAULT_ENTITY_DICT.copy()
                            ),
                            "triples": triples,
                            "eval_passed": False,
                            "eval_scores": eval_data.get("scores", []),
                            "confidence": eval_data.get("confidence", "medium"),
                            "reextract_needed": True,
                        }
                    continue

                all_batch_results[corpus_id] = {
                    "entities": verified_results_dict.get(corpus_id, {}).get(
                        "entities", DEFAULT_ENTITY_DICT.copy()
                    ),
                    "triples": triples,
                    "eval_passed": eval_passed,
                    "eval_scores": eval_data.get("scores", []),
                    "confidence": eval_data.get("confidence", "medium"),
                    "reextract_needed": False,
                }

        if verified_results_dict:
            label_input = {}
            for corpus_id, joint_data in verified_results_dict.items():
                merged_data = all_batch_results.get(corpus_id, {})
                label_input[corpus_id] = {
                    "entities": merged_data.get(
                        "entities", joint_data.get("entities", DEFAULT_ENTITY_DICT.copy())
                    ),
                    "triples": sanitize_triples_for_pipeline(
                        merged_data.get("triples", joint_data.get("triples", [])),
                        context=f"batch_label_input:{corpus_id}",
                    ),
                }

            if label_input:
                label_result = await _run_batch_stage_with_retry(
                    f"Batch {batch_num}/Batch_Label",
                    lambda: batch_label_node(label_input, corpus_texts, dummy_writer),
                    lambda r: (not isinstance(r, dict))
                    or bool(r.get("error"))
                    or (not r.get("batch_label_results") and bool(label_input))
                    or _has_missing_coverage(
                        f"Batch {batch_num}/Batch_Label",
                        _dict_id_set(label_input),
                        _dict_id_set(r.get("batch_label_results", {})),
                    ),
                )
                label_result = label_result or {
                    "batch_label_results": {},
                    "error": "Batch_Label 未返回有效结果",
                }
                if batch_skip_on_repeated_failure and label_result.get("error"):
                    err = str(label_result.get("error"))
                    _mark_many_skipped(
                        list(label_input.keys()), err, "batch_label_failed_after_retries"
                    )
                    logger.warning(
                        f"[Batch {batch_num}] Batch_Label 反复失败，按策略直接跳过本批"
                    )
                    continue

                if config.enable_full_self_check:
                    label_check_result = await _run_batch_stage_with_retry(
                        f"Batch {batch_num}/Batch_Self_Check_Label",
                        lambda: batch_self_check_label_node(
                            label_result["batch_label_results"], corpus_texts, dummy_writer
                        ),
                        lambda r: (not isinstance(r, dict))
                        or bool(r.get("error"))
                        or (
                            not r.get("verified_results")
                            and bool(label_result.get("batch_label_results"))
                        )
                        or _has_missing_coverage(
                            f"Batch {batch_num}/Batch_Self_Check_Label",
                            _dict_id_set(label_result.get("batch_label_results", {})),
                            _list_corpus_id_set(r.get("verified_results", []))
                            | _list_corpus_id_set(r.get("rejected_results", [])),
                        ),
                    )
                    label_check_result = label_check_result or {
                        "verified_results": [],
                        "rejected_results": [],
                        "error": "Batch_Self_Check_Label 未返回有效结果",
                    }
                    if batch_skip_on_repeated_failure and label_check_result.get("error"):
                        err = str(label_check_result.get("error"))
                        _mark_many_skipped(
                            list(label_result.get("batch_label_results", {}).keys()),
                            err,
                            "batch_self_check_label_failed_after_retries",
                        )
                        logger.warning(
                            f"[Batch {batch_num}] Batch_Self_Check_Label 反复失败，按策略直接跳过本批"
                        )
                        continue

                    for r in label_check_result["rejected_results"]:
                        corpus_id = r.get("corpus_id")
                        if not batch_skip_on_repeated_failure and batch_llm_fallback:
                            for corpus in batch_corpus:
                                if corpus["id"] == corpus_id:
                                    cid = str(corpus.get("id"))
                                    if cid not in fallback_corpus_ids:
                                        fallback_corpus_ids.add(cid)
                                        fallback_corpus_list.append(corpus)
                                    logger.warning(
                                        f"[Batch_Self_Check_Label] 语料 {corpus_id} 自检拒绝，加入 fallback"
                                    )
                                    break
                        elif batch_skip_on_repeated_failure:
                            all_batch_results[corpus_id] = {
                                "entities": all_batch_results.get(corpus_id, {}).get(
                                    "entities", DEFAULT_ENTITY_DICT.copy()
                                ),
                                "triples": all_batch_results.get(corpus_id, {}).get(
                                    "triples", []
                                ),
                                "eval_passed": all_batch_results.get(corpus_id, {}).get(
                                    "eval_passed", False
                                ),
                                "eval_scores": all_batch_results.get(corpus_id, {}).get(
                                    "eval_scores", []
                                ),
                                "entity_attrs": {},
                                "relation_attrs": {},
                                "confidence": "error",
                                "error": "batch_self_check_label_rejected",
                                "skip_reason": "batch_self_check_label_rejected",
                            }
                            logger.warning(
                                f"[Batch_Self_Check_Label] 语料 {corpus_id} 自检拒绝，按策略直接跳过"
                            )

                    verified_label_results = {
                        r["corpus_id"]: {
                            "entity_attrs": r.get("entity_attrs", {}),
                            "relation_attrs": r.get("relation_attrs", {}),
                        }
                        for r in label_check_result["verified_results"]
                    }
                else:
                    verified_label_results = label_result["batch_label_results"]

                updated_label_results = {}
                label_confusion_items: List[Dict] = []
                for corpus_id, label_data in verified_label_results.items():
                    current_label = label_data
                    updated_label_results[corpus_id] = current_label

                    source_corpus = batch_corpus_map.get(str(corpus_id))
                    if not (enable_qa_mentor and source_corpus):
                        continue

                    eval_data = all_batch_results.get(corpus_id, {})
                    label_confusion = detect_label_confusion(
                        {
                            "entity_attrs": label_data.get("entity_attrs", {}),
                            "relation_attrs": label_data.get("relation_attrs", {}),
                            "overall_confidence": label_data.get(
                                "confidence", "medium"
                            ),
                        },
                        {
                            "raw_text": source_corpus.get("text", ""),
                            "entities": eval_data.get(
                                "entities", DEFAULT_ENTITY_DICT.copy()
                            ),
                            "triples": eval_data.get("triples", []),
                        },
                        missing_ratio_threshold=mentor_label_missing_ratio_threshold,
                        min_missing_attrs=mentor_label_min_missing_attrs,
                    )

                    if _should_request_mentor(label_confusion):
                        logger.info(
                            f"[Batch-Mentor-Inline] 语料 {corpus_id} 在 label 困惑，加入批量导师请求"
                        )
                        label_confusion_items.append(
                            {
                                "corpus_id": str(corpus_id),
                                "source_corpus": source_corpus,
                                "confusion": label_confusion,
                                "context": (
                                    f"entity_attr_count={len(label_data.get('entity_attrs', {}))}; "
                                    f"relation_attr_count={len(label_data.get('relation_attrs', {}))}"
                                ),
                            }
                        )

                mentor_label_response_map: Dict[str, Dict] = {}
                if label_confusion_items:
                    mentor_label_response_map = await _mentor_batch_answer_queries(
                        source_node="label",
                        query_payloads=label_confusion_items,
                    )

                label_rerun_count = 0
                if mentor_label_response_map:
                    rerun_label_input = {}
                    for corpus_id, mentor_resp in mentor_label_response_map.items():
                        eval_data = all_batch_results.get(corpus_id, {})
                        if not eval_data:
                            continue
                        rerun_label_input[corpus_id] = {
                            "entities": eval_data.get("entities", DEFAULT_ENTITY_DICT.copy()),
                            "triples": sanitize_triples_for_pipeline(
                                eval_data.get("triples", []),
                                context=f"batch_label_mentor_rerun:{corpus_id}",
                            ),
                            "mentor_note": mentor_resp.get("answer", ""),
                        }
                    if rerun_label_input:
                        label_rerun_count = len(rerun_label_input)
                        rerun_label_batch = await batch_label_node(
                            rerun_label_input, corpus_texts, dummy_writer
                        )
                        rerun_label_map = rerun_label_batch.get("batch_label_results", {}) or {}
                        for corpus_id, rerun_label in rerun_label_map.items():
                            label_data = verified_label_results.get(corpus_id, {})
                            updated_label_results[corpus_id] = {
                                "entity_attrs": rerun_label.get(
                                    "entity_attrs", label_data.get("entity_attrs", {})
                                ),
                                "relation_attrs": rerun_label.get(
                                    "relation_attrs", label_data.get("relation_attrs", {})
                                ),
                                "confidence": label_data.get("confidence", "medium"),
                            }
                logger.info(
                    f"[Batch-Mentor-Stats] stage=label confusion={len(label_confusion_items)} "
                    f"mentor_answered={len(mentor_label_response_map)} rerun_batch={label_rerun_count}"
                )

                for corpus_id, label_data in updated_label_results.items():
                    if corpus_id in all_batch_results:
                        all_batch_results[corpus_id]["entity_attrs"] = label_data.get(
                            "entity_attrs", {}
                        )
                        all_batch_results[corpus_id]["relation_attrs"] = label_data.get(
                            "relation_attrs", {}
                        )

            # P21新增：批量实体对齐（可选）
            enable_entity_alignment = getattr(config, "enable_entity_alignment", False)
            if enable_entity_alignment and batch_entity_alignment_node and updated_label_results:
                logger.info("[Batch_Entity_Alignment] 开始批量实体对齐")

                # 获取数据库已有实体（用于对齐匹配）
                existing_entities = []
                try:
                    from settings import settings
                    from kg.postgres_client import PostgresClient
                    pg_config = settings.get_postgres_config()
                    with PostgresClient(**pg_config) as pg_client:
                        # 从 geo_entity_names 表获取已有实体名称
                        with pg_client.conn.cursor() as cur:
                            cur.execute("SELECT name FROM geo_entity_names LIMIT 1000")
                            existing_entities = [row[0] for row in cur.fetchall()]
                    logger.debug(f"[Batch_Entity_Alignment] 获取到 {len(existing_entities)} 个已有实体")
                except Exception as e:
                    logger.warning(f"[Batch_Entity_Alignment] 获取已有实体失败: {e}, 使用空列表")
                    existing_entities = []

                # 调用实体对齐节点
                alignment_result = await _run_batch_stage_with_retry(
                    f"Batch {batch_num}/Batch_Entity_Alignment",
                    lambda: batch_entity_alignment_node(
                        updated_label_results,
                        corpus_texts,
                        existing_entities,
                        dummy_writer,
                    ),
                    lambda r: (not isinstance(r, dict))
                    or bool(r.get("error"))
                    or (not r.get("aligned_results") and bool(updated_label_results))
                    or _has_missing_coverage(
                        f"Batch {batch_num}/Batch_Entity_Alignment",
                        _dict_id_set(updated_label_results),
                        _dict_id_set(r.get("aligned_results", {})),
                    ),
                )
                alignment_result = alignment_result or {
                    "aligned_results": {},
                    "error": "Batch_Entity_Alignment 未返回有效结果",
                }
                if batch_skip_on_repeated_failure and alignment_result.get("error"):
                    err = str(alignment_result.get("error"))
                    _mark_many_skipped(
                        list(updated_label_results.keys()),
                        err,
                        "batch_entity_alignment_failed_after_retries",
                    )
                    logger.warning(
                        f"[Batch {batch_num}] Batch_Entity_Alignment 反复失败，按策略直接跳过本批"
                    )
                    continue

                # 更新结果
                aligned_results = alignment_result.get("aligned_results", {})
                for corpus_id, aligned_data in aligned_results.items():
                    target_corpus_id = (
                        corpus_id
                        if corpus_id in all_batch_results
                        else str(corpus_id)
                    )
                    if target_corpus_id in all_batch_results:
                        # 合并对齐后的实体属性
                        aligned_attrs = aligned_data.get("aligned_entity_attrs", {}) or {}
                        new_entities = aligned_data.get("new_entities", []) or []
                        aligned_entity_names = list(aligned_attrs.keys())

                        if aligned_attrs:
                            original_attrs = all_batch_results[target_corpus_id].get(
                                "entity_attrs", {}
                            )
                            # 对齐属性覆盖原始属性
                            all_batch_results[target_corpus_id]["entity_attrs"] = {
                                **original_attrs,
                                **aligned_attrs,
                            }

                        total_entities = len(aligned_entity_names) + len(new_entities)
                        alignment_rate = (
                            len(aligned_entity_names) / total_entities
                            if total_entities > 0
                            else 0.0
                        )

                        all_batch_results[target_corpus_id]["aligned_entities"] = (
                            aligned_entity_names
                        )
                        all_batch_results[target_corpus_id]["new_entities"] = new_entities
                        all_batch_results[target_corpus_id]["entity_alignment_result"] = {
                            "alignment_items": [],
                            "aligned_entities": aligned_entity_names,
                            "aligned_entity_attrs": aligned_attrs,
                            "new_entities": new_entities,
                            "created_entities": [],
                            "created_count": 0,
                            "skipped_entities": [],
                            "overall_alignment_rate": alignment_rate,
                            "alignment_confidence": aligned_data.get(
                                "confidence", "medium"
                            ),
                        }
                        logger.info(
                            f"[Batch_Entity_Alignment][Corpus {target_corpus_id}] "
                            f"aligned={len(aligned_entity_names)}, "
                            f"new={len(new_entities)}, "
                            f"confidence={aligned_data.get('confidence', 'medium')}"
                        )

                logger.info(f"[Batch_Entity_Alignment] 完成: {len(aligned_results)} 条语料")

    return {
        "batch_results": all_batch_results,
        "cross_corpus_aliases": all_cross_corpus_aliases,
        "fallback_corpus_list": fallback_corpus_list,
        "needs_single_processing": len(fallback_corpus_list) > 0,
    }

def create_qa_mentor_node(llm: Any, config: ExtractionConfig):
    """
    创建QA导师节点 - 使用强模型进行深度语义分析

    职责：
    1. 生成5W1H问答脚手架
    2. 输出导师指导信息（语义关注点、实体优先级、质量标准）
    3. 设定预期约束
    4. 保存推理过程（可选）

    P14改进：支持回答后续节点的查询（双向交流）
    """
    from .schemas import QAMentorScaffoldResult, MentorQueryResponse

    scaffold_parser = PydanticOutputParser(pydantic_object=QAMentorScaffoldResult)
    query_parser = PydanticOutputParser(pydantic_object=MentorQueryResponse)

    async def qa_mentor_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """QA导师节点：深度语义分析 + 导师指导 + 回答查询"""
        corpus_id = state["corpus_id"]

        # P14新增：检查是否有来自后续节点的查询
        mentor_query = state.get("mentor_query")

        if mentor_query:
            # ===== 回答查询模式 =====
            query_source = state.get("query_source_node", "unknown")
            logger.info(
                f"[QA_Mentor] 回答来自 {query_source} 的查询: {mentor_query.get('query_type')}"
            )

            writer(
                {
                    "step": "qa_mentor",
                    "corpus_id": corpus_id,
                    "status": "answering_query",
                    "query_source": query_source,
                    "query_type": mentor_query.get("query_type"),
                }
            )

            try:
                text_for_processing = get_text_for_processing(state)

                # 构造查询上下文
                query_type = mentor_query.get("query_type", "unknown")
                query_content = mentor_query.get("query_content", "")
                involved_entities = mentor_query.get("involved_entities", [])
                involved_relations = mentor_query.get("involved_relations", [])
                current_confidence = mentor_query.get("current_confidence", "medium")

                # 构建当前处理结果的摘要作为上下文
                context = _build_query_context(state, query_source)

                # 获取之前的导师指导
                previous_guidance = state.get("mentor_guidance", {})
                previous_guidance_text = format_mentor_guidance(previous_guidance)

                # 调用导师回答查询
                prompt_text = MENTOR_QUERY_PROMPT.invoke(
                    {
                        "source_node": query_source,
                        "query_type": query_type,
                        "query_content": query_content,
                        "involved_entities": ", ".join(involved_entities)
                        if involved_entities
                        else "(无)",
                        "involved_relations": ", ".join(involved_relations)
                        if involved_relations
                        else "(无)",
                        "current_confidence": current_confidence,
                        "context": context,
                        "raw_text": text_for_processing,
                        "previous_guidance": previous_guidance_text,
                    }
                )
                full_prompt = f"{prompt_text.messages[1].content}\n\n{query_parser.get_format_instructions()}"
                response = await llm.ainvoke(full_prompt)
                query_result: MentorQueryResponse = (
                    safe_parse_json_with_quote_fix(query_parser, response.content)
                )

                logger.info(
                    f"[QA_Mentor] 查询回答完成: 置信度={query_result.response_confidence}, "
                    f"建议修改={query_result.suggests_revision}"
                )

                writer(
                    {
                        "step": "qa_mentor",
                        "corpus_id": corpus_id,
                        "status": "query_answered",
                        "answer": query_result.answer[:100],
                        "confidence": query_result.response_confidence,
                        "return_to": query_result.return_to_node,
                    }
                )

                # 更新指导信息（如果有）
                updated_guidance = {}
                if query_result.updated_guidance:
                    updated_guidance = query_result.updated_guidance.model_dump()
                elif previous_guidance:
                    updated_guidance = previous_guidance

                # 返回到发起查询的节点（优先使用导师建议的目标节点）
                return_to = (
                    query_result.return_to_node
                    if query_result.return_to_node
                    else (query_source if query_source else "joint_ner_re")
                )

                return {
                    "mentor_response": query_result.model_dump(),
                    "mentor_guidance": updated_guidance,
                    "qa_entity_hints": query_result.updated_entity_hints
                    or state.get("qa_entity_hints", []),
                    "qa_relation_hints": query_result.updated_relation_hints
                    or state.get("qa_relation_hints", []),
                    "needs_mentor_help": False,  # 已回答，继续处理
                    "query_count": state.get("query_count", 0) + 1,  # 增加查询计数
                    "return_to_node": return_to,
                    "current_step": getattr(
                        StepEnum, return_to.upper(), StepEnum.JOINT_NER_RE
                    ),
                }

            except Exception as e:
                logger.error(f"[QA_Mentor] 回答查询失败: {e}")
                writer(
                    {
                        "step": "qa_mentor",
                        "corpus_id": corpus_id,
                        "status": "query_error",
                        "error": str(e),
                    }
                )
                # 查询失败时，返回到原节点继续
                return {
                    "mentor_response": {},
                    "needs_mentor_help": False,
                    "query_count": state.get("query_count", 0) + 1,
                    "return_to_node": state.get("query_source_node", "joint_ner_re"),
                    "current_step": getattr(
                        StepEnum,
                        state.get("query_source_node", "joint_ner_re").upper(),
                        StepEnum.JOINT_NER_RE,
                    ),
                }

        else:
            # ===== 初始化指导模式 =====
            logger.info(f"[QA_Mentor] 处理语料: {corpus_id}")

            writer(
                {
                    "step": "qa_mentor",
                    "corpus_id": corpus_id,
                    "status": "started",
                    "message": "开始导师深度分析",
                }
            )

            try:
                # 使用归一化后的文本
                text_for_processing = get_text_for_processing(state)

                # 调用LLM
                prompt_text = QA_MENTOR_PROMPT.invoke(
                    {
                        "normalized_text": text_for_processing,
                    }
                )
                full_prompt = f"{prompt_text.messages[1].content}\n\n{scaffold_parser.get_format_instructions()}"
                response = await llm.ainvoke(full_prompt)
                result: QAMentorScaffoldResult = (
                    safe_parse_json_with_quote_fix(scaffold_parser, response.content)
                )

                logger.info(
                    f"[QA_Mentor] 完成: {len(result.qa_pairs)} 个问答对, "
                    f"{len(result.entity_hints)} 个实体提示, "
                    f"置信度={result.overall_confidence}"
                )

                # 发送完成事件
                writer(
                    {
                        "step": "qa_mentor",
                        "corpus_id": corpus_id,
                        "status": "completed",
                        "qa_count": len(result.qa_pairs),
                        "entity_hints": result.entity_hints,
                        "relation_hints": result.relation_hints,
                        "confidence": result.overall_confidence,
                        "has_mentor_guidance": result.mentor_guidance is not None,
                    }
                )

                # 根据结果决定下一步
                if result.should_skip_detailed_extraction:
                    logger.info(f"[QA_Mentor] 建议跳过详细抽取: {corpus_id}")
                    return {
                        "qa_scaffold_result": result.model_dump(),
                        "semantic_summary": result.semantic_summary,
                        "mentor_guidance": result.mentor_guidance.model_dump()
                        if result.mentor_guidance
                        else {},
                        "reasoning_trace": result.reasoning_trace,
                        "qa_entity_hints": result.entity_hints,
                        "qa_relation_hints": result.relation_hints,
                        "qa_context_dependencies": result.context_dependencies,
                        "needs_mentor_help": False,  # P14新增
                        "current_step": StepEnum.DONE,
                    }
                else:
                    return {
                        "qa_scaffold_result": result.model_dump(),
                        "semantic_summary": result.semantic_summary,
                        "mentor_guidance": result.mentor_guidance.model_dump()
                        if result.mentor_guidance
                        else {},
                        "reasoning_trace": result.reasoning_trace,
                        "qa_entity_hints": result.entity_hints,
                        "qa_relation_hints": result.relation_hints,
                        "qa_context_dependencies": result.context_dependencies,
                        "needs_mentor_help": False,  # P14新增
                        "current_step": StepEnum.JOINT_NER_RE,
                    }

            except Exception as e:
                logger.error(f"[QA_Mentor] 处理失败: {e}")
                writer(
                    {
                        "step": "qa_mentor",
                        "corpus_id": corpus_id,
                        "status": "error",
                        "error": str(e),
                    }
                )
                return {
                    "qa_scaffold_result": {},
                    "semantic_summary": "",
                    "mentor_guidance": {},
                    "qa_entity_hints": [],
                    "qa_relation_hints": [],
                    "qa_context_dependencies": [],
                    "error": str(e),
                    "needs_mentor_help": False,
                    "current_step": StepEnum.JOINT_NER_RE,  # 失败时继续
                }

    return qa_mentor_node


def _build_query_context(state: CorpusState, source_node: str) -> str:
    """构建查询上下文：根据来源节点格式化当前处理结果"""
    context_lines = []

    if source_node == "joint_ner_re":
        # 联合抽取节点：显示抽取结果
        entities = state.get("entities", {})
        triples = state.get("triples", [])
        joint_result = state.get("joint_extraction_result", {})

        if entities:
            entity_lines = []
            for entity_type, names in entities.items():
                if names:
                    entity_lines.append(f"{entity_type}: {', '.join(names[:5])}")
            context_lines.append("当前抽取实体:\n" + "\n".join(entity_lines))

        if triples:
            triple_lines = [
                f"<{t.get('head')}, {t.get('relation')}, {t.get('tail')}>"
                for t in triples[:5]
            ]
            context_lines.append("当前抽取三元组:\n" + "\n".join(triple_lines))

        if joint_result:
            confidence = joint_result.get("overall_confidence", "unknown")
            context_lines.append(f"整体置信度: {confidence}")

    elif source_node == "eval":
        # 评估节点：显示评估结果
        eval_passed = state.get("eval_passed", False)
        corrected_triples = state.get("corrected_triples", [])
        original_triples = state.get("triples", [])

        context_lines.append(f"评估通过: {eval_passed}")
        context_lines.append(f"原始三元组数: {len(original_triples)}")
        context_lines.append(f"修正后三元组数: {len(corrected_triples)}")

        if corrected_triples:
            passed_count = sum(
                1 for t in corrected_triples if t.get("passed_eval", False)
            )
            context_lines.append(f"通过评估数: {passed_count}")

    elif source_node == "label":
        # 标注节点：显示标注结果
        entity_attrs = state.get("entity_attrs", {})
        relation_attrs = state.get("relation_attrs", {})

        context_lines.append(f"已标注实体数: {len(entity_attrs)}")
        context_lines.append(f"已标注关系数: {len(relation_attrs)}")

    return "\n\n".join(context_lines) if context_lines else "(无上下文信息)"


def create_qa_approval_node(llm: Any, config: ExtractionConfig):
    """
    创建QA审批节点 - 审批后续节点的抽取结果

    职责：
    1. 校验联合抽取结果
    2. 校验评估结果
    3. 校验标注结果
    4. 输出审批状态和改进反馈
    5. 整合语义脚手架
    """
    from .schemas import QAApprovalResult

    parser = PydanticOutputParser(pydantic_object=QAApprovalResult)

    async def qa_approval_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """QA审批节点：审批后续节点结果"""
        corpus_id = state["corpus_id"]
        revision_cycle_count = state.get("revision_cycle_count", 0)
        max_revision_cycles = state.get(
            "max_revision_cycles", config.max_revision_cycles
        )

        logger.info(
            f"[QA_Approval] 审批语料: {corpus_id}, 修改轮次: {revision_cycle_count}/{max_revision_cycles}"
        )

        writer(
            {
                "step": "qa_approval",
                "corpus_id": corpus_id,
                "status": "started",
                "revision_cycle": revision_cycle_count,
            }
        )

        try:
            text = get_text_for_processing(state)

            # 格式化各节点结果用于审批
            joint_result = state.get("joint_extraction_result", {})
            eval_result = {
                "eval_passed": state.get("eval_passed", False),
                "corrected_triples": state.get("corrected_triples", []),
            }
            label_result = {
                "entity_attrs": state.get("entity_attrs", {}),
                "relation_attrs": state.get("relation_attrs", {}),
            }

            # 导师指导
            mentor_guidance = state.get("mentor_guidance", {})
            semantic_summary = state.get("semantic_summary", "")

            # 历史反馈
            revision_feedbacks = state.get("revision_feedbacks", [])

            # Self-Check反思结果（用于审批一致性检查）
            reflection_summary = format_reflection_for_approval(state)

            # 调用LLM进行审批
            prompt_text = QA_APPROVAL_PROMPT.invoke(
                {
                    "raw_text": text,
                    "mentor_guidance": format_mentor_guidance(mentor_guidance),
                    "semantic_summary": semantic_summary,
                    "joint_result": format_joint_for_approval(joint_result),
                    "eval_result": format_eval_for_approval(eval_result),
                    "label_result": format_label_for_approval(label_result),
                    "previous_feedbacks": format_revision_feedbacks(revision_feedbacks),
                    "reflection_summary": reflection_summary,
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: QAApprovalResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            logger.info(
                f"[QA_Approval] 完成: overall_status={result.overall_status}, "
                f"retry_suggested={result.retry_suggested}, "
                f"retry_target_nodes={result.retry_target_nodes}"
            )

            writer(
                {
                    "step": "qa_approval",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "overall_status": result.overall_status.value,
                    "overall_confidence": result.overall_confidence,
                    "retry_suggested": result.retry_suggested,
                    "revision_cycle": revision_cycle_count + 1,
                }
            )

            # 收集反馈
            all_feedbacks = state.get("revision_feedbacks", [])
            for f in result.all_feedbacks:
                all_feedbacks.append(f.model_dump())

            # 更新语义脚手架
            integrated_semantic_summary = result.integrated_semantic_summary
            if integrated_semantic_summary:
                semantic_summary = integrated_semantic_summary

            return {
                "qa_approval_result": result.model_dump(),
                "integrated_semantic_summary": integrated_semantic_summary,
                "semantic_summary": semantic_summary,
                "revision_feedbacks": all_feedbacks,
                "revision_cycle_count": revision_cycle_count + 1,
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "pending_approval_nodes": result.retry_target_nodes,
                "current_step": StepEnum.DONE,  # 默认结束，路由会决定是否重试
            }

        except Exception as e:
            logger.error(f"[QA_Approval] 处理失败: {e}")
            writer(
                {
                    "step": "qa_approval",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            return {
                "qa_approval_result": {},
                "error": str(e),
                "revision_cycle_count": revision_cycle_count + 1,
                "current_step": StepEnum.DONE,
            }

    return qa_approval_node


def create_revision_joint_node(llm: Any):
    """
    创建修改联合抽取节点 - 根据QA反馈改进抽取结果

    职责：
    1. 根据反馈补充遗漏实体
    2. 删除幻觉三元组
    3. 修正关系错误
    4. 完善证据
    """
    parser = PydanticOutputParser(pydantic_object=JointExtractionResult)

    async def revision_joint_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """修改联合抽取节点"""
        corpus_id = state["corpus_id"]
        revision_cycle = state.get("revision_cycle_count", 0)
        logger.info(
            f"[Revision_Joint] 修改抽取: {corpus_id}, 修改轮次: {revision_cycle}"
        )

        writer(
            {
                "step": "revision_joint",
                "corpus_id": corpus_id,
                "status": "started",
                "revision_cycle": revision_cycle,
            }
        )

        try:
            text = get_text_for_processing(state)

            # 获取QA反馈
            revision_feedbacks = state.get("revision_feedbacks", [])
            recent_feedbacks = revision_feedbacks[-3:] if revision_feedbacks else []

            # 获取语义脚手架
            semantic_summary = state.get("semantic_summary", "")
            entity_hints = state.get("qa_entity_hints", [])
            relation_hints = state.get("qa_relation_hints", [])

            # 获取导师指导（QA Mentor模式）
            mentor_guidance = state.get("mentor_guidance", {})

            # 获取之前的抽取结果
            previous_entities = state.get("entities", {})
            previous_triples = state.get("triples", [])

            # 调用LLM改进
            prompt_text = REVISION_JOINT_PROMPT.invoke(
                {
                    "raw_text": text,
                    "feedback_summary": format_feedback_summary(
                        revision_feedbacks, revision_cycle
                    ),
                    "feedbacks": format_feedbacks_for_revision(recent_feedbacks),
                    "semantic_summary": semantic_summary,
                    "mentor_guidance": format_mentor_guidance(mentor_guidance),
                    "previous_entities": format_entities(previous_entities),
                    "previous_triples": format_triples(previous_triples),
                    "entity_hints": format_entity_hints(entity_hints),
                    "relation_hints": format_relation_hints(relation_hints),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: JointExtractionResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # 转换为现有格式（v3.4扩展版：6种实体类型）
            entities_dict = {
                "道路": [],
                "POI": [],
                "建筑物": [],
                "街区": [],
                "功能": [],
                "事件": [],
            }
            for e in result.entities:
                entity_type = extract_enum_value(e.type)  # P15改进：使用工具函数
                if entity_type in entities_dict:
                    entities_dict[entity_type].append(e.name)

            triples_list = [
                {
                    "head": t.head,
                    "relation": extract_enum_value(t.relation),  # P15改进：使用工具函数
                    "tail": t.tail,
                    "evidence": t.evidence,
                    "confidence": extract_enum_value(
                        t.confidence
                    ),  # P15改进：使用工具函数
                    "attributes": t.attributes.model_dump(exclude_none=True)
                    if t.attributes
                    else {},
                }
                for t in result.triples
            ]

            logger.info(
                f"[Revision_Joint] 完成: {len(result.entities)}个实体, "
                f"{len(result.triples)}个三元组, 置信度={result.overall_confidence}"
            )

            writer(
                {
                    "step": "revision_joint",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "entity_count": len(result.entities),
                    "triple_count": len(result.triples),
                    "revision_cycle": revision_cycle,
                }
            )

            return {
                "entities": entities_dict,
                "triples": triples_list,
                "joint_extraction_result": result.model_dump(),
                "extraction_strategy": "joint_revision",
                "current_step": StepEnum.EVAL,
            }

        except Exception as e:
            logger.error(f"[Revision_Joint] 处理失败: {e}")
            writer(
                {
                    "step": "revision_joint",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            return {
                "error": str(e),
                "current_step": StepEnum.EVAL,
            }

    return revision_joint_node


# ===== P11新增：实体对齐节点 =====


def create_entity_alignment_node(llm: Any, config: ExtractionConfig):
    """
    创建实体对齐节点 - 将抽取实体与数据库已有实体匹配

    流程：
    1. 从state获取抽取的实体名称
    2. 对每个实体name进行向量嵌入
    3. 查询策略（P12优化）：
       - geo_entity_names（小表）：预加载全部embedding到内存，批量计算
       - amap_poi_wgs84（大表）：利用pgvector HNSW索引批量查询
    4. 合并候选结果，按相似度排序
    5. 相似度判断：
       - >= high_threshold: 直接确认匹配
       - >= threshold && < high_threshold: 交给LLM判断
       - < threshold: 直接跳过（新实体）

    ID映射说明：
    - geo_entity_names：entity_id字段直接作为db_entity_id，neo4j中匹配entity_id属性
    - amap_poi_wgs84：entity_id字段存储原始高德ID(如amap_B0FFLCH14H)，
      直接作为db_entity_id，neo4j中匹配original_id属性

    性能优化策略（P12改进）：
    - geo_entity_names（~1000条）：预加载全部embedding (~3MB)，内存批量计算
    - amap_poi_wgs84（~37000条）：利用pgvector HNSW索引批量查询，避免加载全部到内存
    """
    from .schemas import (
        EntityAlignmentResult,
        EntityAlignmentItem,
        EntityCandidate,
        BatchEntityAlignmentDecisionResult,
    )

    parser = PydanticOutputParser(pydantic_object=EntityAlignmentItem)
    batch_alignment_parser = PydanticOutputParser(
        pydantic_object=BatchEntityAlignmentDecisionResult
    )

    # ===== 函数级缓存（P12性能优化） =====
    # 嵌入模型缓存（避免重复加载）
    _embedding_model_cache = None
    # geo_entity_names embedding缓存（小表，预加载全部）
    _geo_cache = None

    def _get_embedding_model():
        """懒加载嵌入模型"""
        nonlocal _embedding_model_cache
        if _embedding_model_cache is None:
            import os

            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像加速
            from sentence_transformers import SentenceTransformer

            model_name = config.alignment_embedding_model
            logger.info(f"[Entity_Alignment] 加载嵌入模型: {model_name}")
            _embedding_model_cache = SentenceTransformer(model_name)
        return _embedding_model_cache

    def _load_geo_embeddings(conn):
        """
        预加载geo_entity_names全部embedding到内存（小表策略）

        内存占用估算：~1000条 × 768维 × 4字节 ≈ 3MB

        参数：
        - conn: psycopg2原生连接对象

        返回：
        - geo_entities: List[Dict]
        - geo_embeddings_np: numpy数组 (N_geo, dim)
        """
        import numpy as np

        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id, name, type, longitude, latitude, embedding
                FROM geo_entity_names
                WHERE embedding IS NOT NULL
            """)
            geo_rows = cur.fetchall()

        geo_entities = []
        geo_embeddings = []
        for row in geo_rows:
            entity_id, name, type_, lon, lat, emb_str = row
            if emb_str:
                import json

                # 解析embedding向量（pgvector返回字符串格式，使用json.loads替代eval）
                emb_list = json.loads(emb_str) if isinstance(emb_str, str) else emb_str
                geo_entities.append(
                    {
                        "entity_id": entity_id,
                        "name": name,
                        "type": type_ or "",
                        "longitude": lon,
                        "latitude": lat,
                        "source": "geo_entity_names",
                    }
                )
                geo_embeddings.append(emb_list)

        geo_embeddings_np = np.array(geo_embeddings) if geo_embeddings else np.array([])
        logger.info(
            f"[Entity_Alignment] 预加载 geo_entity_names: {len(geo_entities)}条 (~{len(geo_entities) * 768 * 4 / 1024 / 1024:.1f}MB)"
        )

        return geo_entities, geo_embeddings_np

    def _batch_similarity_search_geo(
        query_embeddings_np, geo_embeddings_np, geo_entities, top_k
    ):
        """
        geo实体批量相似度搜索（内存计算）

        参数：
        - query_embeddings_np: (N_query, dim) numpy数组
        - geo_embeddings_np: (N_geo, dim) numpy数组
        - geo_entities: List[Dict]
        - top_k: 每个查询返回的候选数量

        返回：
        - candidates_per_query: List[List[Dict]] 每个查询的top_k候选列表
        """
        import numpy as np

        if len(geo_embeddings_np) == 0 or len(query_embeddings_np) == 0:
            return [[] for _ in range(len(query_embeddings_np))]

        # 归一化向量
        query_norms = np.linalg.norm(query_embeddings_np, axis=1, keepdims=True)
        db_norms = np.linalg.norm(geo_embeddings_np, axis=1, keepdims=True)

        query_normalized = query_embeddings_np / (query_norms + 1e-10)
        db_normalized = geo_embeddings_np / (db_norms + 1e-10)

        # 计算相似度矩阵 (N_query, N_geo)
        similarity_matrix = np.dot(query_normalized, db_normalized.T)

        # 为每个查询选取top_k
        candidates_per_query = []
        for i in range(len(query_embeddings_np)):
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            candidates = []
            for idx in top_indices:
                sim = float(similarities[idx])
                entity = geo_entities[idx].copy()
                entity["similarity"] = sim
                entity["db_entity_id"] = entity.pop("entity_id")
                candidates.append(entity)

            candidates_per_query.append(candidates)

        return candidates_per_query

    def _batch_similarity_search_amap(conn, query_embeddings_list, top_k):
        """
        amap实体批量相似度搜索（数据库查询，利用pgvector HNSW索引）

        策略：单次批量查询，传入所有query embedding，返回每个query的top_k候选

        ID映射说明：
        - amap_poi_wgs84表的entity_id字段存储原始高德ID（如amap_B0FFLCH14H）
        - neo4j中amap节点的original_id属性存储这个原始高德ID
        - 对齐结果中db_entity_id直接使用原始高德ID

        参数：
        - conn: psycopg2原生连接对象
        - query_embeddings_list: List[List[float]] 每个实体的embedding
        - top_k: 每个查询返回的候选数量

        返回：
        - candidates_per_query: List[List[Dict]]
        """
        candidates_per_query = []

        for query_emb in query_embeddings_list:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, entity_id, name, type, longitude, latitude, address,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM amap_poi_wgs84
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """,
                    (query_emb, query_emb, top_k),
                )
                amap_rows = cur.fetchall()

            candidates = []
            for row in amap_rows:
                # row: (id, entity_id, name, type, longitude, latitude, address, similarity)
                # entity_id字段存储原始高德ID（如amap_B0FFLCH14H）
                # neo4j中amap节点的original_id属性存储这个原始高德ID
                amap_table_id, amap_original_id, name, type_, lon, lat, address, sim = (
                    row
                )
                candidates.append(
                    {
                        "db_entity_id": amap_original_id,  # 直接使用原始高德ID，在neo4j中对应original_id属性
                        "name": name,
                        "type": type_ or "",
                        "similarity": sim,
                        "longitude": lon,
                        "latitude": lat,
                        "address": address,
                        "source": "amap_poi_wgs84",
                    }
                )

            candidates_per_query.append(candidates)

        return candidates_per_query

    # ===== 函数级缓存（P12性能优化） =====
    # ===== 函数级缓存（P12性能优化） =====
    # 嵌入模型缓存（避免重复加载）
    _embedding_model_cache = None
    # 数据库embedding缓存（避免重复查询）
    _db_cache = None
    # amap entity_id基数缓存
    _amap_id_base_cache = None

    def _get_embedding_model():
        """懒加载嵌入模型"""
        nonlocal _embedding_model_cache
        if _embedding_model_cache is None:
            import os

            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像加速
            from sentence_transformers import SentenceTransformer

            model_name = config.alignment_embedding_model
            logger.info(f"[Entity_Alignment] 加载嵌入模型: {model_name}")
            _embedding_model_cache = SentenceTransformer(model_name)
        return _embedding_model_cache

    def _load_db_embeddings(pg_client):
        """
        预加载数据库中所有实体embedding到内存

        返回：
        - geo_entities: List[Dict] 每个包含 entity_id, name, type, longitude, latitude, embedding
        - amap_entities: List[Dict] 每个包含 id, entity_id(原始), name, type, longitude, latitude, address, embedding
        - geo_embeddings_np: numpy数组 (N_geo, dim)
        - amap_embeddings_np: numpy数组 (N_amap, dim)
        """
        import numpy as np

        # 查询geo_entity_names
        with pg_client.conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id, name, type, longitude, latitude, embedding
                FROM geo_entity_names
                WHERE embedding IS NOT NULL
            """)
            geo_rows = cur.fetchall()

        geo_entities = []
        geo_embeddings = []
        for row in geo_rows:
            entity_id, name, type_, lon, lat, emb_str = row
            if emb_str:
                import json

                # 解析embedding向量（pgvector返回字符串格式，使用json.loads替代eval）
                emb_list = json.loads(emb_str) if isinstance(emb_str, str) else emb_str
                geo_entities.append(
                    {
                        "entity_id": entity_id,
                        "name": name,
                        "type": type_ or "",
                        "longitude": lon,
                        "latitude": lat,
                        "source": "geo_entity_names",
                    }
                )
                geo_embeddings.append(emb_list)

        # 查询amap_poi_wgs84
        with pg_client.conn.cursor() as cur:
            cur.execute("""
                SELECT id, entity_id, name, type, longitude, latitude, address, embedding
                FROM amap_poi_wgs84
                WHERE embedding IS NOT NULL
            """)
            amap_rows = cur.fetchall()

        amap_entities = []
        amap_embeddings = []
        for row in amap_rows:
            amap_id, original_id, name, type_, lon, lat, address, emb_str = row
            if emb_str:
                emb_list = eval(emb_str) if isinstance(emb_str, str) else emb_str
                amap_entities.append(
                    {
                        "id": amap_id,
                        "original_id": original_id,
                        "name": name,
                        "type": type_ or "",
                        "longitude": lon,
                        "latitude": lat,
                        "address": address,
                        "source": "amap_poi_wgs84",
                    }
                )
                amap_embeddings.append(emb_list)

        # 转换为numpy数组
        geo_embeddings_np = np.array(geo_embeddings) if geo_embeddings else np.array([])
        amap_embeddings_np = (
            np.array(amap_embeddings) if amap_embeddings else np.array([])
        )

        logger.info(
            f"[Entity_Alignment] 预加载 geo_entity_names: {len(geo_entities)}条, amap_poi_wgs84: {len(amap_entities)}条"
        )

        return geo_entities, amap_entities, geo_embeddings_np, amap_embeddings_np

    def _batch_similarity_search(
        query_embeddings_np, db_embeddings_np, db_entities, top_k
    ):
        """
        批量相似度搜索（内存计算）

        使用cosine similarity: similarity = 1 - cosine_distance

        参数：
        - query_embeddings_np: (N_query, dim) numpy数组
        - db_embeddings_np: (N_db, dim) numpy数组
        - db_entities: List[Dict] 数据库实体列表
        - top_k: 每个查询返回的候选数量

        返回：
        - candidates_per_query: List[List[Dict]] 每个查询的top_k候选列表
        """
        import numpy as np

        if len(db_embeddings_np) == 0 or len(query_embeddings_np) == 0:
            return [[] for _ in range(len(query_embeddings_np))]

        # 计算cosine similarity矩阵 (N_query, N_db)
        # cosine_sim = dot(A, B) / (norm(A) * norm(B))
        # 由于embedding通常已归一化，可以直接用 dot 计算相似度

        # 归一化query和db向量
        query_norms = np.linalg.norm(query_embeddings_np, axis=1, keepdims=True)
        db_norms = np.linalg.norm(db_embeddings_np, axis=1, keepdims=True)

        query_normalized = query_embeddings_np / (query_norms + 1e-10)
        db_normalized = db_embeddings_np / (db_norms + 1e-10)

        # 计算相似度矩阵
        similarity_matrix = np.dot(query_normalized, db_normalized.T)  # (N_query, N_db)

        # 为每个查询选取top_k
        candidates_per_query = []
        for i in range(len(query_embeddings_np)):
            similarities = similarity_matrix[i]
            # 获取top_k索引
            if len(similarities) >= top_k:
                top_indices = np.argsort(similarities)[-top_k:][::-1]  # 降序
            else:
                top_indices = np.argsort(similarities)[::-1]

            candidates = []
            for idx in top_indices:
                sim = float(similarities[idx])
                entity = db_entities[idx].copy()
                entity["similarity"] = sim
                candidates.append(entity)

            candidates_per_query.append(candidates)

        return candidates_per_query

    async def entity_alignment_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """实体对齐节点（批量优化版）"""
        corpus_id = state["corpus_id"]
        logger.info(f"[Entity_Alignment] 处理语料: {corpus_id}")

        writer(
            {
                "step": "entity_alignment",
                "corpus_id": corpus_id,
                "status": "started",
                "message": "开始实体对齐",
            }
        )

        try:
            # 连接数据库
            from settings import settings
            from kg.postgres_client import PostgresClient

            pg_config = settings.get_postgres_config()
            with PostgresClient(**pg_config) as pg_client:
                # 获取geo_poi_count并缓存（用于计算amap在neo4j的entity_id）
                nonlocal _amap_id_base_cache, _db_cache
                if _amap_id_base_cache is None:
                    with pg_client.conn.cursor() as cur:
                        cur.execute(
                            "SELECT COUNT(*) FROM geo_entity_names WHERE type = 'poi'"
                        )
                        geo_poi_count = cur.fetchone()[0]
                        _amap_id_base_cache = geo_poi_count
                        logger.info(
                            f"[Entity_Alignment] geo_entity_names poi数量: {geo_poi_count}, amap neo4j ID起始: poi_{_amap_id_base_cache + 1}"
                        )
                amap_entity_id_base = _amap_id_base_cache

                # 预加载数据库embedding（首次调用时加载，后续使用缓存）
                if _db_cache is None:
                    _db_cache = _load_db_embeddings(pg_client)
                geo_entities, amap_entities, geo_embeddings_np, amap_embeddings_np = (
                    _db_cache
                )

                # 获取抽取的实体（从joint_extraction_result或entities）
                joint_result = state.get("joint_extraction_result", {})
                entities_dict = state.get("entities", {})

                # 提取实体名称列表
                entity_names = []
                entity_types = {}
                if joint_result and "entities" in joint_result:
                    # 从联合抽取结果获取
                    for e in joint_result["entities"]:
                        name = e.get("name", "")
                        type_ = e.get("type", "")
                        if name and name not in entity_names:
                            entity_names.append(name)
                            entity_types[name] = type_
                else:
                    # 从传统entities字典获取
                    for type_, names in entities_dict.items():
                        for name in names:
                            if name and name not in entity_names:
                                entity_names.append(name)
                                entity_types[name] = type_

                if not entity_names:
                    logger.info(f"[Entity_Alignment] 无实体需要对齐")
                    return {
                        "entity_alignment_result": {
                            "alignment_items": [],
                            "aligned_entities": [],
                            "new_entities": [],
                            "skipped_entities": [],
                            "overall_alignment_rate": 0.0,
                            "alignment_confidence": "high",
                        },
                        "aligned_entity_ids": {},
                        "new_entity_names": [],
                        "current_step": StepEnum.DONE,
                    }

            # 加载嵌入模型
            model = _get_embedding_model()

            # 生成实体嵌入向量（批量）
            entity_embeddings = model.encode(
                entity_names, show_progress_bar=False, convert_to_numpy=True
            )

            # 对齐配置
            high_threshold = config.alignment_high_confidence_threshold
            low_threshold = config.alignment_similarity_threshold
            top_k = config.alignment_top_k
            use_llm = config.alignment_use_llm_decision

            # 批量相似度搜索
            geo_candidates_per_query = _batch_similarity_search(
                entity_embeddings, geo_embeddings_np, geo_entities, top_k
            )
            amap_candidates_per_query = _batch_similarity_search(
                entity_embeddings, amap_embeddings_np, amap_entities, top_k
            )

            alignment_items = []
            aligned_entities = []
            new_entities = []
            skipped_entities = []
            aligned_ids = {}
            medium_conf_items = []

            def _mark_aligned(
                entity_name: str, alignment_item_ref: Dict, candidate_ref: Dict
            ):
                alignment_item_ref["alignment_status"] = "aligned"
                alignment_item_ref["best_match"] = candidate_ref
                aligned_entities.append(
                    {
                        "name": entity_name,
                        "db_id": candidate_ref["db_entity_id"],
                        "db_name": candidate_ref["name"],
                        "similarity": candidate_ref["similarity"],
                        "source": candidate_ref["source"],
                    }
                )
                aligned_ids[entity_name] = candidate_ref["db_entity_id"]

            def _mark_new_entity(
                entity_name: str, alignment_item_ref: Dict, reason: str
            ):
                alignment_item_ref["alignment_status"] = "new_entity"
                alignment_item_ref["llm_decision"] = reason
                new_entities.append(entity_name)

            # 处理每个实体的候选结果
            for i, name in enumerate(entity_names):
                logger.debug(f"[Entity_Alignment] 对齐实体: {name}")

                # 合并geo和amap候选
                candidates = geo_candidates_per_query[i] + amap_candidates_per_query[i]

                # 为amap候选计算正确的neo4j entity_id
                for c in candidates:
                    if c.get("source") == "amap_poi_wgs84":
                        amap_id = c.get("id")
                        c["db_entity_id"] = f"poi_{amap_entity_id_base + amap_id}"
                        c["db_original_id"] = c.get("original_id")
                        # 清理不需要的字段
                        c.pop("id", None)
                        c.pop("original_id", None)
                    elif c.get("source") == "geo_entity_names":
                        c["db_entity_id"] = c.get("entity_id")
                        c.pop("entity_id", None)

                # 按相似度排序，取top_k
                candidates.sort(key=lambda x: x["similarity"], reverse=True)
                candidates = candidates[:top_k]

                # 判断对齐状态
                best_candidate = candidates[0] if candidates else None
                best_similarity = (
                    best_candidate.get("similarity", 0.0) if best_candidate else 0.0
                )

                alignment_item = {
                    "extracted_name": name,
                    "extracted_type": entity_types.get(name, ""),
                    "candidates": candidates,
                    "best_match": None,
                    "alignment_status": "pending",
                    "llm_decision": None,
                }

                # 高置信度：直接匹配
                if best_similarity >= high_threshold:
                    alignment_item["llm_decision"] = (
                        f"高置信度匹配({best_similarity:.3f}>=0.90)，直接确认"
                    )
                    _mark_aligned(name, alignment_item, best_candidate)

                    logger.debug(
                        f"[Entity_Alignment] {name} -> {best_candidate['name']} (高置信度, 来源: {best_candidate['source']})"
                    )

                # 低置信度：直接跳过（新实体）
                elif best_similarity < low_threshold:
                    _mark_new_entity(
                        name,
                        alignment_item,
                        f"相似度过低({best_similarity:.3f}<0.75)，判定为新实体",
                    )

                    logger.debug(f"[Entity_Alignment] {name} -> 新实体 (低置信度)")

                # 中置信度：收集后批量交给LLM判断（N次 -> 1次）
                elif use_llm and candidates:
                    alignment_item["llm_decision"] = "待批量LLM判断"
                    medium_conf_items.append(
                        {
                            "extracted_name": name,
                            "alignment_item": alignment_item,
                            "candidates": candidates,
                            "extracted_type": entity_types.get(name, ""),
                        }
                    )

                else:
                    # 不使用LLM判断，默认为新实体
                    _mark_new_entity(
                        name,
                        alignment_item,
                        f"中置信度({best_similarity:.3f})，未启用LLM判断",
                    )

                alignment_items.append(alignment_item)

            # 中置信度实体批量LLM对齐（N次 -> 1次）
            if medium_conf_items:
                try:
                    batch_items = []
                    for item in medium_conf_items:
                        candidates_for_format = []
                        for c in item["candidates"]:
                            c_formatted = c.copy()
                            c_formatted["db_name"] = c_formatted.get("name", "")
                            c_formatted["db_type"] = c_formatted.get("type", "")
                            candidates_for_format.append(c_formatted)

                        batch_items.append(
                            {
                                "extracted_name": item["extracted_name"],
                                "extracted_type": item["extracted_type"],
                                "candidates": candidates_for_format,
                            }
                        )

                    prompt_text = BATCH_ENTITY_ALIGNMENT_DECISION_PROMPT.invoke(
                        {
                            "raw_text": state.get("raw_text", ""),
                            "items_json": json.dumps(batch_items, ensure_ascii=False),
                        }
                    )
                    full_prompt = (
                        f"{prompt_text.messages[1].content}\n\n"
                        f"{batch_alignment_parser.get_format_instructions()}"
                    )
                    response = await llm.ainvoke(full_prompt)
                    parsed_result: BatchEntityAlignmentDecisionResult = (
                        safe_parse_json_with_quote_fix(batch_alignment_parser, response.content)
                    )

                    decision_map = {
                        d.extracted_name: d
                        for d in (parsed_result.decisions or [])
                    }
                    valid_statuses = {"aligned", "new_entity", "skip"}

                    for item in medium_conf_items:
                        entity_name = item["extracted_name"]
                        alignment_item = item["alignment_item"]
                        candidates = item["candidates"]
                        decision = decision_map.get(entity_name)

                        if not decision:
                            _mark_new_entity(
                                entity_name,
                                alignment_item,
                                "批量LLM未返回该实体决策，默认新实体",
                            )
                            continue

                        llm_status = (
                            decision.alignment_status
                            if decision.alignment_status in valid_statuses
                            else "new_entity"
                        )
                        best_index = int(decision.best_match_index or -1)
                        llm_decision = decision.llm_decision or "批量LLM判断"

                        alignment_item["alignment_status"] = llm_status
                        alignment_item["llm_decision"] = llm_decision

                        if llm_status == "aligned" and 0 <= best_index < len(candidates):
                            best_match = candidates[best_index]
                            _mark_aligned(entity_name, alignment_item, best_match)
                            logger.debug(
                                f"[Entity_Alignment] {entity_name} -> {best_match['name']} (批量LLM匹配, 来源: {best_match['source']})"
                            )
                        else:
                            new_entities.append(entity_name)
                            if llm_status == "skip":
                                skipped_entities.append(entity_name)
                                logger.debug(
                                    f"[Entity_Alignment] {entity_name} -> 跳过 (批量LLM)"
                                )
                            logger.debug(
                                f"[Entity_Alignment] {entity_name} -> 新实体 (批量LLM)"
                            )

                    logger.info(
                        f"[Entity_Alignment] 中置信度批量LLM判断完成: {len(medium_conf_items)} 个实体，调用 1 次"
                    )
                except Exception as e:
                    logger.warning(
                        f"[Entity_Alignment] 中置信度批量LLM判断失败: {e}，全部降级为新实体"
                    )
                    for item in medium_conf_items:
                        _mark_new_entity(
                            item["extracted_name"],
                            item["alignment_item"],
                            "批量LLM判断异常，默认新实体",
                        )

            # ===== 新实体创建逻辑（P12新增） =====
            # 将未对齐的新实体写入数据库和neo4j
            created_entity_ids = {}
            if new_entities:
                logger.info(
                    f"[Entity_Alignment] 开始创建 {len(new_entities)} 个新实体..."
                )

                new_pg_client = None
                neo4j_client = None
                try:
                    # 重新连接数据库
                    pg_config = settings.get_postgres_config()
                    new_pg_client = PostgresClient(**pg_config)

                    neo4j_config = settings.get_neo4j_config()
                    from kg.neo4j_client import Neo4jClient

                    neo4j_client = Neo4jClient(**neo4j_config)

                    # 批量生成新实体的embedding
                    new_embeddings = model.encode(
                        new_entities, show_progress_bar=False, convert_to_numpy=True
                    )

                    # 准备新实体数据
                    for i, name in enumerate(new_entities):
                        entity_type = entity_types.get(name, "poi")
                        # P15修复：确保 entity_type 是字符串（psycopg2 不接受 Enum 类型）
                        if hasattr(entity_type, "value"):
                            entity_type = entity_type.value
                        entity_type = str(entity_type) if entity_type else "poi"
                        entity_data = {
                            "name": name,
                            "type": entity_type,
                            "aliases": [],
                            "source": "xiaohongshu",
                        }

                        # 写入Postgres geo_entity_names表
                        pg_entity_id = new_pg_client.insert_new_geo_entity(
                            entity_data, new_embeddings[i].tolist()
                        )

                        if pg_entity_id:
                            # 写入Neo4j
                            neo4j_entity_id = neo4j_client.create_new_geo_entity(
                                entity_data
                            )
                            if neo4j_entity_id:
                                created_entity_ids[name] = neo4j_entity_id
                                logger.info(
                                    f"[Entity_Alignment] 新实体已创建: {name} -> {neo4j_entity_id}"
                                )
                            else:
                                # neo4j创建失败，使用postgres的entity_id
                                created_entity_ids[name] = pg_entity_id
                                logger.warning(
                                    f"[Entity_Alignment] Neo4j创建失败，使用PG ID: {name} -> {pg_entity_id}"
                                )
                        else:
                            logger.warning(f"[Entity_Alignment] 新实体创建失败: {name}")

                    logger.success(
                        f"[Entity_Alignment] 新实体创建完成: {len(created_entity_ids)}/{len(new_entities)}"
                    )

                except Exception as create_error:
                    logger.error(f"[Entity_Alignment] 新实体创建异常: {create_error}")
                    import traceback

                    traceback.print_exc()
                finally:
                    # 确保连接始终关闭（修复连接泄漏bug）
                    if new_pg_client is not None:
                        try:
                            new_pg_client.close()
                            logger.debug(
                                "[Entity_Alignment] 新实体创建PostgresClient连接已关闭"
                            )
                        except Exception as close_error:
                            logger.warning(
                                f"[Entity_Alignment] 关闭PostgresClient时出错: {close_error}"
                            )
                    if neo4j_client is not None:
                        try:
                            neo4j_client.close()
                            logger.debug("[Entity_Alignment] Neo4jClient连接已关闭")
                        except Exception as close_error:
                            logger.warning(
                                f"[Entity_Alignment] 关闭Neo4jClient时出错: {close_error}"
                            )

            # 计算整体对齐率
            total_entities = len(entity_names)
            aligned_count = len(aligned_entities)
            alignment_rate = (
                aligned_count / total_entities if total_entities > 0 else 0.0
            )

            # 统计各来源对齐数量
            geo_aligned = sum(
                1 for e in aligned_entities if e.get("source") == "geo_entity_names"
            )
            amap_aligned = sum(
                1 for e in aligned_entities if e.get("source") == "amap_poi_wgs84"
            )

            # 整体置信度判断
            if alignment_rate >= 0.8:
                overall_confidence = "high"
            elif alignment_rate >= 0.5:
                overall_confidence = "medium"
            else:
                overall_confidence = "low"

            logger.info(
                f"[Entity_Alignment] 完成: {aligned_count}/{total_entities} 已对齐, "
                f"{len(new_entities)} 新实体(创建{len(created_entity_ids)}个), 对齐率={alignment_rate:.1%}"
                f"(geo:{geo_aligned}, amap:{amap_aligned})"
            )

            # 输出结构化对齐结果，便于在日志中直接查看每个实体的最终去向
            alignment_log_items = []
            for item in alignment_items:
                best_match = item.get("best_match") or {}
                best_similarity = best_match.get("similarity")
                if best_similarity is not None:
                    try:
                        best_similarity = round(float(best_similarity), 4)
                    except (TypeError, ValueError):
                        best_similarity = str(best_similarity)

                alignment_log_items.append(
                    {
                        "extracted_name": item.get("extracted_name", ""),
                        "extracted_type": item.get("extracted_type", ""),
                        "status": item.get("alignment_status", "pending"),
                        "matched_name": best_match.get("name"),
                        "matched_db_id": best_match.get("db_entity_id"),
                        "similarity": best_similarity,
                        "source": best_match.get("source"),
                        "decision": item.get("llm_decision", ""),
                    }
                )

            logger.info(
                f"[Entity_Alignment] 对齐结果明细: "
                f"{json.dumps(alignment_log_items, ensure_ascii=False, default=str)}"
            )

            writer(
                {
                    "step": "entity_alignment",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "aligned_count": aligned_count,
                    "new_count": len(new_entities),
                    "created_count": len(created_entity_ids),
                    "alignment_rate": alignment_rate,
                    "geo_aligned": geo_aligned,
                    "amap_aligned": amap_aligned,
                    "confidence": overall_confidence,
                }
            )

            # 合并aligned_ids和created_entity_ids
            all_entity_ids = {**aligned_ids, **created_entity_ids}

            return {
                "entity_alignment_result": {
                    "alignment_items": alignment_items,
                    "aligned_entities": aligned_entities,
                    "new_entities": new_entities,
                    "created_entities": [
                        {"name": k, "entity_id": v}
                        for k, v in created_entity_ids.items()
                    ],
                    "skipped_entities": skipped_entities,
                    "overall_alignment_rate": alignment_rate,
                    "alignment_confidence": overall_confidence,
                    "geo_aligned_count": geo_aligned,
                    "amap_aligned_count": amap_aligned,
                    "created_count": len(created_entity_ids),
                },
                "aligned_entity_ids": all_entity_ids,  # 包含已对齐和新创建的实体ID
                "new_entity_names": [
                    n for n in new_entities if n not in created_entity_ids
                ],  # 仅保留创建失败的
                "current_step": StepEnum.DONE,
            }

        except Exception as e:
            logger.error(f"[Entity_Alignment] 处理失败: {e}")
            import traceback

            traceback.print_exc()
            writer(
                {
                    "step": "entity_alignment",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            return {
                "entity_alignment_result": {},
                "aligned_entity_ids": {},
                "new_entity_names": [],
                "error": str(e),
                "current_step": StepEnum.DONE,
            }

    return entity_alignment_node


# ===== P13新增：优化版节点函数（使用V3提示词） =====


def create_joint_ner_re_node_v3(llm: Any):
    """
    创建优化版联合抽取节点（使用RISEN框架提示词）

    改进点：
    1. Token减少约60%（表格化Schema）
    2. RISEN框架结构化（Role→Instructions→Steps→End→Narrowing）
    3. RCoT反向验证（减少幻觉）
    4. TIDD-EC约束规则集中化
    """
    parser = PydanticOutputParser(pydantic_object=JointExtractionResult)

    async def joint_ner_re_node_v3(state: CorpusState, writer: StreamWriter) -> Dict:
        """Joint NER + RE V3: RISEN框架优化版"""
        corpus_id = state["corpus_id"]
        logger.info(f"[Joint_NER_RE_V3] 处理语料: {corpus_id}")

        writer(
            {
                "step": "joint_ner_re",
                "corpus_id": corpus_id,
                "status": "started",
                "version": "v3",
                "message": "开始联合抽取（优化版）",
            }
        )

        try:
            text_for_processing = get_text_for_processing(state)

            # 获取 QA Scaffold 上下文
            qa_entity_hints = state.get("qa_entity_hints", [])
            qa_relation_hints = state.get("qa_relation_hints", [])
            qa_context_dependencies = state.get("qa_context_dependencies", [])
            mentor_guidance = state.get("mentor_guidance", {})

            # 使用动态组装函数（更灵活）
            from .prompts import assemble_optimized_joint_prompt

            full_prompt = assemble_optimized_joint_prompt(
                raw_text=text_for_processing,
                entity_hints=format_entity_hints(qa_entity_hints),
                relation_hints=format_relation_hints(qa_relation_hints),
                mentor_guidance=format_mentor_guidance(mentor_guidance),
            )

            # 添加格式化指令
            full_prompt_with_format = (
                f"{full_prompt}\n\n{parser.get_format_instructions()}"
            )

            response = await llm.ainvoke(full_prompt_with_format)
            result: JointExtractionResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # 转换为现有格式（v3.4扩展版：6种实体类型）
            entities_dict = {
                "道路": [],
                "POI": [],
                "建筑物": [],
                "街区": [],
                "功能": [],
                "事件": [],
            }
            for e in result.entities:
                entity_type = extract_enum_value(e.type)  # P15改进：使用工具函数
                if entity_type in entities_dict:
                    entities_dict[entity_type].append(e.name)

            triples_list = [
                {
                    "head": t.head,
                    "relation": extract_enum_value(t.relation),  # P15改进：使用工具函数
                    "tail": t.tail,
                    "evidence": t.evidence,
                    "confidence": extract_enum_value(
                        t.confidence
                    ),  # P15改进：使用工具函数
                    "attributes": t.attributes.model_dump(exclude_none=True)
                    if t.attributes
                    else {},
                }
                for t in result.triples
            ]

            logger.info(
                f"[Joint_NER_RE_V3] 完成: {len(result.entities)}个实体, "
                f"{len(result.triples)}个三元组, 置信度={result.overall_confidence}"
            )

            writer(
                {
                    "step": "joint_ner_re",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "version": "v3",
                    "entity_count": len(result.entities),
                    "triple_count": len(result.triples),
                    "confidence": result.overall_confidence,
                }
            )

            return {
                "entities": entities_dict,
                "triples": triples_list,
                "joint_extraction_result": result.model_dump(),
                "extraction_strategy": "joint_v3",
                "current_step": StepEnum.SELF_CHECK_JOINT,
            }

        except Exception as e:
            logger.error(f"[Joint_NER_RE_V3] 失败: {e}")
            writer(
                {
                    "step": "joint_ner_re",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "version": "v3",
                    "error": str(e),
                }
            )
            return {
                "entities": DEFAULT_ENTITY_DICT.copy(),  # v3.4修复：使用6种实体类型
                "triples": [],
                "error": str(e),
                "current_step": StepEnum.EVAL,
            }

    return joint_ner_re_node_v3


def create_filter_node_v3(llm: Any):
    """
    创建优化版筛选节点（使用APE框架）

    改进点：
    1. Token减少约67%（精简判断规则）
    2. APE框架（Action→Purpose→Expectation）
    3. 保守策略（无法确定时默认放行）
    """
    parser = PydanticOutputParser(pydantic_object=FilterResult)

    async def filter_node_v2(state: CorpusState, writer: StreamWriter) -> Dict:
        """Filter V2: APE框架优化版"""
        corpus_id = state["corpus_id"]
        logger.info(f"[Filter_V2] 筛选语料: {corpus_id}")

        writer(
            {
                "step": "filter",
                "corpus_id": corpus_id,
                "status": "started",
                "version": "v2",
                "message": "开始文本筛选（优化版）",
            }
        )

        try:
            from .prompts import FILTER_PROMPT_V2

            prompt_text = FILTER_PROMPT_V2.invoke({"raw_text": state["raw_text"]})
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: FilterResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            logger.info(
                f"[Filter_V2] 结果: is_valid={result.is_valid}, "
                f"confidence={result.confidence}, "
                f"region_hint={result.region_hint}"
            )

            writer(
                {
                    "step": "filter",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "version": "v2",
                    "is_valid": result.is_valid,
                    "confidence": result.confidence,
                    "skip_reason": result.skip_reason,
                }
            )

            next_step = StepEnum.NER if result.is_valid else StepEnum.DONE

            return {
                "filter_result": result.model_dump(),
                "current_step": next_step,
            }

        except Exception as e:
            logger.error(f"[Filter_V2] 失败: {e}")
            writer(
                {
                    "step": "filter",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "version": "v2",
                    "error": str(e),
                }
            )
            # 保守策略：筛选失败时默认继续处理
            return {
                "filter_result": {
                    "is_valid": True,
                    "confidence": "low",
                    "skip_reason": None,
                },
                "error": str(e),
                "current_step": StepEnum.NER,
            }

    return filter_node_v2


def create_self_check_joint_node_v3(llm: Any):
    """
    创建优化版联合抽取校验节点（Pre-Mortem + 四维度评分）

    改进点：
    1. Pre-Mortem预失败分析（提前识别风险）
    2. 四维度量化评分（完整性/准确性/真实性/证据性）
    3. RCoT反向验证步骤
    4. 可执行的改进动作列表
    """
    from .schemas import SelfCheckJointResultV2  # P12新增增强版模型

    parser = PydanticOutputParser(pydantic_object=SelfCheckJointResultV2)

    async def self_check_joint_node_v3(
        state: CorpusState, writer: StreamWriter
    ) -> Dict:
        """Self-Check Joint V3: Pre-Mortem + 四维度评分"""
        corpus_id = state["corpus_id"]
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)
        logger.info(
            f"[Self-Check-Joint_V3] 校验语料: {corpus_id}, 重试: {retry_count}/{max_retries}"
        )

        writer(
            {
                "step": "self_check_joint",
                "corpus_id": corpus_id,
                "status": "started",
                "version": "v3",
                "retry_count": retry_count,
            }
        )

        try:
            text = get_text_for_processing(state)
            reflection_history = state.get("reflection_history", [])

            from .prompts import SELF_CHECK_JOINT_PROMPT_V3

            prompt_text = SELF_CHECK_JOINT_PROMPT_V3.invoke(
                {
                    "raw_text": text,
                    "entities": format_joint_entities(
                        state.get("joint_extraction_result", {}).get("entities", [])
                    ),
                    "triples": format_joint_triples(state.get("triples", [])),
                    "semantic_summary": state.get("semantic_summary", ""),
                    "context_dependencies": format_context_dependencies(
                        state.get("qa_context_dependencies", [])
                    ),
                    "previous_reflection": format_reflection_history(
                        reflection_history
                    ),
                    "improvement_attempts": format_improvement_strategy(
                        state.get("improvement_strategy", {})
                    ),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckJointResultV2 = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            # 提取四维度评分
            dimension_scores = result.dimension_scores
            overall_confidence = result.overall_confidence

            logger.info(
                f"[Self-Check-Joint_V3] 完成: 四维度评分={dimension_scores}, "
                f"整体置信度={overall_confidence}, 重试建议={result.retry_suggested}"
            )

            writer(
                {
                    "step": "self_check_joint",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "version": "v3",
                    "dimension_scores": dimension_scores,
                    "confidence": overall_confidence,
                    "retry_suggested": result.retry_suggested,
                }
            )

            return {
                "self_check_joint_result": result.model_dump(),
                "reflection_text": result.reflection_text,
                "improvement_strategy": result.improvement_strategy,
                "reflection_history": reflection_history + [result.reflection_text],
                "retry_count": retry_count + (1 if result.retry_suggested else 0),
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "current_step": StepEnum.EVAL
                if not result.retry_suggested
                else StepEnum.JOINT_NER_RE,
            }

        except Exception as e:
            logger.error(f"[Self-Check-Joint_V3] 失败: {e}")
            writer(
                {
                    "step": "self_check_joint",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "version": "v3",
                    "error": str(e),
                }
            )
            return {
                "self_check_joint_result": {},
                "reflection_text": "",
                "error": str(e),
                "current_step": StepEnum.EVAL,
            }

    return self_check_joint_node_v3


def create_re_node_v3(llm: Any):
    """
    创建优化版关系抽取节点（表格化Schema）

    改进点：
    1. Token减少约60%（表格化关系定义）
    2. RCoT反向验证步骤
    3. TIDD-EC Do/Don't集中约束
    """
    parser = PydanticOutputParser(pydantic_object=RelationExtractionResult)

    async def re_node_v2(state: CorpusState, writer: StreamWriter) -> Dict:
        """RE V2: 表格化Schema优化版"""
        corpus_id = state["corpus_id"]
        logger.info(f"[RE_V2] 处理语料: {corpus_id}")

        writer(
            {
                "step": "re",
                "corpus_id": corpus_id,
                "status": "started",
                "version": "v2",
                "message": "开始关系抽取（优化版）",
            }
        )

        total_entities = sum(len(v) for v in state["entities"].values())
        if total_entities == 0:
            logger.debug(f"[RE_V2] 无实体，跳过")
            writer(
                {
                    "step": "re",
                    "corpus_id": corpus_id,
                    "status": "skipped",
                    "version": "v2",
                    "reason": "无实体",
                }
            )
            return {"current_step": StepEnum.EVAL, "triples": []}

        try:
            text_for_processing = get_text_for_processing(state)
            qa_relation_hints = state.get("qa_relation_hints", [])
            qa_context_dependencies = state.get("qa_context_dependencies", [])

            from .prompts import RE_PROMPT_V2

            prompt_text = RE_PROMPT_V2.invoke(
                {
                    "raw_text": text_for_processing,
                    "entities": format_entities(state["entities"]),
                    "relation_hints": format_relation_hints(qa_relation_hints),
                    "context_dependencies": format_context_dependencies(
                        qa_context_dependencies
                    ),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: RelationExtractionResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            triples = [
                {
                    "head": t.head,
                    "relation": extract_enum_value(t.relation),  # P15改进：使用工具函数
                    "tail": t.tail,
                    "evidence": t.evidence or "",
                    "attributes": t.attributes.model_dump(exclude_none=True)
                    if t.attributes
                    else {},
                }
                for t in result.triples
            ]

            logger.debug(f"[RE_V2] 结果: {len(triples)}个三元组")

            writer(
                {
                    "step": "re",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "version": "v2",
                    "triple_count": len(triples),
                }
            )

            return {
                "triples": triples,
                "current_step": StepEnum.EVAL,
            }

        except Exception as e:
            logger.error(f"[RE_V2] 失败: {e}")
            writer(
                {
                    "step": "re",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "version": "v2",
                    "error": str(e),
                }
            )
            return {
                "triples": [],
                "error": str(e),
                "current_step": StepEnum.DONE,
            }

    return re_node_v2


def create_label_node_v3(llm: Any):
    """
    创建优化版属性标注节点（表格化Schema）

    改进点：
    1. Token减少约60%
    2. 特征标签开放文本设计（v3.3）
    3. TIDD-EC约束规则
    """
    parser = PydanticOutputParser(pydantic_object=LabelResult)

    async def label_node_v2(state: CorpusState, writer: StreamWriter) -> Dict:
        """Label V2: 表格化Schema优化版"""
        corpus_id = state["corpus_id"]
        logger.info(f"[Label_V2] 处理语料: {corpus_id}")

        writer(
            {
                "step": "label",
                "corpus_id": corpus_id,
                "status": "started",
                "version": "v2",
                "message": "开始属性标注（优化版）",
            }
        )

        try:
            text_for_processing = get_text_for_processing(state)

            # 收集所有实体名
            all_entities = []
            for entity_type, names in state["entities"].items():
                for name in names:
                    all_entities.append(name)

            # 格式化关系列表
            relations_list = format_triples(state.get("triples", []))

            from .prompts import LABEL_PROMPT_V2

            prompt_text = LABEL_PROMPT_V2.invoke(
                {
                    "raw_text": text_for_processing,
                    "entities": all_entities,
                    "relations": relations_list,
                    "semantic_summary": state.get("semantic_summary", ""),
                    "entity_hints": format_entity_hints(
                        state.get("qa_entity_hints", [])
                    ),
                    "relation_hints": format_relation_hints(
                        state.get("qa_relation_hints", [])
                    ),
                }
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: LabelResult = safe_parse_json_with_quote_fix(
                parser, response.content
            )

            logger.debug(f"[Label_V2] 完成: {len(result.entities)}个实体属性")

            writer(
                {
                    "step": "label",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "version": "v2",
                    "entity_attr_count": len(result.entities),
                    "relation_attr_count": len(result.relations),
                }
            )

            # 转换属性字典格式（P20改进：添加空值检查）
            entity_attrs = {}
            for name, attrs in result.entities.items():
                if attrs is None:
                    entity_attrs[name] = {}
                elif hasattr(attrs, "model_dump"):
                    entity_attrs[name] = attrs.model_dump(exclude_none=True)
                else:
                    entity_attrs[name] = attrs

            relation_attrs = {}
            for key, attrs in result.relations.items():
                if attrs is None:
                    relation_attrs[key] = {}
                elif hasattr(attrs, "model_dump"):
                    relation_attrs[key] = attrs.model_dump(exclude_none=True)
                else:
                    relation_attrs[key] = attrs

            return {
                "entity_attrs": entity_attrs,
                "relation_attrs": relation_attrs,
                "current_step": StepEnum.DONE,
            }

        except Exception as e:
            logger.error(f"[Label_V2] 失败: {e}")
            writer(
                {
                    "step": "label",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "version": "v2",
                    "error": str(e),
                }
            )
            return {
                "entity_attrs": {},
                "relation_attrs": {},
                "error": str(e),
                "current_step": StepEnum.DONE,
            }

    return label_node_v2


# ===== 版本切换辅助函数 =====


def get_node_creators(prompt_version: str = "v2"):
    """
    根据提示词版本返回对应的节点创建函数

    Args:
        prompt_version: "v2"（原版）或 "v3"（优化版）

    Returns:
        Dict: 各节点的创建函数字典
    """
    if prompt_version == "v3":
        return {
            "filter": create_filter_node_v3,
            "joint_ner_re": create_joint_ner_re_node_v3,
            "self_check_joint": create_self_check_joint_node_v3,
            "re": create_re_node_v3,
            "label": create_label_node_v3,
        }
    else:
        # v2 默认版本
        return {
            "filter": create_filter_node,
            "joint_ner_re": create_joint_ner_re_node,
            "self_check_joint": create_self_check_joint_node,
            "re": create_re_node,
            "label": create_label_node,
        }


# ===== P15新增：批量预处理节点 =====


def create_batch_filter_node(llm: Any):
    """创建批量筛选节点 - 一次LLM调用处理多条语料的筛选判断"""
    from .schemas import BatchFilterResult
    from .prompts import BATCH_FILTER_PROMPT, format_batch_corpus

    parser = PydanticOutputParser(pydantic_object=BatchFilterResult)

    async def batch_filter_node(corpus_list: List[Dict], writer: StreamWriter) -> Dict:
        batch_size = len(corpus_list)
        logger.info(f"[Batch_Filter] 处理 {batch_size} 条语料")
        writer({"step": "batch_filter", "status": "started", "batch_size": batch_size})
        if batch_size == 0:
            return {"batch_results": {}, "processed_corpus": [], "skipped_corpus": []}
        try:
            corpus_list_str = format_batch_corpus(corpus_list)
            prompt_text = BATCH_FILTER_PROMPT.invoke(
                {"batch_size": batch_size, "corpus_list": corpus_list_str}
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result = safe_parse_json_with_quote_fix(parser, response.content)
            batch_results, processed_corpus, skipped_corpus = {}, [], []
            for r in result.results:
                # 使用正确的字段名：is_valid（语义相反）, is_non_wuhan_region
                should_skip = not r.is_valid  # is_valid=False 时应跳过
                batch_results[r.corpus_id] = {
                    "should_skip": should_skip,
                    "is_valid": r.is_valid,
                    "skip_reason": r.skip_reason,
                    "confidence": r.confidence,
                    "is_wuhan": not r.is_non_wuhan_region,  # 语义转换
                    "has_geo_info": r.has_geo_entity or r.has_spatial_relation,
                }
                target_list = skipped_corpus if should_skip else processed_corpus
                for c in corpus_list:
                    if c["id"] == r.corpus_id:
                        target_list.append(
                            {**c, "skip_reason": r.skip_reason} if should_skip else c
                        )
                        break
            logger.info(
                f"[Batch_Filter] 完成: {len(processed_corpus)}保留, {len(skipped_corpus)}跳过"
            )
            writer(
                {
                    "step": "batch_filter",
                    "status": "completed",
                    "processed_count": len(processed_corpus),
                    "skipped_count": len(skipped_corpus),
                }
            )
            return {
                "batch_results": batch_results,
                "processed_corpus": processed_corpus,
                "skipped_corpus": skipped_corpus,
                "batch_filter_result": result.model_dump(),
            }
        except Exception as e:
            logger.error(f"[Batch_Filter] 处理失败: {e}")
            return {
                "batch_results": {},
                "processed_corpus": corpus_list,
                "skipped_corpus": [],
                "error": str(e),
            }

    return batch_filter_node


def create_batch_normalize_node(llm: Any):
    """创建批量归一化节点 - 一次LLM调用处理多条语料的归一化"""
    from .schemas import BatchNormalizeResult
    from .prompts import BATCH_NORMALIZE_PROMPT, format_batch_corpus

    parser = PydanticOutputParser(pydantic_object=BatchNormalizeResult)

    async def batch_normalize_node(
        corpus_list: List[Dict], writer: StreamWriter
    ) -> Dict:
        batch_size = len(corpus_list)
        logger.info(f"[Batch_Normalize] 处理 {batch_size} 条语料")
        writer(
            {"step": "batch_normalize", "status": "started", "batch_size": batch_size}
        )
        if batch_size == 0:
            return {"batch_results": {}, "normalized_corpus": []}
        try:
            corpus_list_str = format_batch_corpus(corpus_list)
            prompt_text = BATCH_NORMALIZE_PROMPT.invoke(
                {"batch_size": batch_size, "corpus_list": corpus_list_str}
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result = safe_parse_json_with_quote_fix(parser, response.content)
            batch_results, normalized_corpus = {}, []
            for r in result.results:
                batch_results[r.corpus_id] = {
                    "normalized_text": r.normalized_text,
                    "aliases": r.aliases,
                    "confidence": r.confidence,
                }
                for c in corpus_list:
                    if c["id"] == r.corpus_id:
                        normalized_corpus.append(
                            {
                                **c,
                                "normalized_text": r.normalized_text,
                                "aliases": r.aliases,
                            }
                        )
                        break
            logger.info(f"[Batch_Normalize] 完成: {len(normalized_corpus)}条")
            writer(
                {
                    "step": "batch_normalize",
                    "status": "completed",
                    "batch_size": len(normalized_corpus),
                }
            )
            return {
                "batch_results": batch_results,
                "normalized_corpus": normalized_corpus,
                "batch_normalize_result": result.model_dump(),
            }
        except Exception as e:
            logger.error(f"[Batch_Normalize] 处理失败: {e}")
            return {
                "batch_results": {},
                "normalized_corpus": corpus_list,
                "error": str(e),
            }

    return batch_normalize_node


def create_batch_qa_scaffold_node(llm: Any):
    """创建批量QA脚手架节点 - 一次LLM调用处理多条语料的QA脚手架构建"""
    from .schemas import BatchQAScaffoldResult
    from .prompts import BATCH_QA_SCAFFOLD_PROMPT, format_batch_corpus

    parser = PydanticOutputParser(pydantic_object=BatchQAScaffoldResult)

    async def batch_qa_scaffold_node(
        corpus_list: List[Dict], writer: StreamWriter
    ) -> Dict:
        batch_size = len(corpus_list)
        logger.info(f"[Batch_QA_Scaffold] 处理 {batch_size} 条语料")
        writer(
            {"step": "batch_qa_scaffold", "status": "started", "batch_size": batch_size}
        )
        if batch_size == 0:
            return {"batch_results": {}, "qa_corpus": []}
        try:
            corpus_list_str = format_batch_corpus(corpus_list)
            prompt_text = BATCH_QA_SCAFFOLD_PROMPT.invoke(
                {"batch_size": batch_size, "corpus_list": corpus_list_str}
            )
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result = safe_parse_json_with_quote_fix(parser, response.content)
            batch_results, qa_corpus = {}, []
            for r in result.results:
                batch_results[r.corpus_id] = {
                    "qa_pairs": r.qa_pairs,
                    "entity_hints": r.entity_hints,
                    "relation_hints": r.relation_hints,
                    "context_dependencies": r.context_dependencies,
                    "confidence": r.overall_confidence,  # 使用正确的字段名
                }
                for c in corpus_list:
                    if c["id"] == r.corpus_id:
                        qa_corpus.append(
                            {
                                **c,
                                "qa_pairs": r.qa_pairs,
                                "entity_hints": r.entity_hints,
                                "relation_hints": r.relation_hints,
                                "context_dependencies": r.context_dependencies,
                            }
                        )
                        break
            logger.info(f"[Batch_QA_Scaffold] 完成: {len(qa_corpus)}条")
            writer(
                {
                    "step": "batch_qa_scaffold",
                    "status": "completed",
                    "batch_size": len(qa_corpus),
                }
            )
            return {
                "batch_results": batch_results,
                "qa_corpus": qa_corpus,
                "batch_qa_scaffold_result": result.model_dump(),
            }
        except Exception as e:
            logger.error(f"[Batch_QA_Scaffold] 处理失败: {e}")
            return {"batch_results": {}, "qa_corpus": corpus_list, "error": str(e)}

    return batch_qa_scaffold_node


async def process_batch_preprocessing(
    llm: Any,
    corpus_list: List[Dict],
    config: ExtractionConfig,
    enable_filter: bool = True,
    enable_normalize: bool = True,
    enable_qa_scaffold: bool = True,
) -> Dict:
    """批量前置节点处理：Filter → Normalize → QA_Scaffold → Self_Check_QA（可选），一次LLM调用处理多条语料

    P17改进：添加 QA_Scaffold 校验节点
    """
    logger.info(f"[Batch_Preprocessing] 开始批量预处理 {len(corpus_list)} 条语料")

    def dummy_writer(e):
        pass

    preprocessing_results, current_corpus, skipped_corpus, fallback_corpus = (
        {},
        corpus_list,
        [],
        [],
    )

    # 构建语料文本映射（用于校验节点）
    corpus_texts = {corpus["id"]: corpus["text"] for corpus in corpus_list}

    if enable_filter:
        filter_node = create_batch_filter_node(llm)
        filter_result = await filter_node(current_corpus, dummy_writer)
        preprocessing_results["filter"] = filter_result.get("batch_filter_result", {})
        current_corpus = filter_result.get("processed_corpus", current_corpus)
        skipped_corpus = filter_result.get("skipped_corpus", [])
        if filter_result.get("error"):
            fallback_corpus.extend(
                [{**c, "error": filter_result.get("error")} for c in current_corpus]
            )
            current_corpus = []

    if enable_normalize and current_corpus:
        normalize_node = create_batch_normalize_node(llm)
        normalize_result = await normalize_node(current_corpus, dummy_writer)
        preprocessing_results["normalize"] = normalize_result.get(
            "batch_normalize_result", {}
        )
        current_corpus = normalize_result.get("normalized_corpus", current_corpus)
        if normalize_result.get("error"):
            fallback_corpus.extend(
                [{**c, "error": normalize_result.get("error")} for c in current_corpus]
            )
            current_corpus = []

    if enable_qa_scaffold and current_corpus:
        qa_node = create_batch_qa_scaffold_node(llm)
        qa_retry_attempts = max(
            int(getattr(config, "batch_qa_scaffold_retry_attempts", 1) or 0), 0
        )
        qa_result = None
        last_qa_error = None

        for attempt in range(qa_retry_attempts + 1):
            qa_result = await qa_node(current_corpus, dummy_writer)
            if not qa_result.get("error"):
                if attempt > 0:
                    logger.info(
                        f"[Batch_Preprocessing] Batch_QA_Scaffold 批量重试成功: 第{attempt + 1}次尝试"
                    )
                break

            last_qa_error = qa_result.get("error")
            if attempt < qa_retry_attempts:
                logger.warning(
                    f"[Batch_Preprocessing] Batch_QA_Scaffold 批量失败，准备重试 "
                    f"{attempt + 1}/{qa_retry_attempts}: {last_qa_error}"
                )
            else:
                logger.error(
                    f"[Batch_Preprocessing] Batch_QA_Scaffold 批量重试耗尽({qa_retry_attempts + 1}次): "
                    f"{last_qa_error}"
                )

        qa_result = qa_result or {"batch_results": {}, "qa_corpus": current_corpus}
        # P17修复：batch_results 是 {corpus_id: {qa_pairs, ...}} 格式，用于后续校验
        preprocessing_results["qa_scaffold"] = qa_result.get("batch_results", {})
        # batch_qa_scaffold_result 是原始Pydantic model_dump，用于日志/调试
        preprocessing_results["qa_scaffold_raw"] = qa_result.get(
            "batch_qa_scaffold_result", {}
        )
        current_corpus = qa_result.get("qa_corpus", current_corpus)
        if qa_result.get("error"):
            fallback_corpus.extend(
                [{**c, "error": qa_result.get("error")} for c in current_corpus]
            )
            current_corpus = []

        # P17新增：QA_Scaffold 校验（仅在 enable_full_self_check 时）
        if (
            config.enable_full_self_check
            and current_corpus
            and preprocessing_results.get("qa_scaffold")
        ):
            qa_self_check_node = create_batch_self_check_qa_node(llm)
            qa_check_result = await qa_self_check_node(
                preprocessing_results["qa_scaffold"], corpus_texts, dummy_writer
            )
            preprocessing_results["qa_self_check"] = qa_check_result

            # 处理校验结果：校验失败的语料加入 fallback
            for r in qa_check_result.get("rejected_results", []):
                corpus_id = r.get("corpus_id")
                if config.batch_llm_fallback:
                    for corpus in current_corpus:
                        if corpus["id"] == corpus_id:
                            fallback_corpus.append(
                                {
                                    **corpus,
                                    "error": r.get("reason")
                                    or "batch_self_check_qa_rejected",
                                }
                            )
                            break

            # 只保留通过校验的语料
            verified_ids = {
                r.get("corpus_id") for r in qa_check_result.get("verified_results", [])
            }
            current_corpus = [c for c in current_corpus if c["id"] in verified_ids]

    logger.info(
        f"[Batch_Preprocessing] 完成: {len(current_corpus)}成功, {len(skipped_corpus)}跳过, {len(fallback_corpus)}fallback"
    )
    return {
        "processed_corpus": current_corpus,
        "skipped_corpus": skipped_corpus,
        "fallback_corpus": fallback_corpus,
        "preprocessing_results": preprocessing_results,
    }
