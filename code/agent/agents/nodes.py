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
from typing import Dict, List, Any, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langgraph.types import StreamWriter

from loguru import logger

from .state import CorpusState, KGState, PhaseEnum, StepEnum, DEFAULT_MAX_RETRIES, RELATION_TYPES
from .config import ExtractionConfig


# ===== P6改进：辅助函数 - 获取处理文本 =====

def _get_text_for_processing(state: CorpusState) -> str:
    """
    获取用于处理的文本，优先使用归一化文本

    Args:
        state: 当前语料状态

    Returns:
        用于 NER/RE/Eval 等后续处理的文本
        - 如果有归一化文本（normalized_text），优先使用
        - 否则使用原始文本（raw_text）
    """
    normalized = state.get("normalized_text", "")
    if normalized and normalized.strip():
        return normalized
    return state.get("raw_text", "")
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
    JointEntity, JointTriple, JointExtractionResult,
    SelfCheckJointResult, SelfCheckQAResult, SelfCheckEvalResult, SelfCheckLabelResult,
    # P12新增：Self-Check增强版模型
    SelfCheckJointResultV2,
    # v3.4新增：关系属性映射常量
    RELATION_ATTRS_MAP,
)
from .prompts import (
    FILTER_PROMPT,  # P5新增
    NORMALIZE_PROMPT,  # P6新增
    QA_SCAFFOLD_PROMPT,  # P8新增
    NER_PROMPT, RE_PROMPT, EVAL_PROMPT_1, EVAL_PROMPT_2,
    EVAL_PROMPT_SIMPLIFIED, LABEL_PROMPT,
    SELF_CHECK_NER_PROMPT, SELF_CHECK_RE_PROMPT,
    format_entities, format_triples, format_verified_entities, format_retry_hint,
    format_entity_hints, format_relation_hints, format_context_dependencies,  # P8新增
    # P9新增：联合抽取和所有Self-Check提示词
    JOINT_NER_RE_PROMPT, JOINT_NER_RE_PROMPT_V2,  # P12新增：改进版提示词
    SELF_CHECK_JOINT_PROMPT, SELF_CHECK_JOINT_PROMPT_V2,  # P12新增：改进版提示词
    SELF_CHECK_QA_PROMPT, SELF_CHECK_EVAL_PROMPT, SELF_CHECK_LABEL_PROMPT,
    format_joint_entities, format_joint_triples, format_qa_pairs_for_check, format_eval_scores_for_check,
    format_reflection_history,
    # P10新增：QA导师提示词和格式化函数
    QA_MENTOR_PROMPT, QA_APPROVAL_PROMPT, REVISION_JOINT_PROMPT,
    format_mentor_guidance, format_feedbacks_for_revision, format_feedback_summary, format_joint_for_approval,
    format_eval_for_approval, format_label_for_approval, format_revision_feedbacks, format_reflection_for_approval,
    # P14新增：导师查询提示词
    MENTOR_QUERY_PROMPT,
    # P11新增：实体对齐提示词和格式化函数
    ENTITY_ALIGNMENT_PROMPT, format_alignment_candidates, format_alignment_result_for_output,
    # P12新增：四维度评分格式化函数
    format_dimension_scores, format_improvement_strategy,
    # P13新增：优化版提示词（RISEN/CARE/TIDD-EC框架）
    JOINT_NER_RE_PROMPT_V3, FILTER_PROMPT_V2, RE_PROMPT_V2, LABEL_PROMPT_V2,
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
        corpus_id = state['corpus_id']
        logger.info(f"[Filter] 筛选语料: {corpus_id}")

        # 发送进度事件
        writer({
            "step": "filter",
            "corpus_id": corpus_id,
            "status": "started",
            "message": "开始文本筛选"
        })

        try:
            # 调用 LLM 进行筛选判断（使用 OutputParser）
            prompt_text = FILTER_PROMPT.invoke({"raw_text": state["raw_text"]})
            # 添加格式化指令
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: FilterResult = parser.parse(response.content)

            logger.info(
                f"[Filter] 结果: is_valid={result.is_valid}, "
                f"confidence={result.confidence}, "
                f"skip_reason={result.skip_reason}, "
                f"is_non_wuhan_region={result.is_non_wuhan_region}, "
                f"region_hint={result.region_hint}"
            )

            # 发送完成事件（包含新字段）
            writer({
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
                "region_hint": result.region_hint
            })

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
            writer({
                "step": "filter",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
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
        corpus_id = state['corpus_id']
        logger.info(f"[Normalize] 归一化语料: {corpus_id}")

        # 发送进度事件
        writer({
            "step": "normalize",
            "corpus_id": corpus_id,
            "status": "started",
            "message": "开始文本归一化"
        })

        try:
            # 使用原始文本进行归一化（使用 OutputParser）
            raw_text = state["raw_text"]
            prompt_text = NORMALIZE_PROMPT.invoke({"raw_text": raw_text})
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: NormalizeResult = parser.parse(response.content)

            logger.info(
                f"[Normalize] 结果: has_changes={result.has_changes}, "
                f"confidence={result.confidence}, "
                f"normalizations={len(result.normalizations)}条"
            )

            # 发送完成事件
            writer({
                "step": "normalize",
                "corpus_id": corpus_id,
                "status": "completed",
                "normalized_text": result.normalized_text,
                "normalizations_count": len(result.normalizations),
                "confidence": result.confidence,
                "has_changes": result.has_changes
            })

            # 输出归一化结果
            # normalized_text 供后续 NER/RE 节点使用
            return {
                "normalize_result": result.model_dump(),
                "normalized_text": result.normalized_text,
                "current_step": StepEnum.NER,
            }

        except Exception as e:
            logger.error(f"[Normalize] 失败: {e}")
            writer({
                "step": "normalize",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
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
        corpus_id = state['corpus_id']

        # 使用归一化后的文本（如果有的话）
        text_for_processing = _get_text_for_processing(state)

        logger.info(f"[QA_Scaffold] 处理语料: {corpus_id}")

        # 发送进度事件
        writer({
            "step": "qa_scaffold",
            "corpus_id": corpus_id,
            "status": "started",
            "message": "开始构建语义脚手架"
        })

        try:
            # 调用LLM生成QA脚手架
            prompt_text = QA_SCAFFOLD_PROMPT.invoke({"normalized_text": text_for_processing})
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: QAScaffoldResult = parser.parse(response.content)

            logger.info(
                f"[QA_Scaffold] 完成: {len(result.qa_pairs)} 个问答对, "
                f"{len(result.entity_hints)} 个实体提示, "
                f"置信度={result.overall_confidence}"
            )

            # 发送完成事件
            writer({
                "step": "qa_scaffold",
                "corpus_id": corpus_id,
                "status": "completed",
                "qa_count": len(result.qa_pairs),
                "entity_hints": result.entity_hints,
                "relation_hints": result.relation_hints,
                "confidence": result.overall_confidence,
                "semantic_summary": result.semantic_summary
            })

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
            writer({
                "step": "qa_scaffold",
                "corpus_id": corpus_id,
                "status": "failed",
                "error": str(e)
            })
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
        corpus_id = state['corpus_id']
        logger.info(f"[NER] 处理语料: {corpus_id}")

        # P3改进：发送进度事件
        writer({
            "step": "ner",
            "corpus_id": corpus_id,
            "status": "started",
            "message": "开始命名实体识别"
        })

        try:
            # P6改进：优先使用归一化文本
            text_for_processing = _get_text_for_processing(state)
            logger.debug(f"[NER] 使用文本: {text_for_processing[:50]}...")

            # P8改进：获取 QA Scaffold 上下文
            qa_entity_hints = state.get("qa_entity_hints", [])
            qa_context_dependencies = state.get("qa_context_dependencies", [])

            # 使用 OutputParser 进行结构化输出
            prompt_text = NER_PROMPT.invoke({
                "raw_text": text_for_processing,
                "entity_hints": format_entity_hints(qa_entity_hints),
                "context_dependencies": format_context_dependencies(qa_context_dependencies),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: EntityRecognitionResult = parser.parse(response.content)

            entity_count = len(result.道路) + len(result.POI) + len(result.建筑物) + len(result.街区)
            logger.debug(f"[NER] 结果: {result}")

            # P3改进：发送完成事件
            writer({
                "step": "ner",
                "corpus_id": corpus_id,
                "status": "completed",
                "entity_count": entity_count,
                "entities": {
                    "道路": result.道路,
                    "POI": result.POI,
                    "建筑物": result.建筑物,
                    "街区": result.街区,
                }
            })

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
            writer({
                "step": "ner",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
            return {
                "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
                "error": str(e),
                "current_step": StepEnum.DONE,  # 出错时直接结束
            }

    return ner_node


def create_re_node(llm: Any):
    """创建RE节点（v2.2改进：支持attributes）"""
    parser = PydanticOutputParser(pydantic_object=RelationExtractionResult)

    async def re_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Step 2: 关系抽取"""
        corpus_id = state['corpus_id']
        logger.info(f"[RE] 处理语料: {corpus_id}")

        # P3改进：发送进度事件
        writer({
            "step": "re",
            "corpus_id": corpus_id,
            "status": "started",
            "message": "开始关系抽取"
        })

        # 检查是否有实体
        total_entities = sum(len(v) for v in state["entities"].values())
        if total_entities == 0:
            logger.debug(f"[RE] 无实体，跳过")
            writer({
                "step": "re",
                "corpus_id": corpus_id,
                "status": "skipped",
                "reason": "无实体"
            })
            return {"current_step": StepEnum.EVAL, "triples": []}

        try:
            # P6改进：优先使用归一化文本
            text_for_processing = _get_text_for_processing(state)

            # P8改进：获取 QA Scaffold 上下文
            qa_relation_hints = state.get("qa_relation_hints", [])
            qa_context_dependencies = state.get("qa_context_dependencies", [])

            # 使用 OutputParser 进行结构化输出
            prompt_text = RE_PROMPT.invoke({
                "raw_text": text_for_processing,
                "entities": format_entities(state["entities"]),
                "relation_hints": format_relation_hints(qa_relation_hints),
                "context_dependencies": format_context_dependencies(qa_context_dependencies),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: RelationExtractionResult = parser.parse(response.content)

            # v2.2改进：提取三元组及属性（Enum转字符串+强类型属性转字典）
            triples = [
                {
                    "head": t.head,
                    "relation": t.relation.value if hasattr(t.relation, 'value') else t.relation,  # Enum转字符串
                    "tail": t.tail,
                    "evidence": t.evidence or "",
                    "attributes": t.attributes.model_dump(exclude_none=True) if t.attributes else {},  # TripleAttributes转字典
                }
                for t in result.triples
            ]

            logger.debug(f"[RE] 结果: {len(triples)}个三元组")

            # P3改进：发送完成事件
            writer({
                "step": "re",
                "corpus_id": corpus_id,
                "status": "completed",
                "triple_count": len(triples),
                "triples": triples
            })

            return {"triples": triples, "current_step": StepEnum.EVAL}
        except Exception as e:
            logger.error(f"[RE] 失败: {e}")
            # P3改进：发送错误事件
            writer({
                "step": "re",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
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
            prompt_text = EVAL_PROMPT_1.invoke({
                "triples": state["triples"],
                "raw_text": state["raw_text"],
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: EvalResultFirst = parser.parse(response.content)

            scores = [
                {
                    "triple": {
                        "head": s.triple.head,
                        "relation": s.triple.relation,
                        "tail": s.triple.tail,
                    },
                    "SEM": s.SEM,
                    "FAC": s.FAC,
                    "CON": s.CON,
                }
                for s in result.scores
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
            prompt_text = EVAL_PROMPT_2.invoke({
                "previous_scores": state["eval_scores"],
                "raw_text": state["raw_text"],
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: EvalResultSecond = parser.parse(response.content)

            # 更新评分
            final_scores = [
                {
                    "triple": {
                        "head": s.triple.head,
                        "relation": s.triple.relation,
                        "tail": s.triple.tail,
                    },
                    "SEM": s.SEM,
                    "FAC": s.FAC,
                    "CON": s.CON,
                }
                for s in result.final_scores
            ] if result.final_scores else state["eval_scores"]

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
                corrected_triples, correction_mapping = apply_corrections(state["triples"], result.corrections)
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
                avg_triple_score = (triple["sem_score"] + triple["fac_score"] + triple["con_score"]) / 3
                triple["passed_eval"] = avg_triple_score >= passed_threshold if avg_triple_score > 0 else False

            # 计算平均评分判断是否通过
            avg_score = sum(
                s["SEM"] + s["FAC"] + s["CON"]
                for s in final_scores
            ) / (len(final_scores) * 3) if final_scores else 0

            logger.debug(f"[Eval2] 平均评分: {avg_score}, 需修正: {result.need_correction}")

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
                "current_step": StepEnum.LABEL,
            }

    return eval_2_node


# P2改进：规则校验函数（不依赖 LLM）
def rule_based_validation(triples: List[Dict], entities: Dict[str, List[str]]) -> List[Dict]:
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
        head_valid = any(head in e or e in head for e in all_entities) if head else False
        tail_valid = any(tail in e or e in tail for e in all_entities) if tail else False

        # 规则2：关系类型有效性
        relation_valid = relation in VALID_RELATIONS

        # 规则3：关系逻辑检查（可选，可扩展）
        # 例如："连接"关系通常连接两个道路或地点
        logic_valid = True  # 默认通过，后续可扩展

        # 记录校验结果
        triple["_rule_valid"] = head_valid and tail_valid and relation_valid and logic_valid
        triple["_rule_issues"] = []
        if not head_valid:
            triple["_rule_issues"].append(f"头实体'{head}'未在NER结果中")
        if not tail_valid:
            triple["_rule_issues"].append(f"尾实体'{tail}'未在NER结果中")
        if not relation_valid:
            triple["_rule_issues"].append(f"关系'{relation}'不在预定义类型中")

        validated_triples.append(triple)

    return validated_triples


def create_eval_simplified_node(llm: Any, eval_threshold: float = 3.5, enable_query: bool = False):
    """
    P2改进：创建简化的单次评估节点
    P3改进：支持 StreamWriter 流式输出
    P14改进：支持向导师发起查询（enable_query=True时）

    合原来两轮评估为单次评估 + 规则校验，减少 LLM 调用成本
    """
    parser = PydanticOutputParser(pydantic_object=EvalResultSimplified)

    async def eval_simplified_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Step 3: 简化评估（单次LLM调用 + 规则校验）"""
        corpus_id = state['corpus_id']
        logger.info(f"[Eval] 处理语料: {corpus_id}")

        # P3改进：发送进度事件
        writer({
            "step": "eval",
            "corpus_id": corpus_id,
            "status": "started",
            "message": "开始三元组评估"
        })

        if not state["triples"]:
            logger.debug(f"[Eval] 无三元组，跳过评估")
            writer({
                "step": "eval",
                "corpus_id": corpus_id,
                "status": "skipped",
                "reason": "无三元组"
            })
            return {
                "eval_scores": [],
                "corrected_triples": [],
                "eval_passed": True,
                "needs_mentor_help": False,  # P14新增
                "current_step": StepEnum.LABEL,
            }

        try:
            # P6改进：优先使用归一化文本
            text_for_processing = _get_text_for_processing(state)

            # P8改进：获取 QA Scaffold 上下文
            semantic_summary = state.get("semantic_summary", "")
            qa_context_dependencies = state.get("qa_context_dependencies", [])

            # P14新增：如果有导师回答，更新语义摘要
            mentor_response = state.get("mentor_response")
            if mentor_response:
                integrated_summary = mentor_response.get("clarification", "")
                if integrated_summary:
                    semantic_summary = f"{semantic_summary}\n导师澄清: {integrated_summary}"
                logger.info(f"[Eval] 使用导师更新的语义理解")

            # 格式化三元组用于提示词
            triples_text = format_triples(state["triples"])

            # 使用 OutputParser 进行结构化输出
            prompt_text = EVAL_PROMPT_SIMPLIFIED.invoke({
                "triples": triples_text,
                "raw_text": text_for_processing,
                "semantic_summary": semantic_summary or "(无语义摘要)",
                "context_dependencies": format_context_dependencies(qa_context_dependencies),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: EvalResultSimplified = parser.parse(response.content)

            # 处理评分
            scores = [
                {
                    "triple": {
                        "head": s.triple.head,
                        "relation": s.triple.relation,
                        "tail": s.triple.tail,
                    },
                    "SEM": s.SEM,
                    "FAC": s.FAC,
                    "CON": s.CON,
                }
                for s in result.scores
            ]

            # 应用 LLM 修正（如果有）
            if result.need_correction and result.corrections:
                corrected_triples = apply_llm_corrections(state["triples"], result.corrections)
            else:
                corrected_triples = list(state["triples"])

            # P2改进：规则校验（不依赖 LLM）
            corrected_triples = rule_based_validation(corrected_triples, state["entities"])

            # 将评分写入三元组
            score_map = {}
            for s in scores:
                key = (s["triple"]["head"], s["triple"]["relation"], s["triple"]["tail"])
                score_map[key] = s

            for triple in corrected_triples:
                key = (triple["head"], triple["relation"], triple["tail"])
                score_data = score_map.get(key, {})
                triple["sem_score"] = score_data.get("SEM", 3)
                triple["fac_score"] = score_data.get("FAC", 3)
                triple["con_score"] = score_data.get("CON", 3)

                # 综合评分和规则校验结果
                avg_score = (triple["sem_score"] + triple["fac_score"] + triple["con_score"]) / 3
                rule_valid = triple.get("_rule_valid", True)
                triple["passed_eval"] = avg_score >= eval_threshold and rule_valid

            # 计算整体通过率
            passed_count = sum(1 for t in corrected_triples if t.get("passed_eval", False))
            overall_passed = passed_count > 0 if corrected_triples else True

            logger.debug(f"[Eval] 评分完成: {len(scores)}个三元组, {passed_count}个通过, 规则校验已应用")

            # P3改进：发送完成事件
            writer({
                "step": "eval",
                "corpus_id": corpus_id,
                "status": "completed",
                "triple_count": len(corrected_triples),
                "passed_count": passed_count,
                "eval_passed": overall_passed
            })

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
                    if confusion:
                        logger.info(f"[Eval] 检测到困惑，请求导师帮助: {confusion['query_type']}")
                        writer({
                            "step": "eval",
                            "corpus_id": corpus_id,
                            "status": "needs_mentor_help",
                            "query_type": confusion["query_type"],
                            "query_content": confusion["query_content"],
                        })

                        return {
                            "eval_scores": scores,
                            "corrected_triples": corrected_triples,
                            "eval_passed": overall_passed,
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
            writer({
                "step": "eval",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
            # 失败时仍应用规则校验
            fallback_triples = rule_based_validation(state["triples"], state["entities"])
            return {
                "eval_scores": [],
                "corrected_triples": fallback_triples,
                "eval_passed": False,
                "error": str(e),
                "needs_mentor_help": False,
                "current_step": StepEnum.LABEL,
            }

    return eval_simplified_node


def apply_llm_corrections(original_triples: List[Dict], corrections: List[Any]) -> List[Dict]:
    """应用 LLM 返回的修正"""
    corrected = list(original_triples)

    for correction in corrections:
        original_key = (correction.original.head, correction.original.relation, correction.original.tail)
        new_triple = {
            "head": correction.corrected.head,
            "relation": correction.corrected.relation,
            "tail": correction.corrected.tail,
            "evidence": "",
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
        corpus_id = state['corpus_id']
        logger.info(f"[Label] 处理语料: {corpus_id}")

        # P3改进：发送进度事件
        writer({
            "step": "label",
            "corpus_id": corpus_id,
            "status": "started",
            "message": "开始属性标注"
        })

        # 收集所有实体名称
        all_entities = []
        for entity_list in state["entities"].values():
            all_entities.extend(entity_list)

        if not all_entities:
            logger.debug(f"[Label] 无实体，跳过")
            writer({
                "step": "label",
                "corpus_id": corpus_id,
                "status": "skipped",
                "reason": "无实体"
            })
            return {
                "needs_mentor_help": False,  # P14新增
                "current_step": StepEnum.DONE
            }

        try:
            # v2.2改进：获取原始文本用于提取情感标签、体验评价
            text_for_processing = _get_text_for_processing(state)

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
            prompt_text = LABEL_PROMPT.invoke({
                "entities": all_entities,
                "relations": format_triples(state["corrected_triples"]),
                "raw_text": text_for_processing,
                "semantic_summary": semantic_summary or "(无语义摘要)",
                "entity_hints": format_entity_hints(qa_entity_hints),
                "relation_hints": format_relation_hints(qa_relation_hints),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: LabelResult = parser.parse(response.content)

            # v3.2精简版：仅提取schema定义的属性
            entity_attrs = {}
            for name, attrs in result.entities.items():
                entity_attrs[name] = {
                    "类别": attrs.类别,
                    "细分": attrs.细分,
                    "特征标签": attrs.特征标签 or [],
                    "推荐指数": attrs.推荐指数,
                    "情感倾向": attrs.情感倾向,
                }

            # v3.4精简版：根据关系类型选择性提取属性
            # 使用 RELATION_ATTRS_MAP 动态过滤，避免硬编码
            relation_attrs = {}
            for key, attrs in result.relations.items():
                normalized_key = normalize_relation_key(key)
                if normalized_key:
                    # 提取关系类型
                    relation_type = normalized_key.split(", ")[1].strip() if ", " in normalized_key else ""

                    # 动态过滤：根据RELATION_ATTRS_MAP获取允许的属性列表
                    allowed_attrs = RELATION_ATTRS_MAP.get(relation_type, [])
                    # 使用model_dump获取所有非空属性
                    attrs_dict = attrs.model_dump(exclude_none=True)
                    # 仅保留允许的属性
                    filtered_attrs = {k: v for k, v in attrs_dict.items() if k in allowed_attrs}

                    relation_attrs[normalized_key] = filtered_attrs
                else:
                    # 无法解析时保留原始 key
                    relation_attrs[key] = {}

            logger.debug(f"[Label] 完成: {len(entity_attrs)}个实体, {len(relation_attrs)}个关系")

            # P3改进：发送完成事件
            writer({
                "step": "label",
                "corpus_id": corpus_id,
                "status": "completed",
                "entity_count": len(entity_attrs),
                "relation_count": len(relation_attrs)
            })

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
                    if confusion:
                        logger.info(f"[Label] 检测到困惑，请求导师帮助: {confusion['query_type']}")
                        writer({
                            "step": "label",
                            "corpus_id": corpus_id,
                            "status": "needs_mentor_help",
                            "query_type": confusion["query_type"],
                            "query_content": confusion["query_content"],
                        })

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
            writer({
                "step": "label",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
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
    match = re.match(r'^<\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*>$', key)
    if match:
        head, relation, tail = match.groups()
        return f"<{head.strip()}, {relation.strip()}, {tail.strip()}>"

    # 格式2: A, 关系, B (没有尖括号)
    parts = [p.strip() for p in key.split(',')]
    if len(parts) == 3:
        return f"<{parts[0]}, {parts[1]}, {parts[2]}>"

    # 无法解析，返回 None
    return None


def apply_corrections(original_triples: List[Dict], corrections: List[Any]) -> tuple:
    """
    应用三元组修正

    Returns:
        (corrected_triples, correction_mapping)
        correction_mapping: {new_triple_key: original_triple_key} 用于继承评分
    """
    corrected = list(original_triples)
    correction_mapping = {}

    for correction in corrections:
        original = correction.original
        original_key = (original.head, original.relation, original.tail)
        new_triple = {
            "head": correction.corrected.head,
            "relation": correction.corrected.relation,
            "tail": correction.corrected.tail,
            "evidence": "",
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

        logger.info(f"[Coordinator] 创建 {worker_count} 个Worker, 每个处理 {actual_corpus_per_worker} 条语料, 共 {corpus_count} 条")

        return {
            "worker_count": worker_count,
            "corpus_partitions": partitions,
            "active_workers": active_workers,
            "current_phase": PhaseEnum.MAP,
        }

    return coordinator_node


def create_aggregator_node(similarity_threshold: float = 0.85):
    """创建聚合器节点"""

    def aggregator_node(state: KGState) -> Dict:
        """REDUCE阶段 - 合并Worker结果"""
        logger.info("[Aggregator] 开始聚合Worker结果")

        all_entities = []
        all_triples = []

        # 收集所有Worker的结果
        for worker_result in state["worker_results"]:
            for corpus_state in worker_result["results"]:
                # 跳过有错误的结果
                if corpus_state.get("error"):
                    logger.warning(f"[Aggregator] 跳过错误语料: {corpus_state.get('corpus_id')}")
                    continue

                corpus_id = corpus_state.get("corpus_id", "unknown")

                # 收集实体
                entities = corpus_state.get("entities", {})
                for entity_type, names in entities.items():
                    for name in names:
                        all_entities.append({
                            "name": name,
                            "type": entity_type,
                            "corpus_id": corpus_id,
                            "attrs": corpus_state.get("entity_attrs", {}).get(name, {}),
                        })

                # 收集三元组，并写入relation_attrs
                relation_attrs = corpus_state.get("relation_attrs", {})
                corrected_triples = corpus_state.get("corrected_triples", [])

                # v3.4精简版：关系类型属性映射（8个关系类型）
                # Schema v3.4定义：
                # - 位于、包含：无关系属性
                # - 相对方位：距离值、方向值（删除联动推荐）
                # - 具有功能：时段、适合人群(开放文本)、具有限制(开放文本列表)、情感倾向
                # - 优于/相似/劣于：维度
                # - 发生事件：无关系属性（属性在事件实体上）
                RELATION_ATTRS_MAP = {
                    # 相对方位关系属性（v3.4：删除联动推荐，仅2个属性）
                    "相对方位": ["距离值", "方向值"],
                    # 功能关系属性（v3.4：开放文本属性）
                    "具有功能": ["时段", "适合人群", "具有限制", "情感倾向"],
                    # 对比关系属性（1个属性）
                    "优于": ["维度"],
                    "相似": ["维度"],
                    "劣于": ["维度"],
                    # 无属性的关系
                    "位于": [],
                    "包含": [],
                    "发生事件": [],  # 属性在事件实体上，非关系属性
                }

                for triple in corrected_triples:
                    triple["_corpus_id"] = corpus_id
                    # 查找关系属性（使用标准格式）
                    triple_key = f"<{triple['head']}, {triple['relation']}, {triple['tail']}>"

                    # 尝试多种 key 格式查找
                    attrs = (
                        relation_attrs.get(triple_key) or
                        relation_attrs.get(f"{triple['head']}, {triple['relation']}, {triple['tail']}") or
                        relation_attrs.get(f"<{triple['head']},{triple['relation']},{triple['tail']}>")
                    )

                    if attrs:
                        # v3.2精简版：relation_type 直接使用7种标准关系类型
                        triple["relation_type"] = triple.get("relation", "")
                        # 根据关系类型选择对应的属性集（Schema v3.2）
                        relation = triple.get("relation", "")
                        attr_fields = RELATION_ATTRS_MAP.get(relation, [])
                        # 提取该关系类型的有效属性
                        triple["relation_attrs"] = {
                            field: attrs.get(field)
                            for field in attr_fields
                            if attrs.get(field) is not None
                        }
                    all_triples.append(triple)

        # 实体去重
        unique_entities, aliases = deduplicate_entities(all_entities, similarity_threshold)

        # 三元组去重
        unique_triples = deduplicate_triples(all_triples)

        logger.info(f"[Aggregator] 完成: {len(unique_entities)}个实体, {len(unique_triples)}个三元组")

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
        name: str,
        name_len: int,
        length_index: Dict[int, List[str]]
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

                unique_entities.append({
                    "name": standard_name,
                    "type": entity_type,
                    "category": entity_attrs.get("细分", ""),
                    "aliases": other_names,
                    "occurrence_count": len(occurrences),
                    "corpus_ids": list(set(o["corpus_id"] for o in occurrences)),
                    "attrs": entity_attrs,
                })

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
            unique_triples.append({
                "head": triple["head"],
                "relation": triple["relation"],
                "tail": triple["tail"],
                "evidence": triple.get("evidence", ""),
                "corpus_ids": [triple.get("_corpus_id", "")],
                "sem_score": triple.get("sem_score", 0),
                "fac_score": triple.get("fac_score", 0),
                "con_score": triple.get("con_score", 0),
                "passed_eval": triple.get("passed_eval", False),  # 默认 False 更安全
                "relation_type": triple.get("relation_type", ""),
                "relation_subtype": triple.get("relation_subtype", ""),
            })
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
        corpus_id = state['corpus_id']
        retry_count = state.get('retry_count', 0)
        logger.info(f"[Self-Check-NER] 校验语料: {corpus_id}, 重试次数: {retry_count}")

        writer({
            "step": "self_check_ner",
            "corpus_id": corpus_id,
            "status": "started",
            "message": "开始实体校验",
            "retry_count": retry_count
        })

        try:
            # P6改进：优先使用归一化文本
            text_for_processing = _get_text_for_processing(state)

            # P8改进：获取 QA Scaffold 上下文
            qa_entity_hints = state.get("qa_entity_hints", [])
            qa_context_dependencies = state.get("qa_context_dependencies", [])
            semantic_summary = state.get("semantic_summary", "")

            # 构建重试提示（如有）
            problem_entities = state.get('problem_entities', [])
            retry_hint = format_retry_hint(problem_entities, [])

            # 使用 OutputParser 进行结构化输出
            prompt_text = SELF_CHECK_NER_PROMPT.invoke({
                "raw_text": text_for_processing,
                "entities": format_entities(state["entities"]),
                "retry_hint": retry_hint,
                "semantic_summary": semantic_summary or "(无语义摘要)",
                "context_dependencies": format_context_dependencies(qa_context_dependencies),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckNERResult = parser.parse(response.content)

            # 应用归一化，生成 final_entities
            final_entities = _apply_entity_normalizations(state["entities"], result)

            # 提取问题实体（供重试参考）
            missing_names = [e.name for e in result.missing_entities]

            writer({
                "step": "self_check_ner",
                "corpus_id": corpus_id,
                "status": "completed",
                "verified_count": len(result.verified_entities),
                "missing_count": len(result.missing_entities),
                "normalization_count": len(result.entity_normalizations),
                "confidence": result.overall_confidence
            })

            return {
                "self_check_ner_result": result.model_dump(),
                "final_entities": final_entities,
                "problem_entities": missing_names,
                # P8修复：移除Self-Check-NER中的计数器增加，由路由函数统一处理
                "current_step": StepEnum.RE,  # Self-Check-NER 在 NER 和 RE 之间
            }

        except Exception as e:
            logger.error(f"[Self-Check-NER] 失败: {e}")
            writer({
                "step": "self_check_ner",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
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
        corpus_id = state['corpus_id']
        retry_count = state.get('retry_count', 0)
        max_retries = state.get('max_retries', DEFAULT_MAX_RETRIES)
        logger.info(f"[Self-Check-RE] 校验语料: {corpus_id}, 重试次数: {retry_count}/{max_retries}")

        writer({
            "step": "self_check_re",
            "corpus_id": corpus_id,
            "status": "started",
            "message": "开始三元组校验",
            "retry_count": retry_count
        })

        try:
            # 构建重试提示（如有）
            problem_triples = state.get('problem_triples', [])
            retry_hint = format_retry_hint([], problem_triples)

            # P8改进：获取 QA Scaffold 上下文
            semantic_summary = state.get("semantic_summary", "")
            qa_context_dependencies = state.get("qa_context_dependencies", [])

            # 获取已校验实体
            verified_entities = state.get('final_entities', [])
            ner_result = state.get('self_check_ner_result', {})
            if ner_result and 'verified_entities' in ner_result:
                verified_entities = ner_result['verified_entities']

            # P6改进：优先使用归一化文本
            text_for_processing = _get_text_for_processing(state)

            # 使用 OutputParser 进行结构化输出
            prompt_text = SELF_CHECK_RE_PROMPT.invoke({
                "raw_text": text_for_processing,
                "triples": format_triples(state.get("triples", [])),  # RE 输出
                "verified_entities": format_verified_entities(verified_entities),
                "retry_hint": retry_hint,
                "semantic_summary": semantic_summary or "(无语义摘要)",
                "context_dependencies": format_context_dependencies(qa_context_dependencies),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckREResult = parser.parse(response.content)

            # 应用修正，生成 corrected_triples（供 Eval 使用）
            corrected_triples = _apply_triple_corrections_for_self_check(
                state.get("triples", []), result  # RE 输出
            )

            # 提取问题三元组（供重试参考）
            problem_triples_list = [
                {
                    "head": t.head,
                    "relation": t.relation,
                    "tail": t.tail,
                    "reason": t.reason
                }
                for t in result.rejected_triples
            ]

            # 计算整体置信度
            overall_confidence = _calculate_overall_confidence(
                state.get('self_check_ner_result', {}),
                result.model_dump()
            )

            # 判断是否需要重试
            retry_count = state.get('retry_count', 0) + 1  # 每次进入 Self-Check 计数
            needs_retry = _should_trigger_retry(
                result, retry_count, max_retries, state.get('self_check_ner_result', {})
            )

            writer({
                "step": "self_check_re",
                "corpus_id": corpus_id,
                "status": "completed",
                "verified_count": len(result.verified_triples),
                "rejected_count": len(result.rejected_triples),
                "corrected_count": len(result.corrected_triples),
                "confidence": overall_confidence,
                "needs_retry": needs_retry
            })

            return {
                "self_check_re_result": result.model_dump(),
                "corrected_triples": corrected_triples,  # 输出给 Eval 使用
                "final_triples": corrected_triples,      # 同时保留 final_triples 兼容
                "problem_triples": problem_triples_list,
                "verification_confidence": overall_confidence,
                "retry_count": retry_count,
                "needs_review": overall_confidence == "low" or retry_count >= max_retries,
                "current_step": StepEnum.EVAL,  # 默认值，实际由路由决定
            }

        except Exception as e:
            logger.error(f"[Self-Check-RE] 失败: {e}")
            writer({
                "step": "self_check_re",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
            return {
                "self_check_re_result": {},
                "corrected_triples": state.get("triples", []),  # 失败时保留原 triples
                "error": str(e),
                "retry_count": min(state.get('retry_count', 0) + 1, max_retries),  # 确保不超过上限
                "retry_suggested": True,  # 异常时建议重试
                "current_step": StepEnum.EVAL,
            }

    return self_check_re_node


# ===== Self-Check 辅助函数 =====

def _apply_entity_normalizations(
    original_entities: Dict[str, List[str]],
    result: SelfCheckNERResult
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
    original_triples: List[Dict],
    result: SelfCheckREResult
) -> List[Dict]:
    """应用三元组修正"""
    final_triples = []

    # 保留验证通过的三元组
    for vt in result.verified_triples:
        triple = {
            "head": vt.head,
            "relation": vt.relation,
            "tail": vt.tail,
            "confidence": vt.confidence,
            "evidence_valid": vt.evidence_valid,
            "evidence_match": vt.evidence_match,
            "passed_eval": True,
        }
        final_triples.append(triple)

    # 应用修正的三元组
    for tc in result.corrected_triples:
        if tc.action == "delete":
            continue  # 删除操作：不添加到最终结果

        corrected_triple = {
            "head": tc.corrected_head or tc.original_head,
            "relation": tc.corrected_relation or tc.original_relation,
            "tail": tc.corrected_tail or tc.original_tail,
            "confidence": "medium",  # 修正后的置信度默认为 medium
            "correction_reason": tc.reason,
            "passed_eval": True,
        }
        final_triples.append(corrected_triple)

    return final_triples


def _calculate_overall_confidence(
    ner_result: Dict,
    re_result: Dict
) -> str:
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
    re_result: SelfCheckREResult,
    retry_count: int,
    max_retries: int,
    ner_result: Dict
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
        corpus_id = state['corpus_id']
        logger.info(f"[Joint_NER_RE] 处理语料: {corpus_id}")

        writer({
            "step": "joint_ner_re",
            "corpus_id": corpus_id,
            "status": "started",
            "message": "开始联合抽取"
        })

        try:
            # 使用归一化文本
            text_for_processing = _get_text_for_processing(state)

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
            prompt_text = JOINT_NER_RE_PROMPT_V2.invoke({
                "raw_text": text_for_processing,
                "entity_hints": format_entity_hints(qa_entity_hints),
                "relation_hints": format_relation_hints(qa_relation_hints),
                "context_dependencies": format_context_dependencies(qa_context_dependencies),
                "mentor_guidance": format_mentor_guidance(mentor_guidance),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: JointExtractionResult = parser.parse(response.content)

            # 转换为现有格式（兼容后续节点）
            # v3.4扩展版：实体类型扩展为6种（新增功能、事件）
            entities_dict = {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []}
            for e in result.entities:
                entity_type = e.type.value if hasattr(e.type, 'value') else e.type
                if entity_type in entities_dict:
                    entities_dict[entity_type].append(e.name)

            triples_list = [
                {
                    "head": t.head,
                    "relation": t.relation.value if hasattr(t.relation, 'value') else t.relation,  # Enum转字符串
                    "tail": t.tail,
                    "evidence": t.evidence,
                    "confidence": t.confidence.value if hasattr(t.confidence, 'value') else t.confidence,  # Enum转字符串
                    "attributes": t.attributes.model_dump(exclude_none=True) if t.attributes else {},  # TripleAttributes转字典
                }
                for t in result.triples
            ]

            logger.info(
                f"[Joint_NER_RE] 完成: {len(result.entities)}个实体, "
                f"{len(result.triples)}个三元组, 置信度={result.overall_confidence}"
            )

            writer({
                "step": "joint_ner_re",
                "corpus_id": corpus_id,
                "status": "completed",
                "entity_count": len(result.entities),
                "triple_count": len(result.triples),
                "confidence": result.overall_confidence
            })

            # P14新增：困惑检测（如果启用了查询功能）
            if enable_query:
                from .prompts import detect_extraction_confusion
                query_count = state.get("query_count", 0)
                max_queries = state.get("max_queries", 2)

                # 检测困惑（限制查询次数防止无限循环）
                if query_count < max_queries:
                    confusion = detect_extraction_confusion(result.model_dump(), dict(state))
                    if confusion:
                        logger.info(f"[Joint_NER_RE] 检测到困惑，请求导师帮助: {confusion['query_type']}")
                        writer({
                            "step": "joint_ner_re",
                            "corpus_id": corpus_id,
                            "status": "needs_mentor_help",
                            "query_type": confusion["query_type"],
                            "query_content": confusion["query_content"],
                        })

                        return {
                            "entities": entities_dict,
                            "triples": triples_list,
                            "joint_extraction_result": result.model_dump(),
                            "extraction_strategy": "joint",
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
                "needs_mentor_help": False,  # P14新增：标记不需要帮助
                "current_step": StepEnum.SELF_CHECK_JOINT,
            }

        except Exception as e:
            logger.error(f"[Joint_NER_RE] 失败: {e}")
            writer({
                "step": "joint_ner_re",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
            return {
                "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
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
        corpus_id = state['corpus_id']
        retry_count = state.get('retry_count', 0)
        max_retries = state.get('max_retries', DEFAULT_MAX_RETRIES)
        logger.info(f"[Self-Check-Joint] 校验语料: {corpus_id}, 重试: {retry_count}/{max_retries}")

        writer({
            "step": "self_check_joint",
            "corpus_id": corpus_id,
            "status": "started",
            "retry_count": retry_count
        })

        try:
            text = _get_text_for_processing(state)

            # 获取反思历史（用于迭代改进）
            reflection_history = state.get("reflection_history", [])
            previous_reflection = format_reflection_history(reflection_history)

            # P12改进：使用增强版提示词（四维度校验+结构化反思）
            prompt_text = SELF_CHECK_JOINT_PROMPT_V2.invoke({
                "raw_text": text,
                "entities": format_joint_entities(
                    state.get("joint_extraction_result", {}).get("entities", [])
                ),
                "triples": format_joint_triples(state.get("triples", [])),
                "semantic_summary": state.get("semantic_summary", ""),
                "context_dependencies": format_context_dependencies(state.get("qa_context_dependencies", [])),
                "previous_reflection": previous_reflection,
                "improvement_attempts": state.get("improvement_strategy", ""),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckJointResult = parser.parse(response.content)

            # 记录反思历史
            reflection_history.append(result.reflection_text)

            logger.info(
                f"[Self-Check-Joint] 完成: confidence={result.overall_confidence}, "
                f"retry_suggested={result.retry_suggested}"
            )
            logger.info(f"[Self-Check-Joint] 反思: {result.reflection_text[:100]}...")

            writer({
                "step": "self_check_joint",
                "corpus_id": corpus_id,
                "status": "completed",
                "confidence": result.overall_confidence,
                "reflection": result.reflection_text[:200],
                "retry_suggested": result.retry_suggested
            })

            return {
                "self_check_joint_result": result.model_dump(),
                "reflection_text": result.reflection_text,
                "improvement_strategy": result.improvement_strategy,
                "reflection_history": reflection_history,
                "retry_count": retry_count + 1,
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "current_step": StepEnum.EVAL,
            }

        except Exception as e:
            logger.error(f"[Self-Check-Joint] 失败: {e}")
            writer({
                "step": "self_check_joint",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
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
        corpus_id = state['corpus_id']
        retry_count = state.get('retry_count', 0)
        logger.info(f"[Self-Check-QA] 校验语料: {corpus_id}, 重试: {retry_count}")

        writer({
            "step": "self_check_qa",
            "corpus_id": corpus_id,
            "status": "started",
            "retry_count": retry_count
        })

        try:
            text = _get_text_for_processing(state)

            qa_result = state.get("qa_scaffold_result", {})

            prompt_text = SELF_CHECK_QA_PROMPT.invoke({
                "raw_text": text,
                "qa_pairs": format_qa_pairs_for_check(qa_result.get("qa_pairs", [])),
                "entity_hints": format_entity_hints(qa_result.get("entity_hints", [])),
                "relation_hints": format_relation_hints(qa_result.get("relation_hints", [])),
                "semantic_summary": qa_result.get("semantic_summary", ""),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckQAResult = parser.parse(response.content)

            logger.info(
                f"[Self-Check-QA] 完成: entity_coverage={result.entity_coverage}, "
                f"relation_coverage={result.relation_coverage}, retry={result.retry_suggested}"
            )

            writer({
                "step": "self_check_qa",
                "corpus_id": corpus_id,
                "status": "completed",
                "entity_coverage": result.entity_coverage,
                "relation_coverage": result.relation_coverage,
                "retry_suggested": result.retry_suggested
            })

            return {
                "self_check_qa_result": result.model_dump(),
                "retry_count": retry_count + 1,
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "current_step": StepEnum.JOINT_NER_RE,
            }

        except Exception as e:
            logger.error(f"[Self-Check-QA] 失败: {e}")
            writer({
                "step": "self_check_qa",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
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
        corpus_id = state['corpus_id']
        retry_count = state.get('retry_count', 0)
        logger.info(f"[Self-Check-Eval] 校验语料: {corpus_id}, 重试: {retry_count}")

        writer({
            "step": "self_check_eval",
            "corpus_id": corpus_id,
            "status": "started",
            "retry_count": retry_count
        })

        try:
            text = _get_text_for_processing(state)

            prompt_text = SELF_CHECK_EVAL_PROMPT.invoke({
                "raw_text": text,
                "eval_scores": format_eval_scores_for_check(state.get("eval_scores", [])),
                "corrected_triples": format_triples(state.get("corrected_triples", [])),
                "eval_passed": state.get("eval_passed", False),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckEvalResult = parser.parse(response.content)

            logger.info(
                f"[Self-Check-Eval] 完成: score_consistency={result.score_consistency}, "
                f"retry={result.retry_suggested}"
            )

            writer({
                "step": "self_check_eval",
                "corpus_id": corpus_id,
                "status": "completed",
                "score_consistency": result.score_consistency,
                "retry_suggested": result.retry_suggested
            })

            return {
                "self_check_eval_result": result.model_dump(),
                "retry_count": retry_count + 1,
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "current_step": StepEnum.LABEL,
            }

        except Exception as e:
            logger.error(f"[Self-Check-Eval] 失败: {e}")
            writer({
                "step": "self_check_eval",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
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
        corpus_id = state['corpus_id']
        retry_count = state.get('retry_count', 0)
        logger.info(f"[Self-Check-Label] 校验语料: {corpus_id}, 重试: {retry_count}")

        writer({
            "step": "self_check_label",
            "corpus_id": corpus_id,
            "status": "started",
            "retry_count": retry_count
        })

        try:
            text = _get_text_for_processing(state)

            entity_attrs = state.get("entity_attrs", {})
            relation_attrs = state.get("relation_attrs", {})

            # 格式化属性用于提示词
            entity_attrs_str = "\n".join(
                [f"- {name}: {attrs}" for name, attrs in entity_attrs.items()]
            ) if entity_attrs else "(无实体属性)"

            relation_attrs_str = "\n".join(
                [f"- {key}: {attrs}" for key, attrs in relation_attrs.items()]
            ) if relation_attrs else "(无关系属性)"

            prompt_text = SELF_CHECK_LABEL_PROMPT.invoke({
                "raw_text": text,
                "entity_attrs": entity_attrs_str,
                "relation_attrs": relation_attrs_str,
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckLabelResult = parser.parse(response.content)

            logger.info(
                f"[Self-Check-Label] 完成: attr_completeness={result.attr_completeness}, "
                f"retry={result.retry_suggested}"
            )

            writer({
                "step": "self_check_label",
                "corpus_id": corpus_id,
                "status": "completed",
                "attr_completeness": result.attr_completeness,
                "retry_suggested": result.retry_suggested
            })

            # 如果校验通过，更新属性为校验后的版本
            if result.verified_entity_attrs:
                entity_attrs = result.verified_entity_attrs
            if result.verified_relation_attrs:
                relation_attrs = result.verified_relation_attrs

            return {
                "self_check_label_result": result.model_dump(),
                "entity_attrs": entity_attrs,
                "relation_attrs": relation_attrs,
                "retry_count": retry_count + 1,
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "current_step": StepEnum.DONE,
            }

        except Exception as e:
            logger.error(f"[Self-Check-Label] 失败: {e}")
            writer({
                "step": "self_check_label",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
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
        corpus_id = state['corpus_id']
        retry_count = state.get('retry_count', 0)
        logger.info(f"[Self-Check-Filter] 校验语料: {corpus_id}, 重试: {retry_count}")

        writer({
            "step": "self_check_filter",
            "corpus_id": corpus_id,
            "status": "started",
            "retry_count": retry_count
        })

        try:
            text = state.get("raw_text", "")
            filter_result = state.get("filter_result", {})

            prompt_text = SELF_CHECK_FILTER_PROMPT.invoke({
                "raw_text": text,
                "is_valid": filter_result.get("is_valid", True),
                "confidence": filter_result.get("confidence", "medium"),
                "skip_reason": filter_result.get("skip_reason", ""),
                "has_geo_entity": filter_result.get("has_geo_entity", False),
                "has_spatial_relation": filter_result.get("has_spatial_relation", False),
                "geo_entity_hint": filter_result.get("geo_entity_hint", ""),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckFilterResult = parser.parse(response.content)

            logger.info(
                f"[Self-Check-Filter] 完成: verified_is_valid={result.verified_is_valid}, "
                f"false_negative={result.false_negative_detected}, retry={result.retry_suggested}"
            )

            writer({
                "step": "self_check_filter",
                "corpus_id": corpus_id,
                "status": "completed",
                "verified_is_valid": result.verified_is_valid,
                "false_negative_detected": result.false_negative_detected,
                "retry_suggested": result.retry_suggested
            })

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
                "filter_result": updated_filter_result if updated_filter_result else state.get("filter_result", {}),
                "retry_count": retry_count + 1,
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "current_step": StepEnum.NORMALIZE if result.verified_is_valid else StepEnum.DONE,
            }

        except Exception as e:
            logger.error(f"[Self-Check-Filter] 失败: {e}")
            writer({
                "step": "self_check_filter",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
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

    async def self_check_normalize_node(state: CorpusState, writer: StreamWriter) -> Dict:
        corpus_id = state['corpus_id']
        retry_count = state.get('retry_count', 0)
        logger.info(f"[Self-Check-Normalize] 校验语料: {corpus_id}, 重试: {retry_count}")

        writer({
            "step": "self_check_normalize",
            "corpus_id": corpus_id,
            "status": "started",
            "retry_count": retry_count
        })

        try:
            raw_text = state.get("raw_text", "")
            normalize_result = state.get("normalize_result", {})

            prompt_text = SELF_CHECK_NORMALIZE_PROMPT.invoke({
                "raw_text": raw_text,
                "normalized_text": normalize_result.get("normalized_text", ""),
                "confidence": normalize_result.get("confidence", "medium"),
                "has_changes": normalize_result.get("has_changes", False),
                "preserved_semantics": normalize_result.get("preserved_semantics", True),
                "normalizations": format_normalizations_for_check(normalize_result.get("normalizations", [])),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckNormalizeResult = parser.parse(response.content)

            logger.info(
                f"[Self-Check-Normalize] 完成: semantics_preserved={result.semantics_preserved}, "
                f"info_added={result.info_added}, retry={result.retry_suggested}"
            )

            writer({
                "step": "self_check_normalize",
                "corpus_id": corpus_id,
                "status": "completed",
                "semantics_preserved": result.semantics_preserved,
                "retry_suggested": result.retry_suggested
            })

            # 如果语义丢失或添加了信息，使用原文
            updated_normalized_text = result.verified_normalized_text
            if result.info_added or result.info_lost:
                logger.warning(f"[Self-Check-Normalize] 检测到语义问题，使用原文")
                updated_normalized_text = raw_text

            return {
                "self_check_normalize_result": result.model_dump(),
                "normalized_text": updated_normalized_text,
                "retry_count": retry_count + 1,
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "current_step": StepEnum.QA_SCAFFOLD,
            }

        except Exception as e:
            logger.error(f"[Self-Check-Normalize] 失败: {e}")
            writer({
                "step": "self_check_normalize",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
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
        corpus_list: List[Dict],
        writer: StreamWriter
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

        writer({
            "step": "batch_joint",
            "status": "started",
            "batch_size": batch_size,
        })

        if batch_size == 0:
            return {
                "batch_results": {},
                "cross_corpus_aliases": [],
                "needs_fallback": False,
            }

        max_retries = 3
        retry_delay = 2.0  # 初始延迟秒数

        for retry in range(max_retries):
            try:
                # 构建批量输入
                corpus_list_str = format_batch_corpus(corpus_list)

                # 调用LLM
                prompt_text = BATCH_JOINT_PROMPT.invoke({
                    "batch_size": batch_size,
                    "corpus_list": corpus_list_str,
                })
                full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"

                response = await llm.ainvoke(full_prompt)
                result: BatchExtractionResult = parser.parse(response.content)

                # 转换为字典格式
                batch_results = {}
                for r in result.results:
                    batch_results[r.corpus_id] = {
                        "entities": r.entities,
                        "triples": r.triples,
                        "confidence": r.confidence,
                        "has_geo_info": r.has_geo_info,
                        "skip_reason": r.skip_reason,
                    }

                logger.info(
                    f"[Batch_Joint] 完成: {len(result.results)}条语料, "
                    f"跨语料别名: {len(result.cross_corpus_aliases)}个, "
                    f"置信度: {result.overall_confidence}"
                )

                writer({
                    "step": "batch_joint",
                    "status": "completed",
                    "batch_size": len(result.results),
                    "cross_corpus_aliases_count": len(result.cross_corpus_aliases),
                    "confidence": result.overall_confidence,
                })

                return {
                    "batch_results": batch_results,
                    "cross_corpus_aliases": result.cross_corpus_aliases,
                    "cross_corpus_relations": result.cross_corpus_relations,
                    "batch_extraction_result": result.model_dump(),
                    "needs_fallback": False,
                }

            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"[Batch_Joint] 重试 {retry + 2}/{max_retries}: {e}")
                    await asyncio.sleep(retry_delay * (retry + 1))  # 指数退避
                    continue
                else:
                    logger.error(f"[Batch_Joint] 最终失败: {e}")
                    writer({
                        "step": "batch_joint",
                        "status": "error",
                        "error": str(e),
                        "batch_size": batch_size,
                    })

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
    创建批量校验节点

    职责：
    1. 校验批量抽取结果的质量
    2. 验证跨语料别名映射
    3. 决定是否需要重试或fallback
    """
    from .schemas import BatchSelfCheckResult
    from .prompts import BATCH_SELF_CHECK_PROMPT, format_batch_results_for_check, format_cross_corpus_aliases

    parser = PydanticOutputParser(pydantic_object=BatchSelfCheckResult)

    async def batch_self_check_node(
        batch_results: Dict,
        cross_corpus_aliases: List[Dict],
        writer: StreamWriter
    ) -> Dict:
        """
        批量校验

        Args:
            batch_results: {corpus_id: {entities, triples, confidence}}
            cross_corpus_aliases: 跨语料别名列表
            writer: StreamWriter

        Returns:
            {
                "verified_results": [...],
                "rejected_results": [...],
                "verified_aliases": [...],
                "retry_suggested": False,
                "fallback_to_single": False,
            }
        """
        logger.info(f"[Batch_Self_Check] 校验 {len(batch_results)} 条语料结果")

        writer({
            "step": "batch_self_check",
            "status": "started",
            "batch_size": len(batch_results),
        })

        if not batch_results:
            return {
                "verified_results": [],
                "rejected_results": [],
                "verified_aliases": [],
                "retry_suggested": False,
                "fallback_to_single": False,
            }

        try:
            # 格式化输入
            results_list = [
                {"corpus_id": cid, **data}
                for cid, data in batch_results.items()
            ]
            results_str = format_batch_results_for_check(results_list)
            aliases_str = format_cross_corpus_aliases(cross_corpus_aliases)

            # 调用LLM
            prompt_text = BATCH_SELF_CHECK_PROMPT.invoke({
                "batch_results": results_str,
                "cross_corpus_aliases": aliases_str,
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"

            response = await llm.ainvoke(full_prompt)
            result: BatchSelfCheckResult = parser.parse(response.content)

            logger.info(
                f"[Batch_Self_Check] 完成: 通过 {len(result.verified_results)} 条, "
                f"拒绝 {len(result.rejected_results)} 条, "
                f"重试建议: {result.retry_suggested}, "
                f"单条fallback: {result.fallback_to_single}"
            )

            writer({
                "step": "batch_self_check",
                "status": "completed",
                "verified_count": len(result.verified_results),
                "rejected_count": len(result.rejected_results),
                "retry_suggested": result.retry_suggested,
                "fallback_to_single": result.fallback_to_single,
            })

            return {
                "verified_results": [r.model_dump() for r in result.verified_results],
                "rejected_results": result.rejected_results,
                "verified_aliases": result.verified_aliases,
                "rejected_aliases": result.rejected_aliases,
                "batch_self_check_result": result.model_dump(),
                "retry_suggested": result.retry_suggested,
                "fallback_to_single": result.fallback_to_single,
            }

        except Exception as e:
            logger.error(f"[Batch_Self_Check] 失败: {e}")
            writer({
                "step": "batch_self_check",
                "status": "error",
                "error": str(e),
            })

            # 校验失败时，保守策略：全部通过但标记低置信度
            return {
                "verified_results": [
                    {"corpus_id": cid, **data, "confidence": "low"}
                    for cid, data in batch_results.items()
                ],
                "rejected_results": [],
                "verified_aliases": cross_corpus_aliases,
                "retry_suggested": False,
                "fallback_to_single": False,
            }

    return batch_self_check_node


# ===== P10新增：批量处理入口函数 =====

async def process_corpus_batch_with_llm(
    llm: Any,
    corpus_list: List[Dict],
    config: ExtractionConfig,
    batch_joint_node: Any = None,
    batch_self_check_node: Any = None,
) -> Dict:
    """
    批量处理语料（一次LLM调用处理batch_llm_size条）

    Args:
        llm: LLM实例
        corpus_list: 语料列表 [{"id": ..., "text": ...}, ...]
        config: ExtractionConfig

    Returns:
        {
            "batch_results": {corpus_id: {entities, triples}},
            "cross_corpus_aliases": [...],
            "fallback_results": [...],  # fallback单条处理的结果
        }
    """
    batch_llm_size = config.batch_llm_size
    enable_batch_llm = config.enable_batch_llm
    batch_llm_fallback = config.batch_llm_fallback

    # 如果不启用批量LLM，直接返回空（使用单条处理）
    if not enable_batch_llm:
        return {
            "batch_results": {},
            "cross_corpus_aliases": [],
            "needs_single_processing": True,
        }

    # 创建节点（如果未提供）
    if batch_joint_node is None:
        batch_joint_node = create_batch_joint_extraction_node(llm, batch_llm_size)
    if batch_self_check_node is None:
        batch_self_check_node = create_batch_self_check_node(llm)

    all_batch_results = {}
    all_cross_corpus_aliases = []
    fallback_corpus_list = []

    # 分批处理（每批batch_llm_size条）
    for i in range(0, len(corpus_list), batch_llm_size):
        batch_corpus = corpus_list[i:i + batch_llm_size]
        batch_num = i // batch_llm_size + 1

        logger.info(f"[Batch {batch_num}] 处理 {len(batch_corpus)} 条语料")

        # 创建StreamWriter
        def dummy_writer(event):
            pass

        # 批量抽取
        extraction_result = await batch_joint_node(batch_corpus, dummy_writer)

        if extraction_result.get("needs_fallback"):
            # 批量抽取失败，需要fallback
            if batch_llm_fallback:
                logger.warning(f"[Batch {batch_num}] 抽取失败，退化为单条处理")
                fallback_corpus_list.extend(batch_corpus)
            else:
                # 不启用fallback，记录失败
                for corpus in batch_corpus:
                    all_batch_results[corpus["id"]] = {
                        "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
                        "triples": [],
                        "confidence": "error",
                        "error": extraction_result.get("fallback_reason", "批量处理失败"),
                    }
            continue

        # 批量校验（可选）
        batch_results = extraction_result["batch_results"]
        cross_corpus_aliases = extraction_result["cross_corpus_aliases"]

        # 如果启用Self-Check，进行校验
        if config.enable_self_check or config.enable_full_self_check:
            check_result = await batch_self_check_node(
                batch_results, cross_corpus_aliases, dummy_writer
            )

            # 处理校验结果
            for r in check_result["verified_results"]:
                all_batch_results[r["corpus_id"]] = r

            # 校验失败的语料，加入fallback列表
            if batch_llm_fallback:
                for r in check_result["rejected_results"]:
                    corpus_id = r.get("corpus_id")
                    # 找到原语料
                    for corpus in batch_corpus:
                        if corpus["id"] == corpus_id:
                            fallback_corpus_list.append(corpus)
                            break

            all_cross_corpus_aliases.extend(check_result["verified_aliases"])

        else:
            # 不校验，直接使用抽取结果
            all_batch_results.update(batch_results)
            all_cross_corpus_aliases.extend(cross_corpus_aliases)

    return {
        "batch_results": all_batch_results,
        "cross_corpus_aliases": all_cross_corpus_aliases,
        "fallback_corpus_list": fallback_corpus_list,
        "needs_single_processing": len(fallback_corpus_list) > 0,
    }


# ===== P10新增：QA导师节点 =====

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
        corpus_id = state['corpus_id']

        # P14新增：检查是否有来自后续节点的查询
        mentor_query = state.get("mentor_query")

        if mentor_query:
            # ===== 回答查询模式 =====
            query_source = state.get("query_source_node", "unknown")
            logger.info(f"[QA_Mentor] 回答来自 {query_source} 的查询: {mentor_query.get('query_type')}")

            writer({
                "step": "qa_mentor",
                "corpus_id": corpus_id,
                "status": "answering_query",
                "query_source": query_source,
                "query_type": mentor_query.get("query_type"),
            })

            try:
                text_for_processing = _get_text_for_processing(state)

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
                prompt_text = MENTOR_QUERY_PROMPT.invoke({
                    "source_node": query_source,
                    "query_type": query_type,
                    "query_content": query_content,
                    "involved_entities": ", ".join(involved_entities) if involved_entities else "(无)",
                    "involved_relations": ", ".join(involved_relations) if involved_relations else "(无)",
                    "current_confidence": current_confidence,
                    "context": context,
                    "raw_text": text_for_processing,
                    "previous_guidance": previous_guidance_text,
                })
                full_prompt = f"{prompt_text.messages[1].content}\n\n{query_parser.get_format_instructions()}"
                response = await llm.ainvoke(full_prompt)
                query_result: MentorQueryResponse = query_parser.parse(response.content)

                logger.info(
                    f"[QA_Mentor] 查询回答完成: 置信度={query_result.response_confidence}, "
                    f"建议修改={query_result.suggests_revision}"
                )

                writer({
                    "step": "qa_mentor",
                    "corpus_id": corpus_id,
                    "status": "query_answered",
                    "answer": query_result.answer[:100],
                    "confidence": query_result.response_confidence,
                    "return_to": query_result.return_to_node,
                })

                # 更新指导信息（如果有）
                updated_guidance = {}
                if query_result.updated_guidance:
                    updated_guidance = query_result.updated_guidance.model_dump()
                elif previous_guidance:
                    updated_guidance = previous_guidance

                # 返回到发起查询的节点
                return_to = query_source if query_source else "joint_ner_re"

                return {
                    "mentor_response": query_result.model_dump(),
                    "mentor_guidance": updated_guidance,
                    "qa_entity_hints": query_result.updated_entity_hints or state.get("qa_entity_hints", []),
                    "qa_relation_hints": query_result.updated_relation_hint or state.get("qa_relation_hints", []),
                    "needs_mentor_help": False,  # 已回答，继续处理
                    "query_count": state.get("query_count", 0) + 1,  # 增加查询计数
                    "return_to_node": return_to,
                    "current_step": getattr(StepEnum, return_to.upper(), StepEnum.JOINT_NER_RE),
                }

            except Exception as e:
                logger.error(f"[QA_Mentor] 回答查询失败: {e}")
                writer({
                    "step": "qa_mentor",
                    "corpus_id": corpus_id,
                    "status": "query_error",
                    "error": str(e)
                })
                # 查询失败时，返回到原节点继续
                return {
                    "mentor_response": {},
                    "needs_mentor_help": False,
                    "query_count": state.get("query_count", 0) + 1,
                    "return_to_node": state.get("query_source_node", "joint_ner_re"),
                    "current_step": getattr(StepEnum, state.get("query_source_node", "joint_ner_re").upper(), StepEnum.JOINT_NER_RE),
                }

        else:
            # ===== 初始化指导模式 =====
            logger.info(f"[QA_Mentor] 处理语料: {corpus_id}")

            writer({
                "step": "qa_mentor",
                "corpus_id": corpus_id,
                "status": "started",
                "message": "开始导师深度分析"
            })

            try:
                # 使用归一化后的文本
                text_for_processing = _get_text_for_processing(state)

                # 调用LLM
                prompt_text = QA_MENTOR_PROMPT.invoke({
                    "normalized_text": text_for_processing,
                })
                full_prompt = f"{prompt_text.messages[1].content}\n\n{scaffold_parser.get_format_instructions()}"
                response = await llm.ainvoke(full_prompt)
                result: QAMentorScaffoldResult = scaffold_parser.parse(response.content)

                logger.info(
                    f"[QA_Mentor] 完成: {len(result.qa_pairs)} 个问答对, "
                    f"{len(result.entity_hints)} 个实体提示, "
                    f"置信度={result.overall_confidence}"
                )

                # 发送完成事件
                writer({
                    "step": "qa_mentor",
                    "corpus_id": corpus_id,
                    "status": "completed",
                    "qa_count": len(result.qa_pairs),
                    "entity_hints": result.entity_hints,
                    "relation_hints": result.relation_hints,
                    "confidence": result.overall_confidence,
                    "has_mentor_guidance": result.mentor_guidance is not None
                })

                # 根据结果决定下一步
                if result.should_skip_detailed_extraction:
                    logger.info(f"[QA_Mentor] 建议跳过详细抽取: {corpus_id}")
                    return {
                        "qa_scaffold_result": result.model_dump(),
                        "semantic_summary": result.semantic_summary,
                        "mentor_guidance": result.mentor_guidance.model_dump() if result.mentor_guidance else {},
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
                        "mentor_guidance": result.mentor_guidance.model_dump() if result.mentor_guidance else {},
                        "reasoning_trace": result.reasoning_trace,
                        "qa_entity_hints": result.entity_hints,
                        "qa_relation_hints": result.relation_hints,
                        "qa_context_dependencies": result.context_dependencies,
                        "needs_mentor_help": False,  # P14新增
                        "current_step": StepEnum.JOINT_NER_RE,
                    }

            except Exception as e:
                logger.error(f"[QA_Mentor] 处理失败: {e}")
                writer({
                    "step": "qa_mentor",
                    "corpus_id": corpus_id,
                    "status": "error",
                    "error": str(e)
                })
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
            triple_lines = [f"<{t.get('head')}, {t.get('relation')}, {t.get('tail')}>" for t in triples[:5]]
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
            passed_count = sum(1 for t in corrected_triples if t.get("passed_eval", False))
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
        corpus_id = state['corpus_id']
        revision_cycle_count = state.get('revision_cycle_count', 0)
        max_revision_cycles = state.get('max_revision_cycles', config.max_revision_cycles)

        logger.info(f"[QA_Approval] 审批语料: {corpus_id}, 修改轮次: {revision_cycle_count}/{max_revision_cycles}")

        writer({
            "step": "qa_approval",
            "corpus_id": corpus_id,
            "status": "started",
            "revision_cycle": revision_cycle_count
        })

        try:
            text = _get_text_for_processing(state)

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
            prompt_text = QA_APPROVAL_PROMPT.invoke({
                "raw_text": text,
                "mentor_guidance": format_mentor_guidance(mentor_guidance),
                "semantic_summary": semantic_summary,
                "joint_result": format_joint_for_approval(joint_result),
                "eval_result": format_eval_for_approval(eval_result),
                "label_result": format_label_for_approval(label_result),
                "previous_feedbacks": format_revision_feedbacks(revision_feedbacks),
                "reflection_summary": reflection_summary,
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: QAApprovalResult = parser.parse(response.content)

            logger.info(
                f"[QA_Approval] 完成: overall_status={result.overall_status}, "
                f"retry_suggested={result.retry_suggested}, "
                f"retry_target_nodes={result.retry_target_nodes}"
            )

            writer({
                "step": "qa_approval",
                "corpus_id": corpus_id,
                "status": "completed",
                "overall_status": result.overall_status.value,
                "overall_confidence": result.overall_confidence,
                "retry_suggested": result.retry_suggested,
                "revision_cycle": revision_cycle_count + 1
            })

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
            writer({
                "step": "qa_approval",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
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
        corpus_id = state['corpus_id']
        revision_cycle = state.get('revision_cycle_count', 0)
        logger.info(f"[Revision_Joint] 修改抽取: {corpus_id}, 修改轮次: {revision_cycle}")

        writer({
            "step": "revision_joint",
            "corpus_id": corpus_id,
            "status": "started",
            "revision_cycle": revision_cycle
        })

        try:
            text = _get_text_for_processing(state)

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
            prompt_text = REVISION_JOINT_PROMPT.invoke({
                "raw_text": text,
                "feedback_summary": format_feedback_summary(revision_feedbacks, revision_cycle),
                "feedbacks": format_feedbacks_for_revision(recent_feedbacks),
                "semantic_summary": semantic_summary,
                "mentor_guidance": format_mentor_guidance(mentor_guidance),
                "previous_entities": format_entities(previous_entities),
                "previous_triples": format_triples(previous_triples),
                "entity_hints": format_entity_hints(entity_hints),
                "relation_hints": format_relation_hints(relation_hints),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: JointExtractionResult = parser.parse(response.content)

            # 转换为现有格式（v3.4扩展版：6种实体类型）
            entities_dict = {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []}
            for e in result.entities:
                entity_type = e.type.value if hasattr(e.type, 'value') else e.type
                if entity_type in entities_dict:
                    entities_dict[entity_type].append(e.name)

            triples_list = [
                {
                    "head": t.head,
                    "relation": t.relation.value if hasattr(t.relation, 'value') else t.relation,
                    "tail": t.tail,
                    "evidence": t.evidence,
                    "confidence": t.confidence.value if hasattr(t.confidence, 'value') else t.confidence,
                    "attributes": t.attributes.model_dump(exclude_none=True) if t.attributes else {},
                }
                for t in result.triples
            ]

            logger.info(
                f"[Revision_Joint] 完成: {len(result.entities)}个实体, "
                f"{len(result.triples)}个三元组, 置信度={result.overall_confidence}"
            )

            writer({
                "step": "revision_joint",
                "corpus_id": corpus_id,
                "status": "completed",
                "entity_count": len(result.entities),
                "triple_count": len(result.triples),
                "revision_cycle": revision_cycle
            })

            return {
                "entities": entities_dict,
                "triples": triples_list,
                "joint_extraction_result": result.model_dump(),
                "extraction_strategy": "joint_revision",
                "current_step": StepEnum.EVAL,
            }

        except Exception as e:
            logger.error(f"[Revision_Joint] 处理失败: {e}")
            writer({
                "step": "revision_joint",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
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
    from .schemas import EntityAlignmentResult, EntityAlignmentItem, EntityCandidate
    parser = PydanticOutputParser(pydantic_object=EntityAlignmentItem)

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
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像加速
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
                geo_entities.append({
                    "entity_id": entity_id,
                    "name": name,
                    "type": type_ or "",
                    "longitude": lon,
                    "latitude": lat,
                    "source": "geo_entity_names"
                })
                geo_embeddings.append(emb_list)
        
        geo_embeddings_np = np.array(geo_embeddings) if geo_embeddings else np.array([])
        logger.info(f"[Entity_Alignment] 预加载 geo_entity_names: {len(geo_entities)}条 (~{len(geo_entities)*768*4/1024/1024:.1f}MB)")
        
        return geo_entities, geo_embeddings_np

    def _batch_similarity_search_geo(query_embeddings_np, geo_embeddings_np, geo_entities, top_k):
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
                cur.execute("""
                    SELECT id, entity_id, name, type, longitude, latitude, address,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM amap_poi_wgs84
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_emb, query_emb, top_k))
                amap_rows = cur.fetchall()
            
            candidates = []
            for row in amap_rows:
                # row: (id, entity_id, name, type, longitude, latitude, address, similarity)
                # entity_id字段存储原始高德ID（如amap_B0FFLCH14H）
                # neo4j中amap节点的original_id属性存储这个原始高德ID
                amap_table_id, amap_original_id, name, type_, lon, lat, address, sim = row
                candidates.append({
                    "db_entity_id": amap_original_id,  # 直接使用原始高德ID，在neo4j中对应original_id属性
                    "name": name,
                    "type": type_ or "",
                    "similarity": sim,
                    "longitude": lon,
                    "latitude": lat,
                    "address": address,
                    "source": "amap_poi_wgs84"
                })
            
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
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像加速
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
                geo_entities.append({
                    "entity_id": entity_id,
                    "name": name,
                    "type": type_ or "",
                    "longitude": lon,
                    "latitude": lat,
                    "source": "geo_entity_names"
                })
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
                amap_entities.append({
                    "id": amap_id,
                    "original_id": original_id,
                    "name": name,
                    "type": type_ or "",
                    "longitude": lon,
                    "latitude": lat,
                    "address": address,
                    "source": "amap_poi_wgs84"
                })
                amap_embeddings.append(emb_list)
        
        # 转换为numpy数组
        geo_embeddings_np = np.array(geo_embeddings) if geo_embeddings else np.array([])
        amap_embeddings_np = np.array(amap_embeddings) if amap_embeddings else np.array([])
        
        logger.info(f"[Entity_Alignment] 预加载 geo_entity_names: {len(geo_entities)}条, amap_poi_wgs84: {len(amap_entities)}条")
        
        return geo_entities, amap_entities, geo_embeddings_np, amap_embeddings_np

    def _batch_similarity_search(query_embeddings_np, db_embeddings_np, db_entities, top_k):
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
        corpus_id = state['corpus_id']
        logger.info(f"[Entity_Alignment] 处理语料: {corpus_id}")

        writer({
            "step": "entity_alignment",
            "corpus_id": corpus_id,
            "status": "started",
            "message": "开始实体对齐"
        })

        try:
            # 连接数据库
            from settings import settings
            from kg.postgres_client import PostgresClient

            pg_config = settings.get_postgres_config()
            pg_client = PostgresClient(**pg_config)

            # 获取geo_poi_count并缓存（用于计算amap在neo4j的entity_id）
            nonlocal _amap_id_base_cache, _db_cache
            if _amap_id_base_cache is None:
                with pg_client.conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM geo_entity_names WHERE type = 'poi'")
                    geo_poi_count = cur.fetchone()[0]
                    _amap_id_base_cache = geo_poi_count
                    logger.info(f"[Entity_Alignment] geo_entity_names poi数量: {geo_poi_count}, amap neo4j ID起始: poi_{_amap_id_base_cache + 1}")
            amap_entity_id_base = _amap_id_base_cache

            # 预加载数据库embedding（首次调用时加载，后续使用缓存）
            if _db_cache is None:
                _db_cache = _load_db_embeddings(pg_client)
            geo_entities, amap_entities, geo_embeddings_np, amap_embeddings_np = _db_cache

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
                pg_client.close()
                return {
                    "entity_alignment_result": {
                        "alignment_items": [],
                        "aligned_entities": [],
                        "new_entities": [],
                        "skipped_entities": [],
                        "overall_alignment_rate": 0.0,
                        "alignment_confidence": "high"
                    },
                    "aligned_entity_ids": {},
                    "new_entity_names": [],
                    "current_step": StepEnum.DONE,
                }

            # 加载嵌入模型
            model = _get_embedding_model()

            # 生成实体嵌入向量（批量）
            entity_embeddings = model.encode(entity_names, show_progress_bar=False, convert_to_numpy=True)

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
                best_similarity = best_candidate.get("similarity", 0.0) if best_candidate else 0.0

                alignment_item = {
                    "extracted_name": name,
                    "extracted_type": entity_types.get(name, ""),
                    "candidates": candidates,
                    "best_match": None,
                    "alignment_status": "pending",
                    "llm_decision": None
                }

                # 高置信度：直接匹配
                if best_similarity >= high_threshold:
                    alignment_item["alignment_status"] = "aligned"
                    alignment_item["best_match"] = best_candidate
                    alignment_item["llm_decision"] = f"高置信度匹配({best_similarity:.3f}>=0.90)，直接确认"

                    aligned_entities.append({
                        "name": name,
                        "db_id": best_candidate["db_entity_id"],
                        "db_name": best_candidate["name"],
                        "similarity": best_similarity,
                        "source": best_candidate["source"]
                    })
                    aligned_ids[name] = best_candidate["db_entity_id"]

                    logger.debug(f"[Entity_Alignment] {name} -> {best_candidate['name']} (高置信度, 来源: {best_candidate['source']})")

                # 低置信度：直接跳过（新实体）
                elif best_similarity < low_threshold:
                    alignment_item["alignment_status"] = "new_entity"
                    alignment_item["llm_decision"] = f"相似度过低({best_similarity:.3f}<0.75)，判定为新实体"

                    new_entities.append(name)

                    logger.debug(f"[Entity_Alignment] {name} -> 新实体 (低置信度)")

                # 中置信度：交给LLM判断
                elif use_llm and candidates:
                    # 格式化候选信息（包含来源标识）
                    # 需要将name字段改为db_name以兼容format_alignment_candidates
                    candidates_for_format = []
                    for c in candidates:
                        c_formatted = c.copy()
                        c_formatted["db_name"] = c_formatted.get("name", "")
                        c_formatted["db_type"] = c_formatted.get("type", "")
                        candidates_for_format.append(c_formatted)
                    
                    candidates_text = format_alignment_candidates(candidates_for_format)

                    # 调用LLM判断
                    prompt_text = ENTITY_ALIGNMENT_PROMPT.invoke({
                        "extracted_name": name,
                        "extracted_type": entity_types.get(name, ""),
                        "raw_text": state.get("raw_text", ""),
                        "candidates": candidates_text,
                    })

                    try:
                        response = await llm.ainvoke(prompt_text.messages[1].content)

                        # 解析LLM输出
                        import re
                        status_match = re.search(r'alignment_status[=:]\s*"?(\w+)"?', response.content, re.IGNORECASE)
                        index_match = re.search(r'best_match_index[=:]\s*(\d+)', response.content, re.IGNORECASE)
                        decision_match = re.search(r'llm_decision[=:]\s*"([^"]+)"', response.content, re.IGNORECASE)

                        llm_status = status_match.group(1) if status_match else "new_entity"
                        best_index = int(index_match.group(1)) if index_match else -1
                        llm_decision = decision_match.group(1) if decision_match else "LLM判断"
                        
                        # 验证输出状态
                        VALID_STATUSES = {"aligned", "new_entity", "skip"}
                        if llm_status not in VALID_STATUSES:
                            llm_status = "new_entity"

                        alignment_item["alignment_status"] = llm_status
                        alignment_item["llm_decision"] = llm_decision

                        if llm_status == "aligned" and best_index >= 0 and best_index < len(candidates):
                            best_match = candidates[best_index]
                            alignment_item["best_match"] = best_match

                            aligned_entities.append({
                                "name": name,
                                "db_id": best_match["db_entity_id"],
                                "db_name": best_match["name"],
                                "similarity": best_match["similarity"],
                                "source": best_match["source"]
                            })
                            aligned_ids[name] = best_match["db_entity_id"]

                            logger.debug(f"[Entity_Alignment] {name} -> {best_match['name']} (LLM判断匹配, 来源: {best_match['source']})")
                        else:
                            new_entities.append(name)
                            logger.debug(f"[Entity_Alignment] {name} -> 新实体 (LLM判断)")

                    except Exception as e:
                        logger.warning(f"[Entity_Alignment] LLM判断失败: {e}, 默认为新实体")
                        alignment_item["alignment_status"] = "new_entity"
                        alignment_item["llm_decision"] = f"LLM判断异常，默认为新实体"
                        new_entities.append(name)

                else:
                    # 不使用LLM判断，默认为新实体
                    alignment_item["alignment_status"] = "new_entity"
                    alignment_item["llm_decision"] = f"中置信度({best_similarity:.3f})，未启用LLM判断"
                    new_entities.append(name)

                alignment_items.append(alignment_item)

            # 关闭数据库连接
            pg_client.close()

            # ===== 新实体创建逻辑（P12新增） =====
            # 将未对齐的新实体写入数据库和neo4j
            created_entity_ids = {}
            if new_entities:
                logger.info(f"[Entity_Alignment] 开始创建 {len(new_entities)} 个新实体...")

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
                    new_embeddings = model.encode(new_entities, show_progress_bar=False, convert_to_numpy=True)

                    # 准备新实体数据
                    for i, name in enumerate(new_entities):
                        entity_type = entity_types.get(name, "poi")
                        entity_data = {
                            "name": name,
                            "type": entity_type,
                            "aliases": [],
                            "source": "xiaohongshu"
                        }

                        # 写入Postgres geo_entity_names表
                        pg_entity_id = new_pg_client.insert_new_geo_entity(
                            entity_data,
                            new_embeddings[i].tolist()
                        )

                        if pg_entity_id:
                            # 写入Neo4j
                            neo4j_entity_id = neo4j_client.create_new_geo_entity(entity_data)
                            if neo4j_entity_id:
                                created_entity_ids[name] = neo4j_entity_id
                                logger.info(f"[Entity_Alignment] 新实体已创建: {name} -> {neo4j_entity_id}")
                            else:
                                # neo4j创建失败，使用postgres的entity_id
                                created_entity_ids[name] = pg_entity_id
                                logger.warning(f"[Entity_Alignment] Neo4j创建失败，使用PG ID: {name} -> {pg_entity_id}")
                        else:
                            logger.warning(f"[Entity_Alignment] 新实体创建失败: {name}")

                    logger.success(f"[Entity_Alignment] 新实体创建完成: {len(created_entity_ids)}/{len(new_entities)}")

                except Exception as create_error:
                    logger.error(f"[Entity_Alignment] 新实体创建异常: {create_error}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # 确保连接始终关闭（修复连接泄漏bug）
                    if new_pg_client is not None:
                        try:
                            new_pg_client.close()
                            logger.debug("[Entity_Alignment] 新实体创建PostgresClient连接已关闭")
                        except Exception as close_error:
                            logger.warning(f"[Entity_Alignment] 关闭PostgresClient时出错: {close_error}")
                    if neo4j_client is not None:
                        try:
                            neo4j_client.close()
                            logger.debug("[Entity_Alignment] Neo4jClient连接已关闭")
                        except Exception as close_error:
                            logger.warning(f"[Entity_Alignment] 关闭Neo4jClient时出错: {close_error}")

            # 计算整体对齐率
            total_entities = len(entity_names)
            aligned_count = len(aligned_entities)
            alignment_rate = aligned_count / total_entities if total_entities > 0 else 0.0

            # 统计各来源对齐数量
            geo_aligned = sum(1 for e in aligned_entities if e.get("source") == "geo_entity_names")
            amap_aligned = sum(1 for e in aligned_entities if e.get("source") == "amap_poi_wgs84")

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

            writer({
                "step": "entity_alignment",
                "corpus_id": corpus_id,
                "status": "completed",
                "aligned_count": aligned_count,
                "new_count": len(new_entities),
                "created_count": len(created_entity_ids),
                "alignment_rate": alignment_rate,
                "geo_aligned": geo_aligned,
                "amap_aligned": amap_aligned,
                "confidence": overall_confidence
            })

            # 合并aligned_ids和created_entity_ids
            all_entity_ids = {**aligned_ids, **created_entity_ids}

            return {
                "entity_alignment_result": {
                    "alignment_items": alignment_items,
                    "aligned_entities": aligned_entities,
                    "new_entities": new_entities,
                    "created_entities": [{"name": k, "entity_id": v} for k, v in created_entity_ids.items()],
                    "skipped_entities": skipped_entities,
                    "overall_alignment_rate": alignment_rate,
                    "alignment_confidence": overall_confidence,
                    "geo_aligned_count": geo_aligned,
                    "amap_aligned_count": amap_aligned,
                    "created_count": len(created_entity_ids)
                },
                "aligned_entity_ids": all_entity_ids,  # 包含已对齐和新创建的实体ID
                "new_entity_names": [n for n in new_entities if n not in created_entity_ids],  # 仅保留创建失败的
                "current_step": StepEnum.DONE,
            }

        except Exception as e:
            logger.error(f"[Entity_Alignment] 处理失败: {e}")
            import traceback
            traceback.print_exc()
            writer({
                "step": "entity_alignment",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
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
        corpus_id = state['corpus_id']
        logger.info(f"[Joint_NER_RE_V3] 处理语料: {corpus_id}")

        writer({
            "step": "joint_ner_re",
            "corpus_id": corpus_id,
            "status": "started",
            "version": "v3",
            "message": "开始联合抽取（优化版）"
        })

        try:
            text_for_processing = _get_text_for_processing(state)

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
            full_prompt_with_format = f"{full_prompt}\n\n{parser.get_format_instructions()}"

            response = await llm.ainvoke(full_prompt_with_format)
            result: JointExtractionResult = parser.parse(response.content)

            # 转换为现有格式（v3.4扩展版：6种实体类型）
            entities_dict = {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []}
            for e in result.entities:
                entity_type = e.type.value if hasattr(e.type, 'value') else e.type
                if entity_type in entities_dict:
                    entities_dict[entity_type].append(e.name)

            triples_list = [
                {
                    "head": t.head,
                    "relation": t.relation.value if hasattr(t.relation, 'value') else t.relation,
                    "tail": t.tail,
                    "evidence": t.evidence,
                    "confidence": t.confidence.value if hasattr(t.confidence, 'value') else t.confidence,
                    "attributes": t.attributes.model_dump(exclude_none=True) if t.attributes else {},
                }
                for t in result.triples
            ]

            logger.info(
                f"[Joint_NER_RE_V3] 完成: {len(result.entities)}个实体, "
                f"{len(result.triples)}个三元组, 置信度={result.overall_confidence}"
            )

            writer({
                "step": "joint_ner_re",
                "corpus_id": corpus_id,
                "status": "completed",
                "version": "v3",
                "entity_count": len(result.entities),
                "triple_count": len(result.triples),
                "confidence": result.overall_confidence
            })

            return {
                "entities": entities_dict,
                "triples": triples_list,
                "joint_extraction_result": result.model_dump(),
                "extraction_strategy": "joint_v3",
                "current_step": StepEnum.SELF_CHECK_JOINT,
            }

        except Exception as e:
            logger.error(f"[Joint_NER_RE_V3] 失败: {e}")
            writer({
                "step": "joint_ner_re",
                "corpus_id": corpus_id,
                "status": "error",
                "version": "v3",
                "error": str(e)
            })
            return {
                "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
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
        corpus_id = state['corpus_id']
        logger.info(f"[Filter_V2] 筛选语料: {corpus_id}")

        writer({
            "step": "filter",
            "corpus_id": corpus_id,
            "status": "started",
            "version": "v2",
            "message": "开始文本筛选（优化版）"
        })

        try:
            from .prompts import FILTER_PROMPT_V2
            prompt_text = FILTER_PROMPT_V2.invoke({"raw_text": state["raw_text"]})
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: FilterResult = parser.parse(response.content)

            logger.info(
                f"[Filter_V2] 结果: is_valid={result.is_valid}, "
                f"confidence={result.confidence}, "
                f"region_hint={result.region_hint}"
            )

            writer({
                "step": "filter",
                "corpus_id": corpus_id,
                "status": "completed",
                "version": "v2",
                "is_valid": result.is_valid,
                "confidence": result.confidence,
                "skip_reason": result.skip_reason,
            })

            next_step = StepEnum.NER if result.is_valid else StepEnum.DONE

            return {
                "filter_result": result.model_dump(),
                "current_step": next_step,
            }

        except Exception as e:
            logger.error(f"[Filter_V2] 失败: {e}")
            writer({
                "step": "filter",
                "corpus_id": corpus_id,
                "status": "error",
                "version": "v2",
                "error": str(e)
            })
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

    async def self_check_joint_node_v3(state: CorpusState, writer: StreamWriter) -> Dict:
        """Self-Check Joint V3: Pre-Mortem + 四维度评分"""
        corpus_id = state['corpus_id']
        retry_count = state.get('retry_count', 0)
        max_retries = state.get('max_retries', DEFAULT_MAX_RETRIES)
        logger.info(f"[Self-Check-Joint_V3] 校验语料: {corpus_id}, 重试: {retry_count}/{max_retries}")

        writer({
            "step": "self_check_joint",
            "corpus_id": corpus_id,
            "status": "started",
            "version": "v3",
            "retry_count": retry_count
        })

        try:
            text = _get_text_for_processing(state)
            reflection_history = state.get("reflection_history", [])

            from .prompts import SELF_CHECK_JOINT_PROMPT_V3
            prompt_text = SELF_CHECK_JOINT_PROMPT_V3.invoke({
                "raw_text": text,
                "entities": format_joint_entities(
                    state.get("joint_extraction_result", {}).get("entities", [])
                ),
                "triples": format_joint_triples(state.get("triples", [])),
                "semantic_summary": state.get("semantic_summary", ""),
                "context_dependencies": format_context_dependencies(state.get("qa_context_dependencies", [])),
                "previous_reflection": format_reflection_history(reflection_history),
                "improvement_attempts": format_improvement_strategy(state.get("improvement_strategy", {})),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: SelfCheckJointResultV2 = parser.parse(response.content)

            # 提取四维度评分
            dimension_scores = result.dimension_scores
            overall_confidence = result.overall_confidence

            logger.info(
                f"[Self-Check-Joint_V3] 完成: 四维度评分={dimension_scores}, "
                f"整体置信度={overall_confidence}, 重试建议={result.retry_suggested}"
            )

            writer({
                "step": "self_check_joint",
                "corpus_id": corpus_id,
                "status": "completed",
                "version": "v3",
                "dimension_scores": dimension_scores,
                "confidence": overall_confidence,
                "retry_suggested": result.retry_suggested,
            })

            return {
                "self_check_joint_result": result.model_dump(),
                "reflection_text": result.reflection_text,
                "improvement_strategy": result.improvement_strategy,
                "reflection_history": reflection_history + [result.reflection_text],
                "retry_count": retry_count + (1 if result.retry_suggested else 0),
                "retry_suggested": result.retry_suggested,
                "retry_reason": result.retry_reason,
                "current_step": StepEnum.EVAL if not result.retry_suggested else StepEnum.JOINT_NER_RE,
            }

        except Exception as e:
            logger.error(f"[Self-Check-Joint_V3] 失败: {e}")
            writer({
                "step": "self_check_joint",
                "corpus_id": corpus_id,
                "status": "error",
                "version": "v3",
                "error": str(e)
            })
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
        corpus_id = state['corpus_id']
        logger.info(f"[RE_V2] 处理语料: {corpus_id}")

        writer({
            "step": "re",
            "corpus_id": corpus_id,
            "status": "started",
            "version": "v2",
            "message": "开始关系抽取（优化版）"
        })

        total_entities = sum(len(v) for v in state["entities"].values())
        if total_entities == 0:
            logger.debug(f"[RE_V2] 无实体，跳过")
            writer({
                "step": "re",
                "corpus_id": corpus_id,
                "status": "skipped",
                "version": "v2",
                "reason": "无实体"
            })
            return {"current_step": StepEnum.EVAL, "triples": []}

        try:
            text_for_processing = _get_text_for_processing(state)
            qa_relation_hints = state.get("qa_relation_hints", [])
            qa_context_dependencies = state.get("qa_context_dependencies", [])

            from .prompts import RE_PROMPT_V2
            prompt_text = RE_PROMPT_V2.invoke({
                "raw_text": text_for_processing,
                "entities": format_entities(state["entities"]),
                "relation_hints": format_relation_hints(qa_relation_hints),
                "context_dependencies": format_context_dependencies(qa_context_dependencies),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: RelationExtractionResult = parser.parse(response.content)

            triples = [
                {
                    "head": t.head,
                    "relation": t.relation.value if hasattr(t.relation, 'value') else t.relation,
                    "tail": t.tail,
                    "evidence": t.evidence or "",
                    "attributes": t.attributes.model_dump(exclude_none=True) if t.attributes else {},
                }
                for t in result.triples
            ]

            logger.debug(f"[RE_V2] 结果: {len(triples)}个三元组")

            writer({
                "step": "re",
                "corpus_id": corpus_id,
                "status": "completed",
                "version": "v2",
                "triple_count": len(triples),
            })

            return {
                "triples": triples,
                "current_step": StepEnum.EVAL,
            }

        except Exception as e:
            logger.error(f"[RE_V2] 失败: {e}")
            writer({
                "step": "re",
                "corpus_id": corpus_id,
                "status": "error",
                "version": "v2",
                "error": str(e)
            })
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
        corpus_id = state['corpus_id']
        logger.info(f"[Label_V2] 处理语料: {corpus_id}")

        writer({
            "step": "label",
            "corpus_id": corpus_id,
            "status": "started",
            "version": "v2",
            "message": "开始属性标注（优化版）"
        })

        try:
            text_for_processing = _get_text_for_processing(state)

            # 收集所有实体名
            all_entities = []
            for entity_type, names in state["entities"].items():
                for name in names:
                    all_entities.append(name)

            # 格式化关系列表
            relations_list = format_triples(state.get("triples", []))

            from .prompts import LABEL_PROMPT_V2
            prompt_text = LABEL_PROMPT_V2.invoke({
                "raw_text": text_for_processing,
                "entities": all_entities,
                "relations": relations_list,
                "semantic_summary": state.get("semantic_summary", ""),
                "entity_hints": format_entity_hints(state.get("qa_entity_hints", [])),
                "relation_hints": format_relation_hints(state.get("qa_relation_hints", [])),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: LabelResult = parser.parse(response.content)

            logger.debug(f"[Label_V2] 完成: {len(result.entities)}个实体属性")

            writer({
                "step": "label",
                "corpus_id": corpus_id,
                "status": "completed",
                "version": "v2",
                "entity_attr_count": len(result.entities),
                "relation_attr_count": len(result.relations),
            })

            # 转换属性字典格式
            entity_attrs = {}
            for name, attrs in result.entities.items():
                entity_attrs[name] = attrs.model_dump(exclude_none=True) if hasattr(attrs, 'model_dump') else attrs

            relation_attrs = {}
            for key, attrs in result.relations.items():
                relation_attrs[key] = attrs.model_dump(exclude_none=True) if hasattr(attrs, 'model_dump') else attrs

            return {
                "entity_attrs": entity_attrs,
                "relation_attrs": relation_attrs,
                "current_step": StepEnum.DONE,
            }

        except Exception as e:
            logger.error(f"[Label_V2] 失败: {e}")
            writer({
                "step": "label",
                "corpus_id": corpus_id,
                "status": "error",
                "version": "v2",
                "error": str(e)
            })
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