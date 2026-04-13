"""
LangGraph节点函数 - 四步骤工作流节点
使用 LangChain PydanticOutputParser 进行结构化输出（兼容 DeepSeek API）
P2改进：简化评估节点，单次评估+规则校验
P3改进：支持 StreamWriter 流式输出
P5改进：添加 Filter 筛选节点
P6改进：添加 Normalize 归一化节点，NER/RE/Eval 节点优先使用归一化文本
"""
import json
import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Any, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langgraph.types import StreamWriter

from loguru import logger

from .state import CorpusState, KGState, PhaseEnum, StepEnum, DEFAULT_MAX_RETRIES, RELATION_TYPES


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
    EntityRecognitionResult,
    RelationExtractionResult,
    EvalResultFirst,
    EvalResultSecond,
    EvalResultSimplified,
    LabelResult,
    SelfCheckNERResult,
    SelfCheckREResult,
)
from .prompts import (
    FILTER_PROMPT,  # P5新增
    NORMALIZE_PROMPT,  # P6新增
    NER_PROMPT, RE_PROMPT, EVAL_PROMPT_1, EVAL_PROMPT_2,
    EVAL_PROMPT_SIMPLIFIED, LABEL_PROMPT,
    SELF_CHECK_NER_PROMPT, SELF_CHECK_RE_PROMPT,
    format_entities, format_triples, format_verified_entities, format_retry_hint,
)


# ===== Filter 筛选节点（P5新增） =====

def create_filter_node(llm: Any):
    """
    创建 Filter 筛选节点

    职责：
    1. 快速判断文本是否包含有价值的地理信息
    2. 筛选无效文本以节省后续处理成本
    3. 输出筛选结果和置信度
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
                f"skip_reason={result.skip_reason}"
            )

            # 发送完成事件
            writer({
                "step": "filter",
                "corpus_id": corpus_id,
                "status": "completed",
                "is_valid": result.is_valid,
                "confidence": result.confidence,
                "skip_reason": result.skip_reason,
                "has_geo_entity": result.has_geo_entity,
                "has_spatial_relation": result.has_spatial_relation,
                "geo_entity_hint": result.geo_entity_hint
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

            # 使用 OutputParser 进行结构化输出
            prompt_text = NER_PROMPT.invoke({"raw_text": text_for_processing})
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

            # 使用 OutputParser 进行结构化输出
            prompt_text = RE_PROMPT.invoke({
                "raw_text": text_for_processing,
                "entities": format_entities(state["entities"]),
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: RelationExtractionResult = parser.parse(response.content)

            # v2.2改进：提取三元组及属性
            triples = [
                {
                    "head": t.head,
                    "relation": t.relation,
                    "tail": t.tail,
                    "evidence": t.evidence or "",
                    "attributes": t.attributes or {},  # 新增：关系属性
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
    # v2.2改进：使用完整的18个关系类型列表
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


def create_eval_simplified_node(llm: Any, eval_threshold: float = 3.5):
    """
    P2改进：创建简化的单次评估节点
    P3改进：支持 StreamWriter 流式输出

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
                "current_step": StepEnum.LABEL,
            }

        try:
            # P6改进：优先使用归一化文本
            text_for_processing = _get_text_for_processing(state)

            # 格式化三元组用于提示词
            triples_text = format_triples(state["triples"])

            # 使用 OutputParser 进行结构化输出
            prompt_text = EVAL_PROMPT_SIMPLIFIED.invoke({
                "triples": triples_text,
                "raw_text": text_for_processing,
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

            return {
                "eval_scores": scores,
                "corrected_triples": corrected_triples,
                "eval_passed": overall_passed,
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


def create_label_node(llm: Any):
    """创建属性标注节点（v2.2改进：扩展实体属性）"""
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
            return {"current_step": StepEnum.DONE}

        try:
            # v2.2改进：获取原始文本用于提取情感标签、体验评价
            text_for_processing = _get_text_for_processing(state)

            # 使用 OutputParser 进行结构化输出
            prompt_text = LABEL_PROMPT.invoke({
                "entities": all_entities,
                "relations": format_triples(state["corrected_triples"]),
                "raw_text": text_for_processing,  # v2.2新增：原始文本
            })
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: LabelResult = parser.parse(response.content)

            # v2.2改进：提取实体属性（含情感标签、体验评价、知名度）
            entity_attrs = {}
            for name, attrs in result.entities.items():
                entity_attrs[name] = {
                    "类别": attrs.类别,
                    "细分": attrs.细分,
                    "情感标签": attrs.情感标签 or [],  # 新增
                    "体验评价": attrs.体验评价 or [],  # 新增
                    "知名度": attrs.知名度 or "",  # 新增
                    "来源可信度": attrs.来源可信度 or "中",  # 新增
                }

            # v2.2改进：提取关系属性（含空间精度、语义类型等）
            relation_attrs = {}
            for key, attrs in result.relations.items():
                normalized_key = normalize_relation_key(key)
                if normalized_key:
                    # 提取所有Label阶段补充的属性
                    relation_attrs[normalized_key] = {
                        "空间精度": attrs.空间精度 or "",
                        "语义类型": attrs.语义类型 or "",
                        "相邻类型": attrs.相邻类型 or "",  # v2.2新增
                        "层级类型": attrs.层级类型 or "",
                        "连接类型": attrs.连接类型 or "",
                        "交通方式": attrs.交通方式 or "",
                        "距离类型": attrs.距离类型 or "",
                        "方向类型": attrs.方向类型 or "",
                        "穿过类型": attrs.穿过类型 or "",
                        "变化类型": attrs.变化类型 or "",
                        "推荐强度": attrs.推荐强度 or "",
                        "推荐场景": attrs.推荐场景 or "",
                        "活动类型": attrs.活动类型 or "",
                        "活动频率": attrs.活动频率 or "",
                        "可达程度": attrs.可达程度 or "",
                        "交通效率": attrs.交通效率 or "",
                        "价格区间": attrs.价格区间 or "",
                        "消费类型": attrs.消费类型 or "",
                        "特征类型": attrs.特征类型 or "",
                        "特征显著性": attrs.特征显著性 or "",
                        "情感强度": attrs.情感强度 or "",
                        "情感类型": attrs.情感类型 or "",
                        "优势程度": attrs.优势程度 or "",  # v2.2改进：拆分为三个独立字段
                        "相似程度": attrs.相似程度 or "",
                        "劣势程度": attrs.劣势程度 or "",
                        "对比可靠性": attrs.对比可靠性 or "",
                        "替代性": attrs.替代性 or "",
                        "风险等级": attrs.风险等级 or "",
                        "事件影响度": attrs.事件影响度 or "",
                        "事件持续性": attrs.事件持续性 or "",
                        "来源可信度": attrs.来源可信度 or "中",
                    }
                else:
                    # 无法解析时保留原始 key
                    relation_attrs[key] = {
                        "来源可信度": attrs.来源可信度 or "中",
                    }

            logger.debug(f"[Label] 完成: {len(entity_attrs)}个实体, {len(relation_attrs)}个关系")

            # P3改进：发送完成事件
            writer({
                "step": "label",
                "corpus_id": corpus_id,
                "status": "completed",
                "entity_count": len(entity_attrs),
                "relation_count": len(relation_attrs)
            })

            return {
                "entity_attrs": entity_attrs,
                "relation_attrs": relation_attrs,
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
                for triple in corrected_triples:
                    triple["_corpus_id"] = corpus_id
                    # 查找关系属性并写入（使用标准格式）
                    triple_key = f"<{triple['head']}, {triple['relation']}, {triple['tail']}>"

                    # 尝试多种 key 格式查找
                    attrs = (
                        relation_attrs.get(triple_key) or
                        relation_attrs.get(f"{triple['head']}, {triple['relation']}, {triple['tail']}") or
                        relation_attrs.get(f"<{triple['head']},{triple['relation']},{triple['tail']}>")
                    )

                    if attrs:
                        triple["relation_type"] = attrs.get("类型", "")
                        triple["relation_subtype"] = attrs.get("细分", "")
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

            # 构建重试提示（如有）
            problem_entities = state.get('problem_entities', [])
            retry_hint = format_retry_hint(problem_entities, [])

            # 使用 OutputParser 进行结构化输出
            prompt_text = SELF_CHECK_NER_PROMPT.invoke({
                "raw_text": text_for_processing,
                "entities": format_entities(state["entities"]),
                "retry_hint": retry_hint,
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
                "retry_count": state.get('retry_count', 0) + 1,
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