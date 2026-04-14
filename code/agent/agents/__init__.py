"""
智能体模块 - 基于LangGraph和LangChain
P2改进：添加配置化支持
P3改进：添加流式输出支持
P5改进：添加Filter筛选节点
P6改进：添加Normalize归一化节点
P7改进：添加分批次处理入口
P8改进：添加QA Scaffold脚手架节点和Self-Check二次对话节点
P9改进：添加联合抽取 + Reflexion机制 + 全节点二次检查 + Filter/Normalize二次检查（可选）
"""
from .config import ExtractionConfig, DEFAULT_CONFIG
from .state import (
    KGState, WorkerResult, CorpusState,
    StepEnum, PhaseEnum, DEFAULT_MAX_RETRIES, ENTITY_TYPES, RELATION_TYPES, ENTITY_CATEGORIES
)
from .schemas import (
    EntityRecognitionResult,
    RelationExtractionResult,
    Triple, TripleScore,
    EvalResultFirst, EvalResultSecond, EvalResultSimplified,
    Correction, LabelResult,
    EntityAttributes, RelationAttributes,
    QAPair, QAScaffoldResult,
    # P9新增：联合抽取模型
    JointEntity, JointTriple, JointExtractionResult,
    # P9新增：所有Self-Check模型
    SelfCheckJointResult, SelfCheckQAResult, SelfCheckEvalResult, SelfCheckLabelResult,
    # P9新增：Filter/Normalize二次检查模型（可选）
    SelfCheckFilterResult, SelfCheckNormalizeResult,
    # P10新增：批量LLM调用模型
    BatchCorpusResult, BatchExtractionResult, BatchSelfCheckResult,
)
from .prompts import (
    NER_PROMPT, RE_PROMPT,
    EVAL_PROMPT_1, EVAL_PROMPT_2, EVAL_PROMPT_SIMPLIFIED, LABEL_PROMPT,
    QA_SCAFFOLD_PROMPT,
    SELF_CHECK_NER_PROMPT, SELF_CHECK_RE_PROMPT,
    # P9新增：联合抽取和所有Self-Check提示词
    JOINT_NER_RE_PROMPT,
    SELF_CHECK_JOINT_PROMPT, SELF_CHECK_QA_PROMPT, SELF_CHECK_EVAL_PROMPT, SELF_CHECK_LABEL_PROMPT,
    # P9新增：Filter/Normalize二次检查提示词（可选）
    SELF_CHECK_FILTER_PROMPT, SELF_CHECK_NORMALIZE_PROMPT,
    format_entities, format_triples,
    format_entity_hints, format_relation_hints, format_context_dependencies,
)
from .nodes import (
    create_ner_node,
    create_re_node,
    create_eval_1_node,
    create_eval_2_node,
    create_eval_simplified_node,
    create_label_node,
    create_coordinator_node,
    create_aggregator_node,
    create_qa_scaffold_node,
    create_filter_node,
    create_normalize_node,
    create_self_check_ner_node,
    create_self_check_re_node,
    rule_based_validation,
    # P9新增：联合抽取和所有Self-Check节点
    create_joint_ner_re_node,
    create_self_check_joint_node,
    create_self_check_qa_node,
    create_self_check_eval_node,
    create_self_check_label_node,
    # P9新增：Filter/Normalize二次检查节点（可选）
    create_self_check_filter_node,
    create_self_check_normalize_node,
    # P10新增：批量LLM调用节点
    create_batch_joint_extraction_node,
    create_batch_self_check_node,
    process_corpus_batch_with_llm,
)
from .workflow import (
    build_corpus_workflow,
    build_distributed_workflow,
    process_corpus,
    process_batch,
    process_corpus_streaming,
    process_batch_streaming,
    process_corpus_in_batches,
)

__all__ = [
    # 配置
    "ExtractionConfig", "DEFAULT_CONFIG",
    # 状态
    "KGState", "WorkerResult", "CorpusState",
    "StepEnum", "PhaseEnum", "DEFAULT_MAX_RETRIES", "ENTITY_TYPES", "RELATION_TYPES", "ENTITY_CATEGORIES",
    # Pydantic模型
    "EntityRecognitionResult",
    "RelationExtractionResult",
    "Triple", "TripleScore",
    "EvalResultFirst", "EvalResultSecond", "EvalResultSimplified",
    "Correction", "LabelResult",
    "EntityAttributes", "RelationAttributes",
    "QAPair", "QAScaffoldResult",
    # P9新增：联合抽取模型
    "JointEntity", "JointTriple", "JointExtractionResult",
    # P9新增：Self-Check模型
    "SelfCheckJointResult", "SelfCheckQAResult", "SelfCheckEvalResult", "SelfCheckLabelResult",
    # P9新增：Filter/Normalize二次检查模型（可选）
    "SelfCheckFilterResult", "SelfCheckNormalizeResult",
    # P10新增：批量LLM调用模型
    "BatchCorpusResult", "BatchExtractionResult", "BatchSelfCheckResult",
    # 提示词
    "NER_PROMPT", "RE_PROMPT",
    "EVAL_PROMPT_1", "EVAL_PROMPT_2", "EVAL_PROMPT_SIMPLIFIED", "LABEL_PROMPT",
    "QA_SCAFFOLD_PROMPT",
    "SELF_CHECK_NER_PROMPT", "SELF_CHECK_RE_PROMPT",
    # P9新增：联合抽取和Self-Check提示词
    "JOINT_NER_RE_PROMPT",
    "SELF_CHECK_JOINT_PROMPT", "SELF_CHECK_QA_PROMPT", "SELF_CHECK_EVAL_PROMPT", "SELF_CHECK_LABEL_PROMPT",
    # P9新增：Filter/Normalize二次检查提示词（可选）
    "SELF_CHECK_FILTER_PROMPT", "SELF_CHECK_NORMALIZE_PROMPT",
    # P10新增：批量LLM调用提示词
    "BATCH_JOINT_PROMPT", "BATCH_SELF_CHECK_PROMPT",
    "format_entities", "format_triples",
    "format_entity_hints", "format_relation_hints", "format_context_dependencies",
    # P10新增：批量格式化函数
    "format_batch_corpus", "format_batch_results_for_check", "format_cross_corpus_aliases",
    # 节点工厂函数
    "create_ner_node",
    "create_re_node",
    "create_eval_1_node",
    "create_eval_2_node",
    "create_eval_simplified_node",
    "create_label_node",
    "create_coordinator_node",
    "create_aggregator_node",
    "create_qa_scaffold_node",
    "create_filter_node",
    "create_normalize_node",
    "create_self_check_ner_node",
    "create_self_check_re_node",
    "rule_based_validation",
    # P9新增：联合抽取和Self-Check节点
    "create_joint_ner_re_node",
    "create_self_check_joint_node",
    "create_self_check_qa_node",
    "create_self_check_eval_node",
    "create_self_check_label_node",
    # P9新增：Filter/Normalize二次检查节点（可选）
    "create_self_check_filter_node",
    "create_self_check_normalize_node",
    # P10新增：批量LLM调用节点
    "create_batch_joint_extraction_node",
    "create_batch_self_check_node",
    "process_corpus_batch_with_llm",
    # 工作流
    "build_corpus_workflow",
    "build_distributed_workflow",
    "process_corpus",
    "process_batch",
    "process_corpus_streaming",
    "process_batch_streaming",
    "process_corpus_in_batches",
]