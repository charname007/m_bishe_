"""
智能体模块 - 基于LangGraph和LangChain
P2改进：添加配置化支持
P3改进：添加流式输出支持
P5改进：添加Filter筛选节点
P6改进：添加Normalize归一化节点
P7改进：添加分批次处理入口
P8改进：添加QA Scaffold脚手架节点和Self-Check二次对话节点
P9改进：添加联合抽取 + Reflexion机制 + 全节点二次检查 + Filter/Normalize二次检查（可选）
P10改进：添加批量LLM调用 + QA导师模式
v3.2改进：精简版8关系体系 + 5实体属性 + 9功能大类 + 可选属性原则
v3.2重构：提取RELATION_VARIANT_MAPPING常量 + CorpusState子状态拆分 + NodeTemplate基类
P12改进：模块化Schema组件 + RISEN/RCoT框架提示词重构 + Self-Check增强
"""
from .config import ExtractionConfig, DEFAULT_CONFIG
from .state import (
    KGState, WorkerResult, CorpusState,
    StepEnum, PhaseEnum, DEFAULT_MAX_RETRIES, ENTITY_TYPES, RELATION_TYPES, ENTITY_CATEGORIES,
    # v3.4新增：默认实体字典（用于初始化和错误处理）
    DEFAULT_ENTITY_DICT,
    # v3.2重构新增：子状态类
    InputState, ConfigState, FilterState, NormalizeState, QAScaffoldState,
    ExtractState, EvalState, SelfCheckState, ReflexionState, RetryState,
    QAMentorState, AlignmentState, OutputState, ControlState,
    # P15新增：状态工厂函数
    create_default_corpus_state, create_default_kg_state,
)
from .node_template import (
    NodeTemplate, RawTextNodeTemplate, NoLLMNodeTemplate, get_text_for_processing,
    # v3.2重构新增：类型别名
    StateDict, ResultDict, NodeFunc,
)
from .schemas import (
    EntityRecognitionResult,
    RelationExtractionResult,
    Triple, TripleScore,
    EvalResultFirst, EvalResultSecond, EvalResultSimplified,
    Correction, LabelResult,
    EntityAttributes, RelationAttributes,
    QAPair, QAScaffoldResult,
    # P5新增：Filter筛选模型
    FilterResult,
    # P6新增：Normalize归一化模型
    NormalizationRecord, NormalizeResult,
    # v3.2新增：精简版枚举
    RelationTypeEnum, FunctionEnum, FeatureTagEnum, CompareDimensionEnum,
    EventCategoryEnum, EventStateEnum, CrowdNodeEnum, LimitNodeEnum,
    DistanceValueEnum, DirectionValueEnum, EmotionNodeEnum, RatingNodeEnum,
    ConfidenceEnum,
    # P9新增：联合抽取模型
    JointEntity, JointTriple, JointExtractionResult,
    # P9新增：所有Self-Check模型
    SelfCheckJointResult, SelfCheckQAResult, SelfCheckEvalResult, SelfCheckLabelResult,
    # P12新增：Self-Check增强版模型（四维度评分）
    DimensionScore, ImprovementAction, SelfCheckJointResultV2,
    # P9新增：Filter/Normalize二次检查模型（可选）
    SelfCheckFilterResult, SelfCheckNormalizeResult,
    # P10新增：批量LLM调用模型
    BatchCorpusResult, BatchExtractionResult, BatchSelfCheckResult,
    # P15新增：批量前置节点模型
    BatchFilterItemResult, BatchFilterResult,
    BatchNormalizeItemResult, BatchNormalizeResult,
    BatchQAScaffoldItemResult, BatchQAScaffoldResult,
    # P15新增：批量Self-Check节点模型
    BatchSelfCheckQACorpusResult, BatchSelfCheckQAResult,
    # P10新增：QA导师模型
    ApprovalStatusEnum, ApprovalFeedback, NodeApprovalResult, QAApprovalResult,
    MentorGuidance, QAMentorScaffoldResult,
    # P11新增：实体对齐模型
    EntityCandidate, EntityAlignmentItem, EntityAlignmentResult,
    # v3.2新增：常量列表
    RELATION_TYPES, FUNCTION_NODES, FEATURE_TAGS, COMPARE_DIMENSIONS, EVENT_CATEGORIES,
    # v3.2重构新增：关系类型映射常量和辅助函数
    RELATION_VARIANT_MAPPING, normalize_relation_type,
    # v3.4新增：关系属性映射常量
    RELATION_ATTRS_MAP,
)
from .prompts import (
    NER_PROMPT, RE_PROMPT,
    EVAL_PROMPT_1, EVAL_PROMPT_2, EVAL_PROMPT_SIMPLIFIED, LABEL_PROMPT,
    QA_SCAFFOLD_PROMPT,
    # P5新增：Filter筛选提示词
    FILTER_PROMPT,
    # P6新增：Normalize归一化提示词
    NORMALIZE_PROMPT,
    SELF_CHECK_NER_PROMPT, SELF_CHECK_RE_PROMPT,
    # P9新增：联合抽取和所有Self-Check提示词
    JOINT_NER_RE_PROMPT,
    SELF_CHECK_JOINT_PROMPT, SELF_CHECK_QA_PROMPT, SELF_CHECK_EVAL_PROMPT, SELF_CHECK_LABEL_PROMPT,
    # P9新增：Filter/Normalize二次检查提示词（可选）
    SELF_CHECK_FILTER_PROMPT, SELF_CHECK_NORMALIZE_PROMPT,
    # P10新增：批量LLM调用提示词
    BATCH_JOINT_PROMPT, BATCH_SELF_CHECK_PROMPT,
    # P15新增：批量预处理提示词
    BATCH_FILTER_PROMPT, BATCH_NORMALIZE_PROMPT, BATCH_QA_SCAFFOLD_PROMPT,
    # P15新增：批量Self-Check提示词
    BATCH_SELF_CHECK_QA_PROMPT,
    # P15新增：批量Eval提示词
    BATCH_EVAL_PROMPT,
    # P10新增：QA导师模式提示词
    QA_MENTOR_PROMPT, QA_APPROVAL_PROMPT, REVISION_JOINT_PROMPT,
    # P11新增：实体对齐提示词
    ENTITY_ALIGNMENT_PROMPT,
    # P12新增：模块化Schema组件
    ENTITY_SCHEMA_CORE, RELATION_SCHEMA_CORE, ENTITY_ATTRIBUTE_SCHEMA, RELATION_ATTRIBUTE_SCHEMA,
    VALIDATION_COT, NEGATIVE_EXAMPLES, EXPERT_ROLE_TEMPLATE,
    # v3.4新增：实体区分规则
    ENTITY_DISTINCTION_RULES,
    # P12新增：重构版提示词
    JOINT_NER_RE_PROMPT_V2, SELF_CHECK_JOINT_PROMPT_V2,
    assemble_joint_extraction_prompt,
    format_entities, format_triples,
    format_entity_hints, format_relation_hints, format_context_dependencies,
    # P10新增：批量格式化函数
    format_batch_corpus, format_batch_results_for_check, format_cross_corpus_aliases,
    # P15新增：批量QA校验格式化函数
    format_batch_qa_results_for_check, format_corpus_texts_for_check,
    # P15新增：批量Eval格式化函数
    format_batch_triples_for_eval,
    # P10新增：QA导师格式化函数
    format_mentor_guidance, format_feedbacks_for_revision, format_feedback_summary, format_joint_for_approval,
    format_eval_for_approval, format_label_for_approval, format_revision_feedbacks, format_reflection_for_approval,
    # P11新增：实体对齐格式化函数
    format_alignment_candidates, format_alignment_result_for_output,
    # P12新增：反思结果格式化函数
    format_dimension_scores, format_improvement_strategy,
)
from .nodes import (
    # P15新增：枚举值提取工具函数
    extract_enum_value, extract_enum_values_from_list,
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
    # P15新增：批量预处理节点
    create_batch_filter_node,
    create_batch_normalize_node,
    create_batch_qa_scaffold_node,
    process_batch_preprocessing,
    # P15新增：批量Self-Check节点
    create_batch_self_check_qa_node,
    # P15新增：批量Eval节点
    create_batch_eval_node,
    # P15新增：批量Self-Check-Eval节点
    create_batch_self_check_eval_node,
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
# P15修复：路由决策器（保留类作为未来迁移基础，删除未使用的v2函数）
from .route_decider import RouteDecider
from .workflow import (
    build_corpus_workflow,
    build_distributed_workflow,
    process_corpus,
    process_batch,
    process_corpus_streaming,
    process_batch_streaming,
    process_corpus_in_batches,
    # P10新增：QA导师工作流
    build_qa_mentor_workflow,
    process_corpus_with_qa_mentor,
)

__all__ = [
    # ===== 配置 =====
    "ExtractionConfig", "DEFAULT_CONFIG",
    
    # ===== 状态 =====
    "KGState", "WorkerResult", "CorpusState",
    "StepEnum", "PhaseEnum", "DEFAULT_MAX_RETRIES",
    "ENTITY_TYPES", "RELATION_TYPES", "ENTITY_CATEGORIES",
    # v3.4新增：默认实体字典
    "DEFAULT_ENTITY_DICT",
    # P15新增：状态工厂函数
    "create_default_corpus_state", "create_default_kg_state",

    # ===== v3.2重构新增：子状态类 =====
    "InputState", "ConfigState", "FilterState", "NormalizeState",
    "QAScaffoldState", "ExtractState", "EvalState", "SelfCheckState",
    "ReflexionState", "RetryState", "QAMentorState", "AlignmentState",
    "OutputState", "ControlState",
    
    # ===== v3.2精简版枚举 =====
    "RelationTypeEnum", "FunctionEnum", "FeatureTagEnum", "CompareDimensionEnum",
    "EventCategoryEnum", "EventStateEnum", "CrowdNodeEnum", "LimitNodeEnum",
    "DistanceValueEnum", "DirectionValueEnum", "EmotionNodeEnum", "RatingNodeEnum",
    "ConfidenceEnum",
    
    # ===== v3.2常量列表 =====
    "FUNCTION_NODES", "FEATURE_TAGS", "COMPARE_DIMENSIONS", "EVENT_CATEGORIES",
    
    # ===== v3.2重构新增：关系类型映射 =====
    "RELATION_VARIANT_MAPPING", "normalize_relation_type",
    # v3.4新增：关系属性映射
    "RELATION_ATTRS_MAP",

    # ===== v3.2重构新增：节点模板基类 =====
    "NodeTemplate", "RawTextNodeTemplate", "NoLLMNodeTemplate",
    "get_text_for_processing", "StateDict", "ResultDict", "NodeFunc",
    
    # ===== Pydantic模型 =====
    "EntityRecognitionResult", "RelationExtractionResult",
    "Triple", "TripleScore",
    "EvalResultFirst", "EvalResultSecond", "EvalResultSimplified",
    "Correction", "LabelResult",
    "EntityAttributes", "RelationAttributes",
    "QAPair", "QAScaffoldResult",
    "FilterResult", "NormalizationRecord", "NormalizeResult",
    "JointEntity", "JointTriple", "JointExtractionResult",
    "SelfCheckJointResult", "SelfCheckQAResult", "SelfCheckEvalResult", "SelfCheckLabelResult",
    # P12新增
    "DimensionScore", "ImprovementAction", "SelfCheckJointResultV2",
    "SelfCheckFilterResult", "SelfCheckNormalizeResult",
    "BatchCorpusResult", "BatchExtractionResult", "BatchSelfCheckResult",
    # P15新增：批量前置节点模型
    "BatchFilterItemResult", "BatchFilterResult",
    "BatchNormalizeItemResult", "BatchNormalizeResult",
    "BatchQAScaffoldItemResult", "BatchQAScaffoldResult",
    # P15新增：批量Self-Check节点模型
    "BatchSelfCheckQACorpusResult", "BatchSelfCheckQAResult",
    # P15新增：批量Eval节点模型
    "BatchEvalCorpusResult", "BatchEvalResult",
    "ApprovalStatusEnum", "ApprovalFeedback", "NodeApprovalResult", "QAApprovalResult",
    "MentorGuidance", "QAMentorScaffoldResult",
    "EntityCandidate", "EntityAlignmentItem", "EntityAlignmentResult",
    
    # ===== 提示词 =====
    "NER_PROMPT", "RE_PROMPT",
    "EVAL_PROMPT_1", "EVAL_PROMPT_2", "EVAL_PROMPT_SIMPLIFIED", "LABEL_PROMPT",
    "QA_SCAFFOLD_PROMPT", "FILTER_PROMPT", "NORMALIZE_PROMPT",
    "SELF_CHECK_NER_PROMPT", "SELF_CHECK_RE_PROMPT",
    "JOINT_NER_RE_PROMPT",
    "SELF_CHECK_JOINT_PROMPT", "SELF_CHECK_QA_PROMPT",
    "SELF_CHECK_EVAL_PROMPT", "SELF_CHECK_LABEL_PROMPT",
    "SELF_CHECK_FILTER_PROMPT", "SELF_CHECK_NORMALIZE_PROMPT",
    "BATCH_JOINT_PROMPT", "BATCH_SELF_CHECK_PROMPT",
    # P15新增：批量预处理提示词
    "BATCH_FILTER_PROMPT", "BATCH_NORMALIZE_PROMPT", "BATCH_QA_SCAFFOLD_PROMPT",
    # P15新增：批量Self-Check提示词
    "BATCH_SELF_CHECK_QA_PROMPT",
    # P15新增：批量Eval提示词
    "BATCH_EVAL_PROMPT",
    "QA_MENTOR_PROMPT", "QA_APPROVAL_PROMPT", "REVISION_JOINT_PROMPT",
    "ENTITY_ALIGNMENT_PROMPT",

    # ===== P12新增：模块化Schema组件 =====
    "ENTITY_SCHEMA_CORE", "RELATION_SCHEMA_CORE",
    "ENTITY_ATTRIBUTE_SCHEMA", "RELATION_ATTRIBUTE_SCHEMA",
    "VALIDATION_COT", "NEGATIVE_EXAMPLES", "EXPERT_ROLE_TEMPLATE",
    # v3.4新增：实体区分规则
    "ENTITY_DISTINCTION_RULES",

    # ===== P12新增：重构版提示词 =====
    "JOINT_NER_RE_PROMPT_V2", "SELF_CHECK_JOINT_PROMPT_V2",
    "assemble_joint_extraction_prompt",

    # ===== 格式化函数 =====
    "format_entities", "format_triples",
    "format_entity_hints", "format_relation_hints", "format_context_dependencies",
    "format_batch_corpus", "format_batch_results_for_check", "format_cross_corpus_aliases",
    # P15新增：批量QA校验格式化函数
    "format_batch_qa_results_for_check", "format_corpus_texts_for_check",
    # P15新增：批量Eval格式化函数
    "format_batch_triples_for_eval",
    "format_mentor_guidance", "format_feedbacks_for_revision", "format_feedback_summary", "format_joint_for_approval",
    "format_eval_for_approval", "format_label_for_approval", "format_revision_feedbacks", "format_reflection_for_approval",
    "format_alignment_candidates", "format_alignment_result_for_output",
    "format_dimension_scores", "format_improvement_strategy",

    # ===== 节点工厂函数 =====
    "extract_enum_value", "extract_enum_values_from_list",  # P15新增：枚举值提取工具
    # P15修复：路由决策器（保留类，删除未使用的v2函数）
    "RouteDecider",
    "create_ner_node", "create_re_node",
    "create_eval_1_node", "create_eval_2_node", "create_eval_simplified_node",
    "create_label_node", "create_coordinator_node", "create_aggregator_node",
    "create_qa_scaffold_node", "create_filter_node", "create_normalize_node",
    "create_self_check_ner_node", "create_self_check_re_node",
    "rule_based_validation",
    "create_joint_ner_re_node",
    "create_self_check_joint_node", "create_self_check_qa_node",
    "create_self_check_eval_node", "create_self_check_label_node",
    "create_self_check_filter_node", "create_self_check_normalize_node",
    "create_batch_joint_extraction_node", "create_batch_self_check_node",
    "process_corpus_batch_with_llm",
    # P15新增：批量预处理节点
    "create_batch_filter_node", "create_batch_normalize_node", "create_batch_qa_scaffold_node",
    "process_batch_preprocessing",
    # P15新增：批量Self-Check节点
    "create_batch_self_check_qa_node",
    # P15新增：批量Eval节点
    "create_batch_eval_node",
    "create_qa_mentor_node", "create_qa_approval_node", "create_revision_joint_node",
    "create_entity_alignment_node",
    # P13新增：优化版节点
    "create_joint_ner_re_node_v3", "create_filter_node_v3",
    "create_self_check_joint_node_v3", "create_re_node_v3", "create_label_node_v3",
    "get_node_creators",

    # ===== 工作流 =====
    "build_corpus_workflow", "build_distributed_workflow",
    "process_corpus", "process_batch",
    "process_corpus_streaming", "process_batch_streaming",
    "process_corpus_in_batches",
    "build_qa_mentor_workflow", "process_corpus_with_qa_mentor",
]