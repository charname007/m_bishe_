"""
智能体模块 - 基于LangGraph和LangChain
"""
from .state import (
    KGState, WorkerResult, CorpusState,
    StepEnum, PhaseEnum, ENTITY_TYPES, RELATION_TYPES, ENTITY_CATEGORIES
)
from .schemas import (
    EntityRecognitionResult,
    RelationExtractionResult,
    Triple, TripleScore,
    EvalResultFirst, EvalResultSecond,
    Correction, LabelResult,
    EntityAttributes, RelationAttributes,
)
from .prompts import (
    NER_PROMPT, RE_PROMPT,
    EVAL_PROMPT_1, EVAL_PROMPT_2, LABEL_PROMPT,
    format_entities, format_triples,
)
from .nodes import (
    create_ner_node,
    create_re_node,
    create_eval_1_node,
    create_eval_2_node,
    create_label_node,
    create_coordinator_node,
    create_aggregator_node,
)
from .workflow import (
    build_corpus_workflow,
    build_distributed_workflow,
    process_corpus,
    process_batch,
)

__all__ = [
    # 状态
    "KGState", "WorkerResult", "CorpusState",
    "StepEnum", "PhaseEnum", "ENTITY_TYPES", "RELATION_TYPES", "ENTITY_CATEGORIES",
    # Pydantic模型
    "EntityRecognitionResult",
    "RelationExtractionResult",
    "Triple", "TripleScore",
    "EvalResultFirst", "EvalResultSecond",
    "Correction", "LabelResult",
    "EntityAttributes", "RelationAttributes",
    # 提示词
    "NER_PROMPT", "RE_PROMPT",
    "EVAL_PROMPT_1", "EVAL_PROMPT_2", "LABEL_PROMPT",
    "format_entities", "format_triples",
    # 节点工厂函数
    "create_ner_node",
    "create_re_node",
    "create_eval_1_node",
    "create_eval_2_node",
    "create_label_node",
    "create_coordinator_node",
    "create_aggregator_node",
    # 工作流
    "build_corpus_workflow",
    "build_distributed_workflow",
    "process_corpus",
    "process_batch",
]