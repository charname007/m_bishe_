"""
知识图谱构建状态定义 - LangGraph TypedDict + Annotated reducer
"""
import time  # KGState 时间戳需要
from typing import TypedDict, List, Dict, Optional, Annotated, Any
from enum import Enum

# P15修复：从 schemas.py 导入 RELATION_TYPES，避免重复定义
from .schemas import RELATION_TYPES


# ===== Reducer函数 - 用于合并状态更新 =====

def merge_list(current: List, new: List) -> List:
    """合并列表（追加新元素）"""
    if new is None:
        return current
    return current + new


def merge_dict(current: Dict, new: Dict) -> Dict:
    """合并字典（更新键值）"""
    if new is None:
        return current
    return {**current, **new}


def replace_value(current: Any, new: Any) -> Any:
    """替换值（直接使用新值）"""
    return new if new is not None else current


# ===== 步骤/阶段枚举 =====

class StepEnum(str, Enum):
    """工作流步骤枚举"""
    FILTER = "filter"                       # P5新增：文本筛选
    SELF_CHECK_FILTER = "self_check_filter" # P9新增：筛选校验（可选）
    NORMALIZE = "normalize"                 # P6新增：文本归一化
    SELF_CHECK_NORMALIZE = "self_check_normalize" # P9新增：归一化校验（可选）
    QA_SCAFFOLD = "qa_scaffold"             # P8新增：QA脚手架
    QA_MENTOR = "qa_mentor"                 # P10新增：QA导师模式
    SELF_CHECK_QA = "self_check_qa"         # P9新增：QA校验
    JOINT_NER_RE = "joint_ner_re"           # P9新增：联合抽取
    SELF_CHECK_JOINT = "self_check_joint"   # P9新增：联合抽取校验（含Reflexion）
    QA_APPROVAL = "qa_approval"             # P10新增：QA审批节点
    REVISION_JOINT = "revision_joint"       # P10新增：修改联合抽取
    REVISION_EVAL = "revision_eval"         # P10新增：修改评估
    REVISION_LABEL = "revision_label"       # P10新增：修改标注
    ENTITY_ALIGNMENT = "entity_alignment"   # P11新增：实体对齐
    NER = "ner"                             # 流水线模式保留
    RE = "re"                               # 流水线模式保留
    SELF_CHECK_NER = "self_check_ner"       # 流水线模式：实体校验
    SELF_CHECK_RE = "self_check_re"         # 流水线模式：三元组校验
    EVAL = "eval"
    SELF_CHECK_EVAL = "self_check_eval"     # P9新增：评估校验
    LABEL = "label"
    SELF_CHECK_LABEL = "self_check_label"   # P9新增：标注校验
    DONE = "done"


class PhaseEnum(str, Enum):
    """分布式处理阶段枚举"""
    INIT = "init"
    MAP = "map"
    REDUCE = "reduce"
    EVALUATE = "evaluate"
    FINALIZE = "finalize"


# ===== 反思循环配置 =====

DEFAULT_MAX_RETRIES = 1  # 默认最大重试次数（P16优化：减少LLM调用成本）


# ===== 子状态定义（拆分 CorpusState） =====

class InputState(TypedDict):
    """输入状态 - 语料基本信息"""
    corpus_id: str
    raw_text: str


class ConfigState(TypedDict):
    """配置状态 - 路由函数使用的标记字段"""
    _config_enable_normalize: Annotated[bool, replace_value]
    """标记：是否启用了 Normalize 节点"""
    _config_enable_qa_scaffold: Annotated[bool, replace_value]
    """标记：是否启用了 QA Scaffold 节点"""
    _config_enable_entity_alignment: Annotated[bool, replace_value]
    """标记：是否启用了 Entity Alignment 节点（P15新增）"""


class FilterState(TypedDict):
    """筛选状态 - Filter 节点结果"""
    filter_result: Annotated[Dict, replace_value]
    """文本筛选结果：is_valid, skip_reason, confidence"""


class NormalizeState(TypedDict):
    """归一化状态 - Normalize 节点结果"""
    normalize_result: Annotated[Dict, replace_value]
    """文本归一化结果：normalized_text, normalizations, confidence"""
    normalized_text: Annotated[str, replace_value]
    """归一化后的文本（供后续节点使用）"""


class QAScaffoldState(TypedDict):
    """QA脚手架状态 - QA Scaffold 节点结果"""
    qa_scaffold_result: Annotated[Dict, replace_value]
    """QA脚手架结果：qa_pairs, semantic_summary, entity_hints, relation_hints"""
    semantic_summary: Annotated[str, replace_value]
    """语义摘要：整合问答后的文本理解"""
    qa_entity_hints: Annotated[List[str], replace_value]
    """实体提示：QA阶段发现的可能实体"""
    qa_relation_hints: Annotated[List[str], replace_value]
    """关系提示：QA阶段发现的可能关系类型"""
    qa_context_dependencies: Annotated[List[str], replace_value]
    """上下文依赖：需要注意的依赖关系"""


class ExtractState(TypedDict):
    """抽取状态 - NER/RE/联合抽取节点结果（v3.4扩展版）

    v3.4改进：
    - entities字典新增'功能'和'事件'键
    - 新增function_entities和event_entities列表存储详细属性

    ⚠️ **字段区别说明（P15新增）**：

    entities 与 function_entities/event_entities 的关系：
    ┌─────────────────────────────────────────────────────────────┐
    │ entities: Dict[str, List[str]]                              │
    │ - 轻量级存储：仅包含实体名称字符串                            │
    │ - 用于快速访问和统计                                         │
    │ - 示例：{"功能": ["餐饮", "购物"], "事件": ["樱花节"]}        │
    ├─────────────────────────────────────────────────────────────┤
    │ function_entities: List[Dict]                                │
    │ - 详细存储：包含完整的 FunctionEntityAttributes              │
    │ - 用于属性标注和数据库写入                                   │
    │ - 示例：[{name: "餐饮", attrs: {功能类型: "餐饮", ...}}]     │
    ├─────────────────────────────────────────────────────────────┤
    │ event_entities: List[Dict]                                   │
    │ - 详细存储：包含完整的 EventEntityAttributes                 │
    │ - 用于属性标注和数据库写入                                   │
    │ - 示例：[{name: "樱花节", attrs: {事件类别: "人文事件", ...}}]│
    └─────────────────────────────────────────────────────────────┘

    数据流向：
    1. Joint_NER_RE 抽取时同时填充 entities 和 function_entities/event_entities
    2. Label 节点从 function_entities/event_entities 读取详细属性
    3. 最终入库使用 function_entities/event_entities 中的完整数据

    其他实体类型（道路、POI、建筑物、街区）：
    - 仅存储在 entities 字典中（名称 + entity_attrs）
    - 无需单独的详细列表（属性较简单）
    """
    entities: Annotated[Dict[str, List[str]], replace_value]
    """实体识别结果：新增'功能'和'事件'键

    格式：{"道路": [...], "POI": [...], "建筑物": [...], "街区": [...], "功能": [...], "事件": [...]}
    用途：快速访问实体名称，统计各类型数量
    """
    triples: Annotated[List[Dict], replace_value]
    """关系抽取结果"""
    joint_extraction_result: Annotated[Dict, replace_value]
    """联合抽取结果（Joint NER+RE）"""
    extraction_strategy: Annotated[str, replace_value]
    """抽取策略标识：joint/pipeline"""
    entity_attrs: Annotated[Dict[str, Dict], replace_value]
    """实体属性

    格式：{"武汉大学": {类别: "POI", 细分: "大学", ...}}
    用途：存储空间实体（道路/POI/建筑物/街区）的属性
    """
    relation_attrs: Annotated[Dict[str, Dict], replace_value]
    """关系属性"""
    # v3.4新增：功能实体和事件实体详细列表
    function_entities: Annotated[List[Dict], merge_list]
    """功能实体详细列表（含属性）

    格式：[{name: "餐饮", function_attrs: {功能类型: "餐饮", 适合人群: ...}}]
    用途：存储语义实体（功能）的完整属性，用于Label节点和数据库写入
    区别：entities["功能"] 仅存储名称，此字段存储完整属性
    """
    event_entities: Annotated[List[Dict], merge_list]
    """事件实体详细列表（含属性）

    格式：[{name: "樱花节", event_attrs: {事件类别: "人文事件", 发生时间: ...}}]
    用途：存储语义实体（事件）的完整属性，用于Label节点和数据库写入
    区别：entities["事件"] 仅存储名称，此字段存储完整属性
    """


class EvalState(TypedDict):
    """评估状态 - Eval 节点结果"""
    eval_scores: Annotated[List[Dict], replace_value]
    """评分列表"""
    eval_passed: Annotated[bool, replace_value]
    """是否通过评估"""
    corrected_triples: Annotated[List[Dict], replace_value]
    """修正后的三元组"""


class SelfCheckState(TypedDict):
    """校验状态 - 所有 Self-Check 节点结果"""
    self_check_ner_result: Annotated[Dict, replace_value]
    """Self-Check-NER 校验结果"""
    self_check_re_result: Annotated[Dict, replace_value]
    """Self-Check-RE 校验结果"""
    self_check_filter_result: Annotated[Dict, replace_value]
    """Self-Check-Filter 校验结果（可选）"""
    self_check_normalize_result: Annotated[Dict, replace_value]
    """Self-Check-Normalize 校验结果（可选）"""
    self_check_qa_result: Annotated[Dict, replace_value]
    """Self-Check-QA 校验结果"""
    self_check_joint_result: Annotated[Dict, replace_value]
    """Self-Check-Joint 校验结果（含Reflexion）"""
    self_check_eval_result: Annotated[Dict, replace_value]
    """Self-Check-Eval 校验结果"""
    self_check_label_result: Annotated[Dict, replace_value]
    """Self-Check-Label 校验结果"""


class ReflexionState(TypedDict):
    """反思状态 - Reflexion 反思机制"""
    reflection_text: Annotated[str, replace_value]
    """自然语言反思建议"""
    improvement_strategy: Annotated[str, replace_value]
    """具体改进策略"""
    reflection_history: Annotated[List[str], merge_list]
    """多轮反思历史（用于迭代改进追踪）"""


class RetryState(TypedDict):
    """重试状态 - 反思循环控制"""
    retry_count: Annotated[int, replace_value]
    """当前重试次数"""
    max_retries: Annotated[int, replace_value]
    """最大重试次数"""
    retry_reason: Annotated[str, replace_value]
    """重试原因描述"""
    retry_suggested: Annotated[bool, replace_value]
    """是否建议重试（由Self-Check节点返回，供路由函数判断）"""
    problem_entities: Annotated[List[str], replace_value]
    """NER问题实体（遗漏的实体名）"""
    problem_triples: Annotated[List[Dict], replace_value]
    """RE问题三元组（幻觉/错误）"""
    needs_review: Annotated[bool, replace_value]
    """是否需要人工复核"""


class QAMentorState(TypedDict):
    """QA导师状态 - QA导师模式（P10新增）"""
    mentor_guidance: Annotated[Dict, replace_value]
    """导师指导信息：后续节点应遵循的指导"""
    qa_approval_result: Annotated[Dict, replace_value]
    """QA审批结果：对各节点结果的审批"""
    integrated_semantic_summary: Annotated[str, replace_value]
    """整合后的语义摘要：QA审批后更新的语义理解"""
    revision_feedbacks: Annotated[List[Dict], merge_list]
    """修改反馈列表：QA给出的改进建议"""
    revision_cycle_count: Annotated[int, replace_value]
    """修改循环计数：当前修改轮次"""
    max_revision_cycles: Annotated[int, replace_value]
    """最大修改轮次：默认3"""
    pending_approval_nodes: Annotated[List[str], merge_list]
    """待审批节点列表：哪些节点需要QA审批"""
    reasoning_trace: Annotated[str, replace_value]
    """推理过程：Reasoner模型的输出（可选保存）"""


class AlignmentState(TypedDict):
    """实体对齐状态 - Entity Alignment 节点结果（P11新增）"""
    entity_alignment_result: Annotated[Dict, replace_value]
    """实体对齐结果：与数据库已有实体的匹配情况"""
    aligned_entity_ids: Annotated[Dict[str, str], replace_value]
    """已对齐的实体ID映射：{抽取实体名: 数据库实体ID}"""
    new_entity_names: Annotated[List[str], replace_value]
    """新实体名称列表：未找到匹配的实体"""


class MentorQueryState(TypedDict):
    """导师查询状态 - 双向交流机制（P14新增）

    允许后续节点（Joint_NER_RE、Eval、Label）向 QA_Mentor 发起查询，
    实现双向交流而非单向状态传递。
    """
    mentor_query: Annotated[Optional[Dict], replace_value]
    """向导师发起的查询：{query_type, query_content, involved_entities, involved_relations}"""
    mentor_response: Annotated[Optional[Dict], replace_value]
    """导师的回答：{answer, updated_guidance, updated_entity_hints, updated_relation_hints}"""
    query_source_node: Annotated[Optional[str], replace_value]
    """问题来源节点：joint_ner_re / eval / label"""
    needs_mentor_help: Annotated[bool, replace_value]
    """是否需要导师帮助：路由函数判断标志"""
    query_count: Annotated[int, replace_value]
    """查询次数计数：防止无限循环"""
    max_queries: Annotated[int, replace_value]
    """最大查询次数：默认2"""
    return_to_node: Annotated[Optional[str], replace_value]
    """导师回答后返回的目标节点"""


class OutputState(TypedDict):
    """最终输出状态 - 校验/归一化后的结果"""
    final_entities: Annotated[List[Dict], replace_value]
    """最终实体列表（包含别名信息）"""
    final_triples: Annotated[List[Dict], replace_value]
    """最终三元组列表（校验后）"""
    verification_confidence: Annotated[str, replace_value]
    """整体置信度: high/medium/low"""


class ControlState(TypedDict):
    """流程控制状态 - 步骤和错误信息"""
    current_step: Annotated[StepEnum, replace_value]
    """当前步骤"""
    error: Annotated[Optional[str], replace_value]
    """错误信息"""


# ===== CorpusState 组合（通过多重继承） =====

class CorpusState(
    InputState,
    ConfigState,
    FilterState,
    NormalizeState,
    QAScaffoldState,
    ExtractState,
    EvalState,
    SelfCheckState,
    ReflexionState,
    RetryState,
    QAMentorState,
    AlignmentState,
    MentorQueryState,
    OutputState,
    ControlState,
):
    """
    单条语料处理状态 - 通过多重 TypedDict 继承组合

    LangGraph 兼容性说明：
    - TypedDict 多重继承会合并所有父类的字段
    - Annotated reducer 保持不变，LangGraph 正常识别
    - 向下兼容现有节点代码，无需修改

    子状态分组（共15个）：
    - InputState: corpus_id, raw_text
    - ConfigState: 配置标记字段
    - FilterState: filter_result
    - NormalizeState: normalize_result, normalized_text
    - QAScaffoldState: QA脚手架相关字段
    - ExtractState: entities, triples, joint_extraction_result, entity_attrs, relation_attrs
    - EvalState: eval_scores, eval_passed, corrected_triples
    - SelfCheckState: 8个 self_check_xxx_result 字段
    - ReflexionState: reflection_text, improvement_strategy, reflection_history
    - RetryState: retry_count, max_retries, problem_entities, problem_triples 等
    - QAMentorState: mentor_guidance, qa_approval_result 等（P10新增）
    - AlignmentState: entity_alignment_result, aligned_entity_ids, new_entity_names（P11新增）
    - MentorQueryState: mentor_query, mentor_response, query_source_node 等（P14新增）
    - OutputState: final_entities, final_triples, verification_confidence
    - ControlState: current_step, error

    字段总数：52个字段
    - 可通过 create_default_corpus_state() 工厂函数初始化
    """
    pass


# ===== P15新增：状态工厂函数 =====

def create_default_corpus_state(
    corpus_id: str,
    raw_text: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    enable_normalize: bool = False,
    enable_qa_scaffold: bool = False,
    enable_entity_alignment: bool = False,
    enable_mentor_query: bool = False,
    max_queries: int = 2,
    max_revision_cycles: int = 3,
) -> CorpusState:
    """
    创建默认 CorpusState 的工厂函数

    解决问题：手动初始化52个字段繁琐且容易遗漏

    Args:
        corpus_id: 语料ID
        raw_text: 原始文本
        max_retries: 最大重试次数（默认3）
        enable_normalize: 是否启用Normalize节点（设置配置标记）
        enable_qa_scaffold: 是否启用QA Scaffold节点
        enable_entity_alignment: 是否启用实体对齐节点
        enable_mentor_query: 是否启用导师查询功能
        max_queries: 最大导师查询次数（默认2）
        max_revision_cycles: 最大修改轮次（默认3）

    Returns:
        初始化后的 CorpusState 字典

    Example:
        >>> state = create_default_corpus_state("test_001", "武汉大学在珞喻路上")
        >>> assert state["corpus_id"] == "test_001"
        >>> assert state["retry_count"] == 0
    """
    return {
        # ===== InputState =====
        "corpus_id": corpus_id,
        "raw_text": raw_text,

        # ===== ConfigState =====
        "_config_enable_normalize": enable_normalize,
        "_config_enable_qa_scaffold": enable_qa_scaffold,
        "_config_enable_entity_alignment": enable_entity_alignment,  # P15新增

        # ===== FilterState =====
        "filter_result": {},

        # ===== NormalizeState =====
        "normalize_result": {},
        "normalized_text": "",

        # ===== QAScaffoldState =====
        "qa_scaffold_result": {},
        "semantic_summary": "",
        "qa_entity_hints": [],
        "qa_relation_hints": [],
        "qa_context_dependencies": [],

        # ===== ExtractState (v3.4扩展版) =====
        "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
        "triples": [],
        "joint_extraction_result": {},
        "extraction_strategy": "",
        "entity_attrs": {},
        "relation_attrs": {},
        "function_entities": [],  # v3.4新增
        "event_entities": [],     # v3.4新增

        # ===== EvalState =====
        "eval_scores": [],
        "eval_passed": False,
        "corrected_triples": [],

        # ===== SelfCheckState =====
        "self_check_ner_result": {},
        "self_check_re_result": {},
        "self_check_filter_result": {},
        "self_check_normalize_result": {},
        "self_check_qa_result": {},
        "self_check_joint_result": {},
        "self_check_eval_result": {},
        "self_check_label_result": {},
        "reflection_text": "",
        "improvement_strategy": "",
        "reflection_history": [],

        # ===== RetryState =====
        "retry_count": 0,
        "max_retries": max_retries,
        "retry_reason": "",
        "retry_suggested": False,
        "problem_entities": [],
        "problem_triples": [],
        "needs_review": False,

        # ===== QAMentorState =====
        "mentor_guidance": {},
        "qa_approval_result": {},
        "integrated_semantic_summary": "",
        "revision_feedbacks": [],
        "revision_cycle_count": 0,
        "max_revision_cycles": max_revision_cycles,
        "pending_approval_nodes": [],
        "reasoning_trace": "",

        # ===== AlignmentState =====
        "entity_alignment_result": {},
        "aligned_entity_ids": {},
        "new_entity_names": [],

        # ===== MentorQueryState =====
        "mentor_query": None,
        "mentor_response": None,
        "query_source_node": None,
        "needs_mentor_help": False,
        "query_count": 0,
        "max_queries": max_queries,
        "return_to_node": None,

        # ===== OutputState =====
        "final_entities": [],
        "final_triples": [],
        "verification_confidence": "medium",

        # ===== ControlState =====
        "current_step": StepEnum.NER,
        "error": None,
    }


def create_default_kg_state(
    batch_id: str,
    corpus_list: List[Dict],
    worker_count: int = 0,
) -> "KGState":
    """
    创建默认 KGState 的工厂函数

    Args:
        batch_id: 批次ID
        corpus_list: 语料列表
        worker_count: Worker数量

    Returns:
        初始化后的 KGState 字典
    """
    return {
        # ===== 输入数据 =====
        "batch_id": batch_id,
        "corpus_list": corpus_list,
        "total_count": len(corpus_list),

        # ===== MAP阶段 =====
        "worker_count": worker_count,
        "corpus_partitions": {},
        "worker_results": [],

        # ===== REDUCE阶段 =====
        "aggregated_entities": [],
        "aggregated_triples": [],
        "entity_aliases": {},

        # ===== FINALIZE阶段 =====
        "neo4j_stats": {},
        "postgres_stats": {},
        "error": None,

        # ===== 状态控制 =====
        "current_phase": PhaseEnum.INIT,
        "active_workers": [],
        "failed_workers": [],

        # ===== 元数据 =====
        "start_time": time.time(),
        "end_time": None,
    }


# ===== Worker处理结果 =====

class WorkerResult(TypedDict):
    """单个Worker的输出结果"""
    worker_id: str
    corpus_ids: List[str]
    results: List[CorpusState]
    processing_time: float
    error: Optional[str]


# ===== 分布式处理状态（整体工作流状态） =====

class KGState(TypedDict):
    """
    知识图谱构建整体状态 - LangGraph StateGraph使用

    使用Annotated reducer来处理列表和字典的合并：
    - merge_list: 追加新元素到列表
    - merge_dict: 更新字典键值
    - replace_value: 直接替换值
    """
    # ===== 输入数据 =====
    batch_id: Annotated[str, replace_value]
    corpus_list: Annotated[List[Dict], replace_value]
    total_count: Annotated[int, replace_value]

    # ===== MAP阶段：Worker并行处理 =====
    worker_count: Annotated[int, replace_value]
    corpus_partitions: Annotated[Dict[str, List[Dict]], replace_value]
    worker_results: Annotated[List[WorkerResult], merge_list]

    # ===== REDUCE阶段：聚合结果 =====
    aggregated_entities: Annotated[List[Dict], replace_value]
    aggregated_triples: Annotated[List[Dict], replace_value]
    entity_aliases: Annotated[Dict[str, List[str]], replace_value]

    # ===== FINALIZE阶段：结果输出 =====
    neo4j_stats: Annotated[Dict, replace_value]
    postgres_stats: Annotated[Dict, replace_value]
    error: Annotated[Optional[str], replace_value]  # 错误标记（None表示成功）

    # ===== 状态控制 =====
    current_phase: Annotated[PhaseEnum, replace_value]
    active_workers: Annotated[List[str], merge_list]
    failed_workers: Annotated[List[str], merge_list]

    # ===== 元数据 =====
    start_time: Annotated[float, replace_value]
    end_time: Annotated[Optional[float], replace_value]


# ===== 实体/关系类型定义 =====

# v3.4新增：默认实体字典（用于初始化和错误处理）
# 所有代码应统一使用此常量，确保实体类型一致性
DEFAULT_ENTITY_DICT = {
    "道路": [],
    "POI": [],
    "建筑物": [],
    "街区": [],
    "功能": [],
    "事件": [],
}

# v3.4扩展版：实体类型扩展为6种（新增功能、事件）
ENTITY_TYPES = {
    # 空间实体（GIS标准）—— 4种
    "道路": "街道、大道、小巷等（如：关山大道）",
    "POI": "具体店名、地标、机构（如：武汉大学、某某咖啡厅）",
    "建筑物": "具体的楼宇、商场主体（如：泛悦汇）",
    "街区": "具有边界感的生活区域（如：街道口、华农校区）",
    # 语义实体（v3.4新增）—— 2种
    "功能": "场所可进行的用途类型（如：餐饮、购物、休闲等）",
    "事件": "发生的具体事件（如：樱花节、封路、开业等）"
}

# RELATION_TYPES 已从 schemas.py 导入（P15修复）
# 原定义位置：schemas.py:968，与 RelationTypeEnum 同源

# v3.3改进：实体类别参考列表（仅供参考，非强制枚举）
# 细分采用开放文本设计，权威分类由数据源（高德POI）在对齐阶段补充
ENTITY_CATEGORIES = {
    "POI": ["餐饮", "交通", "教育", "历史保护", "购物", "医疗", "娱乐", "文化", "酒店", "服务"],
    "建筑物": ["商业综合体", "住宅", "办公楼", "文化设施", "教育设施", "医疗设施"],
    "街区": ["商圈", "校区", "社区", "行政区", "景区"],
    "道路": ["主干道", "次干道", "支路", "小巷", "地铁线路"]
}

# v2.2改进：人群节点枚举
CROWD_NODES = ["亲子/宝妈", "学生党", "情侣", "打工人", "特种兵", "银发族", "宠物主", "独行者", "团建"]

# v2.2改进：限制节点枚举
LIMIT_NODES = ["需预约", "排队久", "停车难", "限流", "谢绝宠物", "只收现金", "时间限制", "人数限制", "消费门槛", "季节限制"]

# v2.2改进：情感节点枚举
EMOTION_NODES = ["正面", "中性", "负面"]

# v2.2改进：评价等级枚举
RATING_NODES = ["超推", "推荐", "一般", "不推荐"]

# v3.2精简版：删除消费等级枚举（由外部商业数据补充）
# 原 CONSUMPTION_NODES 已删除

# v2.2改进：距离值枚举
DISTANCE_VALUES = ["近", "中等", "远"]

# v2.2改进：方向值枚举
DIRECTION_VALUES = ["东", "南", "西", "北", "东北", "西南", "东侧", "西侧", "对面", "旁边"]

# v3.2精简版：事件类别枚举（7个）
EVENT_CATEGORIES = ["自然事件", "人文事件", "商业活动", "社会事件", "业态变更", "停业/关闭", "其他"]

# v2.2改进：事件状态枚举
EVENT_STATES = ["正在进行", "已结束", "计划中", "周期性"]

# v3.2精简版：对比维度枚举（8个，v3.3新增"其他"）
COMPARE_DIMENSIONS = ["价格", "环境", "服务", "人流量", "品质", "交通", "口味", "其他"]

# v3.2精简版：情感标签已合并入特征标签，此枚举已删除
# 原 EMOTION_TAGS 内容已归入 FeatureTagEnum

# v3.2精简版：体验评价已合并入特征标签，此枚举已删除
# 原 EXPERIENCE_EVALUATIONS 内容已归入 FeatureTagEnum

# v3.2精简版：知名度已合并入特征标签，此枚举已删除
# 原 POPULARITY_LEVELS 内容已归入 FeatureTagEnum