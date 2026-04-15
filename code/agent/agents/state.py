"""
知识图谱构建状态定义 - LangGraph TypedDict + Annotated reducer
"""
from typing import TypedDict, List, Dict, Optional, Annotated, Any
from enum import Enum


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

DEFAULT_MAX_RETRIES = 3  # 默认最大重试次数


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
    """抽取状态 - NER/RE/联合抽取节点结果"""
    entities: Annotated[Dict[str, List[str]], replace_value]
    """实体识别结果"""
    triples: Annotated[List[Dict], replace_value]
    """关系抽取结果"""
    joint_extraction_result: Annotated[Dict, replace_value]
    """联合抽取结果（Joint NER+RE）"""
    extraction_strategy: Annotated[str, replace_value]
    """抽取策略标识：joint/pipeline"""
    entity_attrs: Annotated[Dict[str, Dict], replace_value]
    """实体属性"""
    relation_attrs: Annotated[Dict[str, Dict], replace_value]
    """关系属性"""


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
    OutputState,
    ControlState,
):
    """
    单条语料处理状态 - 通过多重 TypedDict 继承组合
    
    LangGraph 兼容性说明：
    - TypedDict 多重继承会合并所有父类的字段
    - Annotated reducer 保持不变，LangGraph 正常识别
    - 向下兼容现有节点代码，无需修改
    
    子状态分组（共14个）：
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
    - OutputState: final_entities, final_triples, verification_confidence
    - ControlState: current_step, error
    """
    pass


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

    # ===== 状态控制 =====
    current_phase: Annotated[PhaseEnum, replace_value]
    active_workers: Annotated[List[str], merge_list]
    failed_workers: Annotated[List[str], merge_list]

    # ===== 元数据 =====
    start_time: Annotated[float, replace_value]
    end_time: Annotated[Optional[float], replace_value]


# ===== 实体/关系类型定义 =====

ENTITY_TYPES = {
    "道路": "街道、大道、小巷等（如：关山大道）",
    "POI": "具体店名、地标、机构（如：武汉大学、某某咖啡厅）",
    "建筑物": "具体的楼宇、商场主体（如：泛悦汇）",
    "街区": "具有边界感的生活区域（如：街道口、华农校区）"
}

# v3.2精简版：8个关系类型（空间基础3 + 社交语义1 + 对比评价3 + 事件1）
# 参考：docs/semantic_schema_v3.2.md
RELATION_TYPES = [
    # 空间基础关系（3个）—— 图谱骨架
    "位于", "包含", "方位",
    # 社交语义关系（1个）—— 图谱血肉
    "具有功能",
    # 对比评价关系（3个）—— 特色
    "优于", "相似", "劣于",
    # 事件关系（1个）
    "发生事件"
]

# v2.2改进：实体类别细分（扩展）
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

# v3.2精简版：对比维度枚举（7个）
COMPARE_DIMENSIONS = ["价格", "环境", "服务", "人流量", "品质", "交通", "口味"]

# v3.2精简版：情感标签已合并入特征标签，此枚举已删除
# 原 EMOTION_TAGS 内容已归入 FeatureTagEnum

# v3.2精简版：体验评价已合并入特征标签，此枚举已删除
# 原 EXPERIENCE_EVALUATIONS 内容已归入 FeatureTagEnum

# v3.2精简版：知名度已合并入特征标签，此枚举已删除
# 原 POPULARITY_LEVELS 内容已归入 FeatureTagEnum