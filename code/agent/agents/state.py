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
    SELF_CHECK_QA = "self_check_qa"         # P9新增：QA校验
    JOINT_NER_RE = "joint_ner_re"           # P9新增：联合抽取
    SELF_CHECK_JOINT = "self_check_joint"   # P9新增：联合抽取校验（含Reflexion）
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


class CorpusState(TypedDict):
    """单条语料处理状态 - 用于单条语料的四步骤工作流"""
    # 输入
    corpus_id: str
    raw_text: str

    # P9新增：配置标记字段（用于路由函数判断后续节点是否启用）
    _config_enable_normalize: Annotated[bool, replace_value]
    """标记：是否启用了 Normalize 节点"""
    _config_enable_qa_scaffold: Annotated[bool, replace_value]
    """标记：是否启用了 QA Scaffold 节点"""

    # Step 0: Filter筛选结果（P5新增）
    filter_result: Annotated[Dict, replace_value]
    """文本筛选结果：is_valid, skip_reason, confidence"""

    # Step 0.5: Normalize归一化结果（P6新增）
    normalize_result: Annotated[Dict, replace_value]
    """文本归一化结果：normalized_text, normalizations, confidence"""
    normalized_text: Annotated[str, replace_value]
    """归一化后的文本（供后续节点使用）"""

    # Step 0.7: QA Scaffold结果（P8新增）
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

    # Step 1: NER结果
    entities: Annotated[Dict[str, List[str]], replace_value]

    # Step 2: RE结果
    triples: Annotated[List[Dict], replace_value]

    # Step 3: 评估结果
    eval_scores: Annotated[List[Dict], replace_value]
    eval_passed: Annotated[bool, replace_value]
    corrected_triples: Annotated[List[Dict], replace_value]

    # Step 3.5: Self-Check 结果（新增）
    self_check_ner_result: Annotated[Dict, replace_value]
    """Self-Check-NER 校验结果"""
    self_check_re_result: Annotated[Dict, replace_value]
    """Self-Check-RE 校验结果"""

    # P9新增：联合抽取结果
    joint_extraction_result: Annotated[Dict, replace_value]
    """联合抽取结果（Joint NER+RE）"""
    extraction_strategy: Annotated[str, replace_value]
    """抽取策略标识：joint/pipeline"""

    # P9新增：所有Self-Check结果
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

    # P9新增：Reflexion反思字段
    reflection_text: Annotated[str, replace_value]
    """自然语言反思建议"""
    improvement_strategy: Annotated[str, replace_value]
    """具体改进策略"""
    reflection_history: Annotated[List[str], merge_list]
    """多轮反思历史（用于迭代改进追踪）"""

    # 最终输出（校验/归一化后）
    final_entities: Annotated[List[Dict], replace_value]
    """最终实体列表（包含别名信息）"""
    final_triples: Annotated[List[Dict], replace_value]
    """最终三元组列表（校验后）"""
    verification_confidence: Annotated[str, replace_value]
    """整体置信度: high/medium/low"""

    # 反思循环控制（新增）
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

    # Step 4: 属性标注
    entity_attrs: Annotated[Dict[str, Dict], replace_value]
    relation_attrs: Annotated[Dict[str, Dict], replace_value]

    # 状态控制
    current_step: Annotated[StepEnum, replace_value]
    error: Annotated[Optional[str], replace_value]


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

# v2.2改进：18个关系类型
RELATION_TYPES = [
    # 空间基础关系（8个）
    "位于", "相邻", "属于", "连接", "距离", "方向", "穿过", "变化为",
    # 社交语义关系（6个）
    "推荐指数", "承载活动", "可达方式", "消费档次", "品类特征", "引发情感",
    # 对比评价关系（3个）
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

# v2.2改进：消费等级枚举
CONSUMPTION_NODES = ["平价", "中档", "高档", "奢侈"]

# v2.2改进：距离值枚举
DISTANCE_VALUES = ["近", "中等", "远"]

# v2.2改进：方向值枚举
DIRECTION_VALUES = ["东", "南", "西", "北", "东北", "西南", "东侧", "西侧", "对面", "旁边"]

# v2.2改进：事件类别枚举
EVENT_CATEGORIES = ["自然事件", "人文事件", "商业事件", "社会事件", "负面事件"]

# v2.2改进：事件状态枚举
EVENT_STATES = ["正在进行", "已结束", "计划中", "周期性"]

# v2.2改进：对比维度枚举
COMPARE_DIMENSIONS = ["价格", "环境", "服务", "人流量", "品质", "氛围", "交通", "停车", "口味", "性价比"]

# v2.2改进：情感标签枚举
EMOTION_TAGS = ["氛围感", "治愈", "高级感", "温暖", "文艺", "复古", "现代", "网红感", "小清新", "赛博朋克感"]

# v2.2改进：体验评价枚举
EXPERIENCE_EVALUATIONS = ["服务好", "环境舒适", "商品丰富", "性价比高", "停车方便", "交通便利", "人流量适中"]

# v2.2改进：知名度枚举
POPULARITY_LEVELS = ["热门", "小众", "隐藏宝藏", "必去", "打卡圣地"]