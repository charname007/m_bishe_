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
    NER = "ner"
    RE = "re"
    EVAL = "eval"
    LABEL = "label"
    DONE = "done"


class PhaseEnum(str, Enum):
    """分布式处理阶段枚举"""
    INIT = "init"
    MAP = "map"
    REDUCE = "reduce"
    EVALUATE = "evaluate"
    FINALIZE = "finalize"


# ===== 单条语料处理状态（LangGraph节点使用） =====

class CorpusState(TypedDict):
    """单条语料处理状态 - 用于单条语料的四步骤工作流"""
    # 输入
    corpus_id: str
    raw_text: str

    # Step 1: NER结果
    entities: Annotated[Dict[str, List[str]], replace_value]

    # Step 2: RE结果
    triples: Annotated[List[Dict], replace_value]

    # Step 3: 评估结果
    eval_scores: Annotated[List[Dict], replace_value]
    eval_passed: Annotated[bool, replace_value]
    corrected_triples: Annotated[List[Dict], replace_value]

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
    cross_corpus_relations: Annotated[List[Dict], merge_list]

    # ===== EVALUATE阶段：质量评估 =====
    evaluator_results: Annotated[List[Dict], merge_list]
    high_confidence_triples: Annotated[List[Dict], merge_list]
    low_confidence_triples: Annotated[List[Dict], merge_list]

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
    total_tokens: Annotated[int, replace_value]


# ===== 实体/关系类型定义 =====

ENTITY_TYPES = {
    "道路": "街道、大道、小巷等（如：关山大道）",
    "POI": "具体店名、地标、机构（如：武汉大学、某某咖啡厅）",
    "建筑物": "具体的楼宇、商场主体（如：泛悦汇）",
    "街区": "具有边界感的生活区域（如：街道口、华农校区）"
}

RELATION_TYPES = ["连接", "位于", "承载活动", "引发情感", "属于"]

ENTITY_CATEGORIES = {
    "POI": ["餐饮", "交通", "教育", "历史保护", "购物", "医疗", "娱乐", "文化"],
    "建筑物": ["商业综合体", "住宅", "办公楼", "文化设施", "教育设施"],
    "街区": ["商圈", "校区", "社区", "行政区"],
    "道路": ["主干道", "次干道", "支路", "小巷"]
}