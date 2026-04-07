"""
知识图谱构建状态定义
"""
from typing import TypedDict, List, Dict, Optional
from enum import Enum


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
    FINALIZE = "finalize"


# ===== 单条语料处理状态 =====
class CorpusState(TypedDict):
    """单条语料处理状态"""
    # 输入
    corpus_id: str                   # 语料ID
    raw_text: str                    # 原始文本

    # Step 1: NER结果
    entities: Dict[str, List[str]]   # {"道路": [...], "POI": [...], "建筑物": [...], "街区": [...]}

    # Step 2: RE结果
    triples: List[Dict]              # [{"head": ..., "relation": ..., "tail": ..., "evidence": ...}]

    # Step 3: 评估结果
    eval_scores: List[Dict]          # [{"triple": ..., "SEM": 1-5, "FAC": 1-5, "CON": 1-5}]
    eval_passed: bool                # 是否通过评估
    corrected_triples: List[Dict]    # 修正后的三元组

    # Step 4: 属性标注
    entity_attrs: Dict[str, Dict]    # {"武汉大学": {"类别": "POI", "细分": "教育"}}
    relation_attrs: Dict[str, str]   # 关系属性

    # 状态控制
    current_step: StepEnum
    error: Optional[str]


# ===== Worker处理结果 =====
class WorkerResult(TypedDict):
    """单个Worker的输出结果"""
    worker_id: str                   # Worker ID
    corpus_ids: List[str]            # 处理的语料ID列表
    results: List[CorpusState]       # 每条语料的处理结果
    processing_time: float           # 处理耗时
    error: Optional[str]             # 错误信息


# ===== 分布式处理状态 =====
class DistributedState(TypedDict):
    """分布式知识图谱构建状态"""

    # 输入
    batch_id: str                    # 批次ID
    corpus_list: List[Dict]          # 语料列表 [{"id": str, "text": str}, ...]
    total_count: int                 # 语料总数

    # MAP阶段
    worker_count: int                # Worker数量
    corpus_partitions: Dict[str, List[Dict]]  # 语料分片 {worker_id: [corpus_items]}
    worker_results: List[WorkerResult]        # Worker输出结果

    # REDUCE阶段
    aggregated_entities: List[Dict]           # 聚合后的实体
    aggregated_triples: List[Dict]            # 聚合后的三元组
    entity_aliases: Dict[str, List[str]]      # 实体别名 {"珞喻路": ["珞瑜路"]}

    # FINALIZE阶段
    neo4j_stats: Dict                         # Neo4j入库统计
    postgres_stats: Dict                      # PostgreSQL入库统计

    # 状态控制
    current_phase: PhaseEnum
    failed_workers: List[str]

    # 元数据
    start_time: float
    end_time: Optional[float]


# ===== 实体类型定义 =====
ENTITY_TYPES = {
    "道路": "街道、大道、小巷等（如：关山大道）",
    "POI": "具体店名、地标、机构（如：武汉大学、某某咖啡厅）",
    "建筑物": "具体的楼宇、商场主体（如：泛悦汇）",
    "街区": "具有边界感的生活区域（如：街道口、华农校区）"
}

# ===== 关系类型定义 =====
RELATION_TYPES = ["连接", "位于", "承载活动", "引发情感", "属于"]

# ===== 实体细分类别 =====
ENTITY_CATEGORIES = {
    "POI": ["餐饮", "交通", "教育", "历史保护", "购物", "医疗", "娱乐", "文化"],
    "建筑物": ["商业综合体", "住宅", "办公楼", "文化设施", "教育设施"],
    "街区": ["商圈", "校区", "社区", "行政区"],
    "道路": ["主干道", "次干道", "支路", "小巷"]
}