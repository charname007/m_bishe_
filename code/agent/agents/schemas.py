"""
Pydantic模型定义 - 用于LangChain with_structured_output
v2.2改进：适配新的18个关系体系和属性标注体系
"""
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum


# ===== 关系类型枚举（18个） =====

class RelationTypeEnum(str, Enum):
    """关系类型枚举"""
    # 空间基础关系（8个）
    LOCATED = "位于"
    ADJACENT = "相邻"
    BELONGS_TO = "属于"
    CONNECTS = "连接"
    DISTANCE = "距离"
    DIRECTION = "方向"
    CROSS = "穿过"
    CHANGED_TO = "变化为"

    # 社交语义关系（6个）
    RECOMMEND_INDEX = "推荐指数"
    HOSTS_ACTIVITY = "承载活动"
    ACCESSIBLE_BY = "可达方式"
    CONSUMPTION_LEVEL = "消费档次"
    CATEGORY_FEATURE = "品类特征"
    TRIGGERS_EMOTION = "引发情感"

    # 对比评价关系（3个）
    BETTER_THAN = "优于"
    SIMILAR_TO = "相似"
    WORSE_THAN = "劣于"

    # 事件关系（1个）
    HAS_EVENT = "发生事件"


# =====人群节点枚举 =====

class CrowdNodeEnum(str, Enum):
    """人群节点枚举"""
    FAMILY = "亲子/宝妈"
    STUDENT = "学生党"
    COUPLE = "情侣"
    WORKER = "打工人"
    SPECIAL_FORCE = "特种兵"
    SENIOR = "银发族"
    PET_OWNER = "宠物主"
    SOLO = "独行者"
    TEAM = "团建"


# ===== 限制节点枚举 =====

class LimitNodeEnum(str, Enum):
    """限制节点枚举"""
    NEED_RESERVATION = "需预约"
    LONG_QUEUE = "排队久"
    HARD_PARKING = "停车难"
    CAPACITY_LIMIT = "限流"
    NO_PETS = "谢绝宠物"
    CASH_ONLY = "只收现金"
    TIME_LIMIT = "时间限制"
    PEOPLE_LIMIT = "人数限制"
    MIN_CONSUMPTION = "消费门槛"
    SEASON_LIMIT = "季节限制"


# ===== 距离值枚举 =====

class DistanceValueEnum(str, Enum):
    """距离值枚举"""
    NEAR = "近"
    MEDIUM = "中等"
    FAR = "远"


# ===== 方向值枚举 =====

class DirectionValueEnum(str, Enum):
    """方向值枚举"""
    EAST = "东"
    SOUTH = "南"
    WEST = "西"
    NORTH = "北"
    NORTHEAST = "东北"
    SOUTHWEST = "西南"
    EAST_SIDE = "东侧"
    WEST_SIDE = "西侧"
    OPPOSITE = "对面"
    BESIDE = "旁边"


# ===== 情感节点枚举 =====

class EmotionNodeEnum(str, Enum):
    """情感节点枚举"""
    POSITIVE = "正面"
    NEUTRAL = "中性"
    NEGATIVE = "负面"


# ===== 评价等级枚举 =====

class RatingNodeEnum(str, Enum):
    """评价等级枚举"""
    SUPER_RECOMMEND = "超推"
    RECOMMEND = "推荐"
    ORDINARY = "一般"
    NOT_RECOMMEND = "不推荐"


# ===== 消费等级枚举 =====

class ConsumptionNodeEnum(str, Enum):
    """消费等级枚举"""
    AFFORDABLE = "平价"
    MID_RANGE = "中档"
    HIGH_END = "高档"
    LUXURY = "奢侈"


# ===== 事件类别枚举 =====

class EventCategoryEnum(str, Enum):
    """事件类别枚举"""
    NATURAL = "自然事件"
    CULTURAL = "人文事件"
    COMMERCIAL = "商业事件"
    SOCIAL = "社会事件"
    NEGATIVE = "负面事件"


# ===== 事件状态枚举 =====

class EventStateEnum(str, Enum):
    """事件状态枚举"""
    ONGOING = "正在进行"
    ENDED = "已结束"
    PLANNED = "计划中"
    PERIODIC = "周期性"


# ===== 置信度枚举 =====

class ConfidenceEnum(str, Enum):
    """置信度枚举"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ===== Filter阶段输出模型 =====

class FilterResult(BaseModel):
    """文本筛选结果 - 判断是否包含有价值的地理信息"""
    is_valid: bool = Field(
        default=True,
        description="是否包含有价值的地理信息，值得继续处理"
    )
    skip_reason: Optional[str] = Field(
        default=None,
        description="跳过原因（仅当 is_valid=False 时有效）"
    )
    confidence: str = Field(
        default="medium",
        description="判断置信度: high/medium/low"
    )
    has_geo_entity: bool = Field(
        default=False,
        description="是否提及地理实体（道路、POI、建筑物、街区）"
    )
    has_spatial_relation: bool = Field(
        default=False,
        description="是否涉及空间关系（位于、旁边、连接等）"
    )
    geo_entity_hint: Optional[str] = Field(
        default=None,
        description="检测到的地理实体提示（如有）"
    )
    # P9改进：武汉地区筛选
    is_non_wuhan_region: bool = Field(
        default=False,
        description="是否明确只包含武汉以外的地区（北京、上海、广州等），无法确定时为False"
    )
    region_hint: Optional[str] = Field(
        default=None,
        description="地区提示：武汉/非武汉/未知"
    )


# ===== Normalize阶段输出模型 =====

class NormalizationRecord(BaseModel):
    """单条归一化记录"""
    raw: str = Field(description="原始文本片段")
    normalized: str = Field(description="归一化后的文本")
    type: str = Field(description="归一化类型: alias/reference/activity/other")
    confidence: str = Field(default="high", description="归一化置信度")


class NormalizeResult(BaseModel):
    """文本归一化结果 - 指代消解和语义标准化"""
    normalized_text: str = Field(
        description="归一化后的完整文本"
    )
    normalizations: List[NormalizationRecord] = Field(
        default_factory=list,
        description="归一化记录列表"
    )
    confidence: str = Field(
        default="medium",
        description="整体归一化置信度: high/medium/low"
    )
    preserved_semantics: bool = Field(
        default=True,
        description="是否保留了原文语义（不添加新信息）"
    )
    has_changes: bool = Field(
        default=False,
        description="是否有实质性改动"
    )


# ===== QA Scaffold阶段输出模型（P8新增） =====

class QAPair(BaseModel):
    """单个问答对 - 5W1H引导生成的问答"""
    question: str = Field(description="5W1H引导问题")
    answer: str = Field(description="基于原文的回答")
    dimension: str = Field(description="维度标签: who/what/when/where/why/how")
    entities_involved: List[str] = Field(
        default_factory=list,
        description="涉及到的实体名称"
    )
    confidence: str = Field(default="medium", description="回答置信度: high/medium/low")


class QAScaffoldResult(BaseModel):
    """QA脚手架输出 - 5W1H问答扩展构建语义脚手架"""
    qa_pairs: List[QAPair] = Field(
        default_factory=list,
        description="5W1H问答对列表"
    )
    semantic_summary: str = Field(
        default="",
        description="语义摘要：整合问答后的文本理解"
    )
    entity_hints: List[str] = Field(
        default_factory=list,
        description="实体提示列表：可能涉及的地理实体"
    )
    relation_hints: List[str] = Field(
        default_factory=list,
        description="关系提示列表：可能存在的关系类型"
    )
    context_dependencies: List[str] = Field(
        default_factory=list,
        description="上下文依赖：需要后续节点注意的依赖关系"
    )
    overall_confidence: str = Field(
        default="medium",
        description="整体脚手架置信度: high/medium/low"
    )
    should_skip_detailed_extraction: bool = Field(
        default=False,
        description="是否建议跳过详细抽取（简单文本无地理信息）"
    )


# ===== NER阶段输出模型 =====

class EntityRecognitionResult(BaseModel):
    """命名实体识别结果"""
    道路: List[str] = Field(default_factory=list, description="道路实体列表")
    POI: List[str] = Field(default_factory=list, description="POI兴趣点列表")
    建筑物: List[str] = Field(default_factory=list, description="建筑物实体列表")
    街区: List[str] = Field(default_factory=list, description="街区实体列表")


# ===== RE阶段输出模型（v2.2改进：添加attributes字段） =====

class TripleAttributes(BaseModel):
    """三元组属性（RE阶段直接抽取，硬校验枚举值+拒绝额外字段）"""
    model_config = ConfigDict(extra='forbid')  # 拒绝未定义的字段

    # 空间关系属性
    联动推荐: Optional[bool] = Field(
        default=None,
        description="相邻关系的联动推荐属性（布尔）"
    )
    距离值: Optional[DistanceValueEnum] = Field(
        default=None,
        description="距离关系的距离值：近/中等/远"
    )
    方向值: Optional[DirectionValueEnum] = Field(
        default=None,
        description="方向关系的方向值：东/南/西/北/东北/西南/东侧/西侧/对面/旁边"
    )
    变化时间: Optional[str] = Field(
        default=None,
        description="变化为关系的变化时间"
    )

    # 社交语义关系属性
    时段: Optional[str] = Field(
        default=None,
        description="承载活动关系的时段属性"
    )
    适合人群: Optional[CrowdNodeEnum] = Field(
        default=None,
        description="承载活动关系的适合人群属性"
    )
    具有限制: Optional[List[LimitNodeEnum]] = Field(
        default=None,
        description="承载活动关系的限制节点列表"
    )
    具体表达: Optional[str] = Field(
        default=None,
        description="引发情感关系的具体情感表达"
    )

    # 对比关系属性
    维度: Optional[List[str]] = Field(
        default=None,
        description="优于/相似/劣于关系的维度列表"
    )

    # 事件关系属性
    事件类别: Optional[EventCategoryEnum] = Field(
        default=None,
        description="发生事件关系的类别"
    )
    状态: Optional[EventStateEnum] = Field(
        default=None,
        description="发生事件关系的状态"
    )
    时间: Optional[str] = Field(
        default=None,
        description="发生事件关系的时间节点"
    )

    # ===== 值映射校验器（处理LLM输出的变体值） =====

    @field_validator('距离值', mode='before')
    @classmethod
    def normalize_distance(cls, v):
        """将距离值变体映射到标准枚举值"""
        if v is None:
            return None
        if isinstance(v, DistanceValueEnum):
            return v
        # 变体值映射
        distance_mapping = {
            '近': '近', '很近': '近', '附近': '近', '旁边': '近', '不远': '近',
            '中等': '中等', '有点远': '中等', '稍微远': '中等', '几百米': '中等',
            '远': '远', '很远': '远', '比较远': '远', '距离较远': '远',
        }
        normalized = distance_mapping.get(str(v))
        if normalized:
            return DistanceValueEnum(normalized)
        raise ValueError(f"无效的距离值: {v}，有效值为: 近/中等/远")

    @field_validator('方向值', mode='before')
    @classmethod
    def normalize_direction(cls, v):
        """将方向值变体映射到标准枚举值"""
        if v is None:
            return None
        if isinstance(v, DirectionValueEnum):
            return v
        # 方向映射（包含变体）
        direction_mapping = {
            '东': '东', '东边': '东', '东侧': '东侧',
            '南': '南', '南边': '南',
            '西': '西', '西边': '西', '西侧': '西侧',
            '北': '北', '北边': '北',
            '东北': '东北',
            '西南': '西南',
            '对面': '对面', '对面是': '对面',
            '旁边': '旁边', '旁边是': '旁边',
        }
        normalized = direction_mapping.get(str(v))
        if normalized:
            return DirectionValueEnum(normalized)
        raise ValueError(f"无效的方向值: {v}，有效值为: 东/南/西/北/东北/西南/东侧/西侧/对面/旁边")

    @field_validator('适合人群', mode='before')
    @classmethod
    def normalize_crowd(cls, v):
        """将人群变体映射到标准枚举值"""
        if v is None:
            return None
        if isinstance(v, CrowdNodeEnum):
            return v
        # 人群映射（包含变体）
        crowd_mapping = {
            '亲子': '亲子/宝妈', '宝妈': '亲子/宝妈', '带孩子': '亲子/宝妈', '亲子/宝妈': '亲子/宝妈',
            '学生党': '学生党', '学生': '学生党', '大学生': '学生党',
            '情侣': '情侣', '约会': '情侣',
            '打工人': '打工人', '上班族': '打工人',
            '特种兵': '特种兵',
            '银发族': '银发族', '老人': '银发族', '老年人': '银发族',
            '宠物主': '宠物主', '带宠物': '宠物主', '遛狗': '宠物主',
            '独行者': '独行者', '独自': '独行者',
            '团建': '团建', '聚会': '团建',
        }
        normalized = crowd_mapping.get(str(v))
        if normalized:
            return CrowdNodeEnum(normalized)
        raise ValueError(f"无效的人群值: {v}")

    @field_validator('具有限制', mode='before')
    @classmethod
    def normalize_limits(cls, v):
        """将限制变体映射到标准枚举值列表"""
        if v is None:
            return None
        if isinstance(v, list) and all(isinstance(item, LimitNodeEnum) for item in v):
            return v
        # 限制映射
        limit_mapping = {
            '需预约': '需预约', '要预约': '需预约', '预约难': '需预约',
            '排队久': '排队久', '排队': '排队久', '排队很长': '排队久',
            '停车难': '停车难', '没车位': '停车难', '停车要排队': '停车难',
            '限流': '限流', '人太多': '限流',
            '谢绝宠物': '谢绝宠物', '不能带宠物': '谢绝宠物',
            '只收现金': '只收现金',
            '时间限制': '时间限制',
            '人数限制': '人数限制',
            '消费门槛': '消费门槛', '最低消费': '消费门槛',
            '季节限制': '季节限制',
        }
        if isinstance(v, list):
            normalized = []
            for item in v:
                mapped = limit_mapping.get(str(item))
                if mapped:
                    normalized.append(LimitNodeEnum(mapped))
                else:
                    raise ValueError(f"无效的限制值: {item}")
            return normalized
        # 单个值
        mapped = limit_mapping.get(str(v))
        if mapped:
            return [LimitNodeEnum(mapped)]
        raise ValueError(f"无效的限制值: {v}")

    @field_validator('事件类别', mode='before')
    @classmethod
    def normalize_event_category(cls, v):
        """将事件类别变体映射到标准枚举值"""
        if v is None:
            return None
        if isinstance(v, EventCategoryEnum):
            return v
        # 事件类别映射
        category_mapping = {
            '自然事件': '自然事件', '樱花盛开': '自然事件', '荷花盛开': '自然事件',
            '人文事件': '人文事件', '樱花节': '人文事件', '音乐节': '人文事件', '夜市': '人文事件', '展览': '人文事件',
            '商业事件': '商业事件', '开业': '商业事件', '打折': '商业事件', '促销': '商业事件',
            '社会事件': '社会事件', '施工': '社会事件', '装修': '社会事件', '关闭': '社会事件',
            '负面事件': '负面事件', '停业': '负面事件', '整顿': '负面事件',
        }
        normalized = category_mapping.get(str(v))
        if normalized:
            return EventCategoryEnum(normalized)
        raise ValueError(f"无效的事件类别: {v}")

    @field_validator('状态', mode='before')
    @classmethod
    def normalize_event_state(cls, v):
        """将事件状态变体映射到标准枚举值"""
        if v is None:
            return None
        if isinstance(v, EventStateEnum):
            return v
        # 状态映射
        state_mapping = {
            '正在进行': '正在进行', '正在举办': '正在进行', '进行中': '正在进行',
            '已结束': '已结束', '结束了': '已结束',
            '计划中': '计划中', '即将举办': '计划中', '快开了': '计划中',
            '周期性': '周期性', '每年': '周期性', '周末': '周期性',
        }
        normalized = state_mapping.get(str(v))
        if normalized:
            return EventStateEnum(normalized)
        raise ValueError(f"无效的事件状态: {v}")


class Triple(BaseModel):
    """单个三元组（v2.2改进：硬校验枚举值+强类型属性）"""
    head: str = Field(description="头实体名称")
    relation: RelationTypeEnum = Field(description="关系类型（18种之一，硬校验）")
    tail: str = Field(description="尾实体名称或枚举节点")
    evidence: Optional[str] = Field(default="", description="文本证据")
    attributes: Optional[TripleAttributes] = Field(
        default=None,
        description="关系属性（强类型约束，根据关系类型不同）"
    )

    @field_validator('relation', mode='before')
    @classmethod
    def normalize_relation(cls, v):
        """将关系类型变体映射到标准枚举值"""
        if v is None:
            raise ValueError("relation 不能为空")
        if isinstance(v, RelationTypeEnum):
            return v
        # 关系类型映射（包含常见变体）
        relation_mapping = {
            '位于': '位于', '在': '位于', '在...上': '位于', '地处': '位于',
            '相邻': '相邻', '旁边': '相邻', '旁边是': '相邻', '隔壁': '相邻',
            '属于': '属于', '隶属于': '属于', '是...的一部分': '属于',
            '连接': '连接', '连通': '连接', '通往': '连接',
            '距离': '距离', '离': '距离', '距离...很近': '距离', '附近': '距离',
            '方向': '方向', '在...东边': '方向', '东边': '方向',
            '穿过': '穿过', '横穿': '穿过', '穿越': '穿过',
            '变化为': '变化为', '变成': '变化为', '改为': '变化为',
            '推荐指数': '推荐指数', '推荐': '推荐指数', '强烈推荐': '推荐指数',
            '承载活动': '承载活动', '可以...': '承载活动', '适合...': '承载活动',
            '可达方式': '可达方式', '交通': '可达方式', '怎么去': '可达方式',
            '消费档次': '消费档次', '消费': '消费档次', '人均': '消费档次',
            '品类特征': '品类特征', '特色': '品类特征', '风格': '品类特征',
            '引发情感': '引发情感', '情感': '引发情感', '感觉': '引发情感',
            '优于': '优于', '比...好': '优于', '比...便宜': '优于',
            '相似': '相似', '和...差不多': '相似', '类似': '相似',
            '劣于': '劣于', '不如': '劣于', '比...差': '劣于',
            '发生事件': '发生事件', '有': '发生事件', '正在': '发生事件',
        }
        normalized = relation_mapping.get(str(v))
        if normalized:
            return RelationTypeEnum(normalized)
        raise ValueError(f"无效的关系类型: {v}，有效值为18个预定义关系类型")


class RelationExtractionResult(BaseModel):
    """关系抽取结果（v2.2改进）"""
    triples: List[Triple] = Field(default_factory=list, description="抽取的三元组列表")


# ===== Eval阶段输出模型 =====

class TripleForEval(BaseModel):
    """用于评估的三元组（硬校验）"""
    head: str = Field(description="头实体名称")
    relation: RelationTypeEnum = Field(description="关系类型")
    tail: str = Field(description="尾实体名称")


class TripleScore(BaseModel):
    """单个三元组的评分"""
    triple: TripleForEval = Field(description="被评分的三元组")
    SEM: int = Field(ge=1, le=5, description="语义准确性评分(1-5)")
    FAC: int = Field(ge=1, le=5, description="事实真实性评分(1-5)")
    CON: int = Field(ge=1, le=5, description="一致性评分(1-5)")


class EvalResultFirst(BaseModel):
    """第一次评估结果"""
    scores: List[TripleScore] = Field(default_factory=list, description="评分列表")


class Correction(BaseModel):
    """三元组修正"""
    original: TripleForEval = Field(description="原始三元组")
    corrected: TripleForEval = Field(description="修正后的三元组")
    reason: str = Field(description="修正原因")


class EvalResultSecond(BaseModel):
    """第二次评估结果（自检）"""
    need_correction: bool = Field(description="是否需要修正")
    corrections: List[Correction] = Field(default_factory=list, description="修正列表")
    final_scores: List[TripleScore] = Field(default_factory=list, description="最终评分")


class EvalResultSimplified(BaseModel):
    """简化的单次评估结果 - 包含评分和可选修正"""
    scores: List[TripleScore] = Field(default_factory=list, description="评分列表")
    need_correction: bool = Field(default=False, description="是否需要修正")
    corrections: List[Correction] = Field(default_factory=list, description="修正列表（仅当need_correction=True时有效）")


# ===== Label阶段输出模型（v2.2改进：扩展实体属性） =====

class EntityAttributes(BaseModel):
    """实体属性（v2.2改进：扩展情感标签、体验评价、知名度）"""
    # 基础分类属性（GIS标准）
    类别: str = Field(description="实体类别：POI/建筑物/街区/道路")
    细分: str = Field(description="细分类别")

    # 文本属性（从语料提取）
    情感标签: List[str] = Field(
        default_factory=list,
        description="情感标签：氛围感/治愈/高级感/温暖/文艺/复古/现代/网红感/小清新"
    )
    体验评价: List[str] = Field(
        default_factory=list,
        description="体验评价：服务好/环境舒适/商品丰富/性价比高/停车方便/交通便利"
    )
    知名度: str = Field(
        default="",
        description="知名度：热门/小众/隐藏宝藏/必去/打卡圣地"
    )

    # 元数据属性（质量控制）
    来源可信度: str = Field(
        default="中",
        description="来源可信度：高/中/低"
    )


class RelationAttributes(BaseModel):
    """关系属性（v2.2改进：根据关系类型定义）"""
    # 空间精度（通用）
    空间精度: Optional[str] = Field(
        default=None,
        description="空间精度：精确/近似/模糊"
    )

    # 语义类型（位于关系）
    语义类型: Optional[str] = Field(
        default=None,
        description="语义类型：内部/区域内/附近/周边"
    )

    # 相邻类型（相邻关系）
    相邻类型: Optional[str] = Field(
        default=None,
        description="相邻类型：直接相邻/邻近/隔街相望"
    )

    # 层级类型（属于关系）
    层级类型: Optional[str] = Field(
        default=None,
        description="层级类型：组成部分/行政隶属/功能隶属"
    )

    # 连接属性（连接关系）
    连接类型: Optional[str] = Field(
        default=None,
        description="连接类型：直达/换乘/途径/沿线"
    )
    交通方式: Optional[str] = Field(
        default=None,
        description="交通方式：地铁/公交/步行/自驾/骑行"
    )

    # 距离属性（距离关系）
    距离类型: Optional[str] = Field(
        default=None,
        description="距离类型：物理距离/感知距离/步行距离"
    )

    # 方向属性（方向关系）
    方向类型: Optional[str] = Field(
        default=None,
        description="方向类型：绝对方位/相对方位/定性方位"
    )

    # 穿过属性（穿过关系）
    穿过类型: Optional[str] = Field(
        default=None,
        description="穿过类型：横穿/纵穿/穿越"
    )

    # 变化属性（变化为关系）
    变化类型: Optional[str] = Field(
        default=None,
        description="变化类型：业态变更/功能转变/建筑改造/关闭拆除"
    )

    # 推荐属性（推荐指数关系）
    推荐强度: Optional[str] = Field(
        default=None,
        description="推荐强度：强烈/一般/较弱"
    )
    推荐场景: Optional[str] = Field(
        default=None,
        description="推荐场景：日常/周末/节假日/约会/团建"
    )

    # 活动属性（承载活动关系）
    活动类型: Optional[str] = Field(
        default=None,
        description="活动类型：体验型/消费型/社交型/休闲型/观赏型"
    )
    活动频率: Optional[str] = Field(
        default=None,
        description="活动频率：高频/中频/低频/季节性"
    )

    # 可达属性（可达方式关系）
    可达程度: Optional[str] = Field(
        default=None,
        description="可达程度：直达/换乘/需步行/不便"
    )
    交通效率: Optional[str] = Field(
        default=None,
        description="交通效率：高效/一般/低效"
    )

    # 消费属性（消费档次关系）
    价格区间: Optional[str] = Field(
        default=None,
        description="价格区间（补充具体数值）：如人均50-100"
    )
    消费类型: Optional[str] = Field(
        default=None,
        description="消费类型：日常消费/休闲消费/高端消费"
    )

    # 品类属性（品类特征关系）
    特征类型: Optional[str] = Field(
        default=None,
        description="特征类型：风格特征/文化特征/历史特征/功能特征"
    )
    特征显著性: Optional[str] = Field(
        default=None,
        description="特征显著性：显著/一般/微弱"
    )

    # 情感属性（引发情感关系）
    情感强度: Optional[str] = Field(
        default=None,
        description="情感强度：强烈/一般/微弱"
    )
    情感类型: Optional[str] = Field(
        default=None,
        description="情感类型：愉悦型/放松型/感动型/浪漫型/负面型"
    )

    # 对比属性（优于/相似/劣于关系）
    优势程度: Optional[str] = Field(
        default=None,
        description="优势程度（优于关系）：明显优势/稍有优势/相当"
    )
    相似程度: Optional[str] = Field(
        default=None,
        description="相似程度（相似关系）：高度相似/部分相似/风格相近"
    )
    劣势程度: Optional[str] = Field(
        default=None,
        description="劣势程度（劣于关系）：明显劣势/稍有劣势/相当"
    )
    对比可靠性: Optional[str] = Field(
        default=None,
        description="对比可靠性：主观对比/客观对比"
    )
    替代性: Optional[str] = Field(
        default=None,
        description="替代性（相似关系）：可替代/部分替代/不可替代"
    )
    风险等级: Optional[str] = Field(
        default=None,
        description="风险等级（劣于关系）：高风险/中风险/低风险"
    )

    # 事件属性（发生事件关系）
    事件影响度: Optional[str] = Field(
        default=None,
        description="事件影响度：重大影响/一般影响/微弱影响"
    )
    事件持续性: Optional[str] = Field(
        default=None,
        description="事件持续性：长期事件/短期事件/周期性事件"
    )

    # 元数据属性（通用）
    来源可信度: str = Field(
        default="中",
        description="来源可信度：高/中/低"
    )


class LabelResult(BaseModel):
    """属性标注结果（v2.2改进）"""
    entities: Dict[str, EntityAttributes] = Field(
        default_factory=dict,
        description="实体属性字典，键为实体名"
    )
    relations: Dict[str, RelationAttributes] = Field(
        default_factory=dict,
        description="关系属性字典，键为三元组字符串如'<A, 关系, B>'"
    )


# ===== Self-Check阶段输出模型 =====

class VerifiedEntity(BaseModel):
    """校验后的实体"""
    name: str = Field(description="实体名称（归一化后）")
    type: str = Field(description="实体类型：道路/POI/建筑物/街区")
    confidence: str = Field(description="置信度: high/medium/low")
    aliases: List[str] = Field(default_factory=list, description="别名/简称列表")
    evidence: Optional[str] = Field(default="", description="原文依据")


class MissingEntity(BaseModel):
    """遗漏实体建议"""
    name: str = Field(description="建议补充的实体名")
    suggested_type: str = Field(description="建议类型：道路?/POI?/建筑物?/街区?")
    reason: str = Field(description="遗漏原因/原文依据")


class EntityNormalization(BaseModel):
    """实体归一化记录"""
    raw: str = Field(description="原始名称（如'武大'）")
    canonical: str = Field(description="归一化名称（如'武汉大学'）")
    confidence: str = Field(description="归一化置信度: high/medium/low")


class SelfCheckNERResult(BaseModel):
    """Self-Check-NER 输出"""
    verified_entities: List[VerifiedEntity] = Field(
        default_factory=list,
        description="校验通过的实体列表"
    )
    missing_entities: List[MissingEntity] = Field(
        default_factory=list,
        description="遗漏实体建议列表"
    )
    entity_normalizations: List[EntityNormalization] = Field(
        default_factory=list,
        description="实体归一化映射列表"
    )
    removed_entities: List[str] = Field(
        default_factory=list,
        description="过滤掉的无关实体（非地理实体）"
    )
    overall_confidence: str = Field(
        default="medium",
        description="整体置信度: high/medium/low"
    )


class VerifiedTriple(BaseModel):
    """校验后的三元组"""
    head: str = Field(description="头实体")
    relation: str = Field(description="关系类型")
    tail: str = Field(description="尾实体")
    confidence: str = Field(description="置信度: high/medium/low")
    evidence_valid: bool = Field(description="证据是否有效")
    evidence_match: Optional[str] = Field(default="", description="证据原文匹配位置")


class RejectedTriple(BaseModel):
    """拒绝的三元组（幻觉或错误）"""
    head: str = Field(description="头实体")
    relation: str = Field(description="关系类型")
    tail: str = Field(description="尾实体")
    reason: str = Field(description="拒绝原因：幻觉/方向错误/证据无效")
    suggested_fix: Optional[str] = Field(default="", description="修正建议")


class TripleCorrectionForSelfCheck(BaseModel):
    """三元组修正记录（Self-Check专用）"""
    original_head: str = Field(description="原始头实体")
    original_relation: str = Field(description="原始关系")
    original_tail: str = Field(description="原始尾实体")
    corrected_head: Optional[str] = Field(default="", description="修正后头实体")
    corrected_relation: Optional[str] = Field(default="", description="修正后关系")
    corrected_tail: Optional[str] = Field(default="", description="修正后尾实体")
    reason: str = Field(description="修正原因")
    action: str = Field(description="操作类型：modify/delete/add")


class SelfCheckREResult(BaseModel):
    """Self-Check-RE 输出"""
    verified_triples: List[VerifiedTriple] = Field(
        default_factory=list,
        description="校验通过的三元组列表"
    )
    rejected_triples: List[RejectedTriple] = Field(
        default_factory=list,
        description="拒绝的三元组列表（幻觉或严重错误）"
    )
    corrected_triples: List[TripleCorrectionForSelfCheck] = Field(
        default_factory=list,
        description="修正的三元组列表"
    )
    overall_confidence: str = Field(
        default="medium",
        description="整体置信度: high/medium/low"
    )
    retry_suggested: bool = Field(
        default=False,
        description="是否建议触发重抽"
    )
    retry_target: Optional[str] = Field(
        default="",
        description="重抽目标：ner/re/none"
    )
    retry_reason: Optional[str] = Field(
        default="",
        description="重抽原因描述"
    )


# ===== 实体类别细分枚举（用于Prompt） =====

ENTITY_CATEGORY_DETAIL = {
    "POI": ["餐饮", "交通", "教育", "历史保护", "购物", "医疗", "娱乐", "文化", "酒店", "服务"],
    "建筑物": ["商业综合体", "住宅", "办公楼", "文化设施", "教育设施", "医疗设施"],
    "街区": ["商圈", "校区", "社区", "行政区", "景区"],
    "道路": ["主干道", "次干道", "支路", "小巷", "地铁线路"]
}

# ===== 关系类型列表（用于Prompt和校验） =====

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

# ===== 情感节点枚举 =====

EMOTION_NODES = ["正面", "中性", "负面"]

# ===== 评价等级枚举 =====

RATING_NODES = ["超推", "推荐", "一般", "不推荐"]

# ===== 消费等级枚举 =====

CONSUMPTION_NODES = ["平价", "中档", "高档", "奢侈"]

# ===== 距离值枚举 =====

DISTANCE_VALUES = ["近", "中等", "远"]

# ===== 方向值枚举 =====

DIRECTION_VALUES = ["东", "南", "西", "北", "东北", "西南", "东侧", "西侧", "对面", "旁边"]

# ===== 事件类别枚举 =====

EVENT_CATEGORIES = ["自然事件", "人文事件", "商业事件", "社会事件", "负面事件"]

# ===== 事件状态枚举 =====

EVENT_STATES = ["正在进行", "已结束", "计划中", "周期性"]

# ===== 对比维度枚举 =====

COMPARE_DIMENSIONS = ["价格", "环境", "服务", "人流量", "品质", "氛围", "交通", "停车", "口味", "性价比"]

# ===== 情感标签枚举 =====

EMOTION_TAGS = ["氛围感", "治愈", "高级感", "温暖", "文艺", "复古", "现代", "网红感", "小清新", "赛博朋克感"]

# ===== 体验评价枚举 =====

EXPERIENCE_EVALUATIONS = ["服务好", "环境舒适", "商品丰富", "性价比高", "停车方便", "交通便利", "人流量适中"]

# ===== 知名度枚举 =====

POPULARITY_LEVELS = ["热门", "小众", "隐藏宝藏", "必去", "打卡圣地"]


# ===== 联合抽取模型（P9新增） =====

class JointEntity(BaseModel):
    """联合抽取的单个实体"""
    name: str = Field(description="实体名称")
    type: str = Field(description="实体类型：道路/POI/建筑物/街区")
    category: str = Field(description="细分类别")
    aliases: List[str] = Field(default_factory=list, description="别名/简称")
    evidence: str = Field(description="原文依据")


class JointTriple(BaseModel):
    """联合抽取的单个三元组（硬校验+强类型属性）"""
    head: str = Field(description="头实体")
    relation: RelationTypeEnum = Field(description="关系类型（18种之一）")
    tail: str = Field(description="尾实体")
    evidence: str = Field(description="原文依据")
    confidence: ConfidenceEnum = Field(description="置信度")
    attributes: Optional[TripleAttributes] = Field(
        default=None,
        description="关系属性（强类型约束，拒绝额外字段）"
    )

    @field_validator('relation', mode='before')
    @classmethod
    def normalize_relation(cls, v):
        """将关系类型变体映射到标准枚举值"""
        if v is None:
            raise ValueError("relation 不能为空")
        if isinstance(v, RelationTypeEnum):
            return v
        # 使用与 Triple 相同的映射逻辑
        relation_mapping = {
            '位于': '位于', '在': '位于', '在...上': '位于', '地处': '位于',
            '相邻': '相邻', '旁边': '相邻', '旁边是': '相邻', '隔壁': '相邻',
            '属于': '属于', '隶属于': '属于', '是...的一部分': '属于',
            '连接': '连接', '连通': '连接', '通往': '连接',
            '距离': '距离', '离': '距离', '距离...很近': '距离', '附近': '距离',
            '方向': '方向', '在...东边': '方向', '东边': '方向',
            '穿过': '穿过', '横穿': '穿过', '穿越': '穿过',
            '变化为': '变化为', '变成': '变化为', '改为': '变化为',
            '推荐指数': '推荐指数', '推荐': '推荐指数', '强烈推荐': '推荐指数',
            '承载活动': '承载活动', '可以...': '承载活动', '适合...': '承载活动',
            '可达方式': '可达方式', '交通': '可达方式', '怎么去': '可达方式',
            '消费档次': '消费档次', '消费': '消费档次', '人均': '消费档次',
            '品类特征': '品类特征', '特色': '品类特征', '风格': '品类特征',
            '引发情感': '引发情感', '情感': '引发情感', '感觉': '引发情感',
            '优于': '优于', '比...好': '优于', '比...便宜': '优于',
            '相似': '相似', '和...差不多': '相似', '类似': '相似',
            '劣于': '劣于', '不如': '劣于', '比...差': '劣于',
            '发生事件': '发生事件', '有': '发生事件', '正在': '发生事件',
        }
        normalized = relation_mapping.get(str(v))
        if normalized:
            return RelationTypeEnum(normalized)
        raise ValueError(f"无效的关系类型: {v}，有效值为18个预定义关系类型")


class JointExtractionResult(BaseModel):
    """联合抽取结果 - 实体和关系同时输出"""
    entities: List[JointEntity] = Field(default_factory=list, description="抽取的实体列表")
    triples: List[JointTriple] = Field(default_factory=list, description="抽取的三元组列表")
    entity_relation_mapping: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="实体-关系映射：{'武汉大学': ['<武汉大学, 位于, 珞喻路>', ...]}"
    )
    overall_confidence: str = Field(default="medium", description="整体置信度")
    extraction_strategy: str = Field(
        default="joint",
        description="抽取策略标识：joint/pipeline"
    )


# ===== Self-Check-Joint模型（P9新增，含Reflexion） =====

class SelfCheckJointResult(BaseModel):
    """联合抽取校验结果 + Reflexion"""
    verified_entities: List[JointEntity] = Field(description="校验通过的实体")
    verified_triples: List[JointTriple] = Field(description="校验通过的三元组")
    rejected_entities: List[str] = Field(description="拒绝的实体（非地理实体）")
    rejected_triples: List[Dict] = Field(description="拒绝的三元组（幻觉/错误）")

    # Reflexion核心：自然语言反思
    reflection_text: str = Field(
        description="自然语言形式的反思建议，如：'本次抽取遗漏了空间方向关系，建议关注方位介词'"
    )
    improvement_strategy: str = Field(
        description="具体改进策略，如：'增加对方位介词的敏感度，检查是否遗漏位于/相邻关系'"
    )

    overall_confidence: str = Field(description="整体置信度")
    retry_suggested: bool = Field(description="是否建议重试")
    retry_reason: str = Field(description="重试原因")


# ===== Self-Check-QA模型（P9新增） =====

class SelfCheckQAResult(BaseModel):
    """QA脚手架校验结果"""
    verified_qa_pairs: List[QAPair] = Field(description="校验通过的问答对")
    rejected_qa_pairs: List[Dict] = Field(description="拒绝的问答对（与原文不符）")

    # QA质量评估
    entity_coverage: str = Field(
        default="medium",
        description="实体覆盖度：high（遗漏≤1）/ medium（遗漏2-3）/ low（遗漏>3）"
    )
    relation_coverage: str = Field(
        default="medium",
        description="关系覆盖度评估"
    )

    # Reflexion反思
    reflection_text: str = Field(
        default="",
        description="自然语言反思建议"
    )
    improvement_strategy: str = Field(
        default="",
        description="改进策略"
    )

    overall_confidence: str = Field(description="整体置信度")
    retry_suggested: bool = Field(description="是否建议重新生成QA")
    retry_reason: str = Field(default="", description="重试原因")


# ===== Self-Check-Eval模型（P9新增） =====

class SelfCheckEvalResult(BaseModel):
    """评估结果校验"""
    verified_triples: List[Dict] = Field(description="校验通过的三元组（包含评分）")
    rejected_triples: List[Dict] = Field(description="拒绝的三元组（评分过低或有错误）")

    # 评分一致性检查
    score_consistency: str = Field(
        default="medium",
        description="评分一致性：high（评分准确）/ medium（有偏差）/ low（评分不合理）"
    )

    # Reflexion反思
    reflection_text: str = Field(
        default="",
        description="自然语言反思建议"
    )
    improvement_strategy: str = Field(
        default="",
        description="改进策略"
    )

    overall_confidence: str = Field(description="整体置信度")
    retry_suggested: bool = Field(description="是否建议重新评估")
    retry_reason: str = Field(default="", description="重试原因")


# ===== Self-Check-Label模型（P9新增） =====

class SelfCheckLabelResult(BaseModel):
    """标注结果校验"""
    verified_entity_attrs: Dict[str, Dict] = Field(description="校验通过的实体属性")
    verified_relation_attrs: Dict[str, Dict] = Field(description="校验通过的关系属性")

    rejected_entity_attrs: List[str] = Field(description="拒绝的实体属性键（不合理）")
    rejected_relation_attrs: List[str] = Field(description="拒绝的关系属性键（不合理）")

    # 属性完整性检查
    attr_completeness: str = Field(
        default="medium",
        description="属性完整性：high（关键属性完整）/ medium（部分缺失）/ low（大量缺失）"
    )

    # Reflexion反思
    reflection_text: str = Field(
        default="",
        description="自然语言反思建议"
    )
    improvement_strategy: str = Field(
        default="",
        description="改进策略"
    )

    overall_confidence: str = Field(description="整体置信度")
    retry_suggested: bool = Field(description="是否建议重新标注")
    retry_reason: str = Field(default="", description="重试原因")


# ===== Self-Check-Filter模型（P9新增，可选） =====

class SelfCheckFilterResult(BaseModel):
    """Filter筛选校验结果"""
    # 筛选判定校验
    verified_is_valid: bool = Field(description="校验后的筛选判定")
    verified_confidence: str = Field(description="校验后的置信度")

    # 误筛检测
    false_negative_detected: bool = Field(
        default=False,
        description="是否检测到误筛（有效文本被判定为无效）"
    )
    false_positive_detected: bool = Field(
        default=False,
        description="是否检测到误判（无效文本被判定为有效）"
    )

    # 问题分析
    geo_entity_missed: List[str] = Field(
        default_factory=list,
        description="遗漏的地理实体（误筛时）"
    )
    invalid_reason: str = Field(
        default="",
        description="误判原因说明"
    )

    # Reflexion反思
    reflection_text: str = Field(
        default="",
        description="自然语言反思建议"
    )
    improvement_strategy: str = Field(
        default="",
        description="改进策略"
    )

    overall_confidence: str = Field(description="整体置信度")
    retry_suggested: bool = Field(description="是否建议重新筛选")
    retry_reason: str = Field(default="", description="重试原因")


# ===== Self-Check-Normalize模型（P9新增，可选） =====

class SelfCheckNormalizeResult(BaseModel):
    """Normalize归一化校验结果"""
    # 归一化质量校验
    verified_normalized_text: str = Field(description="校验后的归一化文本")
    verified_confidence: str = Field(description="校验后的置信度")

    # 语义保留检查
    semantics_preserved: bool = Field(
        default=True,
        description="是否保留了原文语义"
    )
    info_added: bool = Field(
        default=False,
        description="是否添加了原文不存在的信息（不应添加）"
    )
    info_lost: bool = Field(
        default=False,
        description="是否丢失了原文关键信息"
    )

    # 归一化记录校验
    verified_normalizations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="校验通过的归一化记录"
    )
    rejected_normalizations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="拒绝的归一化记录（不合理）"
    )

    # 问题分析
    alias_errors: List[str] = Field(
        default_factory=list,
        description="别名归一化错误"
    )
    reference_errors: List[str] = Field(
        default_factory=list,
        description="指代消解错误"
    )

    # Reflexion反思
    reflection_text: str = Field(
        default="",
        description="自然语言反思建议"
    )
    improvement_strategy: str = Field(
        default="",
        description="改进策略"
    )

    overall_confidence: str = Field(description="整体置信度")
    retry_suggested: bool = Field(description="是否建议重新归一化")
    retry_reason: str = Field(default="", description="重试原因")


# ===== P10新增：批量LLM调用模型 =====

class BatchCorpusResult(BaseModel):
    """单条语料的批量抽取结果"""
    corpus_id: str = Field(description="语料ID")
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="实体字典: {'道路': [...], 'POI': [...], '建筑物': [...], '街区': [...]}"
    )
    triples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="三元组列表: [{'head': ..., 'relation': ..., 'tail': ..., 'evidence': ..., 'attributes': ...}]"
    )
    confidence: str = Field(default="medium", description="置信度: high/medium/low")
    has_geo_info: bool = Field(default=True, description="是否包含地理信息")
    skip_reason: Optional[str] = Field(default=None, description="跳过原因（无地理信息时）")


class BatchExtractionResult(BaseModel):
    """批量抽取结果 - 一次LLM调用处理多条语料"""
    results: List[BatchCorpusResult] = Field(
        default_factory=list,
        description="各语料的抽取结果列表"
    )
    cross_corpus_aliases: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="跨语料发现的别名映射: [{'raw': '武大', 'canonical': '武汉大学', 'corpus_ids': [...]}]"
    )
    cross_corpus_relations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="跨语料发现的相同三元组（去重依据）"
    )
    overall_confidence: str = Field(default="medium", description="整体置信度")
    batch_size: int = Field(description="处理的语料数量")
    extraction_strategy: str = Field(
        default="batch_joint",
        description="抽取策略: batch_joint/batch_pipeline/fallback_single"
    )


class BatchSelfCheckResult(BaseModel):
    """批量校验结果"""
    verified_results: List[BatchCorpusResult] = Field(
        default_factory=list,
        description="校验通过的语料结果"
    )
    rejected_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="校验失败或标记为跳过的语料"
    )
    verified_aliases: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="校验通过的别名映射"
    )
    rejected_aliases: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="校验失败的别名映射"
    )
    overall_confidence: str = Field(default="medium", description="整体置信度")
    retry_suggested: bool = Field(default=False, description="是否建议重新批量处理")
    retry_reason: str = Field(default="", description="重试原因")
    fallback_to_single: bool = Field(
        default=False,
        description="是否建议退化为单条处理"
    )