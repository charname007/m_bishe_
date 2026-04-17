"""
Pydantic模型定义 - 用于LangChain with_structured_output
v3.4改进：实体类型扩展（新增功能实体、事件实体）+ 属性Schema简化（删除联动推荐，开放文本属性）
核心原则：所有属性和关系必须有原文依据（明确出现/暗示表达/语义推断），禁止幻觉
"""

from typing import List, Dict, Optional, Any, Union, ClassVar
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum


# ===== 关系类型枚举（v3.2精简版：8个） =====


class RelationTypeEnum(str, Enum):
    """关系类型枚举（v3.2精简版：8个关系）

    关系体系：
    - 空间基础关系（3个）：位于、包含、相对方位
    - 社交语义关系（1个）：具有功能
    - 对比评价关系（3个）：优于、相似、劣于
    - 事件关系（1个）：发生事件
    """

    # 空间基础关系（3个）—— 图谱骨架
    LOCATED = "位于"  # 空间定位/归属（合并原"属于")
    CONTAINS = "包含"  # 空间包含（位于的反向）
    RELATIVE_ORIENTATION = "相对方位"  # 空间邻近+相对方位（合并原"相邻+距离+方向")

    # 社交语义关系（1个）—— 图谱血肉
    HAS_FUNCTION = "具有功能"  # 场所的功能用途（原"承载活动")

    # 对比评价关系（3个）—— 特色
    BETTER_THAN = "优于"
    SIMILAR_TO = "相似"
    WORSE_THAN = "劣于"

    # 事件关系（1个）
    HAS_EVENT = "发生事件"


# ===== 实体类型枚举（v3.4新增：6种） =====


class EntityTypeEnum(str, Enum):
    """实体类型枚举（v3.4扩展版：6种）

    实体体系：
    - 空间实体（4个）：道路、POI、建筑物、街区 —— GIS标准
    - 语义实体（2个）：功能、事件 —— 社交媒体特色（v3.4新增）
    """

    # 空间实体（GIS标准）
    ROAD = "道路"  # 交通通道
    POI = "POI"  # 具体地点/机构
    BUILDING = "建筑物"  # 建筑设施
    BLOCK = "街区"  # 地理区域
    # 语义实体（v3.4新增）
    FUNCTION = "功能"  # 场所功能用途
    EVENT = "事件"  # 发生的事件


# ===== 关系类型变体映射常量（提取为模块级常量，消除重复） =====

RELATION_VARIANT_MAPPING: Dict[str, str] = {
    # 空间基础关系（3个）
    "位于": "位于",
    "地处": "位于",
    "属于": "位于",
    "隶属于": "位于",
    "是...的一部分": "位于",  # 合并入"位于"
    "包含": "包含",
    "里面有": "包含",
    "内有": "包含",
    "涵盖": "包含",
    "相对方位": "相对方位",
    "方位": "相对方位",
    "空间方位": "相对方位",
    "相邻": "相对方位",
    "旁边": "相对方位",
    "旁边是": "相对方位",
    "隔壁": "相对方位",  # 合并入"相对方位"
    "距离": "相对方位",
    "离": "相对方位",
    "附近": "相对方位",  # 合并入"相对方位"
    "方向": "相对方位",
    "东边": "相对方位",
    "南边": "相对方位",
    "西边": "相对方位",
    "北边": "相对方位",  # 合并入"相对方位"
    # v3.4补充：常见空间邻近词汇（社交媒体高频）
    "连接": "相对方位",
    "靠近": "相对方位",
    "周边": "相对方位",
    "周围": "相对方位",
    "毗邻": "相对方位",
    "紧邻": "相对方位",
    "对门": "相对方位",
    "对面": "相对方位",
    "交叉": "相对方位",
    "交汇": "相对方位",
    "交界": "相对方位",
    "挨着": "相对方位",
    "紧挨": "相对方位",
    # 社交语义关系（1个）
    "具有功能": "具有功能",
    "适合": "具有功能",
    "承载活动": "具有功能",  # 原名改为"具有功能"
    # 注意：移除 '可以' 和 '活动' 映射（过度歧义）
    # '可以' 太常见（如"我可以去"不应生成三元组）
    # '活动' 语义多样（如"樱花节活动"应为事件实体而非关系）
    # 对比评价关系（3个）
    "优于": "优于",
    "比...好": "优于",
    "比...便宜": "优于",
    "相似": "相似",
    "和...差不多": "相似",
    "类似": "相似",
    "劣于": "劣于",
    "不如": "劣于",
    "比...差": "劣于",
    # 事件关系（1个）
    "发生事件": "发生事件",
    # 注意：移除 '有' 和 '正在' 映射（过度歧义）
    # '有' 太常见（如"武汉大学有图书馆"应为'包含'而非'发生事件'）
    # '正在' 不应作为关系词汇（如"我正在吃饭"不应生成三元组）
}


def normalize_relation_type(v: Any) -> RelationTypeEnum:
    """
    将关系类型变体映射到标准枚举值

    Args:
        v: 输入的关系类型（可以是枚举值、字符串或变体）

    Returns:
        RelationTypeEnum: 标准化的关系类型枚举

    Raises:
        ValueError: 输入为空或无法映射到有效关系类型
    """
    if v is None:
        raise ValueError("relation 不能为空")
    if isinstance(v, RelationTypeEnum):
        return v
    normalized = RELATION_VARIANT_MAPPING.get(str(v))
    if normalized:
        return RelationTypeEnum(normalized)
    raise ValueError(
        f"无效的关系类型: {v}，有效值为8个预定义关系类型："
        f"位于/包含/相对方位/具有功能/优于/相似/劣于/发生事件"
    )


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


# ===== 功能节点枚举（v3.2新增：9大类） =====


class FunctionEnum(str, Enum):
    """功能节点枚举（v3.4扩展版：10大类，新增交通）

    v3.4新增说明：
    - 添加 TRANSPORT = "交通" 用于打车/公交/地铁等出行功能
    - 保持向下兼容，原有9大类不变
    """

    DINING = "餐饮"  # 高频：吃饭、探店、下午茶
    SHOPPING = "购物"  # 高频：逛街、买东西
    LEISURE = "休闲"  # 高频：游玩、散步、放松
    SOCIAL = "社交"  # 高频：聚会、打卡、约会
    VIEWING = "观景"  # 高频：赏花、观展、拍照
    TRANSPORT = "交通"  # v3.4新增：打车、公交、地铁等出行功能
    ACCOMMODATION = "住宿"  # 中频：住酒店、民宿体验
    CULTURE = "文化"  # 中频：学习、体验、参观
    WORK = "工作"  # 低频：办公、产业
    OTHER = "其他"  # 兜底


# ===== 特征标签参考枚举（仅供内部参考，不用于校验LLM输出） =====
# Schema v3.3 说明：特征标签采用开放文本设计，实际抽取时可使用任意自然语言表达
# 此枚举仅作为高频词汇参考列表，帮助开发者理解常见特征表达


class FeatureTagEnum(str, Enum):
    """特征标签参考枚举（仅供参考，不应用于校验）

    Schema v3.3 开放文本设计说明：
    - 特征标签不再使用强枚举约束
    - LLM可输出任意自然语言表达的特征描述
    - 此枚举仅记录高频词汇，供开发参考

    实际使用：EntityAttributes.特征标签 为 List[str] 类型，可接受任意文本
    """

    # 氛围/风格类（高频）
    ATMOSPHERE = "氛围感"
    VIRAL = "网红"
    ARTISTIC = "文艺"
    RETRO = "复古"
    NICHE = "小众"
    WARM = "温馨"
    QUIET = "安静"
    LIVELY = "热闹"

    # 定位/特征类（中频）
    OLD_BRAND = "老字号"
    CHAIN = "连锁"
    CREATIVE = "文创"
    HIGH_END = "高端"

    # 知名度类（高频）
    POPULAR = "热门"
    HIDDEN_GEM = "宝藏"
    MUST_VISIT = "打卡圣地"

    # 体验正面类（高频）
    GOOD_SERVICE = "服务好"
    GOOD_ENVIRONMENT = "环境好"
    GOOD_VALUE = "性价比高"
    CONVENIENT_TRANSPORT = "交通便利"


# ===== 对比维度枚举（v3.2新增：7个） =====


class CompareDimensionEnum(str, Enum):
    """对比维度枚举（v3.3：8个，新增"其他"作为兜底）"""

    PRICE = "价格"
    ENVIRONMENT = "环境"
    SERVICE = "服务"
    CROWD = "人流量"
    QUALITY = "品质"
    TRANSPORT = "交通"
    TASTE = "口味"
    OTHER = "其他"  # v3.3新增：无法归纳时的兜底选项


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


# ===== 事件类别枚举（v3.2精简版：7个） =====


class EventCategoryEnum(str, Enum):
    """事件类别枚举（v3.2精简版：7个）"""

    NATURAL = "自然事件"  # 自然现象相关：樱花盛开、荷花盛开
    CULTURAL = "人文事件"  # 文化活动相关：樱花节、音乐节、展览、夜市
    COMMERCIAL = "商业活动"  # 短期商业行为：开业、打折、促销
    SOCIAL = "社会事件"  # 社会相关事件：施工、装修
    BUSINESS_CHANGE = "业态变更"  # 经营性质变化：书店→咖啡厅
    SHUTDOWN = "停业/关闭"  # 经营终止：停业、倒闭、整顿
    OTHER = "其他"  # 无法归类


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
        default=True, description="是否包含有价值的地理信息，值得继续处理"
    )
    skip_reason: Optional[str] = Field(
        default=None, description="跳过原因（仅当 is_valid=False 时有效）"
    )
    confidence: str = Field(default="medium", description="判断置信度: high/medium/low")
    has_geo_entity: bool = Field(
        default=False, description="是否提及地理实体（道路、POI、建筑物、街区）"
    )
    has_spatial_relation: bool = Field(
        default=False, description="是否涉及空间关系（位于、旁边、连接等）"
    )
    geo_entity_hint: Optional[str] = Field(
        default=None, description="检测到的地理实体提示（如有）"
    )
    # P9改进：武汉地区筛选
    is_non_wuhan_region: bool = Field(
        default=False,
        description="是否明确只包含武汉以外的地区（北京、上海、广州等），无法确定时为False",
    )
    region_hint: Optional[str] = Field(
        default=None, description="地区提示：武汉/非武汉/未知"
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

    normalized_text: str = Field(description="归一化后的完整文本")
    normalizations: List[NormalizationRecord] = Field(
        default_factory=list, description="归一化记录列表"
    )
    confidence: str = Field(
        default="medium", description="整体归一化置信度: high/medium/low"
    )
    preserved_semantics: bool = Field(
        default=True, description="是否保留了原文语义（不添加新信息）"
    )
    has_changes: bool = Field(default=False, description="是否有实质性改动")


# ===== QA Scaffold阶段输出模型（P8新增） =====


class QAPair(BaseModel):
    """单个问答对 - 5W1H引导生成的问答"""

    question: str = Field(description="5W1H引导问题")
    answer: str = Field(description="基于原文的回答")
    dimension: str = Field(description="维度标签: who/what/when/where/why/how")
    entities_involved: List[str] = Field(
        default_factory=list, description="涉及到的实体名称"
    )
    confidence: str = Field(default="medium", description="回答置信度: high/medium/low")


class QAScaffoldResult(BaseModel):
    """QA脚手架输出 - 5W1H问答扩展构建语义脚手架"""

    qa_pairs: List[QAPair] = Field(default_factory=list, description="5W1H问答对列表")
    semantic_summary: str = Field(
        default="", description="语义摘要：整合问答后的文本理解"
    )
    entity_hints: List[str] = Field(
        default_factory=list, description="实体提示列表：可能涉及的地理实体"
    )
    relation_hints: List[str] = Field(
        default_factory=list, description="关系提示列表：可能存在的关系类型"
    )
    context_dependencies: List[str] = Field(
        default_factory=list, description="上下文依赖：需要后续节点注意的依赖关系"
    )
    overall_confidence: str = Field(
        default="medium", description="整体脚手架置信度: high/medium/low"
    )
    should_skip_detailed_extraction: bool = Field(
        default=False, description="是否建议跳过详细抽取（简单文本无地理信息）"
    )


# ===== NER阶段输出模型 =====


class EntityRecognitionResult(BaseModel):
    """命名实体识别结果"""

    道路: List[str] = Field(default_factory=list, description="道路实体列表")
    POI: List[str] = Field(default_factory=list, description="POI兴趣点列表")
    建筑物: List[str] = Field(default_factory=list, description="建筑物实体列表")
    街区: List[str] = Field(default_factory=list, description="街区实体列表")


# ===== RE阶段输出模型（v3.2改进：精简版三元组属性） =====


class TripleAttributes(BaseModel):
    """三元组属性（v3.4精简版：删除联动推荐，开放文本属性）

    原文依据包括：明确出现、暗示表达、语义推断。禁止凭空创造（幻觉）。

    v3.4改进：
    - 删除联动推荐属性（信息价值有限）
    - 适合人群改为开放文本（枚举无法穷尽人群表达）
    - 具有限制改为开放文本列表（保留原文时长准确性）

    v3.5改进：
    - 使用 model_validator(mode='before') 清理 LLM 错误输出的额外字段
    - "推荐指数"等字段应放在实体属性中，而非三元组属性
    """

    model_config = ConfigDict(extra="forbid")  # 拒绝未定义的字段

    # ===== LLM幻觉字段清理器（v3.5新增） =====
    # 这些字段应放在实体属性中，LLM有时错误地放在三元组属性中
    # 使用 ClassVar 避免 Pydantic v2 将其误识别为私有模型属性
    HALLUCINATION_FIELDS: ClassVar[set] = {
        "推荐指数",
        "引发情感",
        "特征标签",
        "rating",
        "emotion",
    }

    @model_validator(mode="before")
    @classmethod
    def clean_hallucination_fields(cls, data):
        """清理 LLM 错误输出的额外字段（幻觉字段）

        LLM 有时会错误地将实体属性（如推荐指数、情感倾向）放入三元组属性中。
        此 validator 在验证前删除这些字段，避免 extra='forbid' 报错。
        """
        if isinstance(data, dict):
            # 删除幻觉字段
            hallucination_fields = cls.HALLUCINATION_FIELDS.intersection(data.keys())
            if hallucination_fields:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"[TripleAttributes] LLM幻觉字段被清理: {hallucination_fields} "
                    f"→ 这些字段应放在实体属性(full_entities)而非三元组属性"
                )
                for field in hallucination_fields:
                    data.pop(field)
        return data

    # ===== 相对方位关系属性（v3.4：删除联动推荐） =====
    距离值: Optional[DistanceValueEnum] = Field(
        default=None,
        description="相对方位关系的距离值：近/中等/远（必须有原文依据，如'附近'→近）",
    )
    方向值: Optional[DirectionValueEnum] = Field(
        default=None,
        description="相对方位关系的方向值：东/南/西/北/东北/西南/东侧/西侧/对面/旁边（必须有原文依据）",
    )

    # ===== 功能关系属性（v3.4：开放文本） =====
    时段: Optional[str] = Field(
        default=None,
        description="功能的适用时段：周末/晚上/樱花季/春季/夏季等（必须有原文依据）",
    )
    适合人群: Optional[str] = Field(
        default=None,
        description="功能的适合人群（开放文本，保留原文表达）：带孩子来玩、闺蜜聚会、大学生打卡等（必须有原文依据）",
    )
    具有限制: Optional[List[str]] = Field(
        default=None,
        description="功能的限制条件列表（开放文本，保留原文表达）：排队两小时、停车超级难、需提前预约等（必须有原文依据）",
    )
    情感倾向: Optional[EmotionNodeEnum] = Field(
        default=None, description="功能的情感倾向：正面/中性/负面（必须有原文依据）"
    )
    功能描述: Optional[str] = Field(
        default=None, description="当功能为'其他'时，具体描述功能内容（必须有原文依据）"
    )
    功能类型: Optional[str] = Field(
        default=None,
        description="功能类型标识（用于校验：当功能类型='其他'时必须提供功能描述）",
    )

    # ===== 对比关系属性 =====
    维度: Optional[List[CompareDimensionEnum]] = Field(
        default=None,
        description='优于/相似/劣于关系的维度列表（必须有原文依据，v3.3新增"其他"作为兜底）',
    )
    维度描述: Optional[str] = Field(
        default=None,
        description='当维度包含"其他"时，具体描述对比内容（必须有原文依据）',
    )

    # ===== 值映射校验器（处理LLM输出的变体值） =====

    @field_validator("距离值", mode="before")
    @classmethod
    def normalize_distance(cls, v):
        """将距离值变体映射到标准枚举值"""
        if v is None:
            return None
        if isinstance(v, DistanceValueEnum):
            return v
        # 变体值映射
        distance_mapping = {
            "近": "近",
            "很近": "近",
            "附近": "近",
            "旁边": "近",
            "不远": "近",
            "中等": "中等",
            "有点远": "中等",
            "稍微远": "中等",
            "几百米": "中等",
            "远": "远",
            "很远": "远",
            "比较远": "远",
            "距离较远": "远",
        }
        normalized = distance_mapping.get(str(v))
        if normalized:
            return DistanceValueEnum(normalized)
        raise ValueError(f"无效的距离值: {v}，有效值为: 近/中等/远")

    @field_validator("方向值", mode="before")
    @classmethod
    def normalize_direction(cls, v):
        """将方向值变体映射到标准枚举值"""
        if v is None:
            return None
        if isinstance(v, DirectionValueEnum):
            return v
        # 方向映射（包含变体）
        direction_mapping = {
            "东": "东",
            "东边": "东",
            "东侧": "东侧",
            "南": "南",
            "南边": "南",
            "西": "西",
            "西边": "西",
            "西侧": "西侧",
            "北": "北",
            "北边": "北",
            "东北": "东北",
            "西南": "西南",
            "对面": "对面",
            "对面是": "对面",
            "旁边": "旁边",
            "旁边是": "旁边",
        }
        normalized = direction_mapping.get(str(v))
        if normalized:
            return DirectionValueEnum(normalized)
        raise ValueError(
            f"无效的方向值: {v}，有效值为: 东/南/西/北/东北/西南/东侧/西侧/对面/旁边"
        )

    # v3.4删除：normalize_crowd和normalize_limits validator（改为开放文本）

    @model_validator(mode="after")
    def validate_other_dimension(self):
        """校验：当维度包含"其他"时，建议提供维度描述（v3.4放宽：不再强制要求）

        v3.4变更说明：
        - 原版强制要求：当维度包含'其他'时，必须提供维度描述
        - v3.4放宽：仅作为建议，不再强制校验（LLM输出难以控制）
        - 维度描述仍为可选字段，建议LLM提供但不强制
        """
        # v3.4：放宽校验，不再强制要求维度描述
        return self

    @model_validator(mode="after")
    def validate_other_function(self):
        """校验：当功能类型为'其他'时，必须提供功能描述"""
        if self.功能类型 and self.功能类型 == "其他" and not self.功能描述:
            raise ValueError("当功能类型为'其他'时，必须提供功能描述")
        return self


class Triple(BaseModel):
    """单个三元组（v3.2精简版：8个关系，强类型属性，全部可选）"""

    head: str = Field(description="头实体名称")
    relation: RelationTypeEnum = Field(description="关系类型（6种之一，硬校验）")
    tail: str = Field(description="尾实体名称或预定义节点")
    evidence: Optional[str] = Field(default="", description="文本证据")
    attributes: Optional[TripleAttributes] = Field(
        default=None, description="关系属性（强类型约束，全部可选，语料中出现才标注）"
    )

    @field_validator("relation", mode="before")
    @classmethod
    def normalize_relation(cls, v):
        """将关系类型变体映射到标准枚举值（使用模块级常量）"""
        return normalize_relation_type(v)


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
    final_scores: List[TripleScore] = Field(
        default_factory=list, description="最终评分"
    )


class EvalResultSimplified(BaseModel):
    """简化的单次评估结果 - 包含评分和可选修正"""

    scores: List[TripleScore] = Field(default_factory=list, description="评分列表")
    need_correction: bool = Field(default=False, description="是否需要修正")
    corrections: List[Correction] = Field(
        default_factory=list, description="修正列表（仅当need_correction=True时有效）"
    )


# ===== Label阶段输出模型（v3.3：特征标签开放文本，细分简化） =====


class FunctionEntityAttributes(BaseModel):
    """功能实体属性（v3.4新增）

    功能实体作为独立实体抽取，不再仅作为三元组Tail。
    所有属性必须有原文依据，禁止幻觉。

    v3.4扩展：功能类型从9种扩展为10种，新增'交通'
    """

    model_config = ConfigDict(extra="forbid")

    功能类型: FunctionEnum = Field(
        description="功能大类（v3.4：10种）：餐饮/购物/休闲/社交/观景/交通/住宿/文化/工作/其他"
    )
    功能细分: Optional[str] = Field(
        default=None,
        description="功能细分描述：咖啡厅、火锅、书店、手工艺体验等（必须有原文依据）",
    )
    适合时段: Optional[str] = Field(
        default=None,
        description="功能适用时段：周末、晚上、樱花季、春季等（必须有原文依据）",
    )
    适合人群: Optional[str] = Field(
        default=None,
        description="适合人群（开放文本，保留原文表达）：带孩子来玩、闺蜜聚会、大学生打卡等（必须有原文依据）",
    )
    具有限制: Optional[List[str]] = Field(
        default=None,
        description="限制条件（开放文本列表，保留原文表达）：排队两小时、停车超级难、需提前预约等（必须有原文依据）",
    )
    情感倾向: Optional[EmotionNodeEnum] = Field(
        default=None, description="功能体验情感：正面/中性/负面（必须有原文依据）"
    )
    推荐指数: Optional[RatingNodeEnum] = Field(
        default=None,
        description="功能推荐程度：超推/推荐/一般/不推荐（必须有原文依据）",
    )
    evidence: Optional[str] = Field(default=None, description="原文依据")


class EventEntityAttributes(BaseModel):
    """事件实体属性（v3.4新增）

    事件实体作为独立实体抽取，不再仅作为三元组Tail。
    所有属性必须有原文依据，禁止幻觉。
    """

    model_config = ConfigDict(extra="forbid")

    事件类别: EventCategoryEnum = Field(
        description="事件类别（7种）：自然事件/人文事件/商业活动/社会事件/业态变更/停业关闭/其他"
    )
    事件状态: Optional[EventStateEnum] = Field(
        default=None,
        description="事件当前状态：正在进行/已结束/计划中/周期性（必须有原文依据）",
    )
    发生时间: Optional[str] = Field(
        default=None,
        description="事件发生时间：每年3月、樱花季、2024年等（必须有原文依据）",
    )
    详细描述: Optional[str] = Field(
        default=None,
        description="事件详细描述：樱花盛开、店铺倒闭、施工改造等（必须有原文依据）",
    )
    情感倾向: Optional[EmotionNodeEnum] = Field(
        default=None, description="事件情感：正面/中性/负面（必须有原文依据）"
    )
    关联场所: Optional[str] = Field(
        default=None, description="事件发生的场所（从三元组推断）"
    )
    evidence: Optional[str] = Field(default=None, description="原文依据")


class EntityAttributes(BaseModel):
    """实体属性（v3.3改进）

    v3.3改进：
    - 特征标签：开放文本，保留原文表达
    - 细分：开放文本，仅记录文本中明确提及的分类词
    - 设计原因：社交媒体表达模糊，权威分类由数据源（高德POI）在对齐阶段补充

    原文依据包括：明确出现、暗示表达、语义推断。禁止凭空创造（幻觉）。
    """

    model_config = ConfigDict(extra="forbid")  # 拒绝未定义的字段

    # 基础分类属性（可选）
    类别: Optional[str] = Field(
        default=None,
        description="实体类别（用于NER边界识别，v3.4：道路/POI/建筑物/街区/功能/事件）",
    )
    细分: Optional[str] = Field(
        default=None,
        description="细分类别（开放文本，仅记录文本明确提及的分类）：餐厅/商场/大学等。注：权威分类由数据源在对齐阶段补充",
    )

    # 文本属性（从语料提取，全部可选）
    # v3.3改进：特征标签改为开放文本，不再使用枚举约束
    特征标签: Optional[List[str]] = Field(
        default=None,
        description="特征描述（开放文本）：保留原文表达，如氛围超好、随手拍好看、遛娃神器等（必须有原文依据）",
    )
    推荐指数: Optional[RatingNodeEnum] = Field(
        default=None, description="推荐程度：超推/推荐/一般/不推荐（必须有原文依据）"
    )
    情感倾向: Optional[EmotionNodeEnum] = Field(
        default=None, description="情感倾向：正面/中性/负面（必须有原文依据）"
    )


class RelationAttributes(BaseModel):
    """关系属性（v3.4精简版：删除联动推荐，开放文本属性）

    根据 Schema v3.4，关系属性包括：
    - 相对方位关系属性：距离值、方向值（删除联动推荐）
    - 功能关系属性：时段、适合人群（开放文本）、具有限制（开放文本列表）、情感倾向
    - 对比关系属性：维度

    所有属性可选，语料中出现才标注。
    """

    # ===== 相对方位关系属性（v3.4：删除联动推荐） =====
    距离值: Optional[DistanceValueEnum] = Field(
        default=None, description="相对方位关系的距离值：近/中等/远（语料中出现才标注）"
    )
    方向值: Optional[DirectionValueEnum] = Field(
        default=None,
        description="相对方位关系的方向值：东/南/西/北/东北/西南/东侧/西侧/对面/旁边（语料中出现才标注）",
    )
    # v3.4删除：联动推荐属性（信息价值有限）

    # ===== 功能关系属性（v3.4：开放文本） =====
    时段: Optional[str] = Field(
        default=None,
        description="功能的适用时段：周末/晚上/樱花季等（语料中出现才标注）",
    )
    适合人群: Optional[str] = Field(
        default=None,
        description="功能的适合人群（开放文本，保留原文表达）：带孩子来玩、闺蜜聚会等（语料中出现才标注）",
    )
    具有限制: Optional[List[str]] = Field(
        default=None,
        description="功能的限制条件（开放文本列表）：排队两小时、停车超级难等（语料中出现才标注）",
    )
    情感倾向: Optional[EmotionNodeEnum] = Field(
        default=None, description="功能的情感倾向：正面/中性/负面（语料中出现才标注）"
    )
    功能描述: Optional[str] = Field(
        default=None,
        description="当功能为'其他'时，具体描述功能内容（语料中出现才标注）",
    )
    功能类型: Optional[str] = Field(
        default=None,
        description="功能类型标识（用于校验：当功能类型='其他'时必须提供功能描述）",
    )

    # ===== 对比关系属性 =====
    维度: Optional[List[CompareDimensionEnum]] = Field(
        default=None,
        description='优于/相似/劣于关系的维度列表（语料中出现才标注，v3.3新增"其他"作为兜底）',
    )
    维度描述: Optional[str] = Field(
        default=None,
        description='当维度包含"其他"时，具体描述对比内容（语料中出现才标注）',
    )

    @model_validator(mode="after")
    def validate_other_dimension(self):
        """校验：当维度包含"其他"时，建议提供维度描述（v3.4放宽：不再强制要求）

        v3.4变更说明：
        - 原版强制要求：当维度包含'其他'时，必须提供维度描述
        - v3.4放宽：仅作为建议，不再强制校验（LLM输出难以控制）
        - 维度描述仍为可选字段，建议LLM提供但不强制
        """
        # v3.4：放宽校验，不再强制要求维度描述
        return self

    @model_validator(mode="after")
    def validate_other_function(self):
        """校验：当功能类型为'其他'时，必须提供功能描述"""
        if self.功能类型 and self.功能类型 == "其他" and not self.功能描述:
            raise ValueError("当功能类型为'其他'时，必须提供功能描述")
        return self


# ===== 关系属性映射常量（v3.4新增） =====
# 用途：label_node根据关系类型动态过滤属性，避免无关属性污染输出
# Schema v3.4属性分配：
# - 相对方位：距离值、方向值（删除联动推荐）
# - 具有功能：时段、适合人群、具有限制、情感倾向、功能描述
# - 优于/相似/劣于：维度、维度描述
# - 位于/包含/发生事件：无属性（隐式定义，不在映射中）
RELATION_ATTRS_MAP: Dict[str, List[str]] = {
    # 空间基础关系
    "相对方位": ["距离值", "方向值"],
    # 社交语义关系
    "具有功能": ["时段", "适合人群", "具有限制", "情感倾向", "功能描述"],
    # 对比评价关系（3个共用同一属性集）
    "优于": ["维度", "维度描述"],
    "相似": ["维度", "维度描述"],
    "劣于": ["维度", "维度描述"],
}


class LabelResult(BaseModel):
    """属性标注结果（v3.2精简版）"""

    entities: Dict[str, EntityAttributes] = Field(
        default_factory=dict, description="实体属性字典，键为实体名"
    )
    relations: Dict[str, RelationAttributes] = Field(
        default_factory=dict,
        description="关系属性字典，键为三元组字符串如'<A, 关系, B>'",
    )


# ===== Self-Check阶段输出模型 =====


class VerifiedEntity(BaseModel):
    """校验后的实体（v3.4更新：支持6种实体类型）"""

    name: str = Field(description="实体名称（归一化后）")
    type: str = Field(description="实体类型：道路/POI/建筑物/街区/功能/事件")
    confidence: str = Field(description="置信度: high/medium/low")
    aliases: List[str] = Field(default_factory=list, description="别名/简称列表")
    evidence: Optional[str] = Field(default="", description="原文依据")


class MissingEntity(BaseModel):
    """遗漏实体建议（v3.4更新：支持6种实体类型）"""

    name: str = Field(description="建议补充的实体名")
    suggested_type: str = Field(
        description="建议类型：道路?/POI?/建筑物?/街区?/功能?/事件?"
    )
    reason: str = Field(description="遗漏原因/原文依据")


class EntityNormalization(BaseModel):
    """实体归一化记录"""

    raw: str = Field(description="原始名称（如'武大'）")
    canonical: str = Field(description="归一化名称（如'武汉大学'）")
    confidence: str = Field(description="归一化置信度: high/medium/low")


class SelfCheckNERResult(BaseModel):
    """Self-Check-NER 输出"""

    verified_entities: List[VerifiedEntity] = Field(
        default_factory=list, description="校验通过的实体列表"
    )
    missing_entities: List[MissingEntity] = Field(
        default_factory=list, description="遗漏实体建议列表"
    )
    entity_normalizations: List[EntityNormalization] = Field(
        default_factory=list, description="实体归一化映射列表"
    )
    removed_entities: List[str] = Field(
        default_factory=list, description="过滤掉的无关实体（非地理实体）"
    )
    overall_confidence: str = Field(
        default="medium", description="整体置信度: high/medium/low"
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
        default_factory=list, description="校验通过的三元组列表"
    )
    rejected_triples: List[RejectedTriple] = Field(
        default_factory=list, description="拒绝的三元组列表（幻觉或严重错误）"
    )
    corrected_triples: List[TripleCorrectionForSelfCheck] = Field(
        default_factory=list, description="修正的三元组列表"
    )
    overall_confidence: str = Field(
        default="medium", description="整体置信度: high/medium/low"
    )
    retry_suggested: bool = Field(default=False, description="是否建议触发重抽")
    retry_target: Optional[str] = Field(default="", description="重抽目标：ner/re/none")
    retry_reason: Optional[str] = Field(default="", description="重抽原因描述")


# ===== 实体类别参考列表（仅供Prompt参考，非强制枚举） =====
# v3.3说明：细分采用开放文本设计，以下列表仅作为高德POI分类的参考对照
# 实体入库时通过entity_alignment节点关联数据源，继承权威分类

ENTITY_CATEGORY_DETAIL = {
    "POI": [
        "餐饮",
        "交通",
        "教育",
        "历史保护",
        "购物",
        "医疗",
        "娱乐",
        "文化",
        "酒店",
        "服务",
    ],
    "建筑物": ["商业综合体", "住宅", "办公楼", "文化设施", "教育设施", "医疗设施"],
    "街区": ["商圈", "校区", "社区", "行政区", "景区"],
    "道路": ["主干道", "次干道", "支路", "小巷", "地铁线路"],
}

# ===== 关系类型列表（v3.2精简版：8个） =====

RELATION_TYPES = [
    # 空间基础关系（3个）
    "位于",
    "包含",
    "相对方位",
    # 社交语义关系（1个）
    "具有功能",
    # 对比评价关系（3个）
    "优于",
    "相似",
    "劣于",
    # 事件关系（1个）
    "发生事件",
]

# ===== 功能节点枚举（v3.2新增：9大类） =====

# v3.4：扩展为10大类，新增交通
FUNCTION_NODES = [
    "餐饮",
    "购物",
    "休闲",
    "社交",
    "观景",
    "交通",
    "住宿",
    "文化",
    "工作",
    "其他",
]

# ===== 情感节点枚举 =====

EMOTION_NODES = ["正面", "中性", "负面"]

# ===== 评价等级枚举 =====

RATING_NODES = ["超推", "推荐", "一般", "不推荐"]

# ===== 距离值枚举 =====

DISTANCE_VALUES = ["近", "中等", "远"]

# ===== 方向值枚举 =====

DIRECTION_VALUES = [
    "东",
    "南",
    "西",
    "北",
    "东北",
    "西南",
    "东侧",
    "西侧",
    "对面",
    "旁边",
]

# ===== 事件类别枚举（v3.2精简版：7个） =====

EVENT_CATEGORIES = [
    "自然事件",
    "人文事件",
    "商业活动",
    "社会事件",
    "业态变更",
    "停业/关闭",
    "其他",
]

# ===== 事件状态枚举 =====

EVENT_STATES = ["正在进行", "已结束", "计划中", "周期性"]

# ===== 对比维度枚举（v3.2精简版：7个） =====

COMPARE_DIMENSIONS = ["价格", "环境", "服务", "人流量", "品质", "交通", "口味", "其他"]

# ===== 特征标签参考列表（v3.3：仅供参考，非强制约束） =====
# v3.3改进：特征标签改为开放文本，此列表仅供参考
# 实际抽取时可使用任意自然语言表达，不受此列表限制

FEATURE_TAGS_REFERENCE = [
    # 氛围/情绪类
    "氛围感",
    "氛围好",
    "松弛感",
    "治愈感",
    "安静",
    "私密",
    "解压",
    # 拍照体验类
    "出片",
    "拍照好看",
    "随手拍好看",
    "适合拍照",
    "出片率高",
    # 风格/审美类
    "网红",
    "文艺",
    "复古",
    "小众",
    "ins风",
    "日系",
    "韩风",
    # 定位/特征类
    "老字号",
    "连锁",
    "文创",
    "高端",
    # 知名度类
    "热门",
    "宝藏",
    "打卡圣地",
    # 服务/环境类
    "服务好",
    "环境好",
    # 价格类
    "性价比高",
    "不贵",
    "平价",
    # 便利类
    "交通便利",
    "好停车",
    # 人群适配类
    "亲子友好",
    "遛娃神器",
]

# 保持原有常量名兼容性（指向参考列表）
FEATURE_TAGS = FEATURE_TAGS_REFERENCE


# ===== 联合抽取模型（P9新增） =====


class JointEntity(BaseModel):
    """联合抽取的单个实体（v3.4扩展版）

    v3.4改进：实体类型扩展为6种（新增功能、事件）
    """

    name: str = Field(description="实体名称")
    type: EntityTypeEnum = Field(description="实体类型：道路/POI/建筑物/街区/功能/事件")
    category: Optional[str] = Field(default=None, description="细分类别")
    aliases: List[str] = Field(default_factory=list, description="别名/简称")
    evidence: str = Field(description="原文依据")
    # v3.4新增：功能实体和事件实体属性
    function_attrs: Optional[FunctionEntityAttributes] = Field(
        default=None, description="功能实体属性（仅当type=功能时有效）"
    )
    event_attrs: Optional[EventEntityAttributes] = Field(
        default=None, description="事件实体属性（仅当type=事件时有效）"
    )


class JointTriple(BaseModel):
    """联合抽取的单个三元组（v3.2精简版：8个关系，强类型属性）"""

    head: str = Field(description="头实体")
    relation: RelationTypeEnum = Field(description="关系类型（6种之一）")
    tail: str = Field(description="尾实体")
    evidence: str = Field(description="原文依据")
    confidence: ConfidenceEnum = Field(description="置信度")
    attributes: Optional[TripleAttributes] = Field(
        default=None, description="关系属性（强类型约束，全部可选，语料中出现才标注）"
    )

    @field_validator("relation", mode="before")
    @classmethod
    def normalize_relation(cls, v):
        """将关系类型变体映射到标准枚举值（使用模块级常量）"""
        return normalize_relation_type(v)


class JointExtractionResult(BaseModel):
    """联合抽取结果 - 实体和关系同时输出"""

    entities: List[JointEntity] = Field(
        default_factory=list, description="抽取的实体列表"
    )
    triples: List[JointTriple] = Field(
        default_factory=list, description="抽取的三元组列表"
    )
    entity_relation_mapping: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="实体-关系映射：{'武汉大学': ['<武汉大学, 位于, 珞喻路>', ...]}",
    )
    overall_confidence: str = Field(default="medium", description="整体置信度")
    extraction_strategy: str = Field(
        default="joint", description="抽取策略标识：joint/pipeline"
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


# ===== P12新增：Self-Check增强版模型（四维度评分） =====


class DimensionScore(BaseModel):
    """单维度评分"""

    rating: str = Field(description="评分等级: high/medium/low")
    issues: int = Field(default=0, description="问题数量")
    details: Optional[List[str]] = Field(default=None, description="问题描述列表")


class ImprovementAction(BaseModel):
    """改进动作项"""

    action_type: str = Field(
        description="动作类型: add_entity/delete_triple/fix_type/fix_direction"
    )
    target: str = Field(description="目标实体或三元组")
    details: str = Field(description="具体说明")
    evidence: Optional[str] = Field(default=None, description="原文依据位置")


class SelfCheckJointResultV2(BaseModel):
    """联合抽取校验结果（P12增强版）- 四维度评分 + 结构化反思

    改进点：
    1. dimension_scores: 四维度量化评分（完整性/准确性/真实性/证据性）
    2. improvement_actions: 可执行的改进动作列表
    3. 保持原有的 reflection_text/improvement_strategy 兼容性
    """

    # 校验结果（继承原有字段）
    verified_entities: List[JointEntity] = Field(
        default_factory=list, description="校验通过的实体"
    )
    verified_triples: List[JointTriple] = Field(
        default_factory=list, description="校验通过的三元组"
    )
    rejected_entities: List[str] = Field(
        default_factory=list, description="拒绝的实体（非地理实体）"
    )
    rejected_triples: List[Dict] = Field(
        default_factory=list, description="拒绝的三元组（幻觉/错误）"
    )

    # P12新增：四维度量化评分
    dimension_scores: Dict[str, DimensionScore] = Field(
        default_factory=dict,
        description="四维度评分: {完整性: DimensionScore, 准确性: DimensionScore, 真实性: DimensionScore, 证据性: DimensionScore}",
    )

    # Reflexion核心：自然语言反思（保持兼容）
    reflection_text: str = Field(
        default="", description="结构化反思：遗漏原因分析 + 幻觉原因分析 + 错误分类"
    )
    improvement_strategy: str = Field(default="", description="改进策略摘要")

    # P12新增：可执行的改进动作列表
    improvement_actions: List[ImprovementAction] = Field(
        default_factory=list, description="可执行的改进动作列表"
    )

    # 整体评估
    overall_confidence: str = Field(default="medium", description="整体置信度")
    retry_suggested: bool = Field(default=False, description="是否建议重试")
    retry_reason: str = Field(default="", description="重试原因")


# ===== Self-Check-QA模型（P9新增） =====


class SelfCheckQAResult(BaseModel):
    """QA脚手架校验结果"""

    verified_qa_pairs: List[QAPair] = Field(description="校验通过的问答对")
    rejected_qa_pairs: List[Dict] = Field(description="拒绝的问答对（与原文不符）")

    # QA质量评估
    entity_coverage: str = Field(
        default="medium",
        description="实体覆盖度：high（遗漏≤1）/ medium（遗漏2-3）/ low（遗漏>3）",
    )
    relation_coverage: str = Field(default="medium", description="关系覆盖度评估")

    # Reflexion反思
    reflection_text: str = Field(default="", description="自然语言反思建议")
    improvement_strategy: str = Field(default="", description="改进策略")

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
        description="评分一致性：high（评分准确）/ medium（有偏差）/ low（评分不合理）",
    )

    # Reflexion反思
    reflection_text: str = Field(default="", description="自然语言反思建议")
    improvement_strategy: str = Field(default="", description="改进策略")

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
        description="属性完整性：high（关键属性完整）/ medium（部分缺失）/ low（大量缺失）",
    )

    # Reflexion反思
    reflection_text: str = Field(default="", description="自然语言反思建议")
    improvement_strategy: str = Field(default="", description="改进策略")

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
        default=False, description="是否检测到误筛（有效文本被判定为无效）"
    )
    false_positive_detected: bool = Field(
        default=False, description="是否检测到误判（无效文本被判定为有效）"
    )

    # 问题分析
    geo_entity_missed: List[str] = Field(
        default_factory=list, description="遗漏的地理实体（误筛时）"
    )
    invalid_reason: str = Field(default="", description="误判原因说明")

    # Reflexion反思
    reflection_text: str = Field(default="", description="自然语言反思建议")
    improvement_strategy: str = Field(default="", description="改进策略")

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
    semantics_preserved: bool = Field(default=True, description="是否保留了原文语义")
    info_added: bool = Field(
        default=False, description="是否添加了原文不存在的信息（不应添加）"
    )
    info_lost: bool = Field(default=False, description="是否丢失了原文关键信息")

    # 归一化记录校验
    verified_normalizations: List[Dict[str, Any]] = Field(
        default_factory=list, description="校验通过的归一化记录"
    )
    rejected_normalizations: List[Dict[str, Any]] = Field(
        default_factory=list, description="拒绝的归一化记录（不合理）"
    )

    # 问题分析
    alias_errors: List[str] = Field(default_factory=list, description="别名归一化错误")
    reference_errors: List[str] = Field(
        default_factory=list, description="指代消解错误"
    )

    # Reflexion反思
    reflection_text: str = Field(default="", description="自然语言反思建议")
    improvement_strategy: str = Field(default="", description="改进策略")

    overall_confidence: str = Field(description="整体置信度")
    retry_suggested: bool = Field(description="是否建议重新归一化")
    retry_reason: str = Field(default="", description="重试原因")


# ===== P10新增：批量LLM调用模型 =====


# P15修复：批量抽取使用JointEntity作为实体输出结构
class BatchCorpusResult(BaseModel):
    """单条语料的批量抽取结果（P15修复：使用JointEntity强类型）"""

    corpus_id: str = Field(description="语料ID")
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="实体字典（快速统计）: {'道路': [...], 'POI': [...], '建筑物': [...], '街区': [...], '功能': [...], '事件': [...]}",
    )
    # P15修复：使用JointEntity列表代替Dict，强制LLM输出完整属性
    full_entities: List[JointEntity] = Field(
        default_factory=list,
        description="完整实体列表（使用JointEntity强类型，包含所有属性）",
    )
    triples: List[JointTriple] = Field(
        default_factory=list, description="三元组列表（使用JointTriple强类型）"
    )
    confidence: str = Field(default="medium", description="置信度: high/medium/low")
    has_geo_info: bool = Field(default=True, description="是否包含地理信息")
    skip_reason: Optional[str] = Field(
        default=None, description="跳过原因（无地理信息时）"
    )


class BatchExtractionResult(BaseModel):
    """批量抽取结果 - 一次LLM调用处理多条语料

    v3.5改进：添加 validator 自动计算 batch_size，避免依赖 LLM 输出
    """

    results: List[BatchCorpusResult] = Field(
        default_factory=list, description="各语料的抽取结果列表"
    )
    cross_corpus_aliases: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="跨语料发现的别名映射: [{'raw': '武大', 'canonical': '武汉大学', 'corpus_ids': [...]}]",
    )
    cross_corpus_relations: List[Dict[str, Any]] = Field(
        default_factory=list, description="跨语料发现的相同三元组（去重依据）"
    )
    overall_confidence: str = Field(default="medium", description="整体置信度")
    batch_size: int = Field(description="处理的语料数量")
    extraction_strategy: str = Field(
        default="batch_joint",
        description="抽取策略: batch_joint/batch_pipeline/fallback_single",
    )

    @model_validator(mode="before")
    @classmethod
    def auto_fill_batch_size(cls, data):
        """自动填充 batch_size（从 results 长度推断）

        LLM 经常忽略输出 batch_size 字段，导致验证失败。
        此 validator 在验证前自动计算，避免依赖 LLM 输出冗余信息。
        """
        if isinstance(data, dict):
            if "batch_size" not in data or data.get("batch_size") is None:
                results = data.get("results", [])
                data["batch_size"] = len(results)
        return data


class BatchSelfCheckResult(BaseModel):
    """批量校验结果"""

    verified_results: List[BatchCorpusResult] = Field(
        default_factory=list, description="校验通过的语料结果"
    )
    rejected_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="校验失败或标记为跳过的语料"
    )
    verified_aliases: List[Dict[str, Any]] = Field(
        default_factory=list, description="校验通过的别名映射"
    )
    rejected_aliases: List[Dict[str, Any]] = Field(
        default_factory=list, description="校验失败的别名映射"
    )
    overall_confidence: str = Field(default="medium", description="整体置信度")
    retry_suggested: bool = Field(default=False, description="是否建议重新批量处理")
    retry_reason: str = Field(default="", description="重试原因")
    fallback_to_single: bool = Field(
        default=False, description="是否建议退化为单条处理"
    )


# ===== QA导师架构模型（P10新增） =====


class ApprovalStatusEnum(str, Enum):
    """审批状态枚举"""

    APPROVED = "approved"  # 审批通过
    NEEDS_REVISION = "needs_revision"  # 需要修改
    REJECTED = "rejected"  # 拒绝（严重错误）


class ApprovalFeedback(BaseModel):
    """审批反馈项"""

    target_node: str = Field(description="目标节点: joint_ner_re/eval/label")
    issue_type: str = Field(
        description="问题类型: hallucination/missing_entity/relation_error/attribute_error/overall"
    )
    severity: str = Field(description="严重程度: high/medium/low")
    description: str = Field(description="问题描述")
    suggestion: str = Field(description="改进建议")
    specific_entities: List[str] = Field(
        default_factory=list, description="涉及的具体实体"
    )
    specific_triples: List[Dict[str, Any]] = Field(
        default_factory=list, description="涉及的具体三元组"
    )


class NodeApprovalResult(BaseModel):
    """单个节点的审批结果"""

    node_name: str = Field(description="节点名称")
    approval_status: ApprovalStatusEnum = Field(description="审批状态")
    confidence: str = Field(description="审批置信度: high/medium/low")
    feedbacks: List[ApprovalFeedback] = Field(
        default_factory=list, description="反馈列表"
    )
    approved_items: List[Dict[str, Any]] = Field(
        default_factory=list, description="审批通过的内容"
    )
    rejected_items: List[Dict[str, Any]] = Field(
        default_factory=list, description="拒绝的内容"
    )
    revision_required: bool = Field(default=False, description="是否需要重新执行节点")


class QAApprovalResult(BaseModel):
    """QA审批结果 - 整合所有节点的审批"""

    # 各节点审批结果
    joint_approval: Optional[NodeApprovalResult] = Field(
        default=None, description="联合抽取审批结果"
    )
    eval_approval: Optional[NodeApprovalResult] = Field(
        default=None, description="评估审批结果"
    )
    label_approval: Optional[NodeApprovalResult] = Field(
        default=None, description="标注审批结果"
    )

    # 整体评估
    overall_status: ApprovalStatusEnum = Field(description="整体审批状态")
    overall_confidence: str = Field(description="整体置信度")

    # 整合后的语义脚手架
    integrated_semantic_summary: str = Field(default="", description="整合后的语义摘要")
    integrated_entity_hints: List[str] = Field(
        default_factory=list, description="整合后的实体提示"
    )
    integrated_relation_hints: List[str] = Field(
        default_factory=list, description="整合后的关系提示"
    )

    # 反馈汇总
    all_feedbacks: List[ApprovalFeedback] = Field(
        default_factory=list, description="所有反馈汇总"
    )

    # 重试建议
    retry_suggested: bool = Field(default=False, description="是否建议重试")
    retry_target_nodes: List[str] = Field(
        default_factory=list, description="建议重试的节点列表"
    )
    retry_reason: str = Field(default="", description="重试原因")


class MentorGuidance(BaseModel):
    """导师指导信息 - QA向后续节点发出的指导"""

    guidance_type: str = Field(
        default="initial", description="指导类型: initial/revision/approval"
    )

    # 指导内容
    semantic_focus: List[str] = Field(
        default_factory=list, description="语义关注点：后续节点应重点关注的语义方面"
    )
    entity_priorities: List[str] = Field(
        default_factory=list, description="实体优先级：重要实体的优先级排序"
    )
    relation_priorities: List[str] = Field(
        default_factory=list, description="关系优先级：重要关系的优先级"
    )
    quality_standards: List[str] = Field(
        default_factory=list, description="质量标准：后续节点应达到的质量标准"
    )

    # 预期约束
    expected_entity_types: Dict[str, List[str]] = Field(
        default_factory=dict, description="预期实体类型分布"
    )
    expected_relations: List[str] = Field(
        default_factory=list, description="预期关系类型"
    )

    # 已知问题提示
    known_issues: List[str] = Field(default_factory=list, description="已知问题提示")
    avoid_patterns: List[str] = Field(default_factory=list, description="避免的模式")


class QAMentorScaffoldResult(BaseModel):
    """QA导师脚手架结果 - 扩展原有QAScaffoldResult"""

    # 原有字段（继承）
    qa_pairs: List[Any] = Field(default_factory=list, description="5W1H问答对列表")
    semantic_summary: str = Field(
        default="", description="语义摘要：整合问答后的文本理解"
    )
    entity_hints: List[str] = Field(
        default_factory=list, description="实体提示列表：可能涉及的地理实体"
    )
    relation_hints: List[str] = Field(
        default_factory=list, description="关系提示列表：可能存在的关系类型"
    )
    context_dependencies: List[str] = Field(
        default_factory=list, description="上下文依赖：需要后续节点注意的依赖关系"
    )
    overall_confidence: str = Field(
        default="medium", description="整体脚手架置信度: high/medium/low"
    )
    should_skip_detailed_extraction: bool = Field(
        default=False, description="是否建议跳过详细抽取"
    )

    # 新增导师字段
    mentor_guidance: Optional[MentorGuidance] = Field(
        default=None, description="导师指导信息"
    )
    reasoning_trace: str = Field(default="", description="推理过程（Reasoner模型输出）")
    deep_analysis: str = Field(default="", description="深度语义分析")


# ===== 实体对齐模型（P11新增） =====


class EntityCandidate(BaseModel):
    """实体对齐候选 - 从数据库检索的相似实体"""

    db_entity_id: str = Field(description="数据库中的实体ID")
    db_name: str = Field(description="数据库中的实体名称")
    db_type: str = Field(default="", description="数据库中的实体类型")
    similarity: float = Field(description="相似度分数 (0-1)")
    longitude: Optional[float] = Field(default=None, description="经度")
    latitude: Optional[float] = Field(default=None, description="纬度")
    source: str = Field(default="unknown", description="数据来源")


class EntityAlignmentItem(BaseModel):
    """单个实体的对齐结果"""

    extracted_name: str = Field(description="抽取的实体名称")
    extracted_type: str = Field(default="", description="抽取的实体类型")
    candidates: List[EntityCandidate] = Field(
        default_factory=list, description="候选实体列表（按相似度排序）"
    )
    best_match: Optional[EntityCandidate] = Field(
        default=None, description="最佳匹配（LLM确认后）"
    )
    alignment_status: str = Field(
        default="pending", description="对齐状态: pending/aligned/new_entity/skip"
    )
    llm_decision: Optional[str] = Field(default=None, description="LLM决策说明")


class EntityAlignmentResult(BaseModel):
    """实体对齐节点输出 - 整体对齐结果"""

    alignment_items: List[EntityAlignmentItem] = Field(
        default_factory=list, description="每个实体的对齐结果"
    )
    aligned_entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="已对齐的实体（含DB ID）"
    )
    new_entities: List[str] = Field(
        default_factory=list, description="新实体（未找到匹配）"
    )
    skipped_entities: List[str] = Field(
        default_factory=list, description="跳过的实体（相似度过低）"
    )
    overall_alignment_rate: float = Field(
        default=0.0, description="整体对齐率（已对齐/总实体）"
    )
    alignment_confidence: str = Field(
        default="medium", description="整体对齐置信度: high/medium/low"
    )


# ===== 导师查询响应模型（P14新增：双向交流机制） =====


class QueryTypeEnum(str, Enum):
    """查询类型枚举"""

    ENTITY_AMBIGUITY = "entity_ambiguity"
    """实体歧义：实体类型或名称不确定"""
    RELATION_CONFUSION = "relation_confusion"
    """关系困惑：关系类型或证据不确定"""
    EVIDENCE_MISSING = "evidence_missing"
    """证据缺失：三元组缺乏文本支持"""
    OVERALL_UNCERTAINTY = "overall_uncertainty"
    """整体不确定：整体抽取置信度过低"""
    EVAL_DISAGREEMENT = "eval_disagreement"
    """评估分歧：评估结果与抽取结果不一致"""
    LABEL_CONFUSION = "label_confusion"
    """标注困惑：属性标注不确定"""


class MentorQueryResponse(BaseModel):
    """导师对后续节点查询的响应 - P14新增

    当后续节点（Joint_NER_RE、Eval、Label）遇到困惑时，
    可以向 QA_Mentor 发起查询，导师给出针对性的解答。
    """

    # 响应内容
    answer: str = Field(default="", description="导师的回答：对问题的直接解答")
    clarification: str = Field(default="", description="澄清说明：对困惑点的详细解释")
    recommendation: str = Field(default="", description="推荐方案：导师建议的处理方式")

    # 更新的指导信息
    updated_guidance: Optional[MentorGuidance] = Field(
        default=None, description="更新的导师指导信息"
    )
    updated_entity_hints: Optional[List[str]] = Field(
        default=None, description="更新的实体提示列表"
    )
    updated_relation_hints: Optional[List[str]] = Field(
        default=None, description="更新的关系提示列表"
    )

    # 状态信息
    response_confidence: str = Field(
        default="medium", description="响应置信度: high/medium/low"
    )
    suggests_revision: bool = Field(
        default=False, description="是否建议修改已抽取的结果"
    )
    revision_suggestion: str = Field(default="", description="修改建议的具体内容")

    # 返回路径
    return_to_node: str = Field(
        default="", description="返回的目标节点：joint_ner_re / eval / label"
    )


class MentorQuery(BaseModel):
    """后续节点向导师发起的查询 - P14新增"""

    query_type: QueryTypeEnum = Field(description="查询类型")
    query_content: str = Field(description="查询内容：问题描述")
    involved_entities: List[str] = Field(
        default_factory=list, description="涉及的实体列表"
    )
    involved_relations: List[str] = Field(
        default_factory=list, description="涉及的关系列表"
    )
    current_confidence: str = Field(default="medium", description="当前置信度")
    source_node: str = Field(default="", description="发起查询的节点名称")
    context: str = Field(
        default="", description="查询上下文：当前的抽取/评估/标注结果摘要"
    )
