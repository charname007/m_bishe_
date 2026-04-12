"""
Pydantic模型定义 - 用于LangChain with_structured_output
P2改进：简化评估模型，单次评估返回评分+可选修正
P5改进：添加 Filter 节点模型，用于文本筛选
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


# ===== Filter阶段输出模型（P5新增） =====

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


# ===== Normalize阶段输出模型（P6新增） =====

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


# ===== NER阶段输出模型 =====

class EntityRecognitionResult(BaseModel):
    """命名实体识别结果"""
    道路: List[str] = Field(default_factory=list, description="道路实体列表")
    POI: List[str] = Field(default_factory=list, description="POI兴趣点列表")
    建筑物: List[str] = Field(default_factory=list, description="建筑物实体列表")
    街区: List[str] = Field(default_factory=list, description="街区实体列表")


# ===== RE阶段输出模型 =====

class Triple(BaseModel):
    """单个三元组"""
    head: str = Field(description="头实体名称")
    relation: str = Field(description="关系类型")
    tail: str = Field(description="尾实体名称")
    evidence: Optional[str] = Field(default="", description="文本证据")


class RelationExtractionResult(BaseModel):
    """关系抽取结果"""
    triples: List[Triple] = Field(default_factory=list, description="抽取的三元组列表")


# ===== Eval阶段输出模型 =====

class TripleForEval(BaseModel):
    """用于评估的三元组"""
    head: str = Field(description="头实体名称")
    relation: str = Field(description="关系类型")
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


# P2改进：简化的单次评估模型（合并评分和修正）
class EvalResultSimplified(BaseModel):
    """简化的单次评估结果 - 包含评分和可选修正"""
    scores: List[TripleScore] = Field(default_factory=list, description="评分列表")
    need_correction: bool = Field(default=False, description="是否需要修正")
    corrections: List[Correction] = Field(default_factory=list, description="修正列表（仅当need_correction=True时有效）")


# ===== Label阶段输出模型 =====

class EntityAttributes(BaseModel):
    """实体属性"""
    类别: str = Field(description="实体类别")
    细分: str = Field(description="细分类别")


class RelationAttributes(BaseModel):
    """关系属性"""
    类型: str = Field(description="关系类型")
    细分: str = Field(description="细分类别")


class LabelResult(BaseModel):
    """属性标注结果"""
    entities: Dict[str, EntityAttributes] = Field(
        default_factory=dict,
        description="实体属性字典，键为实体名"
    )
    relations: Dict[str, RelationAttributes] = Field(
        default_factory=dict,
        description="关系属性字典，键为三元组字符串如'<A, 关系, B>'"
    )


# ===== Self-Check阶段输出模型（二次对话验证）=====

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