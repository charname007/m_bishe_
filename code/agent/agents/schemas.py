"""
Pydantic模型定义 - 各阶段结构化输出类型
"""
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


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

class TripleScore(BaseModel):
    """单个三元组的评分"""
    triple: Triple = Field(description="被评分的三元组")
    SEM: int = Field(ge=1, le=5, description="语义准确性评分(1-5)")
    FAC: int = Field(ge=1, le=5, description="事实真实性评分(1-5)")
    CON: int = Field(ge=1, le=5, description="一致性评分(1-5)")


class EvalResultFirst(BaseModel):
    """第一次评估结果"""
    scores: List[TripleScore] = Field(default_factory=list, description="评分列表")


class Correction(BaseModel):
    """三元组修正"""
    original: Triple = Field(description="原始三元组")
    corrected: Triple = Field(description="修正后的三元组")
    reason: str = Field(description="修正原因")


class EvalResultSecond(BaseModel):
    """第二次评估结果（自检）"""
    need_correction: bool = Field(description="是否需要修正")
    corrections: List[Correction] = Field(default_factory=list, description="修正列表")
    final_scores: List[TripleScore] = Field(default_factory=list, description="最终评分")


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


# ===== 批量处理输出模型 =====

class CorpusResult(BaseModel):
    """单条语料处理结果"""
    corpus_id: str = Field(description="语料ID")
    entities: EntityRecognitionResult = Field(description="实体识别结果")
    triples: List[Triple] = Field(default_factory=list, description="关系抽取结果")


class BatchResult(BaseModel):
    """批量处理结果"""
    results: List[CorpusResult] = Field(default_factory=list, description="语料处理结果列表")


# ===== 创建Output Parser =====

from langchain.output_parsers import PydanticOutputParser

# NER解析器
ner_parser = PydanticOutputParser(pydantic_object=EntityRecognitionResult)

# RE解析器
re_parser = PydanticOutputParser(pydantic_object=RelationExtractionResult)

# Eval解析器
eval_first_parser = PydanticOutputParser(pydantic_object=EvalResultFirst)
eval_second_parser = PydanticOutputParser(pydantic_object=EvalResultSecond)

# Label解析器
label_parser = PydanticOutputParser(pydantic_object=LabelResult)

# 批量处理解析器
batch_parser = PydanticOutputParser(pydantic_object=BatchResult)