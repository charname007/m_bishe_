# Agent Workflow 修改建议（参考稿）

更新时间：2026-04-15
适用模块：`agent/agents/`

**目标**
在不大幅增加成本的前提下，提高社交媒体文本中三元组与属性抽取的准确率、召回率和稳定性，同时增强可解释性与可控性。

---

# 已实现架构总结（2026-04-15更新）

## 一、架构版本演进

| 版本 | 代号 | 主要改进 | 实现状态 |
|------|------|----------|----------|
| **P2** | 配置化支持 | ExtractionConfig支持运行时配置和环境变量 | ✅ 已实现 |
| **P3** | 流式输出 | StreamWriter实时进度推送 | ✅ 已实现 |
| **P5** | Filter筛选 | 文本有效性筛选，提前过滤无效语料 | ✅ 已实现 |
| **P6** | Normalize归一化 | 指代消解、别名归一化、口语改写 | ✅ 已实现 |
| **P7** | 分批次处理 | process_corpus_in_batches批量入口 | ✅ 已实现 |
| **P8** | QA Scaffold | 5W1H问答脚手架 + Self-Check二次对话 | ✅ 已实现 |
| **P9** | 联合抽取 | Joint NER+RE + Reflexion + 全节点二次检查 | ✅ 已实现 |
| **P10** | QA导师模式 | 多LLM协作 + 审批修改循环 | ✅ 已实现 |
| **v3.2** | 精简版体系 | 6关系枚举 + 功能节点体系 | ✅ 已实现 |

---

## 二、核心文件架构

### 2.1 文件职责矩阵

| 文件 | 职责 | 主要内容 |
|------|------|----------|
| [config.py](agent/agents/config.py) | 配置管理 | ExtractionConfig类、环境变量加载、默认配置 |
| [state.py](agent/agents/state.py) | 状态定义 | CorpusState/KGState TypedDict、StepEnum/PhaseEnum、Reducer函数 |
| [schemas.py](agent/agents/schemas.py) | 数据模型 | Pydantic模型（30+）、枚举定义（10+）、验证器 |
| [prompts.py](agent/agents/prompts.py) | 提示词 | ChatPromptTemplate（15+）、格式化函数（10+） |
| [nodes.py](agent/agents/nodes.py) | 节点实现 | 工作流节点工厂函数（20+）、辅助函数 |
| [workflow.py](agent/agents/workflow.py) | 工作流构建 | StateGraph构建、路由函数、批量处理入口 |
| [__init__.py](agent/agents/__init__.py) | 模块导出 | 公开API导出列表 |

---

## 三、工作流架构

### 3.1 单条语料工作流（CorpusState）

**联合抽取模式（默认）**：
```
START → Filter(可选) → Normalize(可选) → QA_Scaffold(可选) → Joint_NER_RE → Self-Check_Joint → Eval → Label → END
                                    ↑________________ Reflexion重试循环 ________________|
```

**QA导师模式（P10新增）**：
```
START → Filter(可选) → Normalize(可选) → QA_Mentor → Joint_NER_RE → Eval → Label → QA_Approval → END
         ↑_______________________________________________________________________________|
                                     ↑_______ Revision Loop _______|
```

**流水线模式（备选）**：
```
START → Filter → Normalize → QA_Scaffold → NER → Self-Check-NER → RE → Self-Check-RE → Eval → Label → END
```

### 3.2 批量处理工作流（KGState）

```
START → Coordinator(分片) → Workers(并行) → Aggregator(去重合并) → Finalizer(落库) → END
```

---

## 四、节点功能详解

### 4.1 前置节点（可选）

| 节点 | 功能 | 输入 | 输出 | 配置开关 |
|------|------|------|------|----------|
| **Filter** | 文本有效性筛选 | raw_text | is_valid, has_geo_entity, is_non_wuhan_region | `enable_filter` |
| **Normalize** | 指代消解、别名归一化 | raw_text | normalized_text, normalizations | `enable_normalize` |
| **QA_Scaffold** | 5W1H问答脚手架 | raw_text | qa_pairs, semantic_summary, entity_hints | `enable_qa_scaffold` |

### 4.2 抽取节点

| 节点 | 功能 | 模式 | 输出 |
|------|------|------|------|
| **Joint_NER_RE** | 联合抽取实体+关系 | 联合模式 | entities, triples, entity_relation_mapping |
| **NER** | 命名实体识别 | 流水线模式 | entities: Dict[str, List[str]] |
| **RE** | 关系抽取 | 流水线模式 | triples: List[Dict] |

### 4.3 校验节点

| 节点 | 功能 | 特性 |
|------|------|------|
| **Self-Check_Joint** | 联合抽取校验 | Reflexion反思输出、重试建议 |
| **Self-Check-QA** | QA脚手架校验 | 遗漏实体检查 |
| **Self-Check-Eval** | 评估结果校验 | 评分合理性验证 |
| **Self-Check-Label** | 标注结果校验 | 属性完整性检查 |

### 4.4 QA导师节点（P10新增）

| 节点 | 功能 | LLM | 输出 |
|------|------|-----|------|
| **QA_Mentor** | 深度语义分析 + 导师指导 | Reasoner(强模型) | mentor_guidance, semantic_summary |
| **QA_Approval** | 审批后续节点结果 | Reasoner(强模型) | approval_status, feedbacks, retry_suggested |
| **Revision_Joint** | 根据反馈改进抽取 | Chat(工作模型) | 修正后的entities, triples |

### 4.5 后处理节点

| 节点 | 功能 | 输出 |
|------|------|------|
| **Eval** | 三元组评估 | eval_scores, corrected_triples, eval_passed |
| **Label** | 属性标注 | entity_attrs, relation_attrs |

---

## 五、关系类型体系（v3.2精简版）

### 5.1 关系类型枚举（RelationTypeEnum）

```python
class RelationTypeEnum(str, Enum):
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
```

### 5.2 功能节点枚举（FunctionEnum）

```python
class FunctionEnum(str, Enum):
    """9大功能类别"""
    DINING = "餐饮"
    SHOPPING = "购物"
    ENTERTAINMENT = "娱乐"
    TRANSPORTATION = "交通"
    EDUCATION = "教育"
    MEDICAL = "医疗"
    RESIDENCE = "居住"
    OFFICE = "办公"
    TOURISM = "旅游"
```

### 5.3 实体属性体系

| 属性类型 | 字段名 | 适用实体 |
|----------|--------|----------|
| **功能类别** | function | POI, 建筑物 |
| **特色标签** | feature_tags | POI, 建筑物 |
| **人群适宜** | suitable_crowd | POI, 建筑物 |
| **限制条件** | limitations | POI, 建筑物 |
| **营业状态** | business_status | POI, 建筑物 |

---

## 六、配置参数详解

### 6.1 ExtractionConfig核心参数

```python
@dataclass
class ExtractionConfig:
    # ===== 评估配置 =====
    eval_threshold: float = 3.5          # 评估通过阈值
    similarity_threshold: float = 0.85   # 实体去重相似度阈值

    # ===== 分布式处理 =====
    corpus_per_worker: int = 10          # 每Worker语料数
    max_workers: int = 10                # 最大Worker数
    batch_size: int = 100                # 批次读取数

    # ===== 批量LLM调用（P10） =====
    batch_llm_size: int = 5              # 每LLM调用语料数
    enable_batch_llm: bool = True        # 启用批量模式

    # ===== 可选节点开关 =====
    enable_filter: bool = False          # Filter筛选
    enable_normalize: bool = False       # Normalize归一化
    enable_qa_scaffold: bool = False     # QA脚手架
    enable_qa_mentor: bool = False       # QA导师模式
    use_joint_extraction: bool = True    # 联合抽取模式

    # ===== Self-Check + Reflexion =====
    enable_self_check: bool = False
    enable_full_self_check: bool = False
    enable_reflexion: bool = True
    reflexion_max_retries: int = 3

    # ===== QA导师模式（P10） =====
    qa_llm_model: str = "deepseek-reasoner"   # QA导师模型
    worker_llm_model: str = "deepseek-chat"   # 工作节点模型
    max_revision_cycles: int = 3              # 最大修改轮次
    qa_approval_enabled: bool = False         # QA审批流程
```

---

## 七、Pydantic模型清单

### 7.1 抽取模型

| 模型 | 用途 | 字段数 |
|------|------|--------|
| JointExtractionResult | 联合抽取输出 | 5 |
| JointEntity | 联合抽取实体 | 5 |
| JointTriple | 联合抽取三元组 | 6 |
| EntityRecognitionResult | NER输出（流水线） | 2 |
| RelationExtractionResult | RE输出（流水线） | 2 |

### 7.2 校验模型

| 模型 | 用途 | 特性字段 |
|------|------|----------|
| SelfCheckJointResult | 联合校验 | reflection_text, improvement_strategy |
| SelfCheckQAResult | QA校验 | missing_entities |
| SelfCheckEvalResult | Eval校验 | score_issues |
| SelfCheckLabelResult | Label校验 | attribute_issues |

### 7.3 QA导师模型（P10）

| 模型 | 用途 | 核心字段 |
|------|------|----------|
| QAMentorScaffoldResult | 导师脚手架 | mentor_guidance, reasoning_trace |
| QAApprovalResult | 审批结果 | overall_status, all_feedbacks |
| MentorGuidance | 导师指导 | semantic_focus, quality_standards |
| ApprovalFeedback | 审批反馈 | issue_type, suggestion |
| NodeApprovalResult | 单节点审批 | approval_status, feedbacks |

### 7.4 批量处理模型（P10）

| 模型 | 用途 | 字段 |
|------|------|------|
| BatchCorpusResult | 批量语料 | corpus_list, total_count |
| BatchExtractionResult | 批量抽取 | results, failed_indices |
| BatchSelfCheckResult | 批量校验 | check_results, needs_retry |

---

## 八、提示词模板清单

### 8.1 前置节点提示词

| 提示词 | 节点 | 特点 |
|--------|------|------|
| FILTER_PROMPT | Filter | 快速筛选、武汉区域判断 |
| NORMALIZE_PROMPT | Normalize | 指代消解、别名归一化 |
| QA_SCAFFOLD_PROMPT | QA_Scaffold | 5W1H框架、语义摘要 |
| QA_MENTOR_PROMPT | QA_Mentor | 深度分析、导师指导 |

### 8.2 抽取节点提示词

| 提示词 | 节点 | 特点 |
|--------|------|------|
| JOINT_NER_RE_PROMPT | Joint_NER_RE | 联合抽取、证据输出 |
| NER_PROMPT | NER | 实体识别、四类型分类 |
| RE_PROMPT | RE | 关系抽取、属性推断 |

### 8.3 校验节点提示词

| 提示词 | 节点 | 特点 |
|--------|------|------|
| SELF_CHECK_JOINT_PROMPT | Self-Check-Joint | Reflexion反思、重试建议 |
| SELF_CHECK_QA_PROMPT | Self-Check-QA | 遗漏检查 |
| SELF_CHECK_EVAL_PROMPT | Self-Check-Eval | 评分验证 |
| SELF_CHECK_LABEL_PROMPT | Self-Check-Label | 属性验证 |

### 8.4 QA导师提示词（P10）

| 提示词 | 节点 | 特点 |
|--------|------|------|
| QA_APPROVAL_PROMPT | QA_Approval | 审批标准、反馈生成 |
| REVISION_JOINT_PROMPT | Revision_Joint | 根据反馈改进 |

### 8.5 批量处理提示词（P10）

| 提示词 | 用途 | 特点 |
|--------|------|------|
| BATCH_JOINT_PROMPT | 批量联合抽取 | 多语料并行输入 |
| BATCH_SELF_CHECK_PROMPT | 批量校验 | 跨语料别名检测 |

---

## 九、状态字段详解

### 9.1 CorpusState字段分类

**输入字段**：
- `corpus_id`: 语料ID
- `raw_text`: 原始文本
- `_config_enable_normalize`: 配置标记
- `_config_enable_qa_scaffold`: 配置标记

**前置处理字段**：
- `filter_result`: Filter结果
- `normalize_result`: Normalize结果
- `normalized_text`: 归一化文本
- `qa_scaffold_result`: QA脚手架结果
- `semantic_summary`: 语义摘要
- `qa_entity_hints`: 实体提示
- `qa_relation_hints`: 关系提示

**抽取结果字段**：
- `entities`: 实体字典
- `triples`: 三元组列表
- `joint_extraction_result`: 联合抽取完整结果
- `extraction_strategy`: 抽取策略标识

**校验结果字段**：
- `self_check_joint_result`: 联合校验结果
- `self_check_qa_result`: QA校验结果
- `self_check_eval_result`: Eval校验结果
- `self_check_label_result`: Label校验结果
- `reflection_text`: 反思建议
- `reflection_history`: 反思历史列表
- `improvement_strategy`: 改进策略

**QA导师字段（P10）**：
- `mentor_guidance`: 导师指导
- `qa_approval_result`: 审批结果
- `revision_feedbacks`: 修改反馈列表
- `revision_cycle_count`: 修改循环计数
- `reasoning_trace`: 推理过程

**输出字段**：
- `eval_passed`: 评估是否通过
- `corrected_triples`: 修正后三元组
- `entity_attrs`: 实体属性
- `relation_attrs`: 关系属性
- `final_entities`: 最终实体
- `final_triples`: 最终三元组

### 9.2 KGState字段

- `batch_id`: 批次ID
- `corpus_partitions`: Worker分片映射
- `worker_results`: Worker结果列表
- `aggregated_entities`: 聚合实体
- `aggregated_triples`: 聚合三元组
- `entity_aliases`: 实体别名映射

---

## 十、导出API清单

### 10.1 __all__导出列表（约80项）

```python
__all__ = [
    # 配置
    "ExtractionConfig", "DEFAULT_CONFIG",
    # 状态
    "KGState", "WorkerResult", "CorpusState",
    "StepEnum", "PhaseEnum", "DEFAULT_MAX_RETRIES",
    # 枚举（v3.2）
    "RelationTypeEnum", "FunctionEnum", "FeatureTagEnum",
    "CrowdNodeEnum", "LimitNodeEnum", "DistanceValueEnum",
    "DirectionValueEnum", "EmotionNodeEnum", "RatingNodeEnum",
    # 模型
    "JointEntity", "JointTriple", "JointExtractionResult",
    "SelfCheckJointResult", "SelfCheckQAResult", "SelfCheckEvalResult",
    "QAApprovalResult", "MentorGuidance", "QAMentorScaffoldResult",
    # 提示词
    "JOINT_NER_RE_PROMPT", "QA_MENTOR_PROMPT", "QA_APPROVAL_PROMPT",
    # 节点
    "create_joint_ner_re_node", "create_qa_mentor_node",
    "create_qa_approval_node", "create_revision_joint_node",
    # 工作流
    "build_corpus_workflow", "build_qa_mentor_workflow",
    "process_corpus_with_qa_mentor", "process_corpus_in_batches",
]
```

---

## 十一、运行测试验证

### 11.1 测试脚本位置

[test_run.py](agent/test_run.py) 包含8个测试函数：

| 测试函数 | 测试内容 |
|----------|----------|
| `test_single_corpus` | 基础模式（可选Filter/Normalize/QA） |
| `test_invalid_corpus` | 无效语料筛选测试 |
| `test_joint_extraction_with_full_self_check` | P9联合抽取+Reflexion |
| `test_qa_mentor_mode` | P10 QA导师模式 |

### 11.2 验证结果

| 检查项 | 状态 |
|--------|------|
| Python导入 | ✅ 成功 |
| Pydantic模型实例化 | ✅ 正常 |
| 工作流构建（各种配置） | ✅ 正常 |
| Filter+Normalize同时启用 | ✅ 已修复bug |

---

## 十二、成本分析

### 12.1 LLM调用次数

| 模式 | 前置节点 | 抽取 | 校验 | 后处理 | 总计 |
|------|----------|------|------|--------|------|
| **基础模式** | 0 | 1(Joint) | 0 | 2(Eval+Label) | 3次 |
| **Filter+Normalize+QA** | 3 | 1(Joint) | 0 | 2 | 6次 |
| **联合抽取+Self-Check** | 0 | 1 | 1 | 2 | 4次 |
| **QA导师模式** | 3 | 1 | 0 | 2+2(Approval+Revision) | 7次 |

### 12.2 QA导师模式成本对比

| 项目 | 普通模式 | QA导师模式 | 增加 |
|------|----------|------------|------|
| QA节点 | 1次Chat | 1次Reasoner | ~2x |
| QA审批 | 0 | 1-3次Reasoner | 新增 |
| 总成本 | ~5次Chat | ~4次Reasoner | ~40% |

---

## 十三、后续优化方向

1. **v3.3**: 关系属性细化（维度枚举化）
2. **P11**: 知识库协同（实体链接、KG验证）
3. **P12**: 自适应动态路由（按文本复杂度选择路径）
4. **P13**: 多轮Self-Consistency投票
5. **P14**: 证据span输出（增强可解释性）

---

**文档维护**: 本文档随架构迭代持续更新，记录已实现的改进和待实施的优化方向。