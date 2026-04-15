# Agent Workflow 修改建议（参考稿）

更新时间：2026-04-16
适用模块：`agent/agents/`

**目标**
在不大幅增加成本的前提下，提高社交媒体文本中三元组与属性抽取的准确率、召回率和稳定性，同时增强可解释性与可控性。

---

# 已实现架构总结（2026-04-16更新）

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
| **v3.2** | 精简版体系 | 8关系枚举 + 9功能节点体系 | ✅ 已实现 |
| **P12** | 提示词工程优化 | 模块化Schema + RISEN/RCoT框架 + Self-Check增强 | ✅ 已实现 |

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
    """关系类型枚举（v3.2精简版：8个关系）
    
    关系体系：
    - 空间基础关系（3个）：位于、包含、相对方位
    - 社交语义关系（1个）：具有功能
    - 对比评价关系（3个）：优于、相似、劣于
    - 事件关系（1个）：发生事件
    """
    # 空间基础关系（3个）
    LOCATED = "位于"           # 空间定位/归属（合并原"属于"）
    CONTAINS = "包含"          # 空间包含（位于的反向）
    RELATIVE_ORIENTATION = "相对方位"  # 空间邻近+相对方位（合并原"相邻+距离+方向"）

    # 社交语义关系（1个）
    HAS_FUNCTION = "具有功能"  # 场所的功能用途

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
    """功能节点枚举（v3.2精简版：9大类）"""
    DINING = "餐饮"       # 高频：吃饭、探店、下午茶
    SHOPPING = "购物"     # 高频：逛街、买东西
    LEISURE = "休闲"      # 高频：游玩、散步、放松
    SOCIAL = "社交"       # 高频：聚会、打卡、约会
    VIEWING = "观景"      # 高频：赏花、观展、拍照
    ACCOMMODATION = "住宿"  # 中频：住酒店、民宿体验
    CULTURE = "文化"      # 中频：学习、体验、参观
    WORK = "工作"         # 低频：办公、产业
    OTHER = "其他"        # 兜底
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

## 十四、P12提示词工程优化（2026-04-16新增）

### 14.1 改进背景

基于Prompt Architect框架分析，发现原有提示词存在以下问题：
- **提示词冗长**：JOINT_NER_RE_USER约800+行，Token效率低
- **内容重复严重**：关系定义在多处完整重复
- **角色定义宽泛**："地理语义专家"缺乏精准定义
- **CoT过于笼统**：缺少反向验证步骤
- **示例设计失衡**：缺少反面示例
- **Self-Check反思薄弱**：反思维度单一

### 14.2 模块化Schema组件

新增可复用的Schema模块，解决内容重复问题：

```python
# 核心Schema模块（prompts.py新增）
ENTITY_SCHEMA_CORE       # 实体类型定义（GIS标准）
RELATION_SCHEMA_CORE     # 关系类型定义（8种）
ENTITY_ATTRIBUTE_SCHEMA  # 实体属性定义
RELATION_ATTRIBUTE_SCHEMA # 关系属性定义
VALIDATION_COT           # 带反向验证的思维链
NEGATIVE_EXAMPLES        # 反面示例（禁止产生）
EXPERT_ROLE_TEMPLATE     # 精准角色定义模板
```

**使用方式**：
```python
from agent.agents import assemble_joint_extraction_prompt

# 模块化组装提示词（按需组装）
prompt = assemble_joint_extraction_prompt(
    raw_text="武汉大学在珞喻路上",
    entity_hints=format_entity_hints(["武汉大学", "珞喻路"]),
    relation_hints=format_relation_hints(["位于"]),
    include_negative_examples=True  # 可选是否包含反面示例
)
```

### 14.3 RISEN + RCoT框架重构

应用RISEN框架重构核心提示词：

| RISEN组件 | 改进内容 |
|-----------|----------|
| **R**ole | 精准定义：GIS背景 + 武汉本地知识 + 社交媒体语料分析能力 |
| **I**nstructions | 结构化指令：优先级 + 执行顺序 + 验证要求 |
| **S**teps | 正向抽取5步 + 反向验证4步（RCoT） |
| **E**nd goal | 明确质量标准 + 验收条件 |
| **N**arrowing | 整合边界约束 + 幻觉禁止 + 格式要求 |

**反向验证步骤（RCoT）**：
```
6. 幻觉检查：每个三元组能否在原文找到依据？
7. 实体检查：是否存在泛化词被误识别？
8. 方向检查：头尾实体顺序是否正确？
9. 属性检查：属性值是否有原文依据？
```

### 14.4 反面示例设计

新增反面示例模块，明确禁止产生的内容：

```python
NEGATIVE_EXAMPLES = """
### ❌ 幻觉三元组
输入: "武汉大学樱花很美"
错误: <武汉大学, 发生事件, 樱花节>  ← 原文无"樱花节"
正确: 实体属性: 特征标签=["樱花景观"]

### ❌ 关系方向错误
输入: "群光广场在珞喻路上"
错误: <珞喻路, 位于, 群光广场>  ← 方向颠倒
正确: <群光广场, 位于, 珞喻路>

### ❌ 泛化词误识别
输入: "这边风景不错"
错误: 实体: "这边" [POI]  ← 模糊指代
正确: (无实体) confidence=low
"""
```

### 14.5 Self-Check增强

新增四维度校验结构：

| 维度 | 检查项 | 评分标准 |
|------|--------|----------|
| 完整性 | 遗漏实体数量 | 0=high, 1-2=medium, 3+=low |
| 准确性 | 类型判定错误数 | 0=high, 1-2=medium, 3+=low |
| 真实性 | 幻觉三元组数 | 0=high, 1-2=medium, 3+=low |
| 证据性 | 证据缺失数 | 0=high, 1-2=medium, 3+=low |

**改进策略格式化函数**：
```python
format_dimension_scores(scores)      # 格式化四维度评分
format_improvement_strategy(strategy) # 格式化可执行改进动作列表
```

### 14.6 新增提示词模板

| 提示词 | 用途 | 特点 |
|--------|------|------|
| `JOINT_NER_RE_PROMPT_V2` | 联合抽取（重构版） | RISEN框架 + RCoT验证 + 模块化组装 |
| `SELF_CHECK_JOINT_PROMPT_V2` | 联合校验（增强版） | 四维度校验 + 结构化反思 |

### 14.7 预期效果

| 指标 | 原版 | P12改进版 | 提升 |
|------|------|-----------|------|
| Token效率 | ~800行 | ~300行（模块化） | 降低40-50% |
| 幻觉检测 | 单维度 | 四维度量化 | 提升精确度 |
| 反面警示 | 无 | 4个典型示例 | 边界理解提升 |
| 反思质量 | 文本描述 | 结构化列表 | 可执行性提升 |

### 14.8 使用指南

**启用新版提示词**：
```python
from agent.agents import (
    JOINT_NER_RE_PROMPT_V2,    # 重构版联合抽取
    SELF_CHECK_JOINT_PROMPT_V2, # 增强版校验
    assemble_joint_extraction_prompt,  # 模块化组装函数
    ENTITY_SCHEMA_CORE, RELATION_SCHEMA_CORE,  # Schema组件
    NEGATIVE_EXAMPLES, VALIDATION_COT,  # 验证组件
)
```

**自定义组装**：
```python
# 仅包含核心Schema（不含反面示例）
prompt = assemble_joint_extraction_prompt(
    raw_text=text,
    include_negative_examples=False
)

# 包含完整验证步骤
prompt = assemble_joint_extraction_prompt(
    raw_text=text,
    entity_hints=hints,
    relation_hints=relation_hints,
    include_negative_examples=True
)
```

---

# ===== P12.1改进：提示词实际启用（2026-04-16新增） =====

## 改进背景

P12版本定义了改进版提示词（JOINT_NER_RE_PROMPT_V2、SELF_CHECK_JOINT_PROMPT_V2），
但未在nodes.py中实际使用。本次改进将新版提示词正式启用。

---

## 已完成改进清单

### 1. 节点提示词启用（nodes.py）

| 节点 | 原版提示词 | 改进版提示词 | 改进点 |
|------|-----------|-------------|--------|
| **create_joint_ner_re_node** | JOINT_NER_RE_PROMPT | JOINT_NER_RE_PROMPT_V2 | RCoT反向验证 + 反面示例 + 精准角色定义 |
| **create_self_check_joint_node** | SELF_CHECK_JOINT_PROMPT | SELF_CHECK_JOINT_PROMPT_V2 | 四维度校验 + 结构化反思 + 可执行改进动作 |

### 2. 新增数据模型（schemas.py）

| 模型 | 用途 | 字段数 |
|------|------|--------|
| **DimensionScore** | 单维度评分 | rating, issues, details |
| **ImprovementAction** | 改进动作项 | action_type, target, details, evidence |
| **SelfCheckJointResultV2** | 增强版校验结果 | 继承SelfCheckJointResult + dimension_scores + improvement_actions |

### 3. 导出更新（__init__.py）

新增导出项：
- `DimensionScore`, `ImprovementAction`, `SelfCheckJointResultV2`
- `format_dimension_scores`, `format_improvement_strategy`

---

## 改进效果对比

| 维度 | 原版 | P12.1改进版 |
|------|------|-------------|
| **反向验证** | 无 | 4步RCoT验证（幻觉/实体/方向/属性检查） |
| **反面示例** | 0个 | 4个典型错误示例 |
| **角色定义** | 宽泛"专家" | GIS背景 + 武汉本地知识 + 审慎原则 |
| **反思维度** | 单一文本 | 四维度量化评分 |
| **改进策略** | 文本描述 | 可执行动作列表 |

---

## 验证结果

**导入测试**：✅ 通过
```
from agent.agents import (
    JOINT_NER_RE_PROMPT_V2, SELF_CHECK_JOINT_PROMPT_V2,
    SelfCheckJointResultV2, DimensionScore, ImprovementAction,
    VALIDATION_COT, NEGATIVE_EXAMPLES, EXPERT_ROLE_TEMPLATE,
)
```

**参数兼容性**：✅ 通过
- JOINT_NER_RE_PROMPT_V2 参数: context_dependencies, entity_hints, mentor_guidance, raw_text, relation_hints
- SELF_CHECK_JOINT_PROMPT_V2 参数: context_dependencies, entities, improvement_attempts, previous_reflection, raw_text, semantic_summary, triples

---

## 后续优化方向

1. **P13**: 添加配置开关，允许用户选择使用新版或旧版提示词
2. **P14**: 完整测试对比新版与旧版的抽取质量差异
3. **P15**: 根据实际效果调整四维度评分阈值

---

**维护说明**: 本次改进于2026-04-16实施，已验证通过。