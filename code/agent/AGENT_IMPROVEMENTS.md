# Agent Workflow 修改建议（参考稿）

更新时间：2026-04-10
适用模块：`agent/agents/`

**目标**
在不大幅增加成本的前提下，提高社交媒体文本中三元组与属性抽取的准确率、召回率和稳定性，同时增强可解释性与可控性。

**当前 Workflow 简述**
流程为 NER → RE → Eval → Label，支持简化评估与两轮评估，具备并行处理、实体去重、流式事件与数据库落库能力。
相关文件：`agent/agents/workflow.py`、`agent/agents/nodes.py`、`agent/agents/prompts.py`、`agent/agents/state.py`

**改进建议（优先级从高到低）**

1. 增加"语义归一化/指代消解"前置节点
作用是将省略主语、模糊描述的社交媒体文本改写为显式主语与标准句式，降低后续 NER/RE 误差。
推荐位置：`agent/agents/workflow.py` 在 NER 前插入 `normalize` 节点。

2. 引入"双阶段关系抽取（候选生成 + 验证）"
先宽松生成候选三元组，再由强约束验证器筛选，显著降低幻觉与方向错误。
推荐位置：`agent/agents/nodes.py` 中扩展 `create_re_node`，或新增 `re_verify` 节点。

3. NER/RE 采用多样性投票（Self-Consistency）
对同一文本采样 2-3 次，合并一致结果以提高召回与稳定性。
推荐位置：`agent/agents/nodes.py` 的 `create_ner_node` 和 `create_re_node`。

4. Schema 约束升级（实体类型 × 关系矩阵）
加入关系类型与实体类型的兼容矩阵，例如"连接"仅限道路↔道路/道路↔POI。
推荐位置：`agent/agents/nodes.py` 的 `rule_based_validation`。

5. 证据跨度与置信度校准
输出 `evidence_span` 与 `confidence`，增强可解释性与后处理筛选能力。
推荐位置：`agent/agents/schemas.py` 与 `agent/agents/nodes.py` 的输出结构。

6. 错误驱动局部修复循环
当 Eval 发现问题，仅对问题三元组局部修复，避免全量重跑。
推荐位置：`agent/agents/workflow.py` 添加轻量修复循环节点。

7. 引入地名词表与知识库协同
结合本地 POI/OSM 词表做候选实体提示，再由 LLM disambiguation。
推荐位置：`agent/agents/prompts.py` 与 `create_ner_node`。

8. 事件/情感实体化
将"承载活动/引发情感"中的活动与情感建模为可选实体类型，提升关系语义清晰度。
推荐位置：`agent/agents/state.py` 与 `prompts.py`。

**可快速落地的最小改动（Quick Wins）**

1. 在 `rule_based_validation` 中加入类型-关系矩阵限制。
2. 在 NER/RE 增加 2 次采样一致性投票。
3. 增加 Normalize 节点，仅重写文本，不改变原文本保存逻辑。

**中期演进方向**

1. 双阶段 RE（候选 + 验证）。
2. 引入词表辅助 NER。
3. 证据跨度与置信度校准输出。

**长期方向**

1. 事件/情感实体体系。
2. 多源知识库融合与实体对齐。
3. 动态 Schema 与增量学习。

**评估建议**

1. 指标：三元组准确率、召回率、F1、幻觉率、关系方向错误率。
2. 对比：简化评估 vs 双轮评估；单次抽取 vs 多样性投票。
3. 误差分析：统计规则校验失败类型与分布。

**风险与对策**

1. 成本上升：投票与双阶段验证可能提升 LLM 调用数，建议按配置开关。
2. 稳定性下降：多节点链路增加失败点，建议统一 RetryPolicy 与降级策略。
3. 规则过严：Schema 限制可能误杀长尾关系，建议灰度开关与日志分析。

**下一步建议**

1. 选择一个方向先落地，例如 Normalize 节点或 Schema 约束升级。
2. 增加 20-50 条人工标注的验证集，量化效果增益。
3. 根据评估结果再决定是否上双阶段 RE 或多样性投票。

---

# 详细分析版（新增）

---

## 一、当前 Workflow 架构深度分析

### 1.1 单条语料工作流 (CorpusState)

```text
START → NER → [条件路由] → RE → Eval → Label → END
                      ↓
                    END (NER失败时)
```

**节点职责详解**：

| 节点 | 功能 | 输入 | 输出 |
|------|------|------|------|
| **NER** | 命名实体识别（道路、POI、建筑物、街区） | raw_text | entities: Dict[str, List[str]] |
| **RE** | 关系抽取（连接、位于、承载活动、引发情感、属于） | raw_text + entities | triples: List[Dict] |
| **Eval** | 三元组评估（语义/事实/一致性评分）+ 规则校验 | raw_text + triples | corrected_triples |
| **Label** | 属性标注（实体细分类别、关系细分类型） | entities + corrected_triples | entity_attrs, relation_attrs |

**关键特性**：
- 条件路由：NER失败直接结束，无实体跳过RE
- 简化评估模式：单次LLM评估 + 规则校验（减少调用成本）
- 两轮评估模式：Eval1评分 → Eval2二次验证修正
- RetryPolicy：LLM调用节点自动重试（临时故障处理）
- StreamWriter：实时进度事件推送（前端可监听）

### 1.2 分布式批量工作流 (KGState)

```text
START → Coordinator → Workers(并行) → Aggregator → Finalizer → END
```

**阶段职责详解**：

| 阶段 | 功能 | 核心逻辑 |
|------|------|----------|
| **Coordinator** | 语料分片 | 计算Worker数量，按corpus_per_worker切分 |
| **Workers** | 并行处理 | 每个Worker调用预编译的corpus_workflow处理分片 |
| **Aggregator** | 结果聚合 | 实体去重（多层blocking）、别名发现、三元组合并 |
| **Finalizer** | 数据落库 | Neo4j图数据库 + PostgreSQL关系数据库 |

**实体去重算法详解**（P1优化）：

```text
第一层blocking: 按实体类型分组（Road/POI/Building/Block）
第二层blocking: 按首字符分组
长度索引: 跨block简称检查（如"华农" ↔ "华中农业大学"）
相似度阈值: 0.85（可配置）
```

算法亮点：
- 避免不同类型实体误合并（如POI"武汉大学"与道路"大学路"）
- 首字符分组大幅减少比较次数（O(n*k)而非O(n²)）
- 长度索引支持简称别名发现（如"武大"→"武汉大学"）

### 1.3 数据流与状态管理

**CorpusState**（单条语料）：

```python
corpus_id: str           # 语料唯一标识
raw_text: str            # 原始文本
entities: Dict           # 按类型分类的实体 {"道路": [], "POI": [], ...}
triples: List[Dict]      # 抽取的三元组 [{head, relation, tail, evidence}, ...]
eval_scores: List[Dict]  # 评估评分 [{triple, SEM, FAC, CON}, ...]
corrected_triples: List  # 修正后三元组
entity_attrs: Dict       # 实体属性 {"武汉大学": {"类别": "POI", "细分": "教育"}}
relation_attrs: Dict     # 关系属性
current_step: StepEnum   # 当前步骤 (NER/RE/EVAL/LABEL/DONE)
error: Optional[str]     # 错误信息
```

**KGState**（批量处理）：

```python
batch_id: str                    # 批次ID
corpus_partitions: Dict          # Worker分片映射 {"worker_1": [corpus_list]}
worker_results: List[WorkerResult] # Worker结果（merge_list reducer）
aggregated_entities: List[Dict]  # 聚合实体
aggregated_triples: List[Dict]   # 聚合三元组
entity_aliases: Dict[str, List]  # 实体别名映射 {"武汉大学": ["武大"]}
```

---

## 二、现有设计的问题与局限

### 2.1 架构层面问题

| 问题 | 影响 | 根因 |
|------|------|------|
| **固定流水线** | 无法适应文本复杂度差异 | 所有文本走相同流程，简单文本过度处理，复杂文本处理不足 |
| **NER-RE分离** | 关系抽取缺少实体上下文 | NER结果仅传递实体列表，RE节点丢失了实体识别的置信度、位置信息 |
| **单向数据流** | 无反馈闭环 | Eval发现问题后只能局部修正，无法触发上游重新抽取 |
| **知识库断层** | 未利用已有KG知识 | 每次抽取独立进行，无实体链接、无知识增强 |

### 2.2 抽取质量问题

| 问题 | 典型案例 | 现有处理 |
|------|----------|----------|
| **指代消解缺失** | "樱花开了，很多人在拍照" → 主语省略 | 仅靠提示词中的CoT推断，无前置消解 |
| **幻觉三元组** | 生成原文不存在的关系 | Eval评分+规则校验，但可能遗漏 |
| **方向错误** | A-位于-B vs B-位于-A | Eval一致性评分，但无Schema约束 |
| **粒度不一致** | "武汉大学" vs "武大" | Aggregator阶段去重，但NER已重复识别 |
| **边界模糊** | "街道口商圈" 是街区还是POI？ | 静态四类别，无动态判断 |

### 2.3 效率与成本问题

| 问题 | 影响 | 现有方案 |
|------|------|----------|
| **LLM调用冗余** | 成本高 | 简化评估模式（单次调用） |
| **并行度受限** | 批量处理吞吐瓶颈 | Worker数量上限限制（max_workers） |
| **重复计算** | 相似语料重复抽取 | 无语料相似度检测 |

---

## 三、创新改进建议（按创新度排序）

### 3.1 【高创新】自适应动态路由 Agent

**核心理念**：根据文本特征动态选择处理路径，而非固定流水线

```text
                    ┌─────────────────────────────────────┐
                    │         Text Analyzer Agent         │
                    │   (文本复杂度、实体密度、情感强度)    │
                    └─────────────────────────────────────┘
                              ↓
                    ┌─────────────────────────────────────┐
                    │         Dynamic Router              │
                    └─────────────────────────────────────┘
              ↓                    ↓                    ↓
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │ Simple Path │      │ Normal Path │      │ Deep Path   │
    │ (短文本)    │      │ (中等文本)  │      │ (复杂文本)  │
    │ NER→Label   │      │ NER→RE→Eval │      │ NER→RE→Eval │
    │             │      │             │      │ →Reflect→RE │
    └─────────────┘      └─────────────┘      └─────────────┘
```

**实现方案**：

```python
# agent/agents/nodes.py 新增
def create_text_analyzer_node(llm: Any):
    """分析文本特征，输出路由决策"""
    async def analyzer_node(state: CorpusState) -> Dict:
        features = await analyze_text_features(state["raw_text"])
        return {
            "text_features": features,
            "routing_decision": decide_path(features),
            "current_step": StepEnum.ROUTING
        }
    return analyzer_node

# routing_decision: "simple" | "normal" | "deep"
# 特征维度：length, entity_density, complexity, sentiment_strength
```

**收益**：
- 简单文本跳过RE/Eval，降低50%+成本
- 复杂文本触发反思循环，提高抽取质量
- 实现成本与质量的动态平衡

---

### 3.2 【高创新】反思驱动循环 Agent (Reflect-Driven Loop)

**核心理念**：引入整体反思节点，可触发上游重新抽取

```text
NER → RE → Eval → Reflect → [决策点]
                      ↓
              ┌───────┴───────┐
              │               │
         ┌────↓────┐    ┌─────↓─────┐
         │ ACCEPT  │    │ RE-EXTRACT│
         │ (通过)  │    │ (重抽)    │
         └────┬────┘    └─────┬─────┘
              │               │
              │        ┌──────↓──────┐
              │        │ Selective   │
              │        │ Re-NER/RE   │
              │        │ (仅问题部分)│
              │        └──────┬──────┘
              │               │
              └───────┬───────┘
                      ↓
                    Label → END
```

**实现方案**：

```python
# agent/agents/state.py 扩展
class CorpusState(TypedDict):
    # 新增字段
    reflection_result: Annotated[Dict, replace_value]
    retry_count: Annotated[int, replace_value]  # 最大3次
    problem_entities: Annotated[List[str], replace_value]  # 需重抽实体

# agent/agents/nodes.py 新增
def create_reflect_node(llm: Any):
    """整体反思：评估抽取质量，决定是否重抽"""
    async def reflect_node(state: CorpusState) -> Dict:
        reflection = await llm_reflect(
            raw_text=state["raw_text"],
            entities=state["entities"],
            triples=state["corrected_triples"]
        )
        # reflection包含：整体评分、问题分析、重抽建议
        
        if reflection["overall_score"] < 3.0 and state["retry_count"] < 3:
            return {
                "reflection_result": reflection,
                "problem_entities": reflection["problem_entities"],
                "retry_count": state["retry_count"] + 1,
                "current_step": StepEnum.RE_NER  # 回退到NER
            }
        else:
            return {
                "reflection_result": reflection,
                "current_step": StepEnum.LABEL
            }
    return reflect_node
```

**反思内容**：
- **实体覆盖度**：是否遗漏明显地理实体？
- **关系完整性**：是否遗漏关键空间关系？
- **逻辑一致性**：三元组是否存在矛盾？
- **证据充分性**：每个三元组是否有原文依据？

**收益**：
- 形成反馈闭环，而非单向流水线
- 低质量结果自动重抽，无需人工干预
- 选择性重抽（仅问题实体），避免全量重跑

---

### 3.3 【高创新】知识库增强 Agent (KG-Enhanced Extraction)

**核心理念**：利用已有知识图谱辅助抽取，实现实体链接与知识增强

```text
┌────────────────────────────────────────────────────────────┐
│                    KG Context Provider                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Neo4j Query │  │ Entity Cache│  │ Relation    │        │
│  │ (已知实体)  │  │ (高频实体)  │  │ Patterns    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└────────────────────────────────────────────────────────────┘
                      ↓
              ┌───────────────┐
              │ KG-Enhanced   │
              │ NER Context   │
              │ (候选实体提示)│
              └───────────────┘
                      ↓
                    NER Agent
                      ↓
              ┌───────────────┐
              │ Entity Linking│
              │ (链接到KG实体)│
              └───────────────┘
```

**实现方案**：

```python
# agent/agents/nodes.py 扩展
async def kg_enhanced_ner_node(state: CorpusState, kg_client: Neo4jClient) -> Dict:
    """知识库增强的NER"""
    # 1. 查询KG中相似实体作为候选
    kg_candidates = await kg_client.search_entities_by_text(state["raw_text"])
    
    # 2. 构建增强提示词
    enhanced_prompt = NER_PROMPT.invoke({
        "raw_text": state["raw_text"],
        "kg_candidates": format_kg_candidates(kg_candidates)  # 新增
    })
    
    # 3. LLM抽取 + 实体链接
    result = await structured_llm.ainvoke(enhanced_prompt)
    
    # 4. 尝试链接到KG实体
    linked_entities = []
    for entity in result.entities:
        kg_match = await kg_client.find_matching_entity(entity)
        if kg_match:
            linked_entities.append({
                "name": entity,
                "kg_id": kg_match["id"],
                "aliases": kg_match["aliases"],
                "is_new": False
            })
        else:
            linked_entities.append({
                "name": entity,
                "is_new": True
            })
    
    return {"entities": linked_entities, ...}
```

**知识库增强内容**：
- **候选实体提示**：从KG检索相关实体，作为NER候选
- **实体消歧**：同名实体通过上下文消歧
- **关系模式**：从KG学习高频关系模式，作为RE提示
- **别名发现**：利用KG中的别名辅助实体识别

**收益**：
- 降低幻觉：已知实体直接链接，不重新生成
- 提高一致性：同名实体统一标识
- 知识积累：每次抽取增量更新KG

---

### 3.4 【中创新】联合抽取 Agent (Joint NER-RE)

**核心理念**：端到端联合抽取实体和关系，减少信息流失

```text
当前流水线:
NER → entities → RE → triples
      (信息流失: 实体位置、置信度、上下文)

联合抽取:
┌────────────────────────────────────┐
│        Joint Extraction Agent      │
│  输入: raw_text                    │
│  输出: entities + triples + spans  │
│  (单次LLM调用，保留完整上下文)      │
└────────────────────────────────────┘
```

**实现方案**：

```python
# agent/agents/schemas.py 新增
class JointExtractionResult(BaseModel):
    """联合抽取结果"""
    entities: List[EntityWithSpan]  # 实体+文本位置
    triples: List[TripleWithSpan]   # 三元组+证据span
    confidence: float               # 整体置信度

class EntityWithSpan(BaseModel):
    name: str
    type: str  # Road/POI/Building/Block
    span: Tuple[int, int]  # 文本位置[start, end]
    confidence: float

class TripleWithSpan(BaseModel):
    head: str
    relation: str
    tail: str
    evidence_span: Tuple[int, int]  # 证据位置
    confidence: float

# agent/agents/nodes.py 新增
def create_joint_extraction_node(llm: Any):
    """联合抽取节点"""
    structured_llm = llm.with_structured_output(JointExtractionResult)
    
    JOINT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "地理语义联合抽取专家..."),
        ("human", """从文本中联合抽取实体和关系...
        输出每个实体的文本位置和置信度。
        输出每个三元组的证据位置和置信度。
        
        文本: {raw_text}""")
    ])
    
    async def joint_node(state: CorpusState) -> Dict:
        result = await structured_llm.ainvoke(
            JOINT_PROMPT.invoke({"raw_text": state["raw_text"]})
        )
        return {
            "entities": result.entities,
            "triples": result.triples,
            "joint_confidence": result.confidence,
            "current_step": StepEnum.EVAL  # 跳过RE
        }
    
    return joint_node
```

**收益**：
- 单次LLM调用完成NER+RE，降低成本
- 保留实体位置信息，增强可解释性
- 实体与关系一致性更高

---

### 3.5 【中创新】多粒度层次化抽取 Agent

**核心理念**：句子级→段落级→篇章级的层次化抽取，捕捉多层级语义

```text
┌─────────────────────────────────────────────────────┐
│              Hierarchical Extraction                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────┐                                   │
│  │ Sentence    │ → 实体 + 局部关系                  │
│  │ Level       │   (精确span)                      │
│  └─────────────┘                                   │
│         ↓                                          │
│  ┌─────────────┐                                   │
│  │ Paragraph   │ → 跨句关系 + 隐式主语              │
│  │ Level       │   (指代消解)                      │
│  └─────────────┘                                   │
│         ↓                                          │
│  ┌─────────────┐                                   │
│  │ Document    │ → 整体结构关系                    │
│  │ Level       │   (空间拓扑)                      │
│  └─────────────┘                                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**实现方案**：

```python
# agent/agents/nodes.py 新增
async def hierarchical_extraction(state: CorpusState) -> Dict:
    """层次化抽取"""
    text = state["raw_text"]
    
    # Level 1: 句子级抽取
    sentences = split_sentences(text)
    sentence_results = []
    for sent in sentences:
        sent_result = await sentence_level_extract(sent)
        sentence_results.append(sent_result)
    
    # Level 2: 段落级合并 + 指代消解
    paragraph_entities = merge_entities(sentence_results)
    paragraph_triples = await resolve_references(
        sentence_results, paragraph_entities
    )
    
    # Level 3: 篇章级空间关系推理
    document_triples = await infer_spatial_relations(
        paragraph_entities, paragraph_triples
    )
    
    return {
        "entities": paragraph_entities,
        "triples": paragraph_triples + document_triples,
        "hierarchy_metadata": {
            "sentence_count": len(sentences),
            "cross_sentence_relations": len(document_triples)
        }
    }
```

**收益**：
- 捕捉跨句隐式关系（指代消解）
- 发现篇章级空间拓扑关系
- 保留细粒度证据位置

---

### 3.6 【中创新】Self-Consistency投票 Agent

**核心理念**：多次采样+一致性投票，提高抽取稳定性

```text
┌─────────────────────────────────────────────────────┐
│              Self-Consistency Voting                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│        ┌───────────┐                               │
│        │ Sample 1  │ → entities_1, triples_1       │
│        └───────────┘                               │
│        ┌───────────┐                               │
│        │ Sample 2  │ → entities_2, triples_2       │
│        └───────────┘                               │
│        ┌───────────┐                               │
│        │ Sample 3  │ → entities_3, triples_3       │
│        └───────────┘                               │
│              ↓                                     │
│        ┌───────────┐                               │
│        │ Voting    │ → 合并一致结果                │
│        │ & Merge   │   过滤分歧结果                │
│        └───────────┘                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**实现方案**：

```python
async def self_consistency_extraction(
    state: CorpusState, 
    llm: Any,
    num_samples: int = 3
) -> Dict:
    """多次采样+投票"""
    samples = []
    for i in range(num_samples):
        # 温度=0.3保持多样性
        sample_llm = llm.bind(temperature=0.3)
        result = await ner_re_extract(sample_llm, state)
        samples.append(result)
    
    # 实体投票：出现次数 >= num_samples/2
    entity_votes = defaultdict(int)
    for sample in samples:
        for entity in sample["entities"]:
            entity_votes[entity] += 1
    
    threshold = num_samples // 2 + 1
    consensus_entities = [
        e for e, count in entity_votes.items() 
        if count >= threshold
    ]
    
    # 三元组投票：相似三元组聚类
    triple_clusters = cluster_similar_triples(samples["triples"])
    consensus_triples = [
        t for cluster in triple_clusters
        if len(cluster) >= threshold
        for t in cluster[0]  # 取代表性三元组
    ]
    
    return {
        "entities": consensus_entities,
        "triples": consensus_triples,
        "confidence": len(consensus_entities) / len(entity_votes)  # 一致性分数
    }
```

**收益**：
- 降低单次抽取的随机性
- 过滤低置信度结果（分歧结果）
- 提供一致性置信度分数

---

### 3.7 【中创新】Schema约束矩阵 Agent

**核心理念**：实体类型×关系类型的兼容矩阵，防止类型错误

```python
# agent/agents/config.py 新增
ENTITY_RELATION_SCHEMA = {
    # 关系类型: (head_type允许, tail_type允许)
    "连接": (
        ["道路", "POI"],  # head: 道路或POI
        ["道路", "POI", "街区"]  # tail: 道路/POI/街区
    ),
    "位于": (
        ["POI", "建筑物", "街区"],  # head: 地点
        ["道路", "街区", "POI", "建筑物"]  # tail: 空间范围
    ),
    "承载活动": (
        ["POI", "建筑物", "街区"],  # head: 场所
        ["活动", "情感"]  # tail: 活动/情感（新实体类型）
    ),
    "引发情感": (
        ["POI", "建筑物", "街区", "道路"],  # head: 地理实体
        ["情感"]  # tail: 情感（新实体类型）
    ),
    "属于": (
        ["POI", "建筑物", "街区"],  # head: 子实体
        ["POI", "街区", "道路"]  # tail: 父实体
    ),
}

def validate_schema_compatibility(triple: Dict) -> bool:
    """验证三元组Schema兼容性"""
    relation = triple["relation"]
    head_type = get_entity_type(triple["head"])
    tail_type = get_entity_type(triple["tail"])
    
    if relation not in ENTITY_RELATION_SCHEMA:
        return True  # 未知关系允许
    
    allowed_head, allowed_tail = ENTITY_RELATION_SCHEMA[relation]
    return head_type in allowed_head and tail_type in allowed_tail
```

**收益**：
- 防止类型错误（如"武汉大学-承载活动-樱花"）
- 约束关系方向
- 可扩展为动态Schema学习

---

### 3.8 【低创新但高收益】前置归一化节点

**核心理念**：文本预处理，消解省略主语和模糊描述

```text
START → Normalize → NER → RE → Eval → Label → END
```

**实现方案**：

```python
NORMALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "文本语义归一化专家"),
    ("human", """将社交媒体文本改写为显式主语的标准句式。
    
    规则：
    1. 补充省略的主语（如"拍照很好看" → "XX地点拍照很好看")
    2. 展开模糊描述（如"这里" → 具体地点名）
    3. 保留原文语义，不添加新信息
    
    原文: {raw_text}
    改写后的文本:""")
])

async def normalize_node(state: CorpusState) -> Dict:
    normalized_text = await llm.ainvoke(
        NORMALIZE_PROMPT.invoke({"raw_text": state["raw_text"]})
    )
    return {
        "normalized_text": normalized_text,  # 新字段
        "current_step": StepEnum.NER
    }
```

**收益**：
- 降低NER/RE的推断难度
- 减少幻觉（明确主语）
- 低成本（单次调用）

---

## 四、改进优先级与实施路线

### 4.1 Quick Wins（快速落地）

| 改进 | 文件 | 预期收益 | 实施复杂度 |
|------|------|----------|------------|
| **前置归一化节点** | workflow.py, nodes.py | 降低幻觉15%+ | 低 |
| **Schema约束矩阵** | config.py, nodes.py | 减少类型错误20%+ | 低 |
| **证据span输出** | schemas.py, nodes.py | 增强可解释性 | 低 |

### 4.2 中期演进（1-2周）

| 改进 | 文件 | 预期收益 | 实施复杂度 |
|------|------|----------|------------|
| **Self-Consistency投票** | nodes.py | 提高稳定性10%+ | 中 |
| **联合抽取Agent** | schemas.py, workflow.py | 降低成本20%+ | 中 |
| **知识库增强** | nodes.py, kg_client.py | 提高一致性15%+ | 中 |

### 4.3 长期方向（1个月+）

| 改进 | 文件 | 预期收益 | 实施复杂度 |
|------|------|----------|------------|
| **自适应动态路由** | workflow.py, nodes.py | 成本/质量平衡 | 高 |
| **反思驱动循环** | state.py, workflow.py | 自动质量修复 | 高 |
| **多粒度层次化** | nodes.py | 捕捉跨句关系 | 高 |

---

## 五、架构重构建议

### 5.1 状态扩展

```python
# agent/agents/state.py 扩展 CorpusState
class CorpusState(TypedDict):
    # ===== 新增字段 =====
    
    # 自适应路由
    text_features: Annotated[Dict, replace_value]  # 文本特征
    routing_decision: Annotated[str, replace_value]  # 路由决策
    
    # 反思循环
    reflection_result: Annotated[Dict, replace_value]  # 反思结果
    retry_count: Annotated[int, replace_value]  # 重试次数
    problem_entities: Annotated[List[str], replace_value]  # 问题实体
    
    # 知识库增强
    kg_linked_entities: Annotated[List[Dict], replace_value]  # KG链接
    normalized_text: Annotated[str, replace_value]  # 归一化文本
    
    # 联合抽取
    entity_spans: Annotated[List[Dict], replace_value]  # 实体位置
    evidence_spans: Annotated[List[Dict], replace_value]  # 证据位置
    joint_confidence: Annotated[float, replace_value]  # 联合置信度
```

### 5.2 工作流重构

```python
# agent/agents/workflow.py 重构
def build_adaptive_workflow(llm: Any, config: ExtractionConfig) -> CompiledStateGraph:
    """自适应动态路由工作流"""
    
    builder = StateGraph(CorpusState)
    
    # 添加节点
    builder.add_node("normalize", create_normalize_node(llm))
    builder.add_node("analyzer", create_text_analyzer_node(llm))
    builder.add_node("ner", create_ner_node(llm))
    builder.add_node("re", create_re_node(llm))
    builder.add_node("eval", create_eval_simplified_node(llm))
    builder.add_node("reflect", create_reflect_node(llm))
    builder.add_node("label", create_label_node(llm))
    
    # 动态路由
    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "analyzer")
    builder.add_conditional_edges("analyzer", route_by_complexity)
    
    # Simple path: analyzer → ner → label → END
    # Normal path: analyzer → ner → re → eval → label → END
    # Deep path: analyzer → ner → re → eval → reflect → [ner/re/label]
    
    # 反思循环
    builder.add_conditional_edges("reflect", route_after_reflect)
    
    return builder.compile(checkpointer=InMemorySaver())
```

---

## 六、评估指标建议

### 6.1 核心指标

| 指标 | 定义 | 目标 |
|------|------|------|
| **三元组准确率** | 正确三元组 / 总抽取三元组 | ≥85% |
| **三元组召回率** | 正确三元组 / 应有三元组 | ≥75% |
| **实体F1** | 实体准确率与召回率调和平均 | ≥80% |
| **幻觉率** | 无依据三元组 / 总抽取三元组 | ≤10% |
| **方向错误率** | 方向错误三元组 / 总抽取三元组 | ≤5% |

### 6.2 成本指标

| 指标 | 定义 | 目标 |
|------|------|------|
| **LLM调用次数** | 单条语料平均调用数 | ≤4次 |
| **处理时间** | 单条语料平均处理时间 | ≤5s |
| **Token消耗** | 单条语料平均Token数 | ≤2000 |

### 6.3 对比实验

| 实验组 | 配置 | 评估内容 |
|--------|------|----------|
| **Baseline** | 当前workflow | 基准性能 |
| **Normalize** | +归一化节点 | 幻觉率对比 |
| **Schema** | +约束矩阵 | 类型错误对比 |
| **Voting** | +Self-Consistency(n=3) | 稳定性对比 |
| **Joint** | +联合抽取 | 成本对比 |
| **Reflect** | +反思循环 | 质量对比 |

---

## 七、风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| **成本上升** | Voting/Reflect增加LLM调用 | 配置开关，按文本复杂度启用 |
| **稳定性下降** | 多节点链路增加失败点 | 统一RetryPolicy，降级策略 |
| **知识库依赖** | KG未初始化时无法增强 | 空KG时自动回退到baseline |
| **Schema过严** | 误杀长尾关系类型 | 灰度开关，日志分析 |
| **反思过度** | 无限循环重抽 | 最大retry_count=3限制 |

---

## 九、【新增】双重评审 + 智能跳过方案

### 9.1 方案概述

**核心理念**：自评效率高 + 交叉评价客观性强，两者互补验证，并通过智能跳过机制控制成本。

```text
Phase 1: EXTRACT + SELF_EVAL
┌─────────────────────────────────────────────────┐
│ Worker并行执行:                                  │
│   语料 → NER → RE → SelfEval(自评) → 输出待验证 │
│                                                 │
│ 输出: {corpus_id, triples, self_scores,        │
│        self_corrected, raw_text}               │
└─────────────────────────────────────────────────┘

Phase 2: CROSS_EVAL (等待所有Phase 1完成后)
┌─────────────────────────────────────────────────┐
│ 随机分配交叉评价任务 (智能跳过机制):             │
│   - 自评≥4.5 → 高置信度，跳过交叉评价           │
│   - 自评<2.5 → 低置信度，直接不通过，跳过交叉    │
│   - 自评2.5-4.5 → 中置信度，进行交叉评价        │
│                                                 │
│ EvalWorker_1 ← 语料B的结果 + 原文               │
│ EvalWorker_2 ← 语料A的结果 + 原文               │
└─────────────────────────────────────────────────┘

Phase 3: ARBITRATE + LABEL + REDUCE
┌─────────────────────────────────────────────────┐
│ 仲裁机制:                                       │
│   - 两轮评分差异 ≤1 → 取平均，高置信度          │
│   - 两轮评分差异 1-2 → 取较低分，标记需复核     │
│   - 两轮评分差异 >2 → 直接不通过，待复核        │
│                                                 │
│ Label → Aggregator → Finalizer                  │
└─────────────────────────────────────────────────┘
```

### 9.2 优势分析

| 优势 | 说明 |
|------|------|
| **互补验证** | 自评快速发现明显错误，交叉评价发现隐藏问题 |
| **防止自我辩护** | 交叉评价可纠正LLM"放过"的错误 |
| **渐进修正** | 自评先修正 → 交叉评价再处理 → 仲裁最终决定 |
| **成本可控** | 智能跳过机制减少30-45%交叉评价调用 |
| **置信度分级** | 高/中/低三级置信度，便于后续处理决策 |

### 9.3 潜在问题与解决方案

| 问题 | 说明 | 解决方案 |
|------|------|----------|
| **评分不一致** | 自评5分，交叉评价2分 | 仲裁机制：取较低分（保守）+ 标记复核 |
| **修正冲突** | 自评修正A，交叉评价修正B | 交叉评价修正覆盖自评，或标记冲突 |
| **延迟增加** | 交叉评价需等待所有Worker完成 | 两阶段调度，预编译workflow减少编译开销 |
| **Worker分配** | 需确保不自评自己的结果 | eval_dispatcher节点随机分配+排除约束 |

### 9.4 评分仲裁策略

```python
def arbitrate_scores(
    self_eval: Dict,
    cross_eval: Dict,
    threshold: float = 3.5
) -> Dict:
    """
    仲裁自评和交叉评价结果
    
    策略:
    1. 两轮评分差异小(≤1) → 取平均，高置信度
    2. 两轮评分差异中(1-2) → 取较低分，中等置信度 + 标记需复核
    3. 两轮评分差异大(>2) → 直接不通过，低置信度 + 标记需复核
    """
    arbitration_result = {
        "triple": self_eval["triple"],
        "final_score": 0,
        "confidence": "high",  # high/medium/low
        "needs_review": False,
        "passed": False,
    }
    
    self_avg = (self_eval["SEM"] + self_eval["FAC"] + self_eval["CON"]) / 3
    cross_avg = (cross_eval["SEM"] + cross_eval["FAC"] + cross_eval["CON"]) / 3
    
    score_diff = abs(self_avg - cross_avg)
    
    if score_diff <= 1.0:
        # 评分一致，高置信度
        arbitration_result["final_score"] = (self_avg + cross_avg) / 2
        arbitration_result["confidence"] = "high"
        arbitration_result["passed"] = arbitration_result["final_score"] >= threshold
        
    elif score_diff <= 2.0:
        # 评分差异中等，取较低分（保守）
        arbitration_result["final_score"] = min(self_avg, cross_avg)
        arbitration_result["confidence"] = "medium"
        arbitration_result["needs_review"] = True
        arbitration_result["passed"] = arbitration_result["final_score"] >= threshold
        
    else:
        # 评分差异大，低置信度，标记复核
        arbitration_result["final_score"] = min(self_avg, cross_avg)
        arbitration_result["confidence"] = "low"
        arbitration_result["needs_review"] = True
        arbitration_result["passed"] = False  # 大差异直接不通过，待复核
    
    return arbitration_result
```

### 9.5 智能跳过机制

```python
def should_skip_cross_eval(triple: Dict, self_avg_score: float) -> bool:
    """
    判断是否跳过交叉评价
    
    策略:
    - 规则校验失败 → 直接跳过（成本节省5-10%）
    - 自评高置信度(≥4.5) → 免交叉评价，直接通过（成本节省10-15%）
    - 自评低置信度(<2.5) → 直接不通过，跳过交叉评价（成本节省20-30%）
    - 自评中置信度(2.5-4.5) → 需交叉评价
    """
    # 规则校验失败 → 直接跳过
    if not triple.get("_rule_valid", True):
        triple["passed_eval"] = False
        triple["confidence"] = "rule_failed"
        triple["skip_reason"] = triple.get("_rule_issues", [])
        return True
    
    # 自评高置信度 → 免交叉评价
    if self_avg_score >= 4.5:
        triple["passed_eval"] = True
        triple["confidence"] = "high_self"
        triple["final_score"] = self_avg_score
        return True
    
    # 自评低置信度 → 直接不通过
    if self_avg_score < 2.5:
        triple["passed_eval"] = False
        triple["confidence"] = "low_self"
        triple["final_score"] = self_avg_score
        return True
    
    # 中置信度 → 需交叉评价
    return False


def estimate_cross_eval_skip_rate(self_eval_results: List[Dict]) -> Dict:
    """
    估算交叉评价跳过率（用于成本预估）
    """
    high_confidence = sum(1 for r in self_eval_results if r.get("avg_score", 0) >= 4.5)
    low_confidence = sum(1 for r in self_eval_results if r.get("avg_score", 0) < 2.5)
    rule_failed = sum(1 for r in self_eval_results if not r.get("_rule_valid", True))
    
    total = len(self_eval_results)
    skip_count = high_confidence + low_confidence + rule_failed
    
    return {
        "total_triples": total,
        "skip_count": skip_count,
        "skip_rate": skip_count / total if total > 0 else 0,
        "high_confidence_count": high_confidence,
        "low_confidence_count": low_confidence,
        "rule_failed_count": rule_failed,
        "estimated_cost_reduction": f"{skip_count / total * 100:.1f}%" if total > 0 else "0%"
    }
```

### 9.6 状态扩展

```python
# agent/agents/state.py 扩展 KGState
class KGState(TypedDict):
    # ... 原有字段 ...
    
    # 双重评审支持
    pending_cross_eval: Annotated[Dict[str, Dict], replace_value]
    """待交叉评价的语料结果 {corpus_id: {triples, self_scores, raw_text}}"""
    
    eval_assignments: Annotated[Dict[str, List[str]], replace_value]
    """交叉评价任务分配 {eval_worker_id: [corpus_ids]}"""
    
    cross_eval_results: Annotated[List[Dict], merge_list]
    """交叉评价结果"""
    
    arbitration_results: Annotated[Dict[str, List[Dict]], replace_value]
    """仲裁结果 {corpus_id: [arbitrated_triples]}"""
    
    needs_review_queue: Annotated[List[Dict], merge_list]
    """需要人工复核的三元组队列"""
    
    eval_skip_stats: Annotated[Dict, replace_value]
    """智能跳过统计 {skip_count, skip_rate, cost_reduction}"""
```

### 9.7 工作流实现

```python
# agent/agents/workflow.py 新增双重评审工作流
def build_dual_eval_workflow(llm: Any, config: ExtractionConfig) -> CompiledStateGraph:
    """
    双重评审工作流
    
    Phase 1: Extract + SelfEval
    Phase 2: CrossEval (等待Phase 1完成后，智能跳过)
    Phase 3: Arbitrate + Label + Reduce
    """
    builder = StateGraph(KGState)
    
    # Phase 1
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("extractors", create_extractors_with_self_eval_node(llm, config))
    builder.add_node("eval_dispatcher", create_eval_dispatcher_node(config))
    
    # Phase 2
    builder.add_node("cross_eval_workers", create_cross_eval_workers_node(llm, config))
    builder.add_node("arbitrator", create_arbitrator_node(config))
    
    # Phase 3
    builder.add_node("labeler", create_batch_labeler_node(llm, config))
    builder.add_node("aggregator", create_aggregator_node(config.similarity_threshold))
    builder.add_node("finalizer", finalizer_node)
    
    # 边
    builder.add_edge(START, "coordinator")
    builder.add_edge("coordinator", "extractors")
    builder.add_edge("extractors", "eval_dispatcher")
    builder.add_edge("eval_dispatcher", "cross_eval_workers")
    builder.add_edge("cross_eval_workers", "arbitrator")
    builder.add_edge("arbitrator", "labeler")
    builder.add_edge("labeler", "aggregator")
    builder.add_edge("aggregator", "finalizer")
    builder.add_edge("finalizer", END)
    
    return builder.compile(checkpointer=InMemorySaver())
```

### 9.8 节点实现

```python
# agent/agents/nodes.py 新增节点

def create_extractors_with_self_eval_node(llm: Any, config: ExtractionConfig):
    """Phase 1: 抽取 + 自评（含智能跳过标记）"""
    
    async def extractors_node(state: KGState, writer: StreamWriter) -> Dict:
        # 预编译工作流（包含自评）
        corpus_workflow = build_corpus_workflow(llm, use_simplified_eval=True)
        
        async def process_corpus(corpus: Dict) -> Dict:
            # NER + RE + SelfEval + 规则校验
            initial_state = build_initial_corpus_state(corpus, config)
            thread_config = {"configurable": {"thread_id": f"corpus_{corpus['id']}"}}
            result = await corpus_workflow.ainvoke(initial_state, thread_config)
            
            # 计算自评平均分并标记智能跳过
            triples_with_skip_flag = []
            for triple in result["corrected_triples"]:
                avg_score = (
                    triple.get("sem_score", 0) + 
                    triple.get("fac_score", 0) + 
                    triple.get("con_score", 0)
                ) / 3
                triple["self_avg_score"] = avg_score
                triple["skip_cross_eval"] = should_skip_cross_eval(triple, avg_score)
                triples_with_skip_flag.append(triple)
            
            return {
                "corpus_id": result["corpus_id"],
                "worker_id": f"extractor_{corpus.get('id', 'unknown')}",
                "triples": triples_with_skip_flag,
                "self_scores": result["eval_scores"],
                "raw_text": result["raw_text"],
                "entities": result["entities"],
            }
        
        # 并行处理
        tasks = [process_corpus(c) for c in state["corpus_list"]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计跳过率
        all_triples = []
        for r in results:
            if not isinstance(r, Exception):
                all_triples.extend(r.get("triples", []))
        
        skip_stats = estimate_cross_eval_skip_rate(all_triples)
        logger.info(f"[Extractors] 智能跳过率: {skip_stats['skip_rate']:.1%}")
        
        # 构建待交叉评价数据（仅中置信度）
        pending_cross_eval = {}
        for r in results:
            if not isinstance(r, Exception):
                need_cross_triples = [
                    t for t in r["triples"] 
                    if not t.get("skip_cross_eval", False)
                ]
                if need_cross_triples:
                    pending_cross_eval[r["corpus_id"]] = {
                        "triples": need_cross_triples,
                        "self_scores": r["self_scores"],
                        "raw_text": r["raw_text"],
                        "entities": r["entities"],
                    }
        
        return {
            "pending_cross_eval": pending_cross_eval,
            "extractor_results": [r for r in results if not isinstance(r, Exception)],
            "eval_skip_stats": skip_stats,
            "current_phase": PhaseEnum.CROSS_EVAL,
        }
    
    return extractors_node


def create_eval_dispatcher_node(config: ExtractionConfig):
    """Phase 1→2 过渡：分配交叉评价任务（排除自评）"""
    
    def dispatcher_node(state: KGState) -> Dict:
        corpus_ids = list(state["pending_cross_eval"].keys())
        
        # 获取抽取这些语料的Worker ID
        corpus_to_extractor = {}
        for r in state.get("extractor_results", []):
            corpus_to_extractor[r["corpus_id"]] = r["worker_id"]
        
        # 创建评价Worker
        eval_worker_ids = [f"eval_worker_{i}" for i in range(config.max_workers)]
        
        # 随机分配，确保不自评
        import random
        random.seed(int(time.time()))
        
        eval_assignments = defaultdict(list)
        for corpus_id in corpus_ids:
            # 排除抽取该语料的Worker
            eligible_evaluators = [
                w for w in eval_worker_ids 
                if w != corpus_to_extractor.get(corpus_id, "")
            ]
            
            if eligible_evaluators:
                assigned = random.choice(eligible_evaluators)
                eval_assignments[assigned].append(corpus_id)
        
        logger.info(f"[Dispatcher] 分配 {len(corpus_ids)} 条语料到 {len(eval_assignments)} 个评价Worker")
        
        return {
            "eval_assignments": dict(eval_assignments),
            "active_eval_workers": list(eval_assignments.keys()),
        }
    
    return dispatcher_node


def create_cross_eval_workers_node(llm: Any, config: ExtractionConfig):
    """Phase 2: 交叉评价（仅评价中置信度三元组）"""
    structured_llm = llm.with_structured_output(EvalResultSimplified)
    
    async def cross_eval_node(state: KGState, writer: StreamWriter) -> Dict:
        async def eval_corpus(corpus_id: str, data: Dict) -> Dict:
            writer({
                "step": "cross_eval",
                "corpus_id": corpus_id,
                "status": "started"
            })
            
            messages = EVAL_PROMPT_SIMPLIFIED.invoke({
                "triples": format_triples(data["triples"]),
                "raw_text": data["raw_text"],
            })
            result = await structured_llm.ainvoke(messages)
            
            # 处理评分
            cross_scores = [
                {
                    "triple": {
                        "head": s.triple.head,
                        "relation": s.triple.relation,
                        "tail": s.triple.tail,
                    },
                    "SEM": s.SEM,
                    "FAC": s.FAC,
                    "CON": s.CON,
                }
                for s in result.scores
            ]
            
            # 应用修正
            if result.need_correction and result.corrections:
                corrected_triples = apply_llm_corrections(data["triples"], result.corrections)
            else:
                corrected_triples = data["triples"]
            
            writer({
                "step": "cross_eval",
                "corpus_id": corpus_id,
                "status": "completed",
                "triple_count": len(corrected_triples)
            })
            
            return {
                "corpus_id": corpus_id,
                "cross_scores": cross_scores,
                "cross_corrected_triples": corrected_triples,
                "cross_corrections": result.corrections if result.need_correction else [],
            }
        
        # 每个评价Worker处理分配的语料
        tasks = []
        for worker_id, corpus_ids in state["eval_assignments"].items():
            for corpus_id in corpus_ids:
                data = state["pending_cross_eval"].get(corpus_id)
                if data:
                    tasks.append(eval_corpus(corpus_id, data))
        
        cross_eval_results = await asyncio.gather(*tasks)
        
        logger.info(f"[CrossEval] 完成 {len(cross_eval_results)} 条语料的交叉评价")
        
        return {
            "cross_eval_results": cross_eval_results,
            "current_phase": PhaseEnum.ARBITRATE,
        }
    
    return cross_eval_node


def create_arbitrator_node(config: ExtractionConfig):
    """Phase 3: 仲裁自评和交叉评价，合并跳过的结果"""
    
    def arbitrator_node(state: KGState) -> Dict:
        arbitration_results = {}
        needs_review_queue = []
        
        # 处理每个语料
        for extractor_result in state.get("extractor_results", []):
            corpus_id = extractor_result["corpus_id"]
            all_triples = extractor_result["triples"]
            
            # 查找对应的交叉评价结果
            cross_result = None
            for cr in state.get("cross_eval_results", []):
                if cr["corpus_id"] == corpus_id:
                    cross_result = cr
                    break
            
            arbitrated_triples = []
            
            for triple in all_triples:
                # 跳过交叉评价的：直接使用自评结果
                if triple.get("skip_cross_eval", False):
                    if triple.get("passed_eval", False):
                        arbitrated_triples.append(triple)
                    continue
                
                # 需交叉评价的：进行仲裁
                if cross_result:
                    triple_key = (triple["head"], triple["relation"], triple["tail"])
                    
                    # 构建自评评分
                    self_eval = {
                        "triple": triple_key,
                        "SEM": triple.get("sem_score", 3),
                        "FAC": triple.get("fac_score", 3),
                        "CON": triple.get("con_score", 3),
                    }
                    
                    # 查找交叉评价评分
                    cross_eval = find_score_for_triple(triple_key, cross_result["cross_scores"])
                    
                    if cross_eval:
                        arbitration = arbitrate_scores(self_eval, cross_eval, config.eval_threshold)
                        
                        # 应用仲裁结果
                        triple["final_score"] = arbitration["final_score"]
                        triple["confidence"] = arbitration["confidence"]
                        triple["passed_eval"] = arbitration["passed"]
                        
                        if arbitration["needs_review"]:
                            needs_review_queue.append({
                                "triple": triple,
                                "corpus_id": corpus_id,
                                "arbitration": arbitration,
                                "self_score": self_eval,
                                "cross_score": cross_eval,
                            })
                        
                        if arbitration["passed"]:
                            arbitrated_triples.append(triple)
            
            arbitration_results[corpus_id] = arbitrated_triples
        
        # 统计
        total_passed = sum(len(v) for v in arbitration_results.values())
        total_review = len(needs_review_queue)
        logger.info(f"[Arbitrator] 通过: {total_passed}, 待复核: {total_review}")
        
        return {
            "arbitration_results": arbitration_results,
            "needs_review_queue": needs_review_queue,
            "current_phase": PhaseEnum.LABEL,
        }
    
    return arbitrator_node


def find_score_for_triple(triple_key: tuple, scores: List[Dict]) -> Optional[Dict]:
    """查找三元组对应的评分"""
    for score in scores:
        score_key = (
            score["triple"]["head"],
            score["triple"]["relation"],
            score["triple"]["tail"]
        )
        if score_key == triple_key:
            return score
    return None
```

### 9.9 成本预估

| 场景 | 自评调用 | 交叉评价调用 | 总调用 | 成本对比 |
|------|----------|--------------|--------|----------|
| **Baseline（简化评估）** | 1次 | 0次 | 1次 | 基准 |
| **双重评审（无跳过）** | 1次 | 1次 | 2次 | +100% |
| **双重评审+智能跳过** | 1次 | 0.55-0.7次 | 1.55-1.7次 | +55-70% |

**智能跳过预期效果**：
- 高置信度三元组（≥4.5）：约占15-20%
- 低置信度三元组（<2.5）：约占20-30%
- 规则校验失败：约占5-10%
- **总跳过率**：40-60%
- **实际交叉评价调用**：仅评价剩余40-60%的中置信度三元组

### 9.10 配置扩展

```python
# agent/agents/config.py 扩展
@dataclass
class ExtractionConfig:
    # ... 原有字段 ...
    
    # 双重评审配置
    enable_dual_eval: bool = True
    """是否启用双重评审模式"""
    
    dual_eval_skip_threshold_high: float = 4.5
    """自评高分跳过阈值（≥此值免交叉评价）"""
    
    dual_eval_skip_threshold_low: float = 2.5
    """自评低分跳过阈值（<此值直接不通过）"""
    
    dual_eval_arbitration_diff_high: float = 2.0
    """仲裁评分差异阈值（>此值直接不通过）"""
    
    dual_eval_arbitration_diff_medium: float = 1.0
    """仲裁评分差异阈值（>此值标记复核）"""
```

### 9.11 实施优先级

| 优先级 | 内容 | 预期收益 | 复杂度 |
|--------|------|----------|--------|
| **P0** | 智能跳过机制 | 成本节省40-60% | 低 |
| **P1** | 评分仲裁策略 | 提升评价质量15%+ | 低 |
| **P2** | 交叉评价节点 | 提升客观性20%+ | 中 |
| **P3** | 复核队列管理 | 可追溯+可修复 | 中 |

### 9.12 与现有改进方案的协同

| 协同方案 | 协同效果 |
|----------|----------|
| **前置归一化节点** | 归一化后自评置信度更高，跳过率提升 |
| **Schema约束矩阵** | 规则校验前置，更多三元组可跳过交叉评价 |
| **Self-Consistency投票** | 投票后结果更稳定，自评置信度更高 |
| **反思驱动循环** | 低置信度结果触发反思，而非直接不通过 |

---

## 十、下一步行动建议（更新）

### 8.1 立即实施（本周）

1. **前置归一化节点**：在NER前插入normalize节点
2. **Schema约束矩阵**：扩展rule_based_validation
3. **证据span输出**：扩展schemas.py添加span字段

### 8.2 准备工作

1. **验证集构建**：人工标注50-100条语料的正确三元组
2. **评估脚本**：自动化准确率/召回率计算
3. **配置系统**：支持开关各改进模块

### 8.3 后续迭代

1. 根据验证集评估Quick Wins效果
2. 选择效果最显著的改进继续迭代
3. 逐步实现自适应路由和反思循环

---

## 十一、【新增】二次对话验证方案 (Second Conversation Verification)

### 11.1 方案概述

**核心理念**：采用两阶段对话策略，解决大模型"幻觉"问题，实现简单、成本低、效果好。

```text
Phase 1: 初抽 (Primary Extraction)
┌─────────────────────────────────────────────────┐
│ START → NER → RE → [暂存结果] → END             │
│                                                 │
│ 输出: {entities, triples, raw_text}             │
└─────────────────────────────────────────────────┘

Phase 2: 分离校验 (Separated Self-Check)
┌─────────────────────────────────────────────────┐
│ Self-Check-NER: 专门校验实体（查遗漏、识别别名） │
│ Self-Check-RE: 专门校验三元组（查幻觉、验证关系）│
│                                                 │
│ 输入: Phase1结果 + 原文                         │
│ 输出: 最终结果 + 置信度标记                     │
└─────────────────────────────────────────────────┘
```

### 11.2 优势分析

| 优势 | 说明 |
|------|------|
| **实现极简** | 只需增加2个校验节点，无需多Agent对话机制 |
| **成本可控** | LLM调用从4次增加到6次（+50%，而非+200%） |
| **幻觉检测有效** | Self-Check独立审视初抽结果，不带"自我辩护"偏见 |
| **实体归一化** | Self-Check-NER主动识别别名（"武大"→"武汉大学"） |
| **地理验证** | Self-Check-RE检查三元组地理合理性 |
| **可追溯** | 校验过程输出修正记录，便于人工复核 |

### 11.3 与其他方案对比

| 特性 | 原方案（单轮） | 研讨会模式 | 二次对话验证 |
|------|---------------|-----------|-------------|
| LLM 调用次数 | 4次 | 9次+ | **6次** |
| Agent 间对话 | 无 | 有（复杂） | **无**（简单） |
| 实体归一化 | 后处理合并 | Agent协作确认 | **Self-Check识别** |
| 幻觉检测 | Eval自评（偏见） | Agent质疑 | **Self-Check审视** |
| 实现复杂度 | 低 | 高 | **低** |

### 11.4 流程设计详解

#### Self-Check-NER（实体校验节点）

**职责**：
1. 检查是否有遗漏实体（原文提及但未抽取）
2. 识别别名/简称，建议归一化
3. 过滤无关实体（非地理实体）
4. 给出置信度评估

**输入输出设计**：

```python
# 输入
{
    "raw_text": "武大的樱花开了，很多人在行政楼前拍照...",
    "entities": {"POI": ["武汉大学", "行政楼"], "道路": [], ...}
}

# 输出 (SelfCheckNERResult)
{
    "verified_entities": [
        {"name": "武汉大学", "type": "POI", "confidence": "high", "aliases": ["武大"]},
        {"name": "行政楼", "type": "建筑物", "confidence": "high", "aliases": []}
    ],
    "missing_entities": [
        {"name": "樱花", "suggested_type": "POI?", "reason": "原文提及但未抽取"}
    ],
    "entity_normalizations": [
        {"raw": "武大", "canonical": "武汉大学", "confidence": "high"}
    ],
    "removed_entities": [],
    "overall_confidence": "high"
}
```

**Prompt 设计**：

```python
SELF_CHECK_NER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位"实体校验专家"，负责审视NER抽取结果。
    你的任务是：检查遗漏、识别别名、过滤无关实体。"""),
    ("human", """## 校验任务
    1. 遗漏检查：原文是否提及了地理实体但未抽取？
    2. 别名识别：抽取的实体是否有简称/别名需要归一化？
    3. 无关过滤：抽取的实体是否为非地理实体？
    
    ## 已抽取实体
    {entities}
    
    ## 原始文本
    {raw_text}
    
    请输出校验结果（JSON格式），包含：verified_entities, missing_entities, entity_normalizations""")
])
```

#### Self-Check-RE（三元组校验节点）

**职责**：
1. 幻觉检测：三元组是否在原文中有依据？
2. 关系验证：关系类型和方向是否正确？
3. 证据匹配：证据是否真实存在于原文？
4. 修正建议：发现问题时的修正方案

**输入输出设计**：

```python
# 输入
{
    "raw_text": "武大的樱花开了，很多人在行政楼前拍照...",
    "triples": [
        {"head": "武汉大学", "relation": "位于", "tail": "珞喻路", "evidence": "..."},
        {"head": "行政楼", "relation": "属于", "tail": "武汉大学", "evidence": "行政楼在武汉大学内"}
    ]
}

# 输出 (SelfCheckREResult)
{
    "verified_triples": [
        {
            "head": "行政楼", "relation": "属于", "tail": "武汉大学",
            "confidence": "high",
            "evidence_valid": True,
            "evidence_match": "原文第2句提及"
        }
    ],
    "rejected_triples": [
        {
            "head": "武汉大学", "relation": "位于", "tail": "珞喻路",
            "confidence": "low",
            "reason": "幻觉：原文未提及珞喻路",
            "suggested_fix": "删除或改为<武汉大学, 位于, 珞珈山>"
        }
    ],
    "corrected_triples": [
        {
            "original": {"head": "武汉大学", "relation": "位于", "tail": "珞喻路"},
            "corrected": {"head": "武汉大学", "relation": "位于", "tail": "珞珈山"},
            "reason": "知识库显示武汉大学位于珞珈山"
        }
    ],
    "overall_confidence": "medium"
}
```

**Prompt 设计**：

```python
SELF_CHECK_RE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位"三元组校验专家"，负责审视RE抽取结果。
    你的任务是：检测幻觉、验证关系、匹配证据。"""),
    ("human", """## 校验任务
    1. 幻觉检测：三元组是否在原文中有依据？（无依据则标记为幻觉）
    2. 关系验证：关系类型和方向是否正确？
    3. 证据匹配：标注的证据是否真实存在于原文？
    
    ## 已抽取三元组
    {triples}
    
    ## 原始文本
    {raw_text}
    
    请输出校验结果（JSON格式），包含：verified_triples, rejected_triples, corrected_triples""")
])
```

### 11.5 状态扩展

```python
# agent/agents/state.py 扩展 CorpusState
class CorpusState(TypedDict):
    # ... 原有字段 ...
    
    # 二次对话验证支持
    primary_entities: Annotated[Dict, replace_value]
    """Phase 1 初抽实体"""
    
    primary_triples: Annotated[List[Dict], replace_value]
    """Phase 1 初抽三元组"""
    
    self_check_ner_result: Annotated[Dict, replace_value]
    """Self-Check-NER 校验结果"""
    
    self_check_re_result: Annotated[Dict, replace_value]
    """Self-Check-RE 校验结果"""
    
    final_entities: Annotated[List[Dict], replace_value]
    """最终实体（归一化后）"""
    
    final_triples: Annotated[List[Dict], replace_value]
    """最终三元组（校验后）"""
    
    verification_confidence: Annotated[str, replace_value]
    """整体置信度: high/medium/low"""
```

### 11.6 Schema 扩展

```python
# agent/agents/schemas.py 新增

class VerifiedEntity(BaseModel):
    """校验后的实体"""
    name: str = Field(description="实体名称（归一化后）")
    type: str = Field(description="实体类型")
    confidence: str = Field(description="置信度: high/medium/low")
    aliases: List[str] = Field(default_factory=list, description="别名列表")
    evidence: Optional[str] = Field(description="原文依据")

class EntityNormalization(BaseModel):
    """实体归一化记录"""
    raw: str = Field(description="原始名称")
    canonical: str = Field(description="归一化名称")
    confidence: str = Field(description="归一化置信度")

class SelfCheckNERResult(BaseModel):
    """Self-Check-NER 输出"""
    verified_entities: List[VerifiedEntity] = Field(default_factory=list)
    missing_entities: List[Dict] = Field(default_factory=list, description="遗漏实体建议")
    entity_normalizations: List[EntityNormalization] = Field(default_factory=list)
    removed_entities: List[str] = Field(default_factory=list, description="过滤掉的无关实体")
    overall_confidence: str = Field(default="medium")

class VerifiedTriple(BaseModel):
    """校验后的三元组"""
    head: str
    relation: str
    tail: str
    confidence: str = Field(description="置信度: high/medium/low")
    evidence_valid: bool = Field(description="证据是否有效")
    evidence_match: Optional[str] = Field(description="证据原文匹配")

class RejectedTriple(BaseModel):
    """拒绝的三元组"""
    head: str
    relation: str
    tail: str
    reason: str = Field(description="拒绝原因")
    suggested_fix: Optional[str] = Field(description="修正建议")

class TripleCorrection(BaseModel):
    """三元组修正记录"""
    original: Dict = Field(description="原始三元组")
    corrected: Dict = Field(description="修正后三元组")
    reason: str = Field(description="修正原因")

class SelfCheckREResult(BaseModel):
    """Self-Check-RE 输出"""
    verified_triples: List[VerifiedTriple] = Field(default_factory=list)
    rejected_triples: List[RejectedTriple] = Field(default_factory=list)
    corrected_triples: List[TripleCorrection] = Field(default_factory=list)
    overall_confidence: str = Field(default="medium")
```

### 11.7 工作流实现

```python
# agent/agents/workflow.py 新增二次对话验证工作流
def build_second_conversation_workflow(llm: Any, config: ExtractionConfig) -> CompiledStateGraph:
    """
    二次对话验证工作流
    
    Phase 1: NER → RE → [暂存结果]
    Phase 2: Self-Check-NER → Self-Check-RE → [输出最终结果]
    Phase 3: Label → END
    """
    builder = StateGraph(CorpusState)
    
    # Phase 1 节点
    builder.add_node("ner", create_ner_node(llm), retry_policy=LLM_RETRY_POLICY)
    builder.add_node("re", create_re_node(llm), retry_policy=LLM_RETRY_POLICY)
    
    # Phase 2 节点（新增）
    builder.add_node("self_check_ner", create_self_check_ner_node(llm), retry_policy=LLM_RETRY_POLICY)
    builder.add_node("self_check_re", create_self_check_re_node(llm), retry_policy=LLM_RETRY_POLICY)
    
    # Phase 3 节点
    builder.add_node("label", create_label_node(llm), retry_policy=LLM_RETRY_POLICY)
    
    # 边定义
    builder.add_edge(START, "ner")
    builder.add_conditional_edges("ner", route_after_ner)
    builder.add_edge("re", "self_check_ner")  # RE → Self-Check-NER
    builder.add_edge("self_check_ner", "self_check_re")  # 分离校验顺序
    builder.add_edge("self_check_re", "label")
    builder.add_edge("label", END)
    
    return builder.compile(checkpointer=InMemorySaver())
```

### 11.8 节点实现

```python
# agent/agents/nodes.py 新增节点

def create_self_check_ner_node(llm: Any):
    """创建 Self-Check-NER 节点"""
    structured_llm = llm.with_structured_output(SelfCheckNERResult)
    
    async def self_check_ner_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Phase 2a: 实体校验"""
        corpus_id = state['corpus_id']
        logger.info(f"[Self-Check-NER] 校验语料: {corpus_id}")
        
        writer({
            "step": "self_check_ner",
            "corpus_id": corpus_id,
            "status": "started",
            "message": "开始实体校验"
        })
        
        try:
            # 暂存初抽结果
            primary_entities = state["entities"]
            
            # Self-Check 校验
            messages = SELF_CHECK_NER_PROMPT.invoke({
                "raw_text": state["raw_text"],
                "entities": format_entities(primary_entities)
            })
            
            result: SelfCheckNERResult = await structured_llm.ainvoke(messages)
            
            # 应用归一化
            final_entities = apply_entity_normalizations(result)
            
            writer({
                "step": "self_check_ner",
                "corpus_id": corpus_id,
                "status": "completed",
                "verified_count": len(result.verified_entities),
                "normalization_count": len(result.entity_normalizations)
            })
            
            return {
                "primary_entities": primary_entities,
                "self_check_ner_result": result.model_dump(),
                "final_entities": final_entities,
                "current_step": StepEnum.SELF_CHECK_RE,
            }
            
        except Exception as e:
            logger.error(f"[Self-Check-NER] 失败: {e}")
            writer({
                "step": "self_check_ner",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
            return {
                "self_check_ner_result": {},
                "error": str(e),
                "current_step": StepEnum.SELF_CHECK_RE,
            }
    
    return self_check_ner_node


def create_self_check_re_node(llm: Any):
    """创建 Self-Check-RE 节点"""
    structured_llm = llm.with_structured_output(SelfCheckREResult)
    
    async def self_check_re_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Phase 2b: 三元组校验"""
        corpus_id = state['corpus_id']
        logger.info(f"[Self-Check-RE] 校验语料: {corpus_id}")
        
        writer({
            "step": "self_check_re",
            "corpus_id": corpus_id,
            "status": "started",
            "message": "开始三元组校验"
        })
        
        try:
            # 暂存初抽结果
            primary_triples = state["triples"]
            
            # Self-Check 校验
            messages = SELF_CHECK_RE_PROMPT.invoke({
                "raw_text": state["raw_text"],
                "triples": format_triples(primary_triples)
            })
            
            result: SelfCheckREResult = await structured_llm.ainvoke(messages)
            
            # 应用修正，生成最终三元组
            final_triples = apply_triple_corrections(result)
            
            # 计算整体置信度
            overall_confidence = calculate_overall_confidence(result)
            
            writer({
                "step": "self_check_re",
                "corpus_id": corpus_id,
                "status": "completed",
                "verified_count": len(result.verified_triples),
                "rejected_count": len(result.rejected_triples),
                "confidence": overall_confidence
            })
            
            return {
                "primary_triples": primary_triples,
                "self_check_re_result": result.model_dump(),
                "final_triples": final_triples,
                "verification_confidence": overall_confidence,
                "current_step": StepEnum.LABEL,
            }
            
        except Exception as e:
            logger.error(f"[Self-Check-RE] 失败: {e}")
            writer({
                "step": "self_check_re",
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e)
            })
            return {
                "self_check_re_result": {},
                "error": str(e),
                "current_step": StepEnum.LABEL,
            }
    
    return self_check_re_node


def apply_entity_normalizations(result: SelfCheckNERResult) -> List[Dict]:
    """应用实体归一化"""
    entities = []
    for ve in result.verified_entities:
        entity = {
            "name": ve.name,
            "type": ve.type,
            "confidence": ve.confidence,
            "aliases": ve.aliases,
        }
        entities.append(entity)
    return entities


def apply_triple_corrections(result: SelfCheckREResult) -> List[Dict]:
    """应用三元组修正"""
    triples = []
    
    # 保留验证通过的三元组
    for vt in result.verified_triples:
        triple = {
            "head": vt.head,
            "relation": vt.relation,
            "tail": vt.tail,
            "confidence": vt.confidence,
            "evidence_valid": vt.evidence_valid,
        }
        triples.append(triple)
    
    # 应用修正的三元组
    for tc in result.corrected_triples:
        corrected = tc.corrected
        corrected["correction_reason"] = tc.reason
        triples.append(corrected)
    
    return triples


def calculate_overall_confidence(result: SelfCheckREResult) -> str:
    """计算整体置信度"""
    total = len(result.verified_triples) + len(result.rejected_triples)
    if total == 0:
        return "high"
    
    rejected_ratio = len(result.rejected_triples) / total
    
    if rejected_ratio < 0.1:
        return "high"
    elif rejected_ratio < 0.3:
        return "medium"
    else:
        return "low"
```

### 11.9 StepEnum 扩展

```python
# agent/agents/state.py 扩展 StepEnum
class StepEnum(str, Enum):
    """工作流步骤枚举"""
    NER = "ner"
    RE = "re"
    SELF_CHECK_NER = "self_check_ner"  # 新增
    SELF_CHECK_RE = "self_check_re"    # 新增
    LABEL = "label"
    DONE = "done"
```

### 11.10 成本预估

| 场景 | NER | RE | Self-Check-NER | Self-Check-RE | Label | 总调用 | 成本对比 |
|------|-----|-----|-----------------|---------------|-------|--------|----------|
| **原方案（简化评估）** | 1次 | 1次 | 0次 | 1次(Eval) | 1次 | **4次** | 基准 |
| **二次对话验证** | 1次 | 1次 | 1次 | 1次 | 1次 | **5次** | +25% |
| **二次对话验证+知识库** | 1次 | 1次 | 1次(含KB查询) | 1次(含KB验证) | 1次 | **5次** | +25% |

**说明**：Self-Check-RE 替代了原来的 Eval 节点，所以总调用从 4次增加到 5次（而非 6次）。

### 11.11 可选增强：知识库协同

在 Self-Check 节点中可注入知识库查询，增强验证效果：

```python
def create_self_check_re_node_with_kb(llm: Any, kg_client: Neo4jClient):
    """带知识库验证的 Self-Check-RE 节点"""
    
    async def self_check_re_node(state: CorpusState, writer: StreamWriter) -> Dict:
        # 1. LLM Self-Check
        llm_result = await llm_self_check(state)
        
        # 2. 知识库验证（补充）
        for triple in llm_result.verified_triples:
            # 查询知识库中的已知关系
            kg_relation = await kg_client.find_relation(
                triple.head, triple.tail
            )
            if kg_relation:
                triple["kg_verified"] = True
                triple["kg_relation"] = kg_relation
            else:
                triple["kg_verified"] = False
        
        # 3. 合并结果
        final_triples = merge_llm_and_kb_results(llm_result)
        
        return {
            "final_triples": final_triples,
            "current_step": StepEnum.LABEL,
        }
    
    return self_check_re_node
```

### 11.12 实施优先级

| 优先级 | 内容 | 预期收益 | 复杂度 |
|--------|------|----------|--------|
| **P0** | Self-Check-RE 节点 | 幻觉检测+关系验证 | 低 |
| **P1** | Self-Check-NER 节点 | 别名识别+遗漏检查 | 低 |
| **P2** | 知识库协同验证 | 地理常识验证 | 中 |
| **P3** | 置信度分级输出 | 可追溯+可复核 | 低 |

### 11.13 与现有方案的协同

| 协同方案 | 协同效果 |
|----------|----------|
| **前置归一化节点** | Self-Check负担减轻，验证更准确 |
| **Schema约束矩阵** | Self-Check可利用Schema做类型验证 |
| **双重评审方案** | Self-Check作为Phase2，可与交叉评价叠加 |
| **反思驱动循环** | Self-Check发现的问题可触发反思重抽 |

---

## 十二、方案选择建议（更新）

### 12.1 各方案适用场景

| 方案 | 适用场景 | 成本 | 效果预期 |
|------|----------|------|----------|
| **二次对话验证** | 快速落地，解决幻觉问题 | +25% | 中等提升 |
| **双重评审+智能跳过** | 批量处理，需要客观评价 | +55-70% | 高提升 |
| **研讨会模式** | 研究场景，追求极致效果 | +200% | 最高效果 |
| **轻量协作（知识库增强）** | 已有KG数据，追求一致性 | +0%LLM | 中等提升 |

### 12.2 推荐实施顺序

1. **第一步**：二次对话验证（低成本、高收益）
2. **第二步**：Schema约束矩阵（规则验证、无LLM成本）
3. **第三步**：知识库协同（实体归一化）
4. **第四步**：根据效果决定是否升级到双重评审或研讨会模式

---

## 十三、【新增】苏格拉底式 QA 引导节点（P8）

### 13.1 方案概述

**核心理念**：通过问答驱动（QA-driven）的方式构建"语义脚手架"，在提取三元组之前系统地展开文档级深层语义，捕获直接提取流水线中容易丢失的上下文依赖关系和隐性逻辑链接。

**参考框架**：SocraticKG 框架的 5W1H 引导式问答扩展。

### 13.2 流程对比

```text
传统流程：
Filter → Normalize → NER → RE → Eval → Label

新增QA脚手架流程：
Filter → QA_Scaffold → Normalize → NER → RE → Eval → Label
         ↑
    5W1H问答扩展 → 语义脚手架 → 三元组转化辅助
```

### 13.3 5W1H 问答框架设计

| 维度 | 问题模板 | 目的 |
|------|----------|------|
| **Who（谁）** | 文中提到的地点/人物是谁？ | 捕获实体主体 |
| **What（什么）** | 这个地点有什么特征/功能？ | 捕获实体属性 |
| **When（何时）** | 描述的时间背景是什么？ | 捕获时间维度（如"樱花开了"→春季） |
| **Where（何地）** | 这些地点位于哪里？相互位置关系？ | 捕获空间关系 |
| **Why（为什么）** | 作者为什么提到这些地点？ | 捕获情感/动机 |
| **How（如何）** | 如何到达/体验这些地点？ | 捕获活动/可达方式 |

### 13.4 Pydantic 模型设计

```python
# schemas.py 新增

class QAPair(BaseModel):
    """单个问答对"""
    question: str = Field(description="5W1H引导问题")
    answer: str = Field(description="基于原文的回答")
    dimension: str = Field(description="维度标签: who/what/when/where/why/how")
    entities_involved: List[str] = Field(
        default_factory=list,
        description="涉及到的实体名称"
    )
    confidence: str = Field(default="medium", description="回答置信度")


class QAScaffoldResult(BaseModel):
    """QA脚手架输出"""
    qa_pairs: List[QAPair] = Field(
        default_factory=list,
        description="5W1H问答对列表"
    )
    semantic_summary: str = Field(
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
        description="整体脚手架置信度"
    )
    should_skip_detailed_extraction: bool = Field(
        default=False,
        description="是否建议跳过详细抽取（简单文本）"
    )
```

### 13.5 提示词模板设计

```python
# prompts.py 新增

QA_SCAFFOLD_SYSTEM = """你是一位"地理语义分析师"，擅长通过结构化问答来深入理解文本的地理语义。
你的任务是用5W1H框架生成问答对，构建语义脚手架帮助后续提取更准确。"""

QA_SCAFFOLD_USER = """## 5W1H 引导框架

请针对以下文本生成结构化问答对：

| 维度 | 问题方向 | 重点关注 |
|------|----------|----------|
| **Who** | 涉及的地点/实体是谁？ | 捕获地理实体名称 |
| **What** | 这些地点有什么特征？ | 捕获属性、功能、特色 |
| **When** | 时间背景是什么？ | 季节、时段、事件时机 |
| **Where** | 位于哪里？相互位置？ | 空间关系、邻近、方位 |
| **Why** | 作者为什么提到？ | 情感、推荐、评价动机 |
| **How** | 如何到达/体验？ | 交通方式、活动方式 |

## 任务示例

示例1:
输入: "武大的樱花开了，很多人在行政楼前拍照打卡"

输出: {{
  "qa_pairs": [
    {{
      "question": "文中提到的主要地点是谁？",
      "answer": "武汉大学（简称武大）和武汉大学行政楼",
      "dimension": "who",
      "entities_involved": ["武汉大学", "行政楼"],
      "confidence": "high"
    }},
    {{
      "question": "这些地点有什么特征？",
      "answer": "武汉大学有樱花景观，行政楼是拍照打卡点",
      "dimension": "what",
      "entities_involved": ["武汉大学", "行政楼"],
      "confidence": "high"
    }},
    {{
      "question": "时间背景是什么？",
      "answer": "樱花开放季节，可能是春季",
      "dimension": "when",
      "entities_involved": [],
      "confidence": "medium"
    }},
    {{
      "question": "这些地点的位置关系？",
      "answer": "行政楼在武汉大学内部，'行政楼前'表明具体位置",
      "dimension": "where",
      "entities_involved": ["武汉大学", "行政楼"],
      "confidence": "high"
    }},
    {{
      "question": "作者为什么提到这些地点？",
      "answer": "推荐拍照打卡，表达对樱花景观的正面情感",
      "dimension": "why",
      "entities_involved": ["武汉大学"],
      "confidence": "medium"
    }},
    {{
      "question": "人们如何体验这些地点？",
      "answer": "在行政楼前拍照打卡",
      "dimension": "how",
      "entities_involved": ["行政楼"],
      "confidence": "high"
    }}
  ],
  "semantic_summary": "武汉大学在樱花季吸引游客，行政楼前是热门拍照打卡点",
  "entity_hints": ["武汉大学", "行政楼", "樱花"],
  "relation_hints": ["属于", "承载活动", "引发情感", "位于"],
  "context_dependencies": ["武大是武汉大学简称", "行政楼隶属于武汉大学"],
  "overall_confidence": "high",
  "should_skip_detailed_extraction": false
}}

示例2:
输入: "今天心情不好"

输出: {{
  "qa_pairs": [],
  "semantic_summary": "纯情感表达，无地理信息",
  "entity_hints": [],
  "relation_hints": [],
  "context_dependencies": [],
  "overall_confidence": "high",
  "should_skip_detailed_extraction": true
}}

## 待处理文本
{raw_text}

请输出QA脚手架结果（JSON格式）。"""

QA_SCAFFOLD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SCAFFOLD_SYSTEM),
    ("human", QA_SCAFFOLD_USER),
])
```

### 13.6 节点实现设计

```python
# nodes.py 新增

def create_qa_scaffold_node(llm: Any):
    """创建QA脚手架节点（P8新增）"""
    parser = PydanticOutputParser(pydantic_object=QAScaffoldResult)

    async def qa_scaffold_node(state: CorpusState, writer: StreamWriter) -> Dict:
        """Step 0.5: 5W1H问答扩展，构建语义脚手架"""
        corpus_id = state['corpus_id']
        raw_text = state['raw_text']
        
        logger.info(f"[QA_Scaffold] 处理语料: {corpus_id}")
        
        writer({
            "step": "qa_scaffold",
            "corpus_id": corpus_id,
            "status": "started",
            "message": "开始构建语义脚手架"
        })

        try:
            # 调用LLM生成QA脚手架
            prompt_text = QA_SCAFFOLD_PROMPT.invoke({"raw_text": raw_text})
            full_prompt = f"{prompt_text.messages[1].content}\n\n{parser.get_format_instructions()}"
            response = await llm.ainvoke(full_prompt)
            result: QAScaffoldResult = parser.parse(response.content)

            logger.info(
                f"[QA_Scaffold] 完成: {len(result.qa_pairs)} 个问答对, "
                f"置信度={result.overall_confidence}"
            )

            writer({
                "step": "qa_scaffold",
                "corpus_id": corpus_id,
                "status": "completed",
                "qa_count": len(result.qa_pairs),
                "entity_hints": result.entity_hints,
                "relation_hints": result.relation_hints,
                "confidence": result.overall_confidence
            })

            # 根据结果决定下一步
            if result.should_skip_detailed_extraction:
                # 简单文本，跳过后续处理
                return {
                    "qa_scaffold_result": result.model_dump(),
                    "semantic_summary": result.semantic_summary,
                    "current_step": StepEnum.DONE,
                }
            else:
                # 复杂文本，继续到 Normalize
                return {
                    "qa_scaffold_result": result.model_dump(),
                    "semantic_summary": result.semantic_summary,
                    "qa_entity_hints": result.entity_hints,
                    "qa_relation_hints": result.relation_hints,
                    "qa_context_dependencies": result.context_dependencies,
                    "current_step": StepEnum.NORMALIZE,
                }

        except Exception as e:
            logger.error(f"[QA_Scaffold] 处理失败: {e}")
            # 保守策略：失败时继续处理
            writer({
                "step": "qa_scaffold",
                "corpus_id": corpus_id,
                "status": "failed",
                "error": str(e)
            })
            return {
                "qa_scaffold_result": {},
                "semantic_summary": "",
                "current_step": StepEnum.NORMALIZE,
            }

    return qa_scaffold_node
```

### 13.7 状态扩展设计

```python
# state.py CorpusState 新增字段

class CorpusState(TypedDict):
    # ... 现有字段 ...
    
    # P8新增：QA脚手架结果
    qa_scaffold_result: Annotated[Dict, replace_value]
    """QA脚手架结果：包含qa_pairs、semantic_summary等"""
    
    semantic_summary: Annotated[str, replace_value]
    """语义摘要：整合问答后的文本理解"""
    
    qa_entity_hints: Annotated[List[str], replace_value]
    """实体提示：QA阶段发现的可能实体"""
    
    qa_relation_hints: Annotated[List[str], replace_value]
    """关系提示：QA阶段发现的可能关系类型"""
    
    qa_context_dependencies: Annotated[List[str], replace_value]
    """上下文依赖：需要注意的依赖关系"""
```

### 13.8 下游节点利用方式

**NER节点增强**：

```python
# nodes.py ner_node 修改

async def ner_node(state: CorpusState, writer: StreamWriter) -> Dict:
    text_for_processing = _get_text_for_processing(state)
    
    # P8新增：利用QA脚手架信息
    qa_entity_hints = state.get("qa_entity_hints", [])
    qa_context = state.get("qa_context_dependencies", [])
    
    # 构建增强提示词
    prompt_text = NER_PROMPT.invoke({
        "raw_text": text_for_processing,
        "entity_hints": format_entity_hints(qa_entity_hints),  # 新增辅助函数
        "context_dependencies": format_context_dependencies(qa_context),
    })
    # ...
```

**NER提示词增强**：

```python
# prompts.py NER_USER 新增部分

NER_USER = """## 候选目标
...

## QA脚手架提示（如有）
以下实体和上下文依赖在前置QA分析中被识别，可作为参考：
{entity_hints}

上下文依赖提醒：
{context_dependencies}

## 待处理文本
{raw_text}
"""
```

**RE节点增强**：

```python
# nodes.py re_node 修改

async def re_node(state: CorpusState, writer: StreamWriter) -> Dict:
    text_for_processing = _get_text_for_processing(state)
    
    # P8新增：利用QA脚手架信息
    qa_relation_hints = state.get("qa_relation_hints", [])
    qa_context = state.get("qa_context_dependencies", [])
    
    prompt_text = RE_PROMPT.invoke({
        "raw_text": text_for_processing,
        "entities": format_entities(state["entities"]),
        "relation_hints": format_relation_hint(qa_relation_hints),  # 新增
        "context_dependencies": format_context_dependencies(qa_context),
    })
    # ...
```

### 13.9 Workflow集成设计

```python
# workflow.py build_corpus_workflow 修改

def build_corpus_workflow(
    llm: Any,
    use_simplified_eval: bool = True,
    enable_self_check: bool = False,
    enable_filter: bool = False,
    enable_normalize: bool = False,
    enable_qa_scaffold: bool = False,  # P8新增配置
    max_retries: int = DEFAULT_MAX_RETRIES
) -> CompiledStateGraph:
    """
    流程模式新增：
    - QA+Scaffold模式: Filter → QA_Scaffold → Normalize → NER → RE → Eval → Label
    - 简化QA模式: QA_Scaffold → NER → RE → Eval → Label（跳过Filter/Normalize）
    """
    
    if enable_filter and enable_qa_scaffold and enable_normalize:
        # 完整模式
        filter_node = create_filter_node(llm)
        qa_scaffold_node = create_qa_scaffold_node(llm)
        normalize_node = create_normalize_node(llm)
        
        builder.add_node("filter", filter_node)
        builder.add_node("qa_scaffold", qa_scaffold_node)
        builder.add_node("normalize", normalize_node)
        
        builder.add_edge(START, "filter")
        builder.add_conditional_edges("filter", route_after_filter_to_qa)
        builder.add_edge("qa_scaffold", "normalize")
        builder.add_edge("normalize", "ner")
        
    elif enable_qa_scaffold:
        # 仅QA模式
        qa_scaffold_node = create_qa_scaffold_node(llm)
        builder.add_node("qa_scaffold", qa_scaffold_node)
        builder.add_edge(START, "qa_scaffold")
        builder.add_conditional_edges("qa_scaffold", route_after_qa_scaffold)
```

### 13.10 配置扩展

```python
# config.py ExtractionConfig 新增

enable_qa_scaffold: bool = False
"""是否启用苏格拉底式QA引导节点"""

qa_scaffold_min_text_length: int = 20
"""启用QA脚手架的最小文本长度（过短文本跳过）"""

qa_scaffold_skip_simple: bool = True
"""是否对简单文本跳过QA脚手架（根据Filter confidence判断）"""
```

### 13.11 成本分析与优化策略

| 场景 | 无QA | 有QA | 增量成本 |
|------|------|------|----------|
| 简单文本（<20字） | 4-5次LLM | 4-5次（QA跳过） | 0% |
| 中等文本（20-100字） | 4-5次LLM | 5-6次 | +20% |
| 复杂文本（>100字） | 4-5次LLM | 5-6次 | +20% |

**优化策略**：

1. **智能跳过**：对Filter判定为"简单"或"高置信度"的文本跳过QA
2. **轻量QA**：对中等文本只生成3个核心问答（Who/Where/What）
3. **缓存复用**：相似文本的QA结果可缓存复用
4. **批量QA**：多条相似文本合并到一个QA请求

### 13.12 预期收益

| 维度 | 预期提升 | 说明 |
|------|----------|------|
| **实体召回率** | +5-10% | QA预先识别实体提示 |
| **关系准确率** | +10-15% | QA捕获上下文依赖，减少幻觉 |
| **简称识别** | +15-20% | QA明确"武大=武汉大学"等关系 |
| **隐性关系** | +20% | QA挖掘"樱花→春季→武汉大学"等隐性链 |
| **可解释性** | 显著提升 | QA问答对可作为解释依据 |

### 13.13 实施优先级

| 优先级 | 内容 | 复杂度 | 收益 |
|--------|------|--------|------|
| **P0** | QA_Scaffold节点基础实现 | 低 | 中 |
| **P1** | NER/RE提示词集成QA信息 | 低 | 高 |
| **P2** | 智能跳过策略（简单文本跳过QA） | 低 | 成本优化 |
| **P3** | 轻量QA模式（3问答简化） | 低 | 成本优化 |
| **P4** | QA结果缓存与复用 | 中 | 成本优化 |

### 13.14 与现有方案的协同

| 协同方案 | 协同效果 |
|----------|----------|
| **Filter节点** | QA可参考Filter的geo_entity_hint |
| **Normalize节点** | QA的context_dependencies辅助归一化 |
| **Self-Check节点** | QA问答对可作为验证依据 |
| **Schema约束矩阵** | QA的relation_hints可做Schema预筛选 |
| **知识库协同** | QA的entity_hints可匹配KG已有实体 |

---