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

## 八、下一步行动建议

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