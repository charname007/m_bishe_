# Agent 模块

基于 LangGraph 和 LangChain 的知识图谱构建智能体模块。

## 功能特性

- **四步骤工作流**: NER → RE → Eval (两轮对话) → Label
- **分布式并行处理**: MapReduce 架构，自动计算 Worker 数量
- **结构化输出**: 使用 LangChain `with_structured_output` + Pydantic 模型
- **多数据库输出**: 同时写入 Neo4j (图数据库) 和 PostgreSQL (关系数据库)
- **实体去重与别名发现**: 基于相似度的实体合并

---

## 快速开始

### 安装依赖

```bash
pip install langchain langgraph langchain-openai neo4j psycopg2-binary pydantic loguru
```

### 配置环境变量

创建 `.env` 文件：

```env
# Neo4j 配置 (必需)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# PostgreSQL 配置 (必需)
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=kg
PG_USER=postgres
PG_PASSWORD=your_password
```

---

## 使用示例

### 处理单条语料

```python
import asyncio
from langchain_openai import ChatOpenAI
from agent.agents import process_corpus

async def main():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    corpus = {
        "id": "001",
        "text": "武汉大学在珞喻路上，旁边有群光广场"
    }
    
    result = await process_corpus(llm, corpus)
    print(f"实体: {result['entities']}")
    print(f"三元组: {result['corrected_triples']}")

asyncio.run(main())
```

### 批量处理语料

```python
import asyncio
from langchain_openai import ChatOpenAI
from agent.agents import process_batch

async def main():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    corpus_list = [
        {"id": "001", "text": "武汉大学在珞喻路上"},
        {"id": "002", "text": "群光广场在华师旁边"},
        {"id": "003", "text": "光谷步行街很适合逛街"},
    ]
    
    config = {
        "corpus_per_worker": 10,  # 每个 Worker 处理的语料数
        "max_workers": 5,         # 最大 Worker 数量
    }
    
    result = await process_batch(llm, corpus_list, config)
    print(f"聚合实体数: {len(result['aggregated_entities'])}")
    print(f"聚合三元组数: {len(result['aggregated_triples'])}")

asyncio.run(main())
```

---

## 模块结构

```
agent/
├── agents/
│   ├── __init__.py      # 模块导出
│   ├── state.py         # TypedDict 状态定义
│   ├── schemas.py       # Pydantic 模型定义
│   ├── prompts.py       # ChatPromptTemplate 提示词
│   ├── nodes.py         # LangGraph 节点函数
│   └── workflow.py      # StateGraph 工作流定义
├── kg/
│   ├── neo4j_client.py  # Neo4j 客户端
│   └── postgres_client.py # PostgreSQL 客户端
└── tests/
    ├── test_nodes.py    # 节点函数测试
    └── test_workflow.py # 工作流测试
```

---

## 核心组件详解

### 1. 状态定义 (state.py)

#### CorpusState - 单条语料处理状态

| 字段 | 类型 | 说明 |
|------|------|------|
| `corpus_id` | `str` | 语料唯一标识 |
| `raw_text` | `str` | 原始文本 |
| `entities` | `Dict[str, List[str]]` | 按类型分类的实体 |
| `triples` | `List[Dict]` | 抽取的三元组 |
| `eval_scores` | `List[Dict]` | 评估评分 |
| `corrected_triples` | `List[Dict]` | 修正后的三元组 |
| `entity_attrs` | `Dict[str, Dict]` | 实体属性 |
| `relation_attrs` | `Dict[str, Dict]` | 关系属性 |
| `current_step` | `StepEnum` | 当前步骤 |
| `error` | `Optional[str]` | 错误信息 |

#### KGState - 分布式处理状态

| 字段 | 类型 | 说明 |
|------|------|------|
| `batch_id` | `str` | 批次ID |
| `corpus_list` | `List[Dict]` | 语料列表 |
| `worker_results` | `List[WorkerResult]` | Worker结果 |
| `aggregated_entities` | `List[Dict]` | 聚合实体 |
| `aggregated_triples` | `List[Dict]` | 聚合三元组 |

### 2. 节点函数 (nodes.py)

所有节点使用工厂函数模式创建：

```python
def create_ner_node(llm: Any):
    """创建NER节点"""
    structured_llm = llm.with_structured_output(EntityRecognitionResult)
    
    async def ner_node(state: CorpusState) -> Dict:
        # ... 节点逻辑
        return {"entities": result, "current_step": StepEnum.RE}
    
    return ner_node
```

#### 可用节点

| 函数 | 说明 |
|------|------|
| `create_ner_node` | 命名实体识别 |
| `create_re_node` | 关系抽取 |
| `create_eval_1_node` | 第一次评估 |
| `create_eval_2_node` | 第二次评估（自检） |
| `create_label_node` | 属性标注 |
| `create_coordinator_node` | 调度器（分配语料） |
| `create_aggregator_node` | 聚合器（合并结果） |

### 3. 辅助函数

| 函数 | 说明 |
|------|------|
| `normalize_relation_key` | 规范化关系key格式 |
| `apply_corrections` | 应用三元组修正 |
| `deduplicate_entities` | 实体去重 |
| `deduplicate_triples` | 三元组去重 |
| `is_similar` | 判断名称相似度 |

---

## 工作流架构

### 单条语料工作流

```
START → NER → [条件判断] → RE → Eval1 → Eval2 → Label → END
                          ↓
                         END (失败时)
```

**路由规则**：
- NER 失败 → 直接 END
- 无实体 → 跳过 RE，进入 Eval
- 无三元组 → 跳过评估，进入 Label

### 分布式工作流

```
START → Coordinator → Workers(并行) → Aggregator → Finalizer → END
```

**处理流程**：
1. **Coordinator**: 计算Worker数量，分配语料分片
2. **Workers**: 每个Worker并行处理一个分片
3. **Aggregator**: 合并所有Worker结果，实体去重
4. **Finalizer**: 写入Neo4j和PostgreSQL

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CORPUS_PER_WORKER` | 10 | 每个 Worker 处理的语料数 |
| `MAX_WORKERS` | 10 | 最大 Worker 数量 |
| `EVAL_PASSED_THRESHOLD` | 3.5 | 评估通过阈值 (1-5) |
| `DEFAULT_SIMILARITY_THRESHOLD` | 0.85 | 实体相似度阈值 |
| `MAX_TEXT_LENGTH` | 10000 | 最大文本长度 |

---

## 数据库表结构

### PostgreSQL

#### extraction_batches (批次表)
```sql
CREATE TABLE extraction_batches (
    batch_id VARCHAR(36) PRIMARY KEY,
    corpus_count INTEGER,
    worker_count INTEGER,
    status VARCHAR(20),
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

#### entities (实体表)
```sql
CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(36),
    name VARCHAR(200) NOT NULL,
    type VARCHAR(20),
    category VARCHAR(50),
    aliases TEXT[],
    occurrence_count INTEGER,
    corpus_ids TEXT[],
    created_at TIMESTAMP
);
```

#### triples (三元组表)
```sql
CREATE TABLE triples (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(36),
    head_entity VARCHAR(200),
    relation VARCHAR(50),
    tail_entity VARCHAR(200),
    evidence TEXT,
    sem_score INTEGER,
    fac_score INTEGER,
    con_score INTEGER,
    passed_eval BOOLEAN,
    relation_type VARCHAR(50),
    relation_subtype VARCHAR(50),
    corpus_ids TEXT[],
    created_at TIMESTAMP
);
```

#### corpus_sources (语料来源表)
```sql
CREATE TABLE corpus_sources (
    corpus_id VARCHAR(50) PRIMARY KEY,
    batch_id VARCHAR(36),
    raw_text TEXT,
    entities JSONB,
    triples JSONB,
    processed_at TIMESTAMP
);
```

### Neo4j

- **节点标签**: `Entity`
  - 属性: `name`, `type`, `category`, `aliases`, `corpus_ids`, `source`
- **关系类型**: `RELATION`
  - 属性: `type`, `evidence`, `relation_type`, `relation_subtype`, `corpus_ids`

---

## 运行测试

```bash
cd agent
pytest tests/ -v
```

### 测试覆盖

| 测试类 | 覆盖内容 |
|--------|----------|
| `TestIsSimilar` | 名称相似度判断 |
| `TestNormalizeRelationKey` | 关系key格式化 |
| `TestDeduplicateTriples` | 三元组去重 |
| `TestDeduplicateEntities` | 实体去重 |
| `TestApplyCorrections` | 三元组修正 |
| `TestCoordinatorNode` | 调度器节点 |
| `TestAggregatorNode` | 聚合器节点 |
| `TestValidateCorpusText` | 文本验证 |
| `TestValidateCorpusId` | ID验证 |
| `TestRouteAfterNer` | NER后路由 |

---

## 注意事项

### 安全性

1. **密码配置**: 数据库密码必须通过环境变量配置，不支持硬编码默认值
2. **SQL注入防护**: PostgreSQL使用参数化查询
3. **Cypher注入防护**: Neo4j使用参数化查询

### 性能

1. **批量操作**: Neo4j使用`UNWIND`批量合并，性能提升10x+
2. **并发安全**: 每条语料使用唯一的`thread_id`防止状态串扰
3. **错误隔离**: 单条语料处理失败不会影响其他语料

### 可靠性

1. **异常处理**: 所有LLM调用节点都有try-except包裹
2. **降级策略**: 批量操作失败时自动降级为逐个处理
3. **日志记录**: 完整的info/debug/error级别日志

---

## 实体类型定义

```python
ENTITY_TYPES = {
    "道路": "街道、大道、小巷等（如：关山大道）",
    "POI": "具体店名、地标、机构（如：武汉大学、某某咖啡厅）",
    "建筑物": "具体的楼宇、商场主体（如：泛悦汇）",
    "街区": "具有边界感的生活区域（如：街道口、华农校区）"
}
```

## License

MIT