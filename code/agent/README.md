# Agent 模块

基于 LangGraph 和 LangChain 的知识图谱构建智能体模块。

## 功能特性

- **四步骤工作流**: NER → RE → Eval (两轮对话) → Label
- **分布式并行处理**: MapReduce 架构，自动计算 Worker 数量
- **结构化输出**: 使用 LangChain `with_structured_output` + Pydantic 模型
- **多数据库输出**: 同时写入 Neo4j (图数据库) 和 PostgreSQL (关系数据库)

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

### 使用示例

#### 处理单条语料

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

#### 批量处理语料

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

## 工作流架构

### 单条语料工作流

```
START → NER → [条件判断] → RE → Eval1 → Eval2 → Label → END
                          ↓
                         END (失败时)
```

### 分布式工作流

```
START → Coordinator → Workers(并行) → Aggregator → Finalizer → END
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CORPUS_PER_WORKER` | 10 | 每个 Worker 处理的语料数 |
| `MAX_WORKERS` | 10 | 最大 Worker 数量 |
| `EVAL_PASSED_THRESHOLD` | 3.5 | 评估通过阈值 (1-5) |
| `DEFAULT_SIMILARITY_THRESHOLD` | 0.85 | 实体相似度阈值 |
| `MAX_TEXT_LENGTH` | 10000 | 最大文本长度 |

## 数据库表结构

### PostgreSQL

- `extraction_batches`: 批次记录
- `entities`: 实体表
- `triples`: 三元组表
- `corpus_sources`: 语料来源表

### Neo4j

- 节点标签: `Entity`
- 关系类型: `RELATION`

## 运行测试

```bash
cd agent
pytest tests/ -v
```

## 注意事项

1. **密码配置**: 数据库密码必须通过环境变量配置，不支持硬编码默认值
2. **文本验证**: 语料文本会进行长度验证和危险字符过滤
3. **并发安全**: 每条语料使用唯一的 `thread_id` 防止状态串扰
4. **错误处理**: 单条语料处理失败不会影响其他语料

## License

MIT