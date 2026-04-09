"""
LangGraph工作流定义 - 使用StateGraph构建知识图谱抽取工作流
"""
import asyncio
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, cast

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from loguru import logger

from .state import CorpusState, KGState, StepEnum, PhaseEnum
from .nodes import (
    create_ner_node,
    create_re_node,
    create_eval_1_node,
    create_eval_2_node,
    create_label_node,
    create_coordinator_node,
    create_aggregator_node,
)


# ===== 配置常量 =====

class WorkflowConfig:
    """工作流配置"""
    # Worker配置
    CORPUS_PER_WORKER = 10
    MAX_WORKERS = 10

    # 评估阈值
    EVAL_PASSED_THRESHOLD = 3.5

    # 相似度阈值
    DEFAULT_SIMILARITY_THRESHOLD = 0.85

    # 文本验证配置
    MAX_TEXT_LENGTH = 10000  # 最大文本长度
    MIN_TEXT_LENGTH = 1      # 最小文本长度


def _validate_corpus_text(text: str) -> str:
    """
    验证并清理语料文本

    Args:
        text: 原始文本

    Returns:
        清理后的文本

    Raises:
        ValueError: 文本无效
    """
    if not text or not isinstance(text, str):
        raise ValueError("语料文本不能为空")

    # 去除首尾空白
    text = text.strip()

    # 检查长度
    if len(text) < WorkflowConfig.MIN_TEXT_LENGTH:
        raise ValueError(f"语料文本长度不足（最小 {WorkflowConfig.MIN_TEXT_LENGTH} 字符）")

    if len(text) > WorkflowConfig.MAX_TEXT_LENGTH:
        logger.warning(f"语料文本过长（{len(text)} 字符），将被截断")
        text = text[:WorkflowConfig.MAX_TEXT_LENGTH]

    # 移除危险字符（防止注入攻击）
    # 保留中文、英文、数字、标点符号
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    return text


def _validate_corpus_id(corpus_id: Any) -> str:
    """
    验证语料ID

    Args:
        corpus_id: 原始ID

    Returns:
        验证后的ID字符串
    """
    if corpus_id is None:
        return f"auto_{uuid.uuid4().hex[:8]}"

    corpus_id = str(corpus_id).strip()
    if not corpus_id:
        return f"auto_{uuid.uuid4().hex[:8]}"

    # 限制ID长度
    if len(corpus_id) > 100:
        corpus_id = corpus_id[:100]

    return corpus_id


def _get_database_config() -> Dict[str, Any]:
    """
    获取数据库配置

    Returns:
        包含所有数据库连接参数的字典

    Raises:
        ValueError: 必需的环境变量未设置
    """
    # Neo4j密码是必需的
    neo4j_password = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS") or os.getenv("NEO4J_PWD")
    if not neo4j_password:
        raise ValueError("Neo4j密码未设置，请配置环境变量 NEO4J_PASSWORD")

    # PostgreSQL密码是必需的
    pg_password = os.getenv("PG_PASSWORD")
    if not pg_password:
        raise ValueError("PostgreSQL密码未设置，请配置环境变量 PG_PASSWORD")

    return {
        # Neo4j配置（兼容多种环境变量命名）
        "neo4j_uri": (
            os.getenv("NEO4J_URI") or
            os.getenv("NEO4J_URL") or
            "bolt://localhost:7687"
        ),
        "neo4j_user": (
            os.getenv("NEO4J_USER") or
            os.getenv("NEO4J_USERNAME") or
            "neo4j"
        ),
        "neo4j_password": neo4j_password,
        # PostgreSQL配置
        "pg_host": os.getenv("PG_HOST", "localhost"),
        "pg_port": int(os.getenv("PG_PORT", "5432")),
        "pg_database": os.getenv("PG_DATABASE", "kg"),
        "pg_user": os.getenv("PG_USER", "postgres"),
        "pg_password": pg_password,
    }


# ===== 条件路由函数（模块级，便于测试） =====

def route_after_ner(state: CorpusState) -> str:
    """
    NER后路由：失败则END，成功则继续RE

    Args:
        state: 当前语料状态

    Returns:
        下一个节点名称或END
    """
    if state.get("error") or state.get("current_step") == StepEnum.DONE:
        return END
    return "re"


# ===== 单条语料工作流 =====

def build_corpus_workflow(llm: Any) -> CompiledStateGraph:
    """
    构建单条语料处理工作流

    流程: START → NER → (条件判断) → RE → Eval1 → Eval2 → Label → END
    NER失败时直接END，跳过后续节点
    """
    # 创建节点函数
    ner_node = create_ner_node(llm)
    re_node = create_re_node(llm)
    eval_1_node = create_eval_1_node(llm)
    eval_2_node = create_eval_2_node(llm)
    label_node = create_label_node(llm)

    # 创建StateGraph
    builder = StateGraph(CorpusState)

    # 添加节点
    builder.add_node("ner", ner_node)
    builder.add_node("re", re_node)
    builder.add_node("eval_1", eval_1_node)
    builder.add_node("eval_2", eval_2_node)
    builder.add_node("label", label_node)

    # 定义边 - 使用条件边实现失败跳过
    builder.add_edge(START, "ner")
    builder.add_conditional_edges("ner", route_after_ner)
    builder.add_edge("re", "eval_1")
    builder.add_edge("eval_1", "eval_2")
    builder.add_edge("eval_2", "label")
    builder.add_edge("label", END)

    # 编译并返回
    return builder.compile(checkpointer=InMemorySaver())


# ===== 分布式工作流 =====

def build_distributed_workflow(llm: Any, config: Optional[Dict] = None) -> CompiledStateGraph:
    """
    构建分布式知识图谱构建工作流

    流程: START → Coordinator → Workers(并行) → Aggregator → Finalizer → END
    """
    config = config or {}
    corpus_per_worker = config.get("corpus_per_worker", 10)
    max_workers = config.get("max_workers", 10)

    # 创建节点函数
    coordinator_node = create_coordinator_node(corpus_per_worker, max_workers)
    aggregator_node = create_aggregator_node()

    # Worker处理函数
    async def workers_node(state: KGState) -> Dict:
        """并行执行所有Worker - 按分片并行处理"""
        corpus_workflow = build_corpus_workflow(llm)

        async def process_corpus(corpus: Dict) -> Dict:
            """处理单条语料"""
            try:
                # 验证输入
                corpus_id = _validate_corpus_id(corpus.get("id"))
                raw_text = _validate_corpus_text(corpus.get("text", ""))

                initial_state: CorpusState = {
                    "corpus_id": corpus_id,
                    "raw_text": raw_text,
                    "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
                    "triples": [],
                    "eval_scores": [],
                    "eval_passed": False,
                    "corrected_triples": [],
                    "entity_attrs": {},
                    "relation_attrs": {},
                    "current_step": StepEnum.NER,
                    "error": None,
                }
                # 为每条语料生成唯一的 thread_id，避免并发状态串扰
                thread_config = {"configurable": {"thread_id": f"corpus_{corpus_id}_{uuid.uuid4().hex[:8]}"}}
                result = await corpus_workflow.ainvoke(initial_state, thread_config)  # type: ignore
                return result
            except ValueError as e:
                # 输入验证错误
                logger.warning(f"语料验证失败: {e}")
                return {
                    "corpus_id": _validate_corpus_id(corpus.get("id")),
                    "error": f"输入验证失败: {e}",
                }
            except Exception as e:
                logger.error(f"处理语料失败: {e}")
                return {
                    "corpus_id": _validate_corpus_id(corpus.get("id")),
                    "error": str(e),
                }

        async def process_partition(worker_id: str, corpus_list: List[Dict]) -> Dict:
            """处理单个分片（Worker级别）"""
            start_time = time.time()
            tasks = [process_corpus(corpus) for corpus in corpus_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 分离成功和失败的结果
            success_results = []
            errors = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(str(result))
                    logger.error(f"[{worker_id}] 处理语料失败: {result}")
                else:
                    success_results.append(result)

            processing_time = time.time() - start_time
            logger.info(f"[{worker_id}] 完成: {len(success_results)}/{len(corpus_list)} 条语料, 耗时 {processing_time:.2f}s")

            return {
                "worker_id": worker_id,
                "corpus_ids": [r.get("corpus_id", "unknown") for r in success_results],
                "results": success_results,
                "processing_time": processing_time,
                "error": "; ".join(errors) if errors else None,
            }

        # 按分片并行处理，每个分片一个 Worker
        start_time = time.time()
        partition_tasks = [
            process_partition(worker_id, corpus_list)
            for worker_id, corpus_list in state["corpus_partitions"].items()
        ]
        worker_results = await asyncio.gather(*partition_tasks, return_exceptions=True)

        # 处理结果
        final_worker_results = []
        failed_workers = []
        for i, result in enumerate(worker_results):
            if isinstance(result, Exception):
                worker_ids = list(state["corpus_partitions"].keys())
                failed_workers.append(worker_ids[i])
                logger.error(f"Worker失败: {result}")
            else:
                final_worker_results.append(result)

        total_processing_time = time.time() - start_time
        logger.info(f"[Workers] 全部完成: {len(final_worker_results)} 个Worker成功, {len(failed_workers)} 个失败, 总耗时 {total_processing_time:.2f}s")

        return {
            "worker_results": final_worker_results,
            "failed_workers": failed_workers,
            "current_phase": PhaseEnum.REDUCE,
        }

    # 最终化节点
    async def finalizer_node(state: KGState) -> Dict:
        """FINALIZE阶段 - 输出到数据库"""
        from ..kg.neo4j_client import Neo4jClient
        from ..kg.postgres_client import PostgresClient

        neo4j_stats = {"merged_entities": 0, "merged_relations": 0}
        postgres_stats = {"inserted": 0}

        try:
            # 获取数据库配置
            db_config = _get_database_config()

            # 写入 Neo4j
            with Neo4jClient(
                db_config["neo4j_uri"],
                db_config["neo4j_user"],
                db_config["neo4j_password"]
            ) as neo4j:
                # 创建索引
                neo4j.create_indexes()

                # 批量合并实体
                if state["aggregated_entities"]:
                    entity_stats = neo4j.batch_merge_entities(state["aggregated_entities"])
                    neo4j_stats["merged_entities"] = entity_stats.get("merged", 0)

                # 批量合并关系
                if state["aggregated_triples"]:
                    relation_stats = neo4j.batch_merge_relations(state["aggregated_triples"])
                    neo4j_stats["merged_relations"] = relation_stats.get("merged", 0)

            # 写入 PostgreSQL
            with PostgresClient(
                db_config["pg_host"],
                db_config["pg_port"],
                db_config["pg_database"],
                db_config["pg_user"],
                db_config["pg_password"]
            ) as pg:
                # 创建表结构
                pg.create_tables()

                # 插入批次记录
                pg.insert_batch(state["batch_id"], state["total_count"], state["worker_count"])

                # 插入实体
                if state["aggregated_entities"]:
                    entity_count = pg.insert_entities(state["batch_id"], state["aggregated_entities"])
                    postgres_stats["entities"] = entity_count

                # 插入三元组
                if state["aggregated_triples"]:
                    triple_count = pg.insert_triples(state["batch_id"], state["aggregated_triples"])
                    postgres_stats["triples"] = triple_count

                # 插入语料来源（保留证据链）- 过滤掉失败语料
                all_corpus_states = []
                for worker_result in state["worker_results"]:
                    for corpus_state in worker_result.get("results", []):
                        # 只收集成功处理的语料（无error且有raw_text）
                        if not corpus_state.get("error") and corpus_state.get("raw_text"):
                            all_corpus_states.append(corpus_state)
                if all_corpus_states:
                    corpus_count = pg.insert_corpus_sources(state["batch_id"], all_corpus_states)
                    postgres_stats["corpus_sources"] = corpus_count

                # 更新批次状态
                pg.update_batch_status(state["batch_id"], "completed")

            logger.info(f"[Finalizer] Neo4j: {neo4j_stats}, PostgreSQL: {postgres_stats}")

        except Exception as e:
            logger.error(f"[Finalizer] 数据库写入失败: {e}")
            # 即使失败也继续，返回已处理的结果

        return {
            "neo4j_stats": neo4j_stats,
            "postgres_stats": postgres_stats,
            "current_phase": PhaseEnum.FINALIZE,
            "end_time": time.time(),
        }

    # 创建StateGraph
    builder = StateGraph(KGState)

    # 添加节点
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("workers", workers_node)
    builder.add_node("aggregator", aggregator_node)
    builder.add_node("finalizer", finalizer_node)

    # 定义边
    builder.add_edge(START, "coordinator")
    builder.add_edge("coordinator", "workers")
    builder.add_edge("workers", "aggregator")
    builder.add_edge("aggregator", "finalizer")
    builder.add_edge("finalizer", END)

    # 编译并返回
    return builder.compile(checkpointer=InMemorySaver())


# ===== 便捷函数 =====

async def process_corpus(llm: Any, corpus: Dict) -> CorpusState:
    """处理单条语料的便捷函数"""
    workflow = build_corpus_workflow(llm)

    initial_state: CorpusState = {
        "corpus_id": corpus.get("id", "unknown"),
        "raw_text": corpus.get("text", ""),
        "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
        "triples": [],
        "eval_scores": [],
        "eval_passed": False,
        "corrected_triples": [],
        "entity_attrs": {},
        "relation_attrs": {},
        "current_step": StepEnum.NER,
        "error": None,
    }

    # 使用唯一 thread_id 避免状态串扰
    config = {"configurable": {"thread_id": f"corpus_{corpus.get('id', uuid.uuid4().hex)}"}}
    result = await workflow.ainvoke(initial_state, config)
    return cast(CorpusState, result)


async def process_batch(llm: Any, corpus_list: List[Dict], config: Optional[Dict] = None) -> KGState:
    """批量处理语料的便捷函数"""
    workflow = build_distributed_workflow(llm, config)

    initial_state: KGState = {
        "batch_id": f"batch_{int(time.time())}",
        "corpus_list": corpus_list,
        "total_count": len(corpus_list),
        "worker_count": 0,
        "corpus_partitions": {},
        "worker_results": [],
        "aggregated_entities": [],
        "aggregated_triples": [],
        "entity_aliases": {},
        "neo4j_stats": {},
        "postgres_stats": {},
        "current_phase": PhaseEnum.INIT,
        "active_workers": [],
        "failed_workers": [],
        "start_time": time.time(),
        "end_time": None,
    }

    # 使用唯一 thread_id 避免状态串扰
    thread_config = {"configurable": {"thread_id": f"batch_{uuid.uuid4().hex}"}}
    result = await workflow.ainvoke(initial_state, thread_config)
    return cast(KGState, result)