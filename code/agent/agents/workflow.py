"""
LangGraph工作流定义 - 使用StateGraph构建知识图谱抽取工作流
"""
import asyncio
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

    # 条件路由函数 - 根据current_step和error决定下一步
    def route_after_ner(state: CorpusState) -> str:
        """NER后路由：失败则END，成功则继续RE"""
        if state.get("error") or state.get("current_step") == StepEnum.DONE:
            return END  # 直接返回 END 常量
        return "re"

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
                # 为每条语料生成唯一的 thread_id，避免并发状态串扰
                config = {"configurable": {"thread_id": f"corpus_{corpus.get('id', uuid.uuid4().hex)}"}}
                result = await corpus_workflow.ainvoke(initial_state, config)
                return result
            except Exception as e:
                logger.error(f"处理语料失败: {e}")
                return {
                    "corpus_id": corpus.get("id", "unknown"),
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
        import os

        neo4j_stats = {"merged_entities": 0, "merged_relations": 0}
        postgres_stats = {"inserted": 0}

        # 获取数据库配置（兼容多种环境变量命名）
        neo4j_uri = os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL") or "bolt://localhost:7687"
        neo4j_user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"
        neo4j_password = os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS") or os.getenv("NEO4J_PWD") or "password"

        pg_host = os.getenv("PG_HOST", "localhost")
        pg_port = int(os.getenv("PG_PORT", "5432"))
        pg_database = os.getenv("PG_DATABASE", "kg")
        pg_user = os.getenv("PG_USER", "postgres")
        pg_password = os.getenv("PG_PASSWORD", "password")

        try:
            # 写入 Neo4j
            with Neo4jClient(neo4j_uri, neo4j_user, neo4j_password) as neo4j:
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
            with PostgresClient(pg_host, pg_port, pg_database, pg_user, pg_password) as pg:
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
        "cross_corpus_relations": [],
        "evaluator_results": [],
        "high_confidence_triples": [],
        "low_confidence_triples": [],
        "neo4j_stats": {},
        "postgres_stats": {},
        "current_phase": PhaseEnum.INIT,
        "active_workers": [],
        "failed_workers": [],
        "start_time": time.time(),
        "end_time": None,
        "total_tokens": 0,
    }

    # 使用唯一 thread_id 避免状态串扰
    thread_config = {"configurable": {"thread_id": f"batch_{uuid.uuid4().hex}"}}
    result = await workflow.ainvoke(initial_state, thread_config)
    return cast(KGState, result)