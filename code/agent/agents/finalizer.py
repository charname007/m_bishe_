"""
Finalizer Agent - 输出模块
负责写入Neo4j和PostgreSQL
"""
import time
from typing import Dict, List
from loguru import logger

from .state import DistributedState
from ..kg.neo4j_client import Neo4jClient
from ..kg.postgres_client import PostgresClient


class Finalizer:
    """输出模块 - 同时写入Neo4j和PostgreSQL"""

    def __init__(self, neo4j_client: Neo4jClient, postgres_client: PostgresClient):
        self.neo4j = neo4j_client
        self.postgres = postgres_client

    def finalize(self, state: DistributedState) -> DistributedState:
        """执行输出"""
        logger.info("开始写入数据库...")

        batch_id = state["batch_id"]

        # 1. 写入PostgreSQL批次记录
        self.postgres.insert_batch(
            batch_id=batch_id,
            corpus_count=state["total_count"],
            worker_count=state["worker_count"]
        )

        # 2. 写入Neo4j
        neo4j_stats = self._write_to_neo4j(state)

        # 3. 写入PostgreSQL
        postgres_stats = self._write_to_postgres(state)

        # 4. 更新批次状态
        self.postgres.update_batch_status(batch_id, "completed")

        state["neo4j_stats"] = neo4j_stats
        state["postgres_stats"] = postgres_stats
        state["end_time"] = time.time()

        logger.info(f"写入完成: Neo4j={neo4j_stats}, PostgreSQL={postgres_stats}")

        return state

    def _write_to_neo4j(self, state: DistributedState) -> Dict:
        """写入Neo4j"""
        # 创建索引
        self.neo4j.create_indexes()

        # 合并实体
        entity_stats = self.neo4j.batch_merge_entities(state["aggregated_entities"])

        # 合并关系
        relation_stats = self.neo4j.batch_merge_relations(state["aggregated_triples"])

        return {
            "entities": entity_stats,
            "relations": relation_stats
        }

    def _write_to_postgres(self, state: DistributedState) -> Dict:
        """写入PostgreSQL"""
        batch_id = state["batch_id"]

        # 插入实体
        entity_count = self.postgres.insert_entities(
            batch_id, state["aggregated_entities"]
        )

        # 插入三元组
        triple_count = self.postgres.insert_triples(
            batch_id, state["aggregated_triples"]
        )

        # 插入语料来源
        corpus_count = 0
        for worker_result in state["worker_results"]:
            corpus_count += self.postgres.insert_corpus_sources(
                batch_id, worker_result["results"]
            )

        return {
            "entities": entity_count,
            "triples": triple_count,
            "corpus_sources": corpus_count
        }


class FinalizerAgent:
    """
    Finalizer Agent - 用于LangGraph工作流
    """

    def __init__(self, neo4j_client: Neo4jClient, postgres_client: PostgresClient):
        self.finalizer = Finalizer(neo4j_client, postgres_client)

    def __call__(self, state: DistributedState) -> Dict:
        """LangGraph节点调用入口"""
        result = self.finalizer.finalize(state)
        return {
            "neo4j_stats": result["neo4j_stats"],
            "postgres_stats": result["postgres_stats"],
            "end_time": result["end_time"]
        }


def create_finalizer_node(neo4j_client: Neo4jClient,
                          postgres_client: PostgresClient):
    """创建Finalizer节点"""
    finalizer_agent = FinalizerAgent(neo4j_client, postgres_client)

    def finalizer_node(state: DistributedState) -> Dict:
        return finalizer_agent(state)

    return finalizer_node