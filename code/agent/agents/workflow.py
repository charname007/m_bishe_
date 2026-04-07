"""
分布式知识图谱构建工作流
"""
import asyncio
import time
from typing import List, Dict, Optional, Any
from loguru import logger

from .state import DistributedState, PhaseEnum
from .coordinator import Coordinator, CoordinatorConfig
from .worker import run_worker
from .aggregator import Aggregator
from .finalizer import Finalizer
from ..kg.neo4j_client import Neo4jClient
from ..kg.postgres_client import PostgresClient


class DistributedKGWorkflow:
    """
    分布式知识图谱构建工作流

    流程: Coordinator → Workers(并行) → Aggregator → Finalizer
    """

    def __init__(
        self,
        llm_client: Any,
        neo4j_client: Neo4jClient,
        postgres_client: PostgresClient,
        coordinator_config: Optional[CoordinatorConfig] = None
    ):
        self.llm = llm_client
        self.neo4j = neo4j_client
        self.postgres = postgres_client
        self.coordinator = Coordinator(coordinator_config)
        self.aggregator = Aggregator()
        self.finalizer = Finalizer(neo4j_client, postgres_client)

    async def run(self, corpus_list: List[Dict]) -> DistributedState:
        """
        执行完整工作流

        Args:
            corpus_list: [{"id": str, "text": str}, ...]

        Returns:
            DistributedState: 最终状态
        """
        logger.info(f"开始处理 {len(corpus_list)} 条语料...")

        # Step 1: Coordinator - 任务分发
        state = self._coordinator_dispatch(corpus_list)

        # Step 2: Workers - 并行处理
        state = await self._workers_process(state)

        # Step 3: Aggregator - 结果聚合
        state = self._aggregator_merge(state)

        # Step 4: Finalizer - 输出
        state = self._finalizer_output(state)

        logger.info("工作流执行完成!")
        self._print_summary(state)

        return state

    def _coordinator_dispatch(self, corpus_list: List[Dict]) -> DistributedState:
        """Coordinator: 任务分发"""
        logger.info("[Phase: MAP] Coordinator分发任务...")
        return self.coordinator.distribute(corpus_list)

    async def _workers_process(self, state: DistributedState) -> DistributedState:
        """Workers: 并行处理"""
        logger.info("[Phase: MAP] Workers并行处理...")

        partitions = state["corpus_partitions"]
        worker_ids = list(partitions.keys())

        # 创建并行任务
        tasks = []
        for worker_id in worker_ids:
            corpus_list = partitions[worker_id]
            task = run_worker(worker_id, corpus_list, self.llm)
            tasks.append(task)

        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 收集结果
        worker_results = []
        failed_workers = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"{worker_ids[i]} 失败: {result}")
                failed_workers.append(worker_ids[i])
            else:
                worker_results.append(result)

        state["worker_results"] = worker_results
        state["failed_workers"] = failed_workers
        state["current_phase"] = PhaseEnum.REDUCE

        return state

    def _aggregator_merge(self, state: DistributedState) -> DistributedState:
        """Aggregator: 结果聚合"""
        logger.info("[Phase: REDUCE] Aggregator合并结果...")
        return self.aggregator.aggregate(state)

    def _finalizer_output(self, state: DistributedState) -> DistributedState:
        """Finalizer: 输出"""
        logger.info("[Phase: FINALIZE] Finalizer写入数据库...")
        return self.finalizer.finalize(state)

    def _print_summary(self, state: DistributedState):
        """打印摘要"""
        end_time = state.get("end_time") or time.time()
        duration = end_time - state["start_time"]

        print("\n" + "=" * 60)
        print("知识图谱构建完成摘要")
        print("=" * 60)
        print(f"批次ID: {state['batch_id']}")
        print(f"处理语料: {state['total_count']} 条")
        print(f"Worker数量: {state['worker_count']} 个")
        print(f"失败Worker: {len(state['failed_workers'])} 个")
        print(f"处理耗时: {duration:.2f} 秒")
        print("-" * 60)
        print(f"聚合实体: {len(state['aggregated_entities'])} 个")
        print(f"聚合三元组: {len(state['aggregated_triples'])} 个")
        print(f"发现别名: {len(state['entity_aliases'])} 组")
        print("-" * 60)
        print(f"Neo4j入库: {state['neo4j_stats']}")
        print(f"PostgreSQL入库: {state['postgres_stats']}")
        print("=" * 60 + "\n")


async def run_workflow(
    corpus_list: List[Dict],
    llm_client: Any,
    neo4j_config: Dict,
    postgres_config: Dict,
    coordinator_config: Optional[CoordinatorConfig] = None
) -> DistributedState:
    """
    便捷函数：运行工作流

    Args:
        corpus_list: 语料列表
        llm_client: LLM客户端
        neo4j_config: {"uri": str, "user": str, "password": str}
        postgres_config: {"host": str, "port": int, "database": str, "user": str, "password": str}
        coordinator_config: Coordinator配置

    Returns:
        DistributedState
    """
    # 创建数据库客户端
    neo4j_client = Neo4jClient(**neo4j_config)
    postgres_client = PostgresClient(**postgres_config)

    # 创建表结构
    postgres_client.create_tables()

    try:
        # 创建工作流
        workflow = DistributedKGWorkflow(
            llm_client=llm_client,
            neo4j_client=neo4j_client,
            postgres_client=postgres_client,
            coordinator_config=coordinator_config
        )

        # 执行
        result = await workflow.run(corpus_list)
        return result

    finally:
        # 关闭连接
        neo4j_client.close()
        postgres_client.close()