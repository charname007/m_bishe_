"""
Coordinator Agent - 调度器
负责语料分片和Worker调度
"""
import math
import uuid
import time
from typing import Dict, List, Optional
from loguru import logger

from .state import DistributedState, PhaseEnum


class CoordinatorConfig:
    """调度器配置"""
    WORKER_BATCH_SIZE = 10      # 每个Worker处理语料数
    MAX_WORKERS = 10            # 最大Worker数量


class Coordinator:
    """调度器 - 负责任务分发"""

    def __init__(self, config: Optional[CoordinatorConfig] = None):
        self.config = config or CoordinatorConfig()

    def create_initial_state(self, corpus_list: List[Dict]) -> DistributedState:
        """创建初始状态"""
        batch_id = str(uuid.uuid4())

        return DistributedState(
            batch_id=batch_id,
            corpus_list=corpus_list,
            total_count=len(corpus_list),
            worker_count=0,
            corpus_partitions={},
            worker_results=[],
            aggregated_entities=[],
            aggregated_triples=[],
            entity_aliases={},
            neo4j_stats={},
            postgres_stats={},
            current_phase=PhaseEnum.INIT,
            failed_workers=[],
            start_time=time.time(),
            end_time=None
        )

    def calculate_worker_count(self, corpus_count: int) -> int:
        """计算需要的Worker数量"""
        worker_count = math.ceil(corpus_count / self.config.WORKER_BATCH_SIZE)
        return min(worker_count, self.config.MAX_WORKERS)

    def partition_corpus(self, state: DistributedState) -> DistributedState:
        """语料分片"""
        corpus_list = state["corpus_list"]
        corpus_count = len(corpus_list)

        # 计算Worker数量
        worker_count = self.calculate_worker_count(corpus_count)
        state["worker_count"] = worker_count

        logger.info(f"语料总数: {corpus_count}, 创建 {worker_count} 个Workers")

        # 分片
        partitions = {}
        batch_size = self.config.WORKER_BATCH_SIZE

        for i in range(worker_count):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, corpus_count)
            worker_id = f"worker_{i}"
            partitions[worker_id] = corpus_list[start_idx:end_idx]
            logger.debug(f"{worker_id}: 语料 {start_idx+1}-{end_idx}")

        state["corpus_partitions"] = partitions
        state["current_phase"] = PhaseEnum.MAP

        return state

    def distribute(self, corpus_list: List[Dict]) -> DistributedState:
        """分发任务 - 入口方法"""
        logger.info(f"开始分发 {len(corpus_list)} 条语料...")

        # 创建初始状态
        state = self.create_initial_state(corpus_list)

        # 执行分片
        state = self.partition_corpus(state)

        return state


class CoordinatorAgent:
    """
    Coordinator Agent - 用于LangGraph工作流
    """

    def __init__(self, config: Optional[CoordinatorConfig] = None):
        self.coordinator = Coordinator(config)

    def __call__(self, state: DistributedState) -> Dict:
        """LangGraph节点调用入口"""
        # 如果是初始状态，执行分片
        if state["current_phase"] == PhaseEnum.INIT:
            result = self.coordinator.partition_corpus(state)
            return {
                "worker_count": result["worker_count"],
                "corpus_partitions": result["corpus_partitions"],
                "current_phase": result["current_phase"]
            }

        return {}


def create_coordinator_node(config: Optional[CoordinatorConfig] = None):
    """创建Coordinator节点"""
    coordinator_agent = CoordinatorAgent(config)

    def coordinator_node(state: DistributedState) -> Dict:
        return coordinator_agent(state)

    return coordinator_node