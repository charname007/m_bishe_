#!/usr/bin/env python3
"""
社交媒体语料批处理脚本

功能:
- 从 social_media_notes_sampled 表获取待处理数据
- 每批处理50条，最大500条/运行
- 使用完整workflow模式(所有自检节点启用)
- 状态管理和错误处理
- 输出日志到文件

使用方法:
    python scripts/process_social_media_corpus.py [--max MAX] [--batch-size SIZE]
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime
from uuid import uuid4

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from loguru import logger

# 配置日志输出到文件
log_file = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "output",
    f"corpus_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logger.add(log_file, level="DEBUG", encoding="utf-8")
print(f"Log file: {log_file}")

from langchain_openai import ChatOpenAI
from kg.postgres_client import PostgresClient
from kg.neo4j_client import Neo4jClient
from agent.agents.workflow import build_distributed_workflow
from agent.agents.config import ExtractionConfig


# ===== 配置 =====
DEFAULT_TABLE = "social_media_notes_sampled"
DEFAULT_TEXT_COLUMN = "content_cleaned"
DEFAULT_ID_COLUMN = "note_id"
DEFAULT_TIME_COLUMN = "publish_time"
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_TOTAL = 500


def init_clients():
    """初始化数据库客户端"""
    pg_client = PostgresClient(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", 5432)),
        database=os.getenv("PG_DATABASE", "postgres"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
    )

    neo4j_client = Neo4jClient(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
    )

    return pg_client, neo4j_client


def init_workflow(corpus_per_worker: int = DEFAULT_BATCH_SIZE):
    """初始化LLM和workflow

    Args:
        corpus_per_worker: 每个worker处理的语料数量，默认与batch_size相同
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE_URL")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)

    # 启用所有自检节点 + QA导师双向交互
    config = ExtractionConfig.from_env()
    config.enable_filter = True
    config.enable_normalize = True
    config.enable_qa_scaffold = True
    config.enable_full_self_check = True
    config.enable_entity_alignment = True
    config.enable_batch_llm = True
    config.enable_qa_mentor = True  # P15新增：启用QA导师双向交互
    config.corpus_per_worker = corpus_per_worker  # 与batch_size同步

    workflow = build_distributed_workflow(llm, config)

    return workflow, config


async def process_batch(
    workflow, pg_client, neo4j_client, corpus_list: list, batch_id: str
):
    """处理一批语料"""
    logger.info(f"批次 {batch_id}: 开始处理 {len(corpus_list)} 条语料")

    # 构建初始状态
    initial_state = {
        "batch_id": batch_id,
        "corpus_list": corpus_list,
        "worker_count": 1,  # 只用1个worker处理整个batch
        "total_count": len(corpus_list),
    }

    thread_config = {"configurable": {"thread_id": batch_id}}

    try:
        # 运行workflow
        result = await workflow.ainvoke(initial_state, thread_config)

        # 获取结果（数据库写入由 workflow finalizer_node 内部处理）
        entities = result.get("aggregated_entities", [])
        triples = result.get("aggregated_triples", [])

        logger.info(
            f"批次 {batch_id}: 抽取完成 - {len(entities)} 实体, {len(triples)} 三元组"
        )

        return {"success": True, "entities": len(entities), "triples": len(triples)}

    except Exception as e:
        logger.error(f"批次 {batch_id} 处理失败: {e}")
        pg_client.update_batch_status_with_error(batch_id, "error", str(e))
        return {"success": False, "error": str(e)}


async def main(
    max_total: int = DEFAULT_MAX_TOTAL, batch_size: int = DEFAULT_BATCH_SIZE
):
    """主函数"""
    logger.info("=" * 60)
    logger.info("社交媒体语料批处理")
    logger.info("=" * 60)
    logger.info(f"配置: 批大小={batch_size}, 最大处理={max_total}")

    # 初始化
    pg_client, neo4j_client = init_clients()
    workflow, config = init_workflow(corpus_per_worker=batch_size)  # 同步设置

    # 确保表有status字段
    pg_client.ensure_corpus_status_columns(DEFAULT_TABLE)

    # 显示当前状态统计
    stats = pg_client.get_corpus_status_stats(DEFAULT_TABLE)
    logger.info(f"当前状态统计: {stats}")

    # 计算需要处理的批次
    pending_count = stats.get("pending", 0)
    total_to_process = min(pending_count, max_total)
    batch_count = (total_to_process // batch_size) + (
        1 if total_to_process % batch_size else 0
    )

    logger.info(
        f"待处理: {pending_count} 条, 本次处理: {total_to_process} 条, 共 {batch_count} 批"
    )

    if total_to_process == 0:
        logger.info("没有待处理语料，退出")
        return

    # 处理循环
    processed_total = 0
    success_count = 0
    error_count = 0

    while processed_total < total_to_process:
        logger.info(f"进度: {processed_total}/{total_to_process}")

        # 获取一批待处理语料
        current_batch_size = min(batch_size, total_to_process - processed_total)
        corpus_list = pg_client.get_pending_corpus(
            DEFAULT_TABLE,
            DEFAULT_TEXT_COLUMN,
            DEFAULT_ID_COLUMN,
            DEFAULT_TIME_COLUMN,
            current_batch_size,
        )

        if not corpus_list:
            logger.info("没有更多待处理语料")
            break

        # 生成批次ID
        batch_id = (
            f"social_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        )
        corpus_ids = [c["id"] for c in corpus_list]

        # 标记为processing（批次记录由 workflow finalizer_node 内部创建）
        pg_client.update_corpus_status(
            DEFAULT_TABLE, corpus_ids, "processing", batch_id
        )

        # 处理批次
        result = await process_batch(
            workflow, pg_client, neo4j_client, corpus_list, batch_id
        )

        # 更新语料状态
        if result["success"]:
            pg_client.update_corpus_status(
                DEFAULT_TABLE, corpus_ids, "completed", batch_id
            )
            success_count += len(corpus_list)
        else:
            # 标记失败的语料
            for cid in corpus_ids:
                pg_client.mark_corpus_error(
                    DEFAULT_TABLE, cid, result["error"], batch_id
                )
            error_count += len(corpus_list)

        processed_total += len(corpus_list)

    # 最终统计
    logger.info("=" * 60)
    logger.info("处理完成")
    logger.info(
        f"总计: {processed_total} 条, 成功: {success_count}, 失败: {error_count}"
    )

    # 显示最终状态
    final_stats = pg_client.get_corpus_status_stats(DEFAULT_TABLE)
    logger.info(f"最终状态统计: {final_stats}")

    # 关闭连接
    pg_client.close()
    neo4j_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="社交媒体语料批处理")
    parser.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX_TOTAL,
        help=f"最大处理数量 (默认: {DEFAULT_MAX_TOTAL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"批大小 (默认: {DEFAULT_BATCH_SIZE})",
    )

    args = parser.parse_args()

    asyncio.run(main(max_total=args.max, batch_size=args.batch_size))
