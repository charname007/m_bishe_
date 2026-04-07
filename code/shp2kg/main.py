"""
知识图谱构建主入口
"""
import asyncio
import argparse
from typing import List, Dict
from loguru import logger

from config import settings
from agent.agents import DistributedKGWorkflow, CoordinatorConfig
from agent.kg import Neo4jClient, PostgresClient


def create_llm_client():
    """创建LLM客户端"""
    from openai import AsyncOpenAI

    llm_config = settings.get_llm_config()

    client = AsyncOpenAI(
        api_key=llm_config["api_key"],
        base_url=llm_config["base_url"]
    )

    return client


async def process_corpus_list(corpus_list: List[Dict]):
    """
    处理语料列表

    Args:
        corpus_list: [{"id": str, "text": str}, ...]
    """
    # 创建LLM客户端
    llm_client = create_llm_client()

    # 创建数据库客户端
    neo4j_client = Neo4jClient(**settings.get_neo4j_config())
    postgres_client = PostgresClient(**settings.get_postgres_config())

    # 创建PostgreSQL表结构
    postgres_client.create_tables()

    try:
        # 创建工作流
        workflow = DistributedKGWorkflow(
            llm_client=llm_client,
            neo4j_client=neo4j_client,
            postgres_client=postgres_client,
            coordinator_config=CoordinatorConfig()
        )

        # 执行工作流
        result = await workflow.run(corpus_list)

        return result

    finally:
        neo4j_client.close()
        postgres_client.close()


def load_corpus_from_file(file_path: str) -> List[Dict]:
    """从文件加载语料"""
    import json

    corpus_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 支持多种格式
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    corpus_list.append({"id": str(i), "text": item})
                elif isinstance(item, dict):
                    corpus_list.append({
                        "id": item.get("id", str(i)),
                        "text": item.get("text", item.get("content", ""))
                    })
        elif isinstance(data, dict) and "corpus" in data:
            corpus_list = data["corpus"]

    return corpus_list


# 示例语料
SAMPLE_CORPUS = [
    {"id": "001", "text": "在洪山区的街道口，泛悦汇三楼的这家书店氛围感拉满。"},
    {"id": "002", "text": "武汉大学的樱花开了，大家都在行政楼前合影。"},
    {"id": "003", "text": "群光广场就在珞喻路上，离华中师范大学很近。"},
    {"id": "004", "text": "光谷那边新开了家盒马，比武商超市便宜。"},
    {"id": "005", "text": "华师旁边有个很好吃的火锅店，推荐大家去试试。"},
]


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="知识图谱构建工具")
    parser.add_argument("--input", "-i", help="输入语料文件路径(JSON)")
    parser.add_argument("--demo", action="store_true", help="使用示例语料演示")
    args = parser.parse_args()

    # 加载语料
    if args.demo:
        corpus_list = SAMPLE_CORPUS
        logger.info(f"使用示例语料: {len(corpus_list)} 条")
    elif args.input:
        corpus_list = load_corpus_from_file(args.input)
        logger.info(f"从文件加载语料: {len(corpus_list)} 条")
    else:
        # 默认使用示例语料
        corpus_list = SAMPLE_CORPUS
        logger.info(f"使用示例语料: {len(corpus_list)} 条")

    if not corpus_list:
        logger.error("没有加载到语料，退出")
        return

    # 执行处理
    await process_corpus_list(corpus_list)


if __name__ == "__main__":
    asyncio.run(main())