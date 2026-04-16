"""
测试Workflow - 从数据库social_notes_sample表读取50条数据运行
结果输出到文件，不保存到数据库
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 加载环境变量
load_dotenv(os.path.join(project_root, '.env'))

from langchain_openai import ChatOpenAI
from loguru import logger

from agent.agents.workflow import build_corpus_workflow, process_batch
from agent.agents.state import CorpusState, StepEnum, DEFAULT_MAX_RETRIES
from agent.agents.config import ExtractionConfig, DEFAULT_CONFIG
from kg.postgres_client import PostgresClient


def create_llm():
    """创建 LLM 实例（使用 DeepSeek API）"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE_URL")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 未设置，请检查 .env 文件")

    logger.info(f"[LLM] 使用模型: {model}")
    logger.info(f"[LLM] API Base URL: {base_url}")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


def get_database_config() -> dict:
    """获取数据库配置"""
    pg_password = os.getenv("PG_PASSWORD")
    if not pg_password:
        raise ValueError("PG_PASSWORD 未设置，请检查 .env 文件")

    return {
        "pg_host": os.getenv("PG_HOST", "localhost"),
        "pg_port": int(os.getenv("PG_PORT", "5432")),
        "pg_database": os.getenv("PG_DATABASE", "kg"),
        "pg_user": os.getenv("PG_USER", "postgres"),
        "pg_password": pg_password,
    }


def fetch_sample_corpus(limit: int = 50) -> list:
    """从数据库social_notes_sample表读取样本语料"""
    db_config = get_database_config()

    with PostgresClient(
        db_config["pg_host"],
        db_config["pg_port"],
        db_config["pg_database"],
        db_config["pg_user"],
        db_config["pg_password"]
    ) as pg:
        # 尝试读取social_media_notes_sampled表
        # 根据实际表结构调整列名
        corpus_list = pg.fetch_corpus_for_extraction(
            table_name="social_media_notes_sampled",
            text_column="content_cleaned",  # 文本列名
            id_column="note_id",  # ID列名
            limit=limit,
            offset=0,
        )
        logger.info(f"[数据库] 成功读取 {len(corpus_list)} 条语料")
        return corpus_list


def create_initial_state(corpus_id: str, raw_text: str) -> CorpusState:
    """创建初始状态"""
    return {
        "corpus_id": corpus_id,
        "raw_text": raw_text,
        "_config_enable_normalize": False,
        "_config_enable_qa_scaffold": False,
        "filter_result": {},
        "normalize_result": {},
        "normalized_text": "",
        "qa_scaffold_result": {},
        "semantic_summary": "",
        "qa_entity_hints": [],
        "qa_relation_hints": [],
        "qa_context_dependencies": [],
        "joint_extraction_result": {},
        "extraction_strategy": "",
        "self_check_filter_result": {},
        "self_check_normalize_result": {},
        "self_check_qa_result": {},
        "self_check_joint_result": {},
        "self_check_eval_result": {},
        "self_check_label_result": {},
        "reflection_text": "",
        "improvement_strategy": "",
        "reflection_history": [],
        "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
        "triples": [],
        "eval_scores": [],
        "eval_passed": False,
        "corrected_triples": [],
        "self_check_ner_result": {},
        "self_check_re_result": {},
        "final_entities": [],
        "final_triples": [],
        "verification_confidence": "medium",
        "retry_count": 0,
        "max_retries": DEFAULT_MAX_RETRIES,
        "retry_reason": "",
        "retry_suggested": False,
        "problem_entities": [],
        "problem_triples": [],
        "needs_review": False,
        "entity_attrs": {},
        "relation_attrs": {},
        "current_step": StepEnum.NER,
        "error": None,
    }


async def process_single_corpus(llm, corpus: dict, config: ExtractionConfig) -> dict:
    """处理单条语料"""
    workflow = build_corpus_workflow(
        llm,
        use_simplified_eval=config.use_simplified_eval,
        enable_filter=config.enable_filter,
        enable_normalize=config.enable_normalize,
        enable_qa_scaffold=config.enable_qa_scaffold,
        enable_self_check=config.enable_self_check,
        use_joint_extraction=config.use_joint_extraction,
        enable_full_self_check=config.enable_full_self_check,
        max_retries=config.self_check_max_retries,
        prompt_version=config.prompt_version,
    )

    corpus_id = corpus.get("id", "unknown")
    raw_text = corpus.get("text", "")

    if not raw_text or len(raw_text.strip()) < config.min_text_length:
        return {
            "corpus_id": corpus_id,
            "error": "文本为空或长度不足",
            "raw_text": raw_text,
        }

    initial_state = create_initial_state(corpus_id, raw_text)
    thread_config = {"configurable": {"thread_id": f"test_{corpus_id}_{os.getpid()}"}}

    try:
        result = await workflow.ainvoke(initial_state, thread_config)
        return result
    except Exception as e:
        logger.error(f"处理语料失败 {corpus_id}: {e}")
        return {
            "corpus_id": corpus_id,
            "error": str(e),
            "raw_text": raw_text,
        }


async def run_workflow_test(
    corpus_list: list,
    config: ExtractionConfig,
    output_file: str,
    max_concurrent: int = 5
) -> dict:
    """
    运行Workflow测试

    Args:
        corpus_list: 语料列表
        config: 配置
        output_file: 输出文件路径
        max_concurrent: 最大并发数
    """
    llm = create_llm()
    results = []

    # 创建并发控制信号量
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_limit(corpus: dict, idx: int) -> tuple:
        """带并发限制的处理"""
        async with semaphore:
            logger.info(f"[进度] 处理第 {idx + 1}/{len(corpus_list)} 条语料")
            result = await process_single_corpus(llm, corpus, config)
            return idx, result

    # 并行处理所有语料
    start_time = time.time()
    tasks = [process_with_limit(corpus, i) for i, corpus in enumerate(corpus_list)]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # 按原始顺序整理结果
    for item in raw_results:
        if isinstance(item, Exception):
            logger.error(f"处理任务异常: {item}")
            results.append({"error": str(item)})
        else:
            idx, result = item
            results.append(result)

    elapsed_time = time.time() - start_time

    # 统计结果
    total = len(results)
    success_count = sum(1 for r in results if not r.get("error"))
    error_count = total - success_count

    # 统计实体和三元组
    total_entities = 0
    total_triples = 0
    entity_types_count = {}
    relation_types_count = {}

    for result in results:
        if result.get("error"):
            continue

        # 统计实体
        entities = result.get("entities", {})
        for entity_type, names in entities.items():
            if names:
                count = len(names)
                total_entities += count
                entity_types_count[entity_type] = entity_types_count.get(entity_type, 0) + count

        # 统计三元组
        triples = result.get("triples", [])
        total_triples += len(triples)
        for t in triples:
            relation = t.get("relation", "unknown")
            relation_types_count[relation] = relation_types_count.get(relation, 0) + 1

    stats = {
        "total": total,
        "success": success_count,
        "error": error_count,
        "total_entities": total_entities,
        "total_triples": total_triples,
        "entity_types_count": entity_types_count,
        "relation_types_count": relation_types_count,
        "elapsed_time": elapsed_time,
    }

    # 构建输出数据
    output_data = {
        "test_time": datetime.now().isoformat(),
        "config": config.to_dict(),
        "stats": stats,
        "results": results,
    }

    # 写入文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"[输出] 结果已保存到: {output_path}")
    logger.info(f"[统计] 总计: {total} 条, 成功: {success_count}, 失败: {error_count}")
    logger.info(f"[统计] 实体总数: {total_entities}, 三元组总数: {total_triples}")
    logger.info(f"[统计] 耗时: {elapsed_time:.2f} 秒")

    return stats


async def main():
    """主测试流程"""
    logger.info("=" * 60)
    logger.info("Workflow测试 - 从数据库读取50条数据")
    logger.info("=" * 60)

    # 配置
    config = ExtractionConfig(
        use_simplified_eval=True,
        enable_filter=True,  # 启用Filter筛选无效文本
        enable_normalize=False,
        enable_qa_scaffold=False,
        enable_self_check=False,
        use_joint_extraction=True,  # 使用联合抽取模式
        enable_full_self_check=False,
        max_concurrent_corpus=5,  # 最大并发数
        prompt_version="v2",  # 使用原版提示词
    )

    # 从数据库读取数据
    try:
        corpus_list = fetch_sample_corpus(limit=50)
    except Exception as e:
        logger.error(f"读取数据库失败: {e}")
        logger.info("尝试使用模拟数据进行测试...")
        # 使用模拟数据
        corpus_list = [
            {"id": "mock_001", "text": "武汉大学在珞喻路上，旁边就是东湖风景区"},
            {"id": "mock_002", "text": "光谷广场地铁站附近有很多商场，比如光谷步行街"},
            {"id": "mock_003", "text": "街道口的群光广场很适合逛街，比汉街更热闹"},
            {"id": "mock_004", "text": "华农校园很大，在狮子山那边"},
            {"id": "mock_005", "text": "今天心情不好，不想出门"},  # 无地理信息的文本
        ]

    if not corpus_list:
        logger.warning("没有可用的语料数据")
        return

    # 输出文件路径
    output_file = os.path.join(project_root, "agent", "test_output", f"workflow_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # 运行测试
    stats = await run_workflow_test(
        corpus_list=corpus_list,
        config=config,
        output_file=output_file,
        max_concurrent=5,
    )

    logger.info("=" * 60)
    logger.info("测试完成")
    logger.info("=" * 60)

    return stats


if __name__ == "__main__":
    asyncio.run(main())