"""
测试Workflow - 从数据库social_notes_sample表读取数据运行完整流程
测试范围：Filter → Normalize → QA Scaffold → Joint NER+RE → Self-Check → Eval → Label → Entity Alignment
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
    """创建初始状态（v3.4扩展版：包含功能实体和事件实体）"""
    return {
        "corpus_id": corpus_id,
        "raw_text": raw_text,
        "_config_enable_normalize": True,  # 启用Normalize节点
        "_config_enable_qa_scaffold": True,  # 启用QA Scaffold节点
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
        # v3.4扩展：实体类型增加功能、事件
        "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
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
        # v3.4新增：功能实体和事件实体详细列表
        "function_entities": [],
        "event_entities": [],
        # P11新增：实体对齐状态
        "entity_alignment_result": {},
        "aligned_entity_ids": {},
        "new_entity_names": [],
        # P10新增：QA导师状态
        "mentor_guidance": {},
        "qa_approval_result": {},
        "integrated_semantic_summary": "",
        "revision_feedbacks": [],
        "revision_cycle_count": 0,
        "max_revision_cycles": 3,
        "pending_approval_nodes": [],
        "reasoning_trace": "",
        "current_step": StepEnum.NER,
        "error": None,
    }


async def process_single_corpus(llm, corpus: dict, config: ExtractionConfig) -> dict:
    """处理单条语料（完整流程：含实体对齐）"""
    workflow = build_corpus_workflow(
        llm,
        use_simplified_eval=config.use_simplified_eval,
        enable_filter=config.enable_filter,
        enable_normalize=config.enable_normalize,
        enable_qa_scaffold=config.enable_qa_scaffold,
        enable_self_check=config.enable_self_check,
        use_joint_extraction=config.use_joint_extraction,
        enable_full_self_check=config.enable_full_self_check,
        enable_entity_alignment=config.enable_entity_alignment,  # P11新增：实体对齐
        config=config,  # 传入完整配置对象
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

    # 统计实体和三元组（v3.4扩展版）
    total_entities = 0
    total_triples = 0
    entity_types_count = {}
    relation_types_count = {}

    # 统计实体对齐结果（P11新增）
    total_aligned_entities = 0
    total_new_entities = 0
    alignment_success_count = 0

    # 统计功能实体和事件实体（v3.4新增）
    total_function_entities = 0
    total_event_entities = 0

    # 统计各节点执行情况
    filter_valid_count = 0
    filter_invalid_count = 0
    normalize_count = 0
    qa_scaffold_count = 0
    self_check_count = 0

    for result in results:
        if result.get("error"):
            continue

        # 统计实体（含v3.4新增类型）
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

        # 统计功能实体和事件实体
        function_entities = result.get("function_entities", [])
        event_entities = result.get("event_entities", [])
        total_function_entities += len(function_entities)
        total_event_entities += len(event_entities)

        # 统计实体对齐结果
        aligned_entity_ids = result.get("aligned_entity_ids", {})
        new_entity_names = result.get("new_entity_names", [])
        if aligned_entity_ids:
            total_aligned_entities += len(aligned_entity_ids)
            alignment_success_count += 1
        if new_entity_names:
            total_new_entities += len(new_entity_names)

        # 统计各节点执行情况
        filter_result = result.get("filter_result", {})
        if filter_result:
            if filter_result.get("is_valid", True):
                filter_valid_count += 1
            else:
                filter_invalid_count += 1

        if result.get("normalize_result"):
            normalize_count += 1
        if result.get("qa_scaffold_result"):
            qa_scaffold_count += 1
        if result.get("self_check_joint_result"):
            self_check_count += 1

    stats = {
        "total": total,
        "success": success_count,
        "error": error_count,
        "total_entities": total_entities,
        "total_triples": total_triples,
        "entity_types_count": entity_types_count,
        "relation_types_count": relation_types_count,
        # v3.4新增统计
        "total_function_entities": total_function_entities,
        "total_event_entities": total_event_entities,
        # P11新增统计
        "alignment_stats": {
            "success_count": alignment_success_count,
            "total_aligned": total_aligned_entities,
            "total_new": total_new_entities,
        },
        # 节点执行统计
        "node_stats": {
            "filter_valid": filter_valid_count,
            "filter_invalid": filter_invalid_count,
            "normalize": normalize_count,
            "qa_scaffold": qa_scaffold_count,
            "self_check": self_check_count,
        },
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
    logger.info(f"[统计] 功能实体: {total_function_entities}, 事件实体: {total_event_entities}")
    logger.info(f"[统计] 实体对齐: 成功 {alignment_success_count}, 已对齐 {total_aligned_entities}, 新实体 {total_new_entities}")
    logger.info(f"[统计] 耗时: {elapsed_time:.2f} 秒")

    # 验证各节点输出完整性
    validate_node_outputs(results, stats)

    return stats


def validate_node_outputs(results: list, stats: dict):
    """验证各节点输出完整性"""
    logger.info("=" * 40)
    logger.info("节点输出验证")
    logger.info("=" * 40)

    # 验证v3.4实体类型扩展
    entity_types = stats.get("entity_types_count", {})
    has_function = "功能" in entity_types and entity_types["功能"] > 0
    has_event = "事件" in entity_types and entity_types["事件"] > 0

    if has_function:
        logger.info(f"[✓] 功能实体识别成功: {entity_types['功能']} 个")
    else:
        logger.warning("[⚠] 未识别到功能实体，请检查提示词是否正确传递ENTITY_DISTINCTION_RULES")

    if has_event:
        logger.info(f"[✓] 事件实体识别成功: {entity_types['事件']} 个")
    else:
        logger.warning("[⚠] 未识别到事件实体")

    # 验证Filter节点
    node_stats = stats.get("node_stats", {})
    if node_stats.get("filter_valid", 0) > 0:
        logger.info(f"[✓] Filter节点有效文本: {node_stats['filter_valid']} 条")
    if node_stats.get("filter_invalid", 0) > 0:
        logger.info(f"[✓] Filter节点无效文本: {node_stats['filter_invalid']} 条（已过滤）")

    # 验证Normalize节点
    if node_stats.get("normalize", 0) > 0:
        logger.info(f"[✓] Normalize节点执行: {node_stats['normalize']} 条")
    else:
        logger.warning("[⚠] Normalize节点未执行，检查是否启用")

    # 验证QA Scaffold节点
    if node_stats.get("qa_scaffold", 0) > 0:
        logger.info(f"[✓] QA Scaffold节点执行: {node_stats['qa_scaffold']} 条")
    else:
        logger.warning("[⚠] QA Scaffold节点未执行")

    # 验证Self-Check节点
    if node_stats.get("self_check", 0) > 0:
        logger.info(f"[✓] Self-Check节点执行: {node_stats['self_check']} 条")
    else:
        logger.warning("[⚠] Self-Check节点未执行")

    # 验证实体对齐
    alignment_stats = stats.get("alignment_stats", {})
    if alignment_stats.get("success_count", 0) > 0:
        logger.info(f"[✓] 实体对齐成功: {alignment_stats['success_count']} 条")
        logger.info(f"[✓] 已对齐实体: {alignment_stats['total_aligned']} 个, 新实体: {alignment_stats['total_new']} 个")
    else:
        logger.warning("[⚠] 实体对齐未执行或无匹配结果")

    # 验证关系属性过滤
    relation_types = stats.get("relation_types_count", {})
    logger.info(f"[关系] 类型分布: {relation_types}")

    logger.info("=" * 40)


async def main():
    """主测试流程 - 测试完整Workflow功能（含实体对齐）"""
    logger.info("=" * 60)
    logger.info("Workflow完整测试 - 测试所有节点功能")
    logger.info("流程: Filter → Normalize → QA Scaffold → Joint NER+RE → Self-Check → Eval → Label → Entity Alignment")
    logger.info("=" * 60)

    # 配置 - 启用所有节点（暂时禁用实体对齐以加快测试）
    config = ExtractionConfig(
        use_simplified_eval=True,
        enable_filter=True,  # P5：Filter筛选无效文本
        enable_normalize=True,  # P6：Normalize归一化文本
        enable_qa_scaffold=True,  # P8：QA Scaffold语义脚手架
        enable_self_check=True,  # P9：Self-Check校验节点
        use_joint_extraction=True,  # P9：联合抽取模式
        enable_full_self_check=False,  # 不启用所有二次检查（可选）
        enable_entity_alignment=False,  # P11：暂时禁用实体对齐（embedding加载耗时较长）
        max_concurrent_corpus=3,  # 最大并发数（降低以避免API限流）
        prompt_version="v2",  # 使用原版提示词（稳定）
    )

    # 从数据库读取数据
    try:
        corpus_list = fetch_sample_corpus(limit=10)  # 先测试10条数据
    except Exception as e:
        logger.error(f"读取数据库失败: {e}")
        logger.info("尝试使用模拟数据进行测试...")
        # 使用模拟数据（包含功能实体和事件实体的测试场景）
        corpus_list = [
            {"id": "mock_001", "text": "武汉大学在珞喻路上，旁边就是东湖风景区，很适合周末散步"},
            {"id": "mock_002", "text": "光谷广场地铁站附近有很多商场，购物方便，适合逛街"},
            {"id": "mock_003", "text": "街道口的群光广场很适合逛街，比汉街更热闹，樱花季人多"},
            {"id": "mock_004", "text": "华农校园很大，在狮子山那边，每年樱花节很多人去打卡"},
            {"id": "mock_005", "text": "今天心情不好，不想出门"},  # 无地理信息的文本（Filter应过滤）
            {"id": "mock_006", "text": "这家咖啡厅停业了，改成了一家书店"},  # 事件实体测试
            {"id": "mock_007", "text": "头皮护理店在商场二楼，排队很久"},  # 功能实体测试
            {"id": "mock_008", "text": "EHD双店长来武汉了，活动很精彩"},  # 事件实体测试
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
        max_concurrent=3,  # 降低并发数以避免API限流
    )

    # 输出详细统计结果
    logger.info("=" * 60)
    logger.info("测试完成 - 统计结果")
    logger.info("=" * 60)
    logger.info(f"[总体] 处理: {stats['total']} 条, 成功: {stats['success']}, 失败: {stats['error']}")
    logger.info(f"[实体] 总数: {stats['total_entities']}, 类型分布: {stats['entity_types_count']}")
    logger.info(f"[功能实体] {stats['total_function_entities']} 个, [事件实体] {stats['total_event_entities']} 个")
    logger.info(f"[三元组] 总数: {stats['total_triples']}, 关系分布: {stats['relation_types_count']}")
    logger.info(f"[对齐] 成功: {stats['alignment_stats']['success_count']}, 已对齐: {stats['alignment_stats']['total_aligned']}, 新实体: {stats['alignment_stats']['total_new']}")
    logger.info(f"[节点] Filter有效: {stats['node_stats']['filter_valid']}, Filter无效: {stats['node_stats']['filter_invalid']}")
    logger.info(f"[节点] Normalize: {stats['node_stats']['normalize']}, QA Scaffold: {stats['node_stats']['qa_scaffold']}, Self-Check: {stats['node_stats']['self_check']}")
    logger.info(f"[耗时] {stats['elapsed_time']:.2f} 秒")
    logger.info("=" * 60)

    return stats


if __name__ == "__main__":
    asyncio.run(main())