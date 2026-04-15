"""
测试完整工作流程: Filter → Normalize → QA → Joint_NER_RE → Eval → Label → Entity_Alignment
从 PostgreSQL social_media_notes_sampled 表获取50条数据，结果输出到JSON文件
"""
import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent  # agent/tests -> agent -> project_root
sys.path.insert(0, str(project_root / "agent"))  # 添加 agent 目录
sys.path.insert(0, str(project_root))  # 添加项目根目录

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from agents import (
    ExtractionConfig,
    build_corpus_workflow,
    CorpusState,
    StepEnum,
)
from kg.postgres_client import PostgresClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_llm(model: str = "deepseek-chat", temperature: float = 0.0):
    """创建 LLM 实例"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE_URL")

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 未设置")

    logger.info(f"[LLM] 使用模型: {model}, Base URL: {base_url}")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def create_test_config() -> ExtractionConfig:
    """创建测试配置：启用 Filter + Normalize + QA + Joint NER RE + Entity Alignment"""
    config = ExtractionConfig(
        # 启用所有前置节点
        enable_filter=True,
        enable_normalize=True,
        enable_qa_scaffold=True,

        # 使用联合抽取模式
        use_joint_extraction=True,
        use_simplified_eval=True,

        # 不启用 Self-Check 和 Reflexion（简化流程）
        enable_self_check=False,
        enable_full_self_check=False,
        enable_reflexion=False,

        # 启用实体对齐（不保存到数据库，只输出结果）
        enable_entity_alignment=True,
        alignment_similarity_threshold=0.75,
        alignment_high_confidence_threshold=0.90,
        alignment_top_k=5,
        alignment_use_llm_decision=True,  # 中置信度时使用LLM判断
        alignment_embedding_model="shibing624/text2vec-base-chinese",

        # 不启用 QA导师模式
        enable_qa_mentor=False,

        # 其他参数
        eval_threshold=3.5,
        max_text_length=2000,
        qa_scaffold_min_text_length=10,
    )

    logger.info(f"[Config] 配置: Filter={config.enable_filter}, Normalize={config.enable_normalize}, "
                f"QA={config.enable_qa_scaffold}, Joint={config.use_joint_extraction}, "
                f"EntityAlignment={config.enable_entity_alignment}")

    return config


def connect_postgres() -> PostgresClient:
    """连接 PostgreSQL 数据库"""
    pg = PostgresClient(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        database=os.getenv("PG_DATABASE", "bishe"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "postgres"),
    )
    logger.info(f"[PostgreSQL] 连接成功")
    return pg


def fetch_corpus(pg: PostgresClient, limit: int = 50) -> List[Dict]:
    """从 social_media_notes_sampled 表获取语料"""
    corpus_list = pg.fetch_corpus_for_extraction(
        table_name="social_media_notes_sampled",
        text_column="content_cleaned",  # 文本列名
        id_column="note_id",  # ID列名
        limit=limit,
        offset=0,
    )
    logger.info(f"[Fetch] 获取到 {len(corpus_list)} 条语料")
    return corpus_list


async def process_single_corpus(
    workflow: Any,
    corpus: Dict,
    thread_id: str
) -> Dict:
    """处理单条语料"""
    corpus_id = corpus["id"]
    raw_text = corpus["text"]

    # 构建初始状态
    initial_state: CorpusState = {
        "corpus_id": corpus_id,
        "raw_text": raw_text,

        # Filter 初始状态
        "filter_result": {},

        # Normalize 初始状态
        "normalize_result": {},
        "normalized_text": "",

        # QA Scaffold 初始状态
        "qa_scaffold_result": {},
        "semantic_summary": "",
        "qa_entity_hints": [],
        "qa_relation_hints": [],
        "qa_context_dependencies": [],

        # Extract 初始状态
        "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
        "triples": [],
        "joint_extraction_result": {},

        # Eval 初始状态
        "eval_scores": [],
        "eval_passed": False,
        "corrected_triples": [],

        # Label 初始状态
        "entity_attrs": {},
        "relation_attrs": {},

        # Self-Check 初始状态（不启用）
        "self_check_ner_result": {},
        "self_check_re_result": {},
        "self_check_joint_result": {},
        "self_check_qa_result": {},
        "self_check_eval_result": {},
        "self_check_label_result": {},

        # Reflexion 初始状态（不启用）
        "reflection_text": "",
        "improvement_strategy": "",
        "reflection_history": [],

        # Retry 初始状态
        "retry_count": 0,
        "max_retries": 3,
        "retry_reason": "",
        "problem_entities": [],
        "problem_triples": [],
        "needs_review": False,

        # QA Mentor 初始状态（不启用）
        "mentor_guidance": {},
        "qa_approval_result": {},
        "revision_feedbacks": [],

        # Entity Alignment 初始状态（不启用）
        "entity_alignment_result": {},
        "aligned_entity_ids": {},
        "new_entity_names": [],

        # Output 初始状态
        "final_entities": [],
        "final_triples": [],
        "verification_confidence": "medium",

        # Config 初始状态
        "enable_normalize_flag": True,
        "enable_qa_scaffold_flag": True,

        # Control 初始状态
        "current_step": StepEnum.NER,
        "error": None,
    }

    thread_config = {"configurable": {"thread_id": thread_id}}

    # 流式执行工作流
    events = []
    async for event in workflow.astream(initial_state, thread_config, stream_mode="custom"):
        events.append(event)
        step = event.get("step", "unknown")
        status = event.get("status", "unknown")
        logger.info(f"[{corpus_id}] {step}: {status}")

    # 获取最终状态
    final_state = await workflow.aget_state(thread_config)
    final_values = final_state.values

    return {
        "corpus_id": corpus_id,
        "raw_text": raw_text,
        "events": events,
        "final_state": {
            # Filter 结果
            "filter_result": final_values.get("filter_result", {}),

            # Normalize 结果
            "normalized_text": final_values.get("normalized_text", ""),
            "normalize_result": final_values.get("normalize_result", {}),

            # QA Scaffold 结果
            "semantic_summary": final_values.get("semantic_summary", ""),
            "qa_entity_hints": final_values.get("qa_entity_hints", []),
            "qa_relation_hints": final_values.get("qa_relation_hints", []),

            # Joint NER RE 结果
            "entities": final_values.get("entities", {}),
            "triples": final_values.get("triples", []),

            # Eval 结果
            "corrected_triples": final_values.get("corrected_triples", []),
            "eval_passed": final_values.get("eval_passed", False),

            # Label 结果
            "entity_attrs": final_values.get("entity_attrs", {}),
            "relation_attrs": final_values.get("relation_attrs", {}),

            # Entity Alignment 结果（不保存到数据库）
            "entity_alignment_result": final_values.get("entity_alignment_result", {}),
            "aligned_entity_ids": final_values.get("aligned_entity_ids", {}),
            "new_entity_names": final_values.get("new_entity_names", []),

            # 错误信息
            "error": final_values.get("error"),
        }
    }


async def run_test():
    """运行测试"""
    logger.info("=" * 60)
    logger.info("开始测试完整工作流程")
    logger.info("流程: Filter → Normalize → QA → Joint_NER_RE → Eval → Label → Entity_Alignment")
    logger.info("=" * 60)

    # 1. 创建配置
    config = create_test_config()

    # 2. 创建 LLM
    llm = create_llm(model="deepseek-chat", temperature=0.0)

    # 3. 构建工作流（传入config用于实体对齐节点）
    workflow = build_corpus_workflow(
        llm,
        use_simplified_eval=config.use_simplified_eval,
        enable_self_check=config.enable_self_check,
        enable_filter=config.enable_filter,
        enable_normalize=config.enable_normalize,
        enable_qa_scaffold=config.enable_qa_scaffold,
        use_joint_extraction=config.use_joint_extraction,
        enable_full_self_check=config.enable_full_self_check,
        enable_entity_alignment=config.enable_entity_alignment,
        config=config,  # 传入完整配置对象（用于实体对齐）
        max_retries=config.self_check_max_retries,
    )

    logger.info("[Workflow] 工作流构建完成")

    # 4. 连接数据库并获取语料
    pg = connect_postgres()
    corpus_list = fetch_corpus(pg, limit=5)  # 先测试5条验证实体对齐

    if not corpus_list:
        logger.error("未获取到任何语料，测试终止")
        pg.close()
        return

    # 5. 处理每条语料
    results = []
    total = len(corpus_list)

    for i, corpus in enumerate(corpus_list):
        logger.info(f"\n[{i+1}/{total}] 处理语料: {corpus['id']}")
        logger.info(f"原文长度: {len(corpus['text'])} 字符")

        thread_id = f"test_{corpus['id']}_{datetime.now().strftime('%H%M%S')}"

        try:
            result = await process_single_corpus(workflow, corpus, thread_id)
            results.append(result)

            # 打印简要结果
            entities = result["final_state"]["entities"]
            triples = result["final_state"]["corrected_triples"]
            entity_count = sum(len(v) for v in entities.values())
            aligned_ids = result["final_state"]["aligned_entity_ids"]
            new_names = result["final_state"]["new_entity_names"]
            logger.info(f"[结果] 实体: {entity_count}个, 三元组: {len(triples)}个, "
                        f"对齐: {len(aligned_ids)}个, 新实体: {len(new_names)}个")

        except Exception as e:
            logger.error(f"[错误] 处理失败: {e}")
            results.append({
                "corpus_id": corpus["id"],
                "raw_text": corpus["text"],
                "error": str(e),
            })

    # 6. 关闭数据库连接
    pg.close()

    # 7. 保存结果到文件
    output_dir = project_root / "agent" / "test_output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_time": datetime.now().isoformat(),
            "config": config.to_dict(),
            "total_corpus": total,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"\n[完成] 结果已保存到: {output_file}")

    # 8. 打印统计信息
    success_count = sum(1 for r in results if "final_state" in r and r["final_state"].get("triples"))
    total_entities = sum(
        sum(len(v) for v in r["final_state"]["entities"].values())
        for r in results if "final_state" in r
    )
    total_triples = sum(
        len(r["final_state"]["corrected_triples"])
        for r in results if "final_state" in r
    )

    # 实体对齐统计
    total_aligned = sum(
        len(r["final_state"]["aligned_entity_ids"])
        for r in results if "final_state" in r
    )
    total_new_entities = sum(
        len(r["final_state"]["new_entity_names"])
        for r in results if "final_state" in r
    )
    geo_aligned = sum(
        r["final_state"]["entity_alignment_result"].get("geo_aligned_count", 0)
        for r in results if "final_state" in r and r["final_state"].get("entity_alignment_result")
    )
    amap_aligned = sum(
        r["final_state"]["entity_alignment_result"].get("amap_aligned_count", 0)
        for r in results if "final_state" in r and r["final_state"].get("entity_alignment_result")
    )

    logger.info("\n" + "=" * 60)
    logger.info("测试统计:")
    logger.info(f"  总语料数: {total}")
    logger.info(f"  成功抽取数: {success_count}")
    logger.info(f"  总实体数: {total_entities}")
    logger.info(f"  总三元组数: {total_triples}")
    logger.info("---")
    logger.info("实体对齐统计:")
    logger.info(f"  已对齐实体: {total_aligned}")
    logger.info(f"  新实体: {total_new_entities}")
    logger.info(f"  geo_entity_names匹配: {geo_aligned}")
    logger.info(f"  amap_poi_wgs84匹配: {amap_aligned}")
    if total_entities > 0:
        alignment_rate = total_aligned / total_entities
        logger.info(f"  对齐率: {alignment_rate:.1%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_test())