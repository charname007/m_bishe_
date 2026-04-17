#!/usr/bin/env python3
"""
社交媒体语料批处理脚本。

功能：
- 从 social_media_notes_sampled 表获取待处理数据
- 按批次处理语料（默认 50）
- 运行完整 workflow（含自检节点）
- 管理语料状态并记录错误

使用方法：
    python scripts/process_social_media_corpus.py --max 5 --batch-size 5
"""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

from dotenv import load_dotenv
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from agent.agents.config import ExtractionConfig
from agent.agents.workflow import build_distributed_workflow
from kg.neo4j_client import Neo4jClient
from kg.postgres_client import PostgresClient
from langchain_openai import ChatOpenAI


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(
    ROOT_DIR,
    "scripts",
    "output",
    f"corpus_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logger.add(LOG_FILE, level="DEBUG", encoding="utf-8")
print(f"Log file: {LOG_FILE}")


# ===== 配置 =====
DEFAULT_TABLE = "social_media_notes_sampled"
DEFAULT_TEXT_COLUMN = "content_cleaned"
DEFAULT_ID_COLUMN = "note_id"
DEFAULT_TIME_COLUMN = "publish_time"
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_TOTAL = 500
DETAIL_LOG_SAMPLE_LIMIT = 5
DETAIL_LOG_TEXT_PREVIEW = 120


def _safe_preview(text: Any, limit: int = DETAIL_LOG_TEXT_PREVIEW) -> str:
    """文本预览，避免日志过长。"""
    if text is None:
        return ""
    value = str(text)
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _safe_count(value: Any) -> int:
    """安全获取长度。"""
    if value is None:
        return 0
    try:
        return len(value)
    except Exception:
        return 0


def _entity_type_counts(entities: Any) -> Dict[str, int]:
    """统计实体类型数量。"""
    if not isinstance(entities, dict):
        return {}
    counts = {}
    for entity_type, names in entities.items():
        if isinstance(names, list):
            counts[str(entity_type)] = len(names)
    return counts


def _compact_eval_scores(eval_scores: Any) -> list:
    """压缩评分信息，只保留关键信息并截断数量。"""
    if not isinstance(eval_scores, list):
        return []

    compact = []
    for score in eval_scores[:DETAIL_LOG_SAMPLE_LIMIT]:
        if isinstance(score, dict):
            compact.append(
                {
                    "dimension": score.get("dimension") or score.get("name"),
                    "score": score.get("score"),
                    "passed": score.get("passed"),
                    "confidence": score.get("confidence"),
                    "comment": _safe_preview(score.get("comment") or score.get("reason")),
                }
            )
        else:
            compact.append(_safe_preview(score))
    return compact


def _compact_alignment_items(alignment_items: Any) -> list:
    """压缩实体对齐明细，避免日志过长。"""
    if not isinstance(alignment_items, list):
        return []

    compact = []
    for item in alignment_items[:DETAIL_LOG_SAMPLE_LIMIT]:
        if not isinstance(item, dict):
            compact.append(_safe_preview(item))
            continue

        best_match = item.get("best_match") or {}
        similarity = best_match.get("similarity")
        if similarity is not None:
            try:
                similarity = round(float(similarity), 4)
            except Exception:
                similarity = str(similarity)

        compact.append(
            {
                "name": item.get("extracted_name"),
                "status": item.get("alignment_status"),
                "match_name": best_match.get("name"),
                "db_id": best_match.get("db_entity_id"),
                "similarity": similarity,
                "source": best_match.get("source"),
                "decision": _safe_preview(item.get("llm_decision")),
            }
        )
    return compact


def log_corpus_node_details(batch_id: str, worker_id: str, corpus_state: Dict[str, Any]):
    """输出单条语料的各节点详细结果日志。"""
    if not isinstance(corpus_state, dict):
        return

    corpus_id = str(corpus_state.get("corpus_id", "unknown"))
    prefix = f"[Batch {batch_id}][{worker_id}][Corpus {corpus_id}]"

    # Filter
    filter_result = corpus_state.get("filter_result") or {}
    if filter_result:
        logger.info(
            f"{prefix}[Filter] {json.dumps({
                'is_valid': filter_result.get('is_valid'),
                'skip_reason': filter_result.get('skip_reason'),
                'confidence': filter_result.get('confidence'),
                'reason': _safe_preview(filter_result.get('reason')),
            }, ensure_ascii=False)}"
        )

    # Normalize
    normalize_result = corpus_state.get("normalize_result") or {}
    if normalize_result:
        logger.info(
            f"{prefix}[Normalize] {json.dumps({
                'has_changes': normalize_result.get('has_changes'),
                'confidence': normalize_result.get('confidence'),
                'normalizations_count': _safe_count(normalize_result.get('normalizations')),
                'normalized_preview': _safe_preview(corpus_state.get('normalized_text')),
            }, ensure_ascii=False)}"
        )

    # QA Scaffold
    qa_result = corpus_state.get("qa_scaffold_result") or {}
    if qa_result:
        logger.info(
            f"{prefix}[QA_Scaffold] {json.dumps({
                'qa_pairs': _safe_count(qa_result.get('qa_pairs')),
                'entity_hints': _safe_count(qa_result.get('entity_hints') or corpus_state.get('qa_entity_hints')),
                'relation_hints': _safe_count(qa_result.get('relation_hints') or corpus_state.get('qa_relation_hints')),
                'context_dependencies': _safe_count(qa_result.get('context_dependencies') or corpus_state.get('qa_context_dependencies')),
                'confidence': qa_result.get('overall_confidence') or qa_result.get('confidence'),
            }, ensure_ascii=False)}"
        )

    # Joint Extraction
    joint_result = corpus_state.get("joint_extraction_result") or {}
    entities = corpus_state.get("entities") or joint_result.get("entities") or {}
    triples = (
        corpus_state.get("corrected_triples")
        or corpus_state.get("triples")
        or joint_result.get("triples")
        or []
    )
    if entities or triples or joint_result:
        entity_counts = _entity_type_counts(entities)
        logger.info(
            f"{prefix}[Joint_NER_RE] {json.dumps({
                'entity_total': sum(entity_counts.values()),
                'entity_types': entity_counts,
                'triple_count': _safe_count(triples),
                'confidence': joint_result.get('overall_confidence') or corpus_state.get('verification_confidence'),
            }, ensure_ascii=False)}"
        )

    # Self-Check 节点
    self_check_filter = corpus_state.get("self_check_filter_result") or {}
    if self_check_filter:
        logger.info(
            f"{prefix}[Self_Check_Filter] {json.dumps({
                'confidence': self_check_filter.get('overall_confidence') or self_check_filter.get('confidence'),
                'retry_suggested': self_check_filter.get('retry_suggested'),
                'issues': _safe_count(self_check_filter.get('issues')),
            }, ensure_ascii=False)}"
        )

    self_check_normalize = corpus_state.get("self_check_normalize_result") or {}
    if self_check_normalize:
        logger.info(
            f"{prefix}[Self_Check_Normalize] {json.dumps({
                'confidence': self_check_normalize.get('overall_confidence') or self_check_normalize.get('confidence'),
                'retry_suggested': self_check_normalize.get('retry_suggested'),
                'issues': _safe_count(self_check_normalize.get('issues')),
            }, ensure_ascii=False)}"
        )

    self_check_qa = corpus_state.get("self_check_qa_result") or {}
    if self_check_qa:
        logger.info(
            f"{prefix}[Self_Check_QA] {json.dumps({
                'confidence': self_check_qa.get('overall_confidence') or self_check_qa.get('confidence'),
                'retry_suggested': self_check_qa.get('retry_suggested'),
                'issues': _safe_count(self_check_qa.get('issues')),
            }, ensure_ascii=False)}"
        )

    self_check_joint = corpus_state.get("self_check_joint_result") or {}
    if self_check_joint:
        logger.info(
            f"{prefix}[Self_Check_Joint] {json.dumps({
                'confidence': self_check_joint.get('overall_confidence') or self_check_joint.get('confidence'),
                'retry_suggested': self_check_joint.get('retry_suggested'),
                'improvement_actions': _safe_count(self_check_joint.get('improvement_actions')),
                'reflection': _safe_preview(self_check_joint.get('reflection_text')),
            }, ensure_ascii=False)}"
        )

    # Eval / Self-Check Eval
    eval_scores = corpus_state.get("eval_scores") or []
    if eval_scores or ("eval_passed" in corpus_state):
        logger.info(
            f"{prefix}[Eval] {json.dumps({
                'eval_passed': corpus_state.get('eval_passed'),
                'score_count': _safe_count(eval_scores),
                'score_samples': _compact_eval_scores(eval_scores),
            }, ensure_ascii=False)}"
        )

    self_check_eval = corpus_state.get("self_check_eval_result") or {}
    if self_check_eval:
        logger.info(
            f"{prefix}[Self_Check_Eval] {json.dumps({
                'confidence': self_check_eval.get('overall_confidence') or self_check_eval.get('confidence'),
                'retry_suggested': self_check_eval.get('retry_suggested'),
                'improvement_actions': _safe_count(self_check_eval.get('improvement_actions')),
            }, ensure_ascii=False)}"
        )

    # Label / Self-Check Label
    entity_attrs = corpus_state.get("entity_attrs") or {}
    relation_attrs = corpus_state.get("relation_attrs") or {}
    if entity_attrs or relation_attrs:
        logger.info(
            f"{prefix}[Label] {json.dumps({
                'entity_attrs_count': _safe_count(entity_attrs),
                'relation_attrs_count': _safe_count(relation_attrs),
            }, ensure_ascii=False)}"
        )

    self_check_label = corpus_state.get("self_check_label_result") or {}
    if self_check_label:
        logger.info(
            f"{prefix}[Self_Check_Label] {json.dumps({
                'confidence': self_check_label.get('overall_confidence') or self_check_label.get('confidence'),
                'retry_suggested': self_check_label.get('retry_suggested'),
                'improvement_actions': _safe_count(self_check_label.get('improvement_actions')),
            }, ensure_ascii=False)}"
        )

    # Entity Alignment
    alignment_result = corpus_state.get("entity_alignment_result") or {}
    if alignment_result:
        logger.info(
            f"{prefix}[Entity_Alignment] {json.dumps({
                'aligned_count': _safe_count(alignment_result.get('aligned_entities')),
                'new_count': _safe_count(alignment_result.get('new_entities')),
                'created_count': alignment_result.get('created_count') or _safe_count(alignment_result.get('created_entities')),
                'skipped_count': _safe_count(alignment_result.get('skipped_entities')),
                'alignment_rate': alignment_result.get('overall_alignment_rate'),
                'confidence': alignment_result.get('alignment_confidence'),
                'samples': _compact_alignment_items(alignment_result.get('alignment_items')),
            }, ensure_ascii=False)}"
        )

    # 总结
    logger.info(
        f"{prefix}[Summary] {json.dumps({
            'current_step': corpus_state.get('current_step'),
            'batch_processed': corpus_state.get('batch_processed'),
            'skip_reason': corpus_state.get('skip_reason'),
            'error': corpus_state.get('error'),
        }, ensure_ascii=False)}"
    )


def init_clients():
    """初始化数据库客户端。"""
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


def init_workflow():
    """初始化 LLM 和 workflow。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE_URL")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)

    qa_model = os.getenv("QA_LLM_MODEL", os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"))
    qa_temperature = float(os.getenv("QA_LLM_TEMPERATURE", 0.7))
    qa_llm = ChatOpenAI(
        model=qa_model,
        api_key=api_key,
        base_url=base_url,
        temperature=qa_temperature,
    )

    config = ExtractionConfig.from_env()
    config.enable_filter = True
    config.enable_normalize = True
    config.enable_qa_scaffold = True
    config.enable_full_self_check = True
    config.enable_entity_alignment = True
    config.enable_batch_llm = True
    config.enable_qa_mentor = True
    # 批处理优先：禁止退化为单条处理，阶段失败后直接跳过
    config.batch_llm_fallback = False
    config.batch_skip_on_repeated_failure = True
    config.batch_stage_retry_attempts = 1
    config.batch_qa_scaffold_retry_attempts = 1

    workflow = build_distributed_workflow(llm, config, qa_llm=qa_llm)
    return workflow, config


def verify_runtime_dependencies(pg_client: PostgresClient, neo4j_client: Neo4jClient):
    """启动前验证关键依赖连通性。"""
    with pg_client.conn.cursor() as cursor:
        cursor.execute("SELECT 1")
        cursor.fetchone()

    neo4j_client.driver.verify_connectivity()


def safe_close(client, client_name: str):
    """安全关闭客户端。"""
    if client is None:
        return
    try:
        client.close()
    except Exception as error:
        logger.warning(f"关闭{client_name}失败: {error}")


async def process_batch(
    workflow,
    pg_client: PostgresClient,
    neo4j_client: Neo4jClient,
    corpus_list: list,
    batch_id: str,
):
    """处理一批语料。"""
    logger.info(f"批次 {batch_id}: 开始处理 {len(corpus_list)} 条语料")

    # P15修复：从corpus_list提取corpus_ids
    corpus_ids = [corpus["id"] for corpus in corpus_list]

    initial_state = {
        "batch_id": batch_id,
        "corpus_list": corpus_list,
        "worker_count": 1,
        "total_count": len(corpus_list),
    }
    thread_config = {"configurable": {"thread_id": batch_id}}

    try:
        result = await workflow.ainvoke(initial_state, thread_config)

        finalizer_error = result.get("error")
        worker_results = result.get("worker_results", [])
        entities = result.get("aggregated_entities", [])
        triples = result.get("aggregated_triples", [])

        if finalizer_error:
            logger.error(f"批次 {batch_id} Finalizer 失败: {finalizer_error}")
            pg_client.update_batch_status_with_error(batch_id, "error", finalizer_error)
            # P15修复：Finalizer失败时也需要更新语料状态为error
            for corpus_id in corpus_ids:
                pg_client.mark_corpus_error(DEFAULT_TABLE, corpus_id, finalizer_error, batch_id)
            return {"success": False, "error": finalizer_error}

        per_corpus_errors = {}
        for worker_result in worker_results:
            worker_id = str(worker_result.get("worker_id", "worker_unknown"))
            for corpus_state in worker_result.get("results", []):
                corpus_error = corpus_state.get("error")
                if corpus_error:
                    per_corpus_errors[str(corpus_state.get("corpus_id"))] = str(corpus_error)
                log_corpus_node_details(batch_id, worker_id, corpus_state)

        logger.info(
            f"批次 {batch_id}: 抽取完成 - {len(entities)} 实体, {len(triples)} 三元组"
        )
        return {
            "success": True,
            "entities": len(entities),
            "triples": len(triples),
            "per_corpus_errors": per_corpus_errors,
            "worker_results": worker_results,
        }

    except Exception as error:
        logger.error(f"批次 {batch_id} 处理失败: {error}")
        pg_client.update_batch_status_with_error(batch_id, "error", str(error))
        return {"success": False, "error": str(error)}


async def main(max_total: int = DEFAULT_MAX_TOTAL, batch_size: int = DEFAULT_BATCH_SIZE):
    """主函数。"""
    logger.info("=" * 60)
    logger.info("社交媒体语料批处理")
    logger.info("=" * 60)
    logger.info(f"配置: 批大小={batch_size}, 最大处理={max_total}")

    pg_client = None
    neo4j_client = None

    try:
        pg_client, neo4j_client = init_clients()
        verify_runtime_dependencies(pg_client, neo4j_client)
        workflow, config = init_workflow()
        logger.info(
            "Batch-Only策略: "
            f"batch_llm_fallback={config.batch_llm_fallback}, "
            f"batch_skip_on_repeated_failure={config.batch_skip_on_repeated_failure}, "
            f"batch_stage_retry_attempts={config.batch_stage_retry_attempts}, "
            f"batch_qa_scaffold_retry_attempts={config.batch_qa_scaffold_retry_attempts}"
        )

        pg_client.ensure_corpus_status_columns(DEFAULT_TABLE)

        stats = pg_client.get_corpus_status_stats(DEFAULT_TABLE)
        logger.info(f"当前状态统计: {stats}")

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
            return 0

        processed_total = 0
        success_count = 0
        error_count = 0

        while processed_total < total_to_process:
            logger.info(f"进度: {processed_total}/{total_to_process}")

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

            batch_id = f"social_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
            corpus_ids = [corpus["id"] for corpus in corpus_list]

            pg_client.update_corpus_status(DEFAULT_TABLE, corpus_ids, "processing", batch_id)

            result = await process_batch(
                workflow,
                pg_client,
                neo4j_client,
                corpus_list,
                batch_id,
            )

            if result["success"]:
                per_corpus_errors = result.get("per_corpus_errors", {})
                corpus_id_map = {str(corpus_id): corpus_id for corpus_id in corpus_ids}
                processed_id_keys = {
                    corpus_key for corpus_key in corpus_id_map if corpus_key in per_corpus_errors
                }
                for worker_result in result.get("worker_results", []):
                    for corpus_state in worker_result.get("results", []):
                        corpus_key = str(corpus_state.get("corpus_id"))
                        if corpus_key in corpus_id_map:
                            processed_id_keys.add(corpus_key)

                missing_id_keys = set(corpus_id_map.keys()) - processed_id_keys
                for corpus_key in missing_id_keys:
                    per_corpus_errors[corpus_key] = "处理结果缺失（可能为worker失败）"

                error_ids = {
                    corpus_id_map[corpus_key]
                    for corpus_key in corpus_id_map
                    if corpus_key in per_corpus_errors
                }
                success_ids = [corpus_id for corpus_id in corpus_ids if corpus_id not in error_ids]

                if success_ids:
                    pg_client.update_corpus_status(
                        DEFAULT_TABLE,
                        success_ids,
                        "completed",
                        batch_id,
                    )

                if error_ids:
                    error_counter = Counter(per_corpus_errors[str(corpus_id)] for corpus_id in error_ids)
                    summary_error = "; ".join(
                        f"{message} x{count}" for message, count in error_counter.items()
                    )
                    for corpus_id in error_ids:
                        pg_client.mark_corpus_error(
                            DEFAULT_TABLE,
                            corpus_id,
                            per_corpus_errors[str(corpus_id)],
                            batch_id,
                        )
                    logger.warning(
                        f"批次 {batch_id} 存在部分失败: {len(error_ids)}/{len(corpus_list)} 条, {summary_error}"
                    )

                success_count += len(success_ids)
                error_count += len(error_ids)
            else:
                for corpus_id in corpus_ids:
                    pg_client.mark_corpus_error(
                        DEFAULT_TABLE,
                        corpus_id,
                        result["error"],
                        batch_id,
                    )
                error_count += len(corpus_list)

            processed_total += len(corpus_list)

        logger.info("=" * 60)
        logger.info("处理完成")
        logger.info(
            f"总计: {processed_total} 条, 成功: {success_count}, 失败: {error_count}"
        )

        final_stats = pg_client.get_corpus_status_stats(DEFAULT_TABLE)
        logger.info(f"最终状态统计: {final_stats}")
        return 0

    except Exception as error:
        logger.error(f"脚本运行失败: {error}")
        return 1

    finally:
        safe_close(pg_client, "PostgreSQL客户端")
        safe_close(neo4j_client, "Neo4j客户端")


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
    raise SystemExit(asyncio.run(main(max_total=args.max, batch_size=args.batch_size)))
