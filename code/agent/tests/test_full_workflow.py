"""
完整工作流测试 - Filter → Normalize → QA_Scaffold → Joint_NER_RE → Eval → Label
处理50条语料，记录时间和结果
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import psycopg2
from langchain_openai import ChatOpenAI
from loguru import logger

from agent.agents.config import ExtractionConfig
from agent.agents.workflow import process_batch, build_distributed_workflow


def fetch_corpus_from_db(limit: int = 50):
    """从数据库读取语料"""
    conn = psycopg2.connect(
        host=os.getenv('PG_HOST', 'localhost'),
        port=int(os.getenv('PG_PORT', '5432')),
        database=os.getenv('PG_DATABASE', 'bishe'),
        user=os.getenv('PG_USER', 'postgres'),
        password=os.getenv('PG_PASSWORD')
    )

    cur = conn.cursor()

    # 读取数据
    cur.execute("""
        SELECT note_id, nickname, content_cleaned, publish_time
        FROM social_media_notes_sampled
        WHERE content_cleaned IS NOT NULL
        AND content_cleaned != ''
        AND LENGTH(content_cleaned) > 20
        ORDER BY publish_time DESC
        LIMIT %s
    """, (limit,))

    rows = cur.fetchall()

    corpus_list = []
    for row in rows:
        note_id = row[0] or f"unknown_{len(corpus_list)}"
        nickname = row[1] or ""
        content = row[2] or ""
        publish_time = row[3]

        # 清理特殊字符
        content = content.replace('\u200b', '').replace('\x00', '')

        corpus_list.append({
            "id": note_id,
            "text": content,
            "nickname": nickname,
            "publish_time": str(publish_time) if publish_time else "",
        })

    conn.close()

    logger.info(f"从数据库读取 {len(corpus_list)} 条语料")
    return corpus_list


async def test_full_workflow(corpus_list: list, output_file: str):
    """测试完整工作流"""

    # 创建LLM
    llm = ChatOpenAI(
        model='deepseek-chat',
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_API_BASE_URL'),
        temperature=0,
    )

    # 配置完整工作流
    config = ExtractionConfig(
        # 批量处理 - 禁用以使用完整流程（包括Label属性标注）
        batch_llm_size=5,
        enable_batch_llm=False,  # 禁用批量模式，使用完整流程
        batch_llm_fallback=True,
        # 完整流程节点
        enable_filter=True,
        enable_normalize=True,
        enable_qa_scaffold=True,
        use_joint_extraction=True,
        use_simplified_eval=True,
        # Worker并行
        max_workers=4,
        max_concurrent_corpus=10,
        corpus_per_worker=12,  # 50条 / 4个worker ≈ 12条/worker
    )

    logger.info(f"完整工作流配置:")
    logger.info(f"  - Filter: 启用")
    logger.info(f"  - Normalize: 启用")
    logger.info(f"  - QA_Scaffold: 启用")
    logger.info(f"  - Joint_NER_RE: 启用")
    logger.info(f"  - Eval: 启用 (简化模式)")
    logger.info(f"  - Label: 启用")
    logger.info(f"  - max_workers: {config.max_workers}")
    logger.info(f"  - batch_llm_size: {config.batch_llm_size}")
    logger.info(f"  - 总语料: {len(corpus_list)}")

    start_time = time.time()

    # 使用分布式工作流处理
    result = await process_batch(llm, corpus_list, config)

    elapsed_time = time.time() - start_time

    # 构建输出结果
    output = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "corpus_count": len(corpus_list),
            "config": {
                "batch_llm_size": config.batch_llm_size,
                "enable_batch_llm": config.enable_batch_llm,
                "enable_filter": config.enable_filter,
                "enable_normalize": config.enable_normalize,
                "enable_qa_scaffold": config.enable_qa_scaffold,
                "use_joint_extraction": config.use_joint_extraction,
                "max_workers": config.max_workers,
                "max_concurrent_corpus": config.max_concurrent_corpus,
            }
        },
        "timing": {
            "total_seconds": elapsed_time,
            "total_minutes": elapsed_time / 60,
            "avg_per_corpus": elapsed_time / len(corpus_list),
        },
        "statistics": {
            "total_entities": len(result.get("aggregated_entities", [])),
            "total_triples": len(result.get("aggregated_triples", [])),
            "worker_count": result.get("worker_count", 0),
            "neo4j_stats": result.get("neo4j_stats", {}),
        },
        "worker_results": [],
        "aggregated_entities": result.get("aggregated_entities", []),
        "aggregated_triples": result.get("aggregated_triples", []),
        "entity_aliases": result.get("entity_aliases", {}),
    }

    # 收集每个Worker的详细结果
    for worker_result in result.get("worker_results", []):
        worker_id = worker_result.get("worker_id", "unknown")
        processing_time = worker_result.get("processing_time", 0)
        corpus_ids = worker_result.get("corpus_ids", [])
        results = worker_result.get("results", [])

        worker_output = {
            "worker_id": worker_id,
            "processing_time": processing_time,
            "corpus_count": len(corpus_ids),
            "corpus_ids": corpus_ids,
            "corpus_results": []
        }

        # 每条语料的详细结果
        for corpus_result in results:
            corpus_id = corpus_result.get("corpus_id", "unknown")
            entities = corpus_result.get("entities", {})
            triples = corpus_result.get("triples", [])
            entity_attrs = corpus_result.get("entity_attrs", {})
            relation_attrs = corpus_result.get("relation_attrs", {})
            eval_passed = corpus_result.get("eval_passed", False)
            confidence = corpus_result.get("verification_confidence", "medium")
            error = corpus_result.get("error")

            # 找到原语料
            original_text = ""
            for c in corpus_list:
                if c["id"] == corpus_id:
                    original_text = c["text"][:100] + "..." if len(c["text"]) > 100 else c["text"]
                    break

            corpus_output = {
                "corpus_id": corpus_id,
                "original_text": original_text,
                "entities": entities,
                "triples": triples,
                "entity_attrs": entity_attrs,
                "relation_attrs": relation_attrs,
                "eval_passed": eval_passed,
                "confidence": confidence,
                "error": error,
            }

            worker_output["corpus_results"].append(corpus_output)

        output["worker_results"].append(worker_output)

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print("\n" + "=" * 70)
    print("完整工作流测试结果")
    print("=" * 70)
    print(f"总语料数: {len(corpus_list)}")
    print(f"Worker数: {result.get('worker_count', 0)}")
    print("-" * 50)
    print(f"处理时间: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    print(f"平均每条语料: {elapsed_time/len(corpus_list):.2f}秒")
    print("-" * 50)
    print(f"聚合实体数: {len(result.get('aggregated_entities', []))}")
    print(f"聚合三元组数: {len(result.get('aggregated_triples', []))}")
    print(f"实体别名映射: {len(result.get('entity_aliases', {}))} 个")
    print("-" * 50)

    # 打印部分结果
    print("\n前5条语料抽取结果:")
    all_results = []
    for wr in result.get("worker_results", []):
        all_results.extend(wr.get("results", []))

    for i, corpus_result in enumerate(all_results[:5], 1):
        corpus_id = corpus_result.get("corpus_id", "unknown")
        entities = corpus_result.get("entities", {})
        triples = corpus_result.get("triples", [])
        entity_attrs = corpus_result.get("entity_attrs", {})
        relation_attrs = corpus_result.get("relation_attrs", {})
        confidence = corpus_result.get("verification_confidence", "medium")

        # 找原文本
        original_text = ""
        for c in corpus_list:
            if c["id"] == corpus_id:
                original_text = c["text"][:50] + "..."
                break

        print(f"\n[{i}] ID: {corpus_id}")
        print(f"    原文: {original_text}")
        print(f"    置信度: {confidence}")

        entity_strs = []
        for etype, names in entities.items():
            if names:
                entity_strs.append(f"{etype}: {names[:3]}...")
        print(f"    实体: {', '.join(entity_strs)}")

        # 显示实体属性（如果有）
        if entity_attrs:
            print(f"    实体属性 ({len(entity_attrs)}个):")
            for name, attrs in list(entity_attrs.items())[:2]:
                attr_str = ", ".join([f"{k}:{v}" for k, v in attrs.items() if v])
                if attr_str:
                    print(f"      - {name}: {attr_str}")

        if triples:
            print(f"    三元组 ({len(triples)}个):")
            for t in triples[:3]:
                attrs = t.get('attributes', {})
                attr_str = ""
                if attrs:
                    attr_str = " | 属性: " + ", ".join([f"{k}={v}" for k, v in attrs.items() if v])
                print(f"      - <{t.get('head')}, {t.get('relation')}, {t.get('tail')}>{attr_str}")

    print("\n" + "-" * 50)
    print(f"结果已保存到: {output_file}")
    print("=" * 70)

    return output


async def main():
    """主函数"""
    # 从数据库读取50条语料
    corpus_list = fetch_corpus_from_db(limit=50)

    if not corpus_list:
        logger.error("未能读取语料数据")
        return

    # 输出文件路径
    output_file = os.path.join(project_root, "agent", "tests", "full_workflow_result.json")

    # 测试完整工作流
    await test_full_workflow(corpus_list, output_file)


if __name__ == "__main__":
    asyncio.run(main())