"""
P10批量处理测试 - 从PostgreSQL读取50条数据进行批量抽取
"""
import asyncio
import os
import sys
import time
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录（需要向上两级到达code目录）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import psycopg2
from langchain_openai import ChatOpenAI
from loguru import logger

from agent.agents.config import ExtractionConfig
from agent.agents.nodes import (
    create_batch_joint_extraction_node,
    create_batch_self_check_node,
    process_corpus_batch_with_llm,
)
from agent.agents.workflow import build_distributed_workflow


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

    # 读取数据，确保content_cleaned不为空
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

        # 构建语料字典
        corpus_list.append({
            "id": note_id,
            "text": content,
            "nickname": nickname,
            "publish_time": str(publish_time) if publish_time else "",
        })

    conn.close()

    logger.info(f"从数据库读取 {len(corpus_list)} 条语料")
    return corpus_list


async def test_batch_extraction(corpus_list: list, batch_llm_size: int = 5):
    """测试批量抽取"""

    # 创建LLM
    llm = ChatOpenAI(
        model='deepseek-chat',
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_API_BASE_URL'),
        temperature=0,
    )

    # 配置
    config = ExtractionConfig(
        batch_llm_size=batch_llm_size,
        enable_batch_llm=True,
        batch_llm_fallback=True,
    )

    logger.info(f"批量处理配置: batch_llm_size={batch_llm_size}, 总语料={len(corpus_list)}")

    # 创建批量节点
    batch_joint_node = create_batch_joint_extraction_node(llm, batch_llm_size)
    batch_self_check_node = create_batch_self_check_node(llm)

    # 分批处理
    total_entities = 0
    total_triples = 0
    all_results = {}
    all_aliases = []

    start_time = time.time()
    llm_call_count = 0

    for i in range(0, len(corpus_list), batch_llm_size):
        batch_corpus = corpus_list[i:i + batch_llm_size]
        batch_num = i // batch_llm_size + 1
        total_batches = (len(corpus_list) - 1) // batch_llm_size + 1

        logger.info(f"[Batch {batch_num}/{total_batches}] 处理 {len(batch_corpus)} 条语料")

        # 执行批量抽取
        def dummy_writer(event):
            pass

        extraction_result = await batch_joint_node(batch_corpus, dummy_writer)
        llm_call_count += 1

        if extraction_result.get("needs_fallback"):
            logger.warning(f"[Batch {batch_num}] 抽取失败，退化为单条处理")
            # Fallback处理（这里简化为跳过）
            continue

        # 统计结果
        batch_results = extraction_result.get("batch_results", {})
        for corpus_id, data in batch_results.items():
            entities = data.get("entities", {})
            entity_count = sum(len(v) for v in entities.values())
            triple_count = len(data.get("triples", []))

            total_entities += entity_count
            total_triples += triple_count
            all_results[corpus_id] = data

        # 收集跨语料别名
        aliases = extraction_result.get("cross_corpus_aliases", [])
        all_aliases.extend(aliases)

    elapsed_time = time.time() - start_time

    # 输出结果统计
    print("\n" + "=" * 60)
    print("批量处理测试结果")
    print("=" * 60)
    print(f"总语料数: {len(corpus_list)}")
    print(f"batch_llm_size: {batch_llm_size}")
    print(f"LLM调用次数: {llm_call_count}")
    print(f"处理时间: {elapsed_time:.2f}秒")
    print(f"平均每批次耗时: {elapsed_time/llm_call_count:.2f}秒")
    print("-" * 40)
    print(f"成功处理语料: {len(all_results)} 条")
    print(f"总实体数: {total_entities}")
    print(f"总三元组数: {total_triples}")
    print(f"跨语料别名发现: {len(all_aliases)} 个")
    print("-" * 40)

    # 输出部分详细结果
    print("\n部分语料抽取结果 (前5条):")
    for i, (corpus_id, data) in enumerate(list(all_results.items())[:5], 1):
        entities = data.get("entities", {})
        triples = data.get("triples", [])
        confidence = data.get("confidence", "medium")

        # 找到原语料
        original = None
        for c in corpus_list:
            if c["id"] == corpus_id:
                original = c
                break

        if original:
            # 清理特殊字符
            text_preview = original["text"][:50].replace('\u200b', '').replace('\x00', '') + "..."
            print(f"\n[{i}] ID: {corpus_id}")
            print(f"    原文: {text_preview}")
            print(f"    置信度: {confidence}")

            # 实体
            entity_strs = []
            for etype, names in entities.items():
                if names:
                    entity_strs.append(f"{etype}: {names}")
            print(f"    实体: {', '.join(entity_strs)}")

            # 三元组
            if triples:
                print(f"    三元组 ({len(triples)}个):")
                for t in triples[:3]:
                    print(f"      - <{t.get('head')}, {t.get('relation')}, {t.get('tail')}>")

    # 输出跨语料别名
    if all_aliases:
        print("\n跨语料别名发现:")
        for alias in all_aliases[:5]:
            raw = alias.get("raw", "")
            canonical = alias.get("canonical", "")
            print(f"  - '{raw}' → '{canonical}'")

    # 对比：单条处理需要的LLM调用次数
    single_llm_calls = len(corpus_list) * 4  # NER + RE + Eval + Label
    batch_llm_calls = llm_call_count
    saved_calls = single_llm_calls - batch_llm_calls
    saved_percent = (saved_calls / single_llm_calls) * 100 if single_llm_calls > 0 else 0

    print("\n" + "-" * 40)
    print("效率对比:")
    print(f"  单条处理LLM调用: {single_llm_calls} 次")
    print(f"  批量处理LLM调用: {batch_llm_calls} 次")
    print(f"  节省调用次数: {saved_calls} 次 ({saved_percent:.1f}%)")
    print("=" * 60)


async def main():
    """主函数"""
    # 从数据库读取50条语料
    corpus_list = fetch_corpus_from_db(limit=50)

    if not corpus_list:
        logger.error("未能读取语料数据")
        return

    # 测试批量处理，batch_llm_size=5
    await test_batch_extraction(corpus_list, batch_llm_size=5)


if __name__ == "__main__":
    asyncio.run(main())