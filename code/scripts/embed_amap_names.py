"""
将 amap_entity_names 表中的 name 字段嵌入为词向量
使用 transformers 直接加载模型（绕过 sklearn/scipy numpy 问题）
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from loguru import logger

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
import psycopg2
from psycopg2.extras import execute_values
from transformers import AutoTokenizer, AutoModel
import torch


def load_model(model_name: str):
    """加载嵌入模型"""
    logger.info(f"加载嵌入模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def embed_batch(tokenizer, model, names: List[str]) -> np.ndarray:
    """批量生成嵌入向量"""
    encoded = tokenizer(names, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = model(**encoded)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings


def run_embedding(batch_size: int = 100, log_interval: int = 10):
    """运行嵌入流程"""
    pg_config = settings.get_postgres_config()
    model_name = settings.get_embedding_config()["model"]

    # 加载模型
    tokenizer, model = load_model(model_name)
    dim = model.config.hidden_size  # 768 for BERT-base
    logger.info(f"模型维度: {dim}")

    # 连接数据库
    conn = psycopg2.connect(
        host=pg_config['host'],
        port=pg_config['port'],
        database=pg_config['database'],
        user=pg_config['user'],
        password=pg_config['password']
    )

    try:
        # 确保 embedding 列存在
        with conn.cursor() as cur:
            cur.execute(f"ALTER TABLE amap_entity_names ADD COLUMN IF NOT EXISTS embedding VECTOR({dim})")
            conn.commit()

        # 统计总数
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM amap_entity_names WHERE embedding IS NULL")
            total = cur.fetchone()[0]

        logger.info(f"待处理记录: {total} 条")

        if total == 0:
            logger.success("所有记录已嵌入，无需处理")
            return

        processed = 0
        last_log_percent = -log_interval

        while True:
            # 获取一批数据
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, name FROM amap_entity_names
                    WHERE embedding IS NULL
                    ORDER BY id
                    LIMIT %s
                """, (batch_size,))
                records = cur.fetchall()

            if not records:
                break

            ids = [r[0] for r in records]
            names = [r[1] for r in records]

            # 生成嵌入
            embeddings = embed_batch(tokenizer, model, names)

            # 批量更新
            data = [(int(id_), emb.tolist()) for id_, emb in zip(ids, embeddings)]
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    UPDATE amap_entity_names AS g
                    SET embedding = v.embedding::vector
                    FROM (VALUES %s) AS v(id, embedding)
                    WHERE g.id = v.id
                    """,
                    data,
                    template="(%s, %s::float[])"
                )
                conn.commit()

            processed += len(records)
            current_percent = int(processed / total * 100)

            if current_percent >= last_log_percent + log_interval:
                logger.info(f"已处理: {processed}/{total} ({current_percent}%)")
                last_log_percent = current_percent

        logger.success(f"嵌入完成，共处理 {processed} 条记录")

    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="amap_entity_names 嵌入")
    parser.add_argument("--batch-size", type=int, default=100, help="批量处理大小")
    parser.add_argument("--log-interval", type=int, default=10, help="日志打印间隔百分比")

    args = parser.parse_args()
    run_embedding(batch_size=args.batch_size, log_interval=args.log_interval)