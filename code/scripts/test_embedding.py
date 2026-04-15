"""
测试嵌入脚本是否能正常运行
对 amap_entity_names 表进行少量测试
使用 transformers 直接加载模型，绕过 sklearn/scipy numpy 问题
"""
import os
import sys
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
import psycopg2
from psycopg2.extras import execute_values
from loguru import logger
import numpy as np

def test():
    # 获取数据库配置
    pg_config = settings.get_postgres_config()

    # 1. 测试数据库连接
    logger.info("测试数据库连接...")
    conn = psycopg2.connect(
        host=pg_config['host'],
        port=pg_config['port'],
        database=pg_config['database'],
        user=pg_config['user'],
        password=pg_config['password']
    )

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM amap_entity_names")
        count = cur.fetchone()[0]
        logger.info(f"amap_entity_names 表记录数: {count}")

        cur.execute("SELECT COUNT(*) FROM amap_entity_names WHERE embedding IS NULL")
        no_emb = cur.fetchone()[0]
        logger.info(f"未嵌入记录数: {no_emb}")

    conn.close()
    logger.success("数据库连接测试通过")

    # 2. 测试模型加载（使用 transformers 直接加载）
    logger.info("测试模型加载...")
    from transformers import AutoTokenizer, AutoModel
    import torch

    model_name = settings.get_embedding_config()["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    logger.success("模型加载测试通过")

    # 3. 测试嵌入生成
    logger.info("测试嵌入生成...")
    test_names = ["武汉大学", "华中科技大学", "光谷广场"]

    # 编码文本
    encoded = tokenizer(test_names, padding=True, truncation=True, return_tensors="pt")

    # 获取嵌入
    with torch.no_grad():
        outputs = model(**encoded)
        # 使用 mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

    dim = embeddings.shape[1]
    logger.info(f"生成嵌入向量: shape={embeddings.shape}, 维度={dim}")
    logger.success("嵌入生成测试通过")

    # 4. 测试数据库写入
    logger.info("测试数据库写入...")
    conn = psycopg2.connect(
        host=pg_config['host'],
        port=pg_config['port'],
        database=pg_config['database'],
        user=pg_config['user'],
        password=pg_config['password']
    )

    with conn.cursor() as cur:
        # 确保 embedding 列存在
        cur.execute(f"ALTER TABLE amap_entity_names ADD COLUMN IF NOT EXISTS embedding VECTOR({dim})")
        conn.commit()

        # 获取一条测试记录
        cur.execute("SELECT id, name FROM amap_entity_names WHERE embedding IS NULL LIMIT 1")
        record = cur.fetchone()

        if record:
            id_, name = record
            logger.info(f"测试记录: id={id_}, name={name}")

            # 生成嵌入
            encoded = tokenizer(name, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model(**encoded)
                emb = outputs.last_hidden_state.mean(dim=1).numpy()[0]

            # 更新
            execute_values(
                cur,
                """
                UPDATE amap_entity_names AS g
                SET embedding = v.embedding::vector
                FROM (VALUES %s) AS v(id, embedding)
                WHERE g.id = v.id
                """,
                [(id_, emb.tolist())],
                template="(%s, %s::float[])"
            )
            conn.commit()
            logger.success(f"成功更新记录 id={id_}")
        else:
            logger.warning("没有未嵌入的记录，跳过写入测试")

    conn.close()
    logger.success("所有测试通过！")

if __name__ == "__main__":
    test()