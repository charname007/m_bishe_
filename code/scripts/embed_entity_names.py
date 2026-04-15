"""
将 geo_entity_names 表中的 name 字段嵌入为词向量
使用 sentence-transformers 本地模型
支持 ModelScope 镜像源加速下载

依赖关系：
- 与 extract_entity_names.py 共享 geo_entity_names 表
- 维度配置统一从 settings.EMBEDDING_DIM 获取

表结构：
- geo_entity_names 表包含：id, entity_id, name, type, embedding, created_at
- embedding 列为 VECTOR(dim) 类型，维度由模型决定
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from loguru import logger

# 设置 ModelScope 镜像源（国内加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
from kg.postgres_client import PostgresClient
from psycopg2.extras import execute_values

# 合理的维度范围
MIN_EMBEDDING_DIM = 64
MAX_EMBEDDING_DIM = 4096


def get_embedding_model(model_name: str, use_modelscope: bool = False):
    """
    加载嵌入模型

    Args:
        model_name: 模型名称
        use_modelscope: 是否使用 ModelScope 模型

    Returns:
        SentenceTransformer 模型实例
    """
    if use_modelscope:
        from modelscope import snapshot_download
        model_dir = snapshot_download(model_name)
        logger.info(f"从 ModelScope 加载模型: {model_name} -> {model_dir}")
    else:
        model_dir = model_name
        logger.info(f"加载嵌入模型: {model_name} (镜像源: hf-mirror.com)")

    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_dir)


def validate_embedding_dimension(dim: int) -> int:
    """
    验证嵌入维度是否合理

    Args:
        dim: 嵌入维度

    Returns:
        验证后的维度

    Raises:
        ValueError: 维度不在合理范围内
    """
    if not (MIN_EMBEDDING_DIM <= dim <= MAX_EMBEDDING_DIM):
        raise ValueError(
            f"嵌入维度 {dim} 不在合理范围内 "
            f"[{MIN_EMBEDDING_DIM}, {MAX_EMBEDDING_DIM}]"
        )
    return dim


def modify_embedding_column(pg_client: PostgresClient, new_dim: int):
    """
    修改 embedding 列的向量维度

    注意：pgvector 的 vector 类型不支持直接 ALTER，需要重建列
    会删除现有的 embedding 数据

    Args:
        pg_client: PostgreSQL 客户端
        new_dim: 新的向量维度
    """
    # 验证维度范围（防止 SQL 注入）
    validated_dim = validate_embedding_dimension(new_dim)

    with pg_client.conn.cursor() as cur:
        # 删除旧列并重建（使用验证后的维度）
        cur.execute("ALTER TABLE geo_entity_names DROP COLUMN IF EXISTS embedding")
        cur.execute(f"ALTER TABLE geo_entity_names ADD COLUMN embedding VECTOR({validated_dim})")

        # 创建向量索引 (使用 HNSW 索引，适合相似度搜索)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_geo_entity_embedding
            ON geo_entity_names USING hnsw (embedding vector_cosine_ops)
        """)

        pg_client.conn.commit()
        logger.info(f"已修改 embedding 列维度为 {validated_dim}")


def fetch_names_without_embedding(
    pg_client: PostgresClient,
    batch_size: int = 100
) -> List[Tuple[int, str]]:
    """
    获取没有 embedding 的记录

    Args:
        pg_client: PostgreSQL 客户端
        batch_size: 每批获取数量

    Returns:
        [(id, name), ...]
    """
    with pg_client.conn.cursor() as cur:
        cur.execute("""
            SELECT id, name FROM geo_entity_names
            WHERE embedding IS NULL
            ORDER BY id
            LIMIT %s
        """, (batch_size,))
        return cur.fetchall()


def count_names_without_embedding(pg_client: PostgresClient) -> int:
    """
    统计未嵌入的记录总数

    Args:
        pg_client: PostgreSQL 客户端

    Returns:
        未嵌入记录数量
    """
    with pg_client.conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM geo_entity_names WHERE embedding IS NULL")
        return cur.fetchone()[0]


def embed_batch(model, names: List[str]) -> np.ndarray:
    """
    批量生成嵌入向量

    Args:
        model: SentenceTransformer 模型
        names: 名称列表

    Returns:
        numpy array of embeddings, shape (len(names), dim)
    """
    return model.encode(names, show_progress_bar=False, convert_to_numpy=True)


def update_embeddings(pg_client: PostgresClient, ids: List[int], embeddings: np.ndarray):
    """
    批量更新 embedding 字段

    Args:
        pg_client: PostgreSQL 客户端
        ids: 记录 ID 列表
        embeddings: 嵌入向量数组
    """
    # 将 numpy 数组转换为 Python list，格式为 (id, embedding_vector)
    data = [(int(id_), emb.tolist()) for id_, emb in zip(ids, embeddings)]

    with pg_client.conn.cursor() as cur:
        # 使用 execute_values 批量更新
        # 注意：PostgreSQL 不支持在 VALUES 子句中定义列类型
        # 使用更简洁的SQL语句
        execute_values(
            cur,
            """
            UPDATE geo_entity_names AS g
            SET embedding = v.embedding::vector
            FROM (VALUES %s) AS v(id, embedding)
            WHERE g.id = v.id
            """,
            data,
            template="(%s, %s::float[])"
        )
        pg_client.conn.commit()


def run_embedding_process(
    model_name: Optional[str] = None,
    batch_size: int = 100,
    modify_column: bool = True,
    use_modelscope: bool = False,
    log_interval: int = 10  # 每 N% 打印一次日志
):
    """
    运行嵌入流程

    Args:
        model_name: 嵌入模型名称，默认从 settings 获取
        batch_size: 每批处理数量
        modify_column: 是否修改表结构（首次运行需要）
        use_modelscope: 是否使用 ModelScope 模型
        log_interval: 日志打印间隔百分比
    """
    # ModelScope 推荐模型
    modelscope_model = "damo/nlp_gte_sentence-embedding_chinese-base"

    # 确定使用的模型
    if use_modelscope:
        model_name = modelscope_model
    elif model_name is None:
        model_name = settings.get_embedding_config()["model"]

    # 加载模型
    model = get_embedding_model(model_name, use_modelscope=use_modelscope)
    actual_dim = model.get_sentence_embedding_dimension()
    logger.info(f"模型实际维度: {actual_dim}")

    # 连接数据库，使用 try-finally 确保连接关闭
    pg_config = settings.get_postgres_config()
    pg_client = PostgresClient(**pg_config)

    try:
        # 首次运行需要修改表结构
        if modify_column:
            modify_embedding_column(pg_client, actual_dim)

        # 统计总数（每次循环重新获取，避免并发问题）
        total = count_names_without_embedding(pg_client)
        logger.info(f"待处理记录: {total} 条")

        if total == 0:
            logger.success("所有记录已嵌入，无需处理")
            return

        # 记录上次打印日志的百分比
        last_log_percent = -log_interval

        # 分批处理
        processed = 0
        while True:
            # 每次循环重新统计，避免并发竞态条件
            remaining = count_names_without_embedding(pg_client)
            if remaining == 0:
                break

            # 获取一批数据
            records = fetch_names_without_embedding(pg_client, batch_size)
            if not records:
                break

            ids = [r[0] for r in records]
            names = [r[1] for r in records]

            # 生成嵌入
            embeddings = embed_batch(model, names)

            # 更新数据库
            update_embeddings(pg_client, ids, embeddings)

            processed += len(records)
            current_percent = int(processed / total * 100)

            # 控制日志频率，每 log_interval% 打印一次
            if current_percent >= last_log_percent + log_interval:
                logger.info(f"已处理: {processed}/{total} ({current_percent}%)")
                last_log_percent = current_percent

        logger.success(f"嵌入完成，共处理 {processed} 条记录")

    except Exception as e:
        logger.error(f"嵌入过程发生错误: {e}")
        raise
    finally:
        pg_client.close()


def search_similar(
    pg_client: PostgresClient,
    query: str,
    model,
    limit: int = 10
) -> List[Tuple[str, str, float]]:
    """
    语义相似度搜索

    Args:
        pg_client: PostgreSQL 客户端
        query: 查询文本
        model: 嵌入模型
        limit: 返回数量

    Returns:
        [(name, type, similarity), ...] 相似度范围 [0, 1]
    """
    # 生成查询向量
    query_embedding = model.encode(query, convert_to_numpy=True)

    with pg_client.conn.cursor() as cur:
        # 使用余弦距离 <=> 搜索，返回 1 - distance 作为相似度
        cur.execute("""
            SELECT name, type, 1 - (embedding <=> %s::vector) as similarity
            FROM geo_entity_names
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding.tolist(), query_embedding.tolist(), limit))

        return cur.fetchall()


def demo_search(query: str = "北京路步行街", use_modelscope: bool = False):
    """
    演示相似度搜索

    Args:
        query: 查询文本
        use_modelscope: 是否使用 ModelScope 模型
    """
    modelscope_model = "damo/nlp_gte_sentence-embedding_chinese-base"
    embedding_config = settings.get_embedding_config()
    model_name = modelscope_model if use_modelscope else embedding_config["model"]
    model = get_embedding_model(model_name, use_modelscope=use_modelscope)

    pg_config = settings.get_postgres_config()
    pg_client = PostgresClient(**pg_config)

    try:
        results = search_similar(pg_client, query, model)

        logger.info(f"查询: '{query}'")
        for name, type_, sim in results:
            logger.info(f"  {name} ({type_}) - 相似度: {sim:.3f}")
    finally:
        pg_client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="地理实体名称嵌入")
    parser.add_argument("--model", default=None, help="嵌入模型名称")
    parser.add_argument("--batch-size", type=int, default=100, help="批量处理大小")
    parser.add_argument("--modify-column", action="store_true", help="修改表结构")
    parser.add_argument("--search", default=None, help="演示搜索查询")
    parser.add_argument("--modelscope", action="store_true", help="使用 ModelScope 模型源")
    parser.add_argument("--log-interval", type=int, default=10, help="日志打印间隔百分比")

    args = parser.parse_args()

    if args.search:
        demo_search(args.search, use_modelscope=args.modelscope)
    else:
        run_embedding_process(
            model_name=args.model,
            batch_size=args.batch_size,
            modify_column=args.modify_column,
            use_modelscope=args.modelscope,
            log_interval=args.log_interval
        )