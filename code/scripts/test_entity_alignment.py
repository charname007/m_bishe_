"""
实体对齐测试：geo_entity_names (Shapefile POI) vs amap_poi_wgs84 (高德POI)
策略：坐标距离阈值 + 语义相似度阈值
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import math
from loguru import logger

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
import psycopg2
from psycopg2.extras import execute_values


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点间距离（米）"""
    R = 6371000  # 地球半径（米）
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def test_alignment(
    distance_threshold: float = 200,  # 米
    similarity_threshold: float = 0.40,  # 余弦相似度（text2vec模型实际阈值较低）
    limit: int = 100
):
    """测试实体对齐"""
    pg_config = settings.get_postgres_config()

    conn = psycopg2.connect(
        host=pg_config['host'], port=pg_config['port'],
        database=pg_config['database'], user=pg_config['user'], password=pg_config['password']
    )

    try:
        # 1. 统计基本信息
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM geo_entity_names WHERE type = 'poi' AND embedding IS NOT NULL")
            geo_poi_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM amap_poi_wgs84 WHERE embedding IS NOT NULL")
            amap_count = cur.fetchone()[0]

            logger.info(f"geo_entity_names POI: {geo_poi_count}")
            logger.info(f"amap_poi_wgs84: {amap_count}")

        if amap_count == 0:
            logger.warning("amap_poi_wgs84 嵌入数据为空，请先运行 embed_amap_poi_wgs84.py")
            return

        # 2. 对齐测试 - 使用嵌入相似度 + 坐标距离
        logger.info(f"开始对齐测试 (距离阈值: {distance_threshold}m, 相似度阈值: {similarity_threshold})")

        # 获取 geo_entity_names 的 POI 样本
        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id, name, longitude, latitude, embedding
                FROM geo_entity_names
                WHERE type = 'poi'
                  AND embedding IS NOT NULL
                  AND longitude IS NOT NULL
                  AND latitude IS NOT NULL
                ORDER BY id
                LIMIT %s
            """, (limit,))
            geo_pois = cur.fetchall()

        logger.info(f"测试样本: {len(geo_pois)} 个 geo POI")

        aligned_count = 0
        alignment_results = []

        for geo_id, geo_name, geo_lon, geo_lat, geo_emb in geo_pois:
            # 在 amap_poi_wgs84 中查找候选匹配
            # 先用嵌入相似度筛选，再用坐标距离验证
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT entity_id, name, longitude, latitude,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM amap_poi_wgs84
                    WHERE embedding IS NOT NULL
                      AND longitude BETWEEN %s - 0.001 AND %s + 0.001
                      AND latitude BETWEEN %s - 0.001 AND %s + 0.001
                    ORDER BY embedding <=> %s::vector
                    LIMIT 5
                """, (
                    geo_emb, geo_lon, geo_lon, geo_lat, geo_lat, geo_emb
                ))
                candidates = cur.fetchall()

            for amap_id, amap_name, amap_lon, amap_lat, similarity in candidates:
                if similarity < similarity_threshold:
                    continue

                distance = haversine_distance(geo_lat, geo_lon, amap_lat, amap_lon)

                if distance <= distance_threshold:
                    aligned_count += 1
                    alignment_results.append({
                        'geo_id': geo_id,
                        'geo_name': geo_name,
                        'amap_id': amap_id,
                        'amap_name': amap_name,
                        'distance': distance,
                        'similarity': similarity
                    })
                    break  # 只取第一个匹配

        logger.info(f"对齐结果: {aligned_count}/{len(geo_pois)} ({aligned_count/len(geo_pois)*100:.1f}%)")

        # 显示部分对齐结果
        for r in alignment_results[:10]:
            logger.info(f"  {r['geo_name']} <-> {r['amap_name']} | d={r['distance']:.0f}m, sim={r['similarity']:.3f}")

        # 3. 全量对齐统计（使用批量查询）
        logger.info("执行全量对齐统计...")

        with conn.cursor() as cur:
            # 使用更高效的批量对齐查询
            cur.execute("""
                WITH geo_poi AS (
                    SELECT entity_id, name, longitude, latitude, embedding
                    FROM geo_entity_names
                    WHERE type = 'poi' AND embedding IS NOT NULL AND longitude IS NOT NULL
                )
                SELECT
                    g.entity_id, g.name, g.longitude, g.latitude,
                    a.entity_id, a.name, a.longitude, a.latitude,
                    1 - (g.embedding <=> a.embedding) as similarity
                FROM geo_poi g
                CROSS JOIN LATERAL (
                    SELECT entity_id, name, longitude, latitude, embedding
                    FROM amap_poi_wgs84
                    WHERE embedding IS NOT NULL
                      AND longitude BETWEEN g.longitude - 0.001 AND g.longitude + 0.001
                      AND latitude BETWEEN g.latitude - 0.001 AND g.latitude + 0.001
                    ORDER BY embedding <=> g.embedding
                    LIMIT 1
                ) a
                WHERE 1 - (g.embedding <=> a.embedding) >= %s
            """, (similarity_threshold,))

            full_results = cur.fetchall()

        logger.success(f"全量对齐: {len(full_results)} 个匹配 (相似度 >= {similarity_threshold})")

        # 保存对齐结果到新表
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS entity_alignment (
                    id SERIAL PRIMARY KEY,
                    geo_entity_id VARCHAR(100),
                    geo_name VARCHAR(200),
                    geo_longitude DOUBLE PRECISION,
                    geo_latitude DOUBLE PRECISION,
                    amap_entity_id VARCHAR(100),
                    amap_name VARCHAR(200),
                    amap_longitude DOUBLE PRECISION,
                    amap_latitude DOUBLE PRECISION,
                    similarity DOUBLE PRECISION,
                    distance_m DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()

            # 插入对齐结果
            if full_results:
                data = []
                for r in full_results:
                    geo_id, geo_name, geo_lon, geo_lat, amap_id, amap_name, amap_lon, amap_lat, sim = r
                    dist = haversine_distance(geo_lat, geo_lon, amap_lat, amap_lon)
                    data.append((geo_id, geo_name, geo_lon, geo_lat, amap_id, amap_name, amap_lon, amap_lat, sim, dist))

                execute_values(
                    cur,
                    """
                    INSERT INTO entity_alignment
                    (geo_entity_id, geo_name, geo_longitude, geo_latitude,
                     amap_entity_id, amap_name, amap_longitude, amap_latitude, similarity, distance_m)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                    """,
                    data
                )
                conn.commit()

                logger.success(f"保存 {len(data)} 条对齐结果到 entity_alignment 表")

    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="实体对齐测试")
    parser.add_argument("--distance", type=float, default=100, help="距离阈值（米）")
    parser.add_argument("--similarity", type=float, default=0.85, help="相似度阈值")
    parser.add_argument("--limit", type=int, default=100, help="测试样本数量")

    args = parser.parse_args()
    test_alignment(
        distance_threshold=args.distance,
        similarity_threshold=args.similarity,
        limit=args.limit
    )