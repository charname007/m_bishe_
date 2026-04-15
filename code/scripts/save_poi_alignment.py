"""
实体对齐结果保存
将 geo_entity_names (Shapefile POI) 与 amap_poi_wgs84 (高德POI) 的匹配结果保存到表中
"""
import os
import sys
from pathlib import Path
import math
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
import psycopg2
from psycopg2.extras import execute_values
from loguru import logger


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点间距离（米）"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def save_alignment(
    similarity_threshold: float = 0.30,
    spatial_threshold_deg: float = 0.002  # 约200m
):
    """保存对齐结果到表"""
    pg_config = settings.get_postgres_config()
    conn = psycopg2.connect(
        host=pg_config['host'], port=pg_config['port'],
        database=pg_config['database'], user=pg_config['user'], password=pg_config['password']
    )

    try:
        with conn.cursor() as cur:
            # 创建对齐结果表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS poi_alignment (
                    id SERIAL PRIMARY KEY,
                    geo_entity_id VARCHAR(100) NOT NULL,
                    geo_name VARCHAR(200),
                    geo_type VARCHAR(50),
                    geo_longitude DOUBLE PRECISION,
                    geo_latitude DOUBLE PRECISION,
                    amap_entity_id VARCHAR(100) NOT NULL,
                    amap_name VARCHAR(200),
                    amap_type VARCHAR(50),
                    amap_longitude DOUBLE PRECISION,
                    amap_latitude DOUBLE PRECISION,
                    similarity DOUBLE PRECISION,
                    distance_m DOUBLE PRECISION,
                    spatial_threshold_m DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(geo_entity_id, amap_entity_id)
                )
            """)

            # 创建索引
            cur.execute("CREATE INDEX IF NOT EXISTS idx_poi_alignment_geo ON poi_alignment(geo_entity_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_poi_alignment_amap ON poi_alignment(amap_entity_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_poi_alignment_sim ON poi_alignment(similarity)")
            conn.commit()
            logger.info("表 poi_alignment 已创建")

            # 清空旧数据（可选）
            cur.execute("TRUNCATE TABLE poi_alignment")
            conn.commit()

            # 执行对齐查询
            logger.info(f"执行对齐查询 (相似度阈值: {similarity_threshold}, 空间阈值: {spatial_threshold_deg*111:.0f}m)")

            cur.execute("""
                SELECT
                    g.entity_id, g.name, g.type, g.longitude, g.latitude,
                    a.entity_id, a.name, a.type, a.longitude, a.latitude,
                    1 - (g.embedding <=> a.embedding) as similarity
                FROM geo_entity_names g
                CROSS JOIN LATERAL (
                    SELECT entity_id, name, type, longitude, latitude, embedding
                    FROM amap_poi_wgs84
                    WHERE longitude BETWEEN g.longitude - %s AND g.longitude + %s
                    AND latitude BETWEEN g.latitude - %s AND g.latitude + %s
                    AND embedding IS NOT NULL
                    ORDER BY embedding <=> g.embedding
                    LIMIT 1
                ) a
                WHERE g.type = 'poi'
                AND g.embedding IS NOT NULL
                AND g.longitude IS NOT NULL
                AND 1 - (g.embedding <=> a.embedding) >= %s
            """, (spatial_threshold_deg, spatial_threshold_deg, spatial_threshold_deg, spatial_threshold_deg, similarity_threshold))

            matches = cur.fetchall()
            logger.info(f"找到 {len(matches)} 个匹配")

            # 计算距离并准备插入数据
            data = []
            for row in matches:
                geo_id, geo_name, geo_type, geo_lon, geo_lat, \
                amap_id, amap_name, amap_type, amap_lon, amap_lat, sim = row

                dist_m = haversine_distance(geo_lat, geo_lon, amap_lat, amap_lon)
                spatial_threshold_m = spatial_threshold_deg * 111  # 近似转换

                data.append((
                    geo_id, geo_name, geo_type, geo_lon, geo_lat,
                    amap_id, amap_name, amap_type, amap_lon, amap_lat,
                    sim, dist_m, spatial_threshold_m
                ))

            # 批量插入
            if data:
                execute_values(
                    cur,
                    """
                    INSERT INTO poi_alignment
                    (geo_entity_id, geo_name, geo_type, geo_longitude, geo_latitude,
                     amap_entity_id, amap_name, amap_type, amap_longitude, amap_latitude,
                     similarity, distance_m, spatial_threshold_m)
                    VALUES %s
                    """,
                    data
                )
                conn.commit()
                logger.success(f"保存 {len(data)} 条对齐结果")

            # 统计结果
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE similarity >= 0.40) as high_sim,
                    COUNT(*) FILTER (WHERE distance_m <= 50) as close_match,
                    AVG(similarity) as avg_sim,
                    AVG(distance_m) as avg_dist
                FROM poi_alignment
            """)
            stats = cur.fetchone()
            logger.info(f"统计: 总匹配={stats[0]}, 高相似度(>=0.40)={stats[1]}, 近距离(<=50m)={stats[2]}")
            logger.info(f"平均相似度={stats[3]:.3f}, 平均距离={stats[4]:.1f}m")

    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="保存POI对齐结果")
    parser.add_argument("--similarity", type=float, default=0.30, help="相似度阈值")
    parser.add_argument("--spatial", type=float, default=0.002, help="空间阈值(度)")

    args = parser.parse_args()
    save_alignment(similarity_threshold=args.similarity, spatial_threshold_deg=args.spatial)