"""
改进的对齐策略：加入名称字符串相似度过滤
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


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def jaccard_similarity(s1: str, s2: str) -> float:
    """计算Jaccard相似度（字符集合）"""
    if not s1 or not s2:
        return 0.0
    set1 = set(s1)
    set2 = set(s2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def name_match_score(geo_name: str, amap_name: str) -> float:
    """
    名称匹配分数（综合字符串相似度）
    检查：
    1. 是否包含相同核心名称（去掉店名后缀）
    2. Jaccard相似度
    """
    # 提取核心名称（去掉括号内的店名后缀）
    def extract_core(name):
        if '(' in name:
            return name.split('(')[0].strip()
        if '（' in name:
            return name.split('（')[0].strip()
        return name.strip()

    geo_core = extract_core(geo_name)
    amap_core = extract_core(amap_name)

    # 核心名称完全匹配
    if geo_core == amap_core:
        return 1.0

    # 核心名称包含关系
    if geo_core in amap_core or amap_core in geo_core:
        return 0.8

    # Jaccard相似度
    jaccard = jaccard_similarity(geo_core, amap_core)

    return jaccard


def save_alignment_with_name_filter(
    similarity_threshold: float = 0.30,
    spatial_threshold_deg: float = 0.002,
    name_threshold: float = 0.3  # 名称相似度阈值
):
    """保存对齐结果，加入名称过滤"""
    pg_config = settings.get_postgres_config()
    conn = psycopg2.connect(
        host=pg_config['host'], port=pg_config['port'],
        database=pg_config['database'], user=pg_config['user'], password=pg_config['password']
    )

    try:
        with conn.cursor() as cur:
            # 创建新表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS poi_alignment_filtered (
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
                    embedding_sim DOUBLE PRECISION,
                    name_sim DOUBLE PRECISION,
                    distance_m DOUBLE PRECISION,
                    is_high_quality BOOLEAN,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(geo_entity_id, amap_entity_id)
                )
            """)
            conn.commit()

            cur.execute("TRUNCATE TABLE poi_alignment_filtered")
            conn.commit()

            # 查询候选
            cur.execute("""
                SELECT
                    g.entity_id, g.name, g.type, g.longitude, g.latitude,
                    a.entity_id, a.name, a.type, a.longitude, a.latitude,
                    1 - (g.embedding <=> a.embedding) as embedding_sim
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
            logger.info(f"候选匹配: {len(matches)} 个")

            # 应用名称过滤
            filtered_data = []
            for row in matches:
                geo_id, geo_name, geo_type, geo_lon, geo_lat, \
                amap_id, amap_name, amap_type, amap_lon, amap_lat, emb_sim = row

                # 计算名称相似度
                name_sim = name_match_score(geo_name, amap_name)
                dist_m = haversine_distance(geo_lat, geo_lon, amap_lat, amap_lon)

                # 判断是否高质量匹配
                is_high_quality = name_sim >= name_threshold or emb_sim >= 0.45

                # 只保存满足条件的匹配
                if is_high_quality:
                    filtered_data.append((
                        geo_id, geo_name, geo_type, geo_lon, geo_lat,
                        amap_id, amap_name, amap_type, amap_lon, amap_lat,
                        emb_sim, name_sim, dist_m, is_high_quality
                    ))

            if filtered_data:
                execute_values(
                    cur,
                    """
                    INSERT INTO poi_alignment_filtered
                    (geo_entity_id, geo_name, geo_type, geo_longitude, geo_latitude,
                     amap_entity_id, amap_name, amap_type, amap_longitude, amap_latitude,
                     embedding_sim, name_sim, distance_m, is_high_quality)
                    VALUES %s
                    """,
                    filtered_data
                )
                conn.commit()
                logger.success(f"保存 {len(filtered_data)} 条高质量匹配")

            # 统计
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE name_sim >= 0.5) as name_match,
                    COUNT(*) FILTER (WHERE embedding_sim >= 0.45) as high_emb,
                    AVG(embedding_sim) as avg_emb,
                    AVG(name_sim) as avg_name
                FROM poi_alignment_filtered
            """)
            stats = cur.fetchone()
            logger.info(f"统计: 总匹配={stats[0]}, 名称匹配={stats[1]}, 高嵌入相似={stats[2]}")
            logger.info(f"平均嵌入相似度={stats[3]:.3f}, 平均名称相似度={stats[4]:.3f}")

    finally:
        conn.close()


if __name__ == "__main__":
    save_alignment_with_name_filter(
        similarity_threshold=0.30,
        spatial_threshold_deg=0.002,
        name_threshold=0.3
    )