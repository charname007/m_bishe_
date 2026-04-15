"""
分析为什么只匹配到16个实体
"""
import os
import sys
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
import psycopg2

pg_config = settings.get_postgres_config()
conn = psycopg2.connect(
    host=pg_config['host'], port=pg_config['port'],
    database=pg_config['database'], user=pg_config['user'], password=pg_config['password']
)

with conn.cursor() as cur:
    # 1. 统计geo POI附近有无amap POI（空间范围 0.002度 ≈ 200m）
    cur.execute("""
        SELECT
            COUNT(*) as total_geo,
            SUM(CASE WHEN amap_count > 0 THEN 1 ELSE 0 END) as has_nearby_amap,
            SUM(CASE WHEN amap_count = 0 THEN 1 ELSE 0 END) as no_nearby_amap
        FROM (
            SELECT g.entity_id,
                   (SELECT COUNT(*) FROM amap_poi_wgs84
                    WHERE longitude BETWEEN g.longitude - 0.002 AND g.longitude + 0.002
                    AND latitude BETWEEN g.latitude - 0.002 AND g.latitude + 0.002
                    AND embedding IS NOT NULL) as amap_count
            FROM geo_entity_names g
            WHERE g.type = 'poi' AND g.embedding IS NOT NULL AND g.longitude IS NOT NULL
        ) t
    """)
    stats = cur.fetchone()
    print(f'geo POI总数: {stats[0]}')
    print(f'附近有amap POI (200m内): {stats[1]} ({stats[1]/stats[0]*100:.1f}%)')
    print(f'附近无amap POI: {stats[2]} ({stats[2]/stats[0]*100:.1f}%)')

    # 2. 对于附近有amap的，统计相似度分布
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE sim >= 0.50) as sim_50_plus,
            COUNT(*) FILTER (WHERE sim >= 0.40) as sim_40_plus,
            COUNT(*) FILTER (WHERE sim >= 0.35) as sim_35_plus,
            COUNT(*) FILTER (WHERE sim >= 0.30) as sim_30_plus,
            COUNT(*) FILTER (WHERE sim >= 0.25) as sim_25_plus,
            COUNT(*) FILTER (WHERE sim >= 0.20) as sim_20_plus,
            AVG(sim) as avg_sim,
            MAX(sim) as max_sim
        FROM (
            SELECT g.entity_id, g.name, g.longitude, g.latitude,
                   1 - (g.embedding <=> a.embedding) as sim
            FROM geo_entity_names g
            CROSS JOIN LATERAL (
                SELECT embedding
                FROM amap_poi_wgs84
                WHERE longitude BETWEEN g.longitude - 0.002 AND g.longitude + 0.002
                AND latitude BETWEEN g.latitude - 0.002 AND g.latitude + 0.002
                AND embedding IS NOT NULL
                ORDER BY embedding <=> g.embedding
                LIMIT 1
            ) a
            WHERE g.type = 'poi' AND g.embedding IS NOT NULL AND g.longitude IS NOT NULL
        ) t
    """)
    sim_stats = cur.fetchone()
    print(f'\n相似度分布（附近有amap的 {stats[1]} 个geo POI）:')
    print(f'  >= 0.50: {sim_stats[0]}')
    print(f'  >= 0.40: {sim_stats[1]} (当前阈值) → 匹配16个')
    print(f'  >= 0.35: {sim_stats[2]}')
    print(f'  >= 0.30: {sim_stats[3]}')
    print(f'  >= 0.25: {sim_stats[4]}')
    print(f'  >= 0.20: {sim_stats[5]}')
    print(f'  平均相似度: {sim_stats[6]:.3f}')
    print(f'  最高相似度: {sim_stats[7]:.3f}')

    # 3. 分析未匹配原因：看几个无附近amap的geo POI
    cur.execute("""
        SELECT g.entity_id, g.name, g.longitude, g.latitude
        FROM geo_entity_names g
        WHERE g.type = 'poi' AND g.embedding IS NOT NULL AND g.longitude IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM amap_poi_wgs84
            WHERE longitude BETWEEN g.longitude - 0.002 AND g.longitude + 0.002
            AND latitude BETWEEN g.latitude - 0.002 AND g.latitude + 0.002
            AND embedding IS NOT NULL
        )
        ORDER BY g.id
        LIMIT 10
    """)
    no_amap = cur.fetchall()
    print(f'\n无附近amap的geo POI样本:')
    for row in no_amap:
        print(f'  {row[1]} ({row[2]:.4f}, {row[3]:.4f})')

conn.close()