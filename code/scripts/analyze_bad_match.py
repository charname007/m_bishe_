"""
分析特定匹配案例：为什么川菜故事匹配到了曾有味鲜辣小炒
"""
import os
import sys
from pathlib import Path
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
import psycopg2
import numpy as np

pg_config = settings.get_postgres_config()
conn = psycopg2.connect(
    host=pg_config['host'], port=pg_config['port'],
    database=pg_config['database'], user=pg_config['user'], password=pg_config['password']
)

with conn.cursor() as cur:
    # 查找'川菜故事'
    cur.execute("""
        SELECT entity_id, name, longitude, latitude, embedding
        FROM geo_entity_names
        WHERE name = '川菜故事'
    """)
    geo = cur.fetchone()

    if geo:
        geo_id, geo_name, geo_lon, geo_lat, geo_emb = geo
        print(f'geo POI: {geo_name}')
        print(f'  坐标: ({geo_lon:.4f}, {geo_lat:.4f})')

        # 查看附近200m内所有amap候选及其相似度
        cur.execute("""
            SELECT entity_id, name, longitude, latitude,
                   1 - (embedding <=> %s::vector) as sim
            FROM amap_poi_wgs84
            WHERE longitude BETWEEN %s - 0.002 AND %s + 0.002
            AND latitude BETWEEN %s - 0.002 AND %s + 0.002
            AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 10
        """, (geo_emb, geo_lon, geo_lon, geo_lat, geo_lat, geo_emb))
        candidates = cur.fetchall()

        print(f'\n附近200m内所有amap候选（按相似度排序）:')
        for i, (amap_id, amap_name, amap_lon, amap_lat, sim) in enumerate(candidates, 1):
            # 计算距离
            dist_deg = np.sqrt((geo_lon - amap_lon)**2 + (geo_lat - amap_lat)**2)
            dist_m = dist_deg * 111000  # 转换为米
            print(f'  {i}. {amap_name}')
            print(f'     sim={sim:.3f}, dist={dist_m:.0f}m')

        # 检查高德是否有"川菜故事"相关的POI（不限空间）
        cur.execute("""
            SELECT entity_id, name, longitude, latitude,
                   1 - (embedding <=> %s::vector) as sim
            FROM amap_poi_wgs84
            WHERE embedding IS NOT NULL
            AND (name LIKE '%川菜%' OR name LIKE '%故事%')
            ORDER BY embedding <=> %s::vector
            LIMIT 5
        """, (geo_emb, geo_emb))
        related = cur.fetchall()

        print(f'\n高德数据库中包含"川菜"或"故事"的POI:')
        for amap_id, amap_name, amap_lon, amap_lat, sim in related:
            dist_deg = np.sqrt((geo_lon - amap_lon)**2 + (geo_lat - amap_lat)**2)
            dist_m = dist_deg * 111000
            print(f'  {amap_name}')
            print(f'     sim={sim:.3f}, dist={dist_m:.0f}m')

    else:
        print('未找到"川菜故事"')

conn.close()