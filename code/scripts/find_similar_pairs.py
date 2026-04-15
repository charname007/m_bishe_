"""
寻找名称相似的POI对来验证对齐效果
"""
import os
import sys
from pathlib import Path
import numpy as np

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
    # Find POI pairs with similar names using vector search only (no spatial filter first)
    print('Testing pure embedding similarity (no spatial filter):')

    cur.execute("""
        SELECT g.entity_id, g.name, g.longitude, g.latitude,
               a.entity_id, a.name, a.longitude, a.latitude,
               1 - (g.embedding <=> a.embedding) as similarity
        FROM geo_entity_names g
        CROSS JOIN LATERAL (
            SELECT entity_id, name, longitude, latitude, embedding
            FROM amap_poi_wgs84
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> g.embedding
            LIMIT 5
        ) a
        WHERE g.type = 'poi' AND g.embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT 10
    """)
    top_similar = cur.fetchall()

    print(f'\nTop similarity pairs (pure embedding):')
    for p in top_similar:
        geo_id, geo_name, geo_lon, geo_lat, amap_id, amap_name, amap_lon, amap_lat, sim = p
        dist_deg = np.sqrt((geo_lon - amap_lon)**2 + (geo_lat - amap_lat)**2)
        dist_km = dist_deg * 111
        print(f'  {geo_name} <-> {amap_name}')
        print(f'    similarity: {sim:.4f}, distance: {dist_km:.2f} km')

    # Now test with both spatial + embedding filter
    print('\n\nTesting combined spatial + embedding filter:')
    cur.execute("""
        SELECT g.entity_id, g.name, g.longitude, g.latitude,
               a.entity_id, a.name, a.longitude, a.latitude,
               1 - (g.embedding <=> a.embedding) as similarity
        FROM geo_entity_names g
        CROSS JOIN LATERAL (
            SELECT entity_id, name, longitude, latitude, embedding
            FROM amap_poi_wgs84
            WHERE embedding IS NOT NULL
              AND longitude BETWEEN g.longitude - 0.002 AND g.longitude + 0.002
              AND latitude BETWEEN g.latitude - 0.002 AND g.latitude + 0.002
            ORDER BY embedding <=> g.embedding
            LIMIT 1
        ) a
        WHERE g.type = 'poi' AND g.embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT 10
    """)
    combined = cur.fetchall()

    print(f'\nTop similarity pairs (within ~200m):')
    for p in combined:
        geo_id, geo_name, geo_lon, geo_lat, amap_id, amap_name, amap_lon, amap_lat, sim = p
        dist_deg = np.sqrt((geo_lon - amap_lon)**2 + (geo_lat - amap_lat)**2)
        dist_km = dist_deg * 111
        print(f'  {geo_name} <-> {amap_name}')
        print(f'    similarity: {sim:.4f}, distance: {dist_km:.2f} km')

conn.close()