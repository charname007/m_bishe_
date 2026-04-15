"""
诊断实体对齐问题：检查嵌入相似度
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
    # Check embedding counts
    cur.execute("""
        SELECT COUNT(*) FROM geo_entity_names
        WHERE type = 'poi' AND embedding IS NOT NULL
    """)
    geo_emb_count = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM amap_poi_wgs84 WHERE embedding IS NOT NULL
    """)
    amap_emb_count = cur.fetchone()[0]

    print(f'geo_entity_names POI with embedding: {geo_emb_count}')
    print(f'amap_poi_wgs84 with embedding: {amap_emb_count}')

    # Get a geo POI sample near an amap POI
    cur.execute("""
        SELECT g.entity_id, g.name, g.longitude, g.latitude, g.embedding,
               a.entity_id, a.name, a.longitude, a.latitude, a.embedding
        FROM geo_entity_names g
        CROSS JOIN LATERAL (
            SELECT entity_id, name, longitude, latitude, embedding
            FROM amap_poi_wgs84
            WHERE longitude BETWEEN g.longitude - 0.01 AND g.longitude + 0.01
              AND latitude BETWEEN g.latitude - 0.01 AND g.latitude + 0.01
            ORDER BY sqrt(pow(longitude - g.longitude, 2) + pow(latitude - g.latitude, 2))
            LIMIT 1
        ) a
        WHERE g.type = 'poi' AND g.embedding IS NOT NULL
        LIMIT 5
    """)
    pairs = cur.fetchall()

    print(f'\nClose POI pairs (within ~1km):')
    for p in pairs:
        geo_id, geo_name, geo_lon, geo_lat, geo_emb, amap_id, amap_name, amap_lon, amap_lat, amap_emb = p

        # Calculate distance
        dist_deg = np.sqrt((geo_lon - amap_lon)**2 + (geo_lat - amap_lat)**2)
        dist_km = dist_deg * 111

        # Parse embeddings
        if isinstance(geo_emb, str):
            geo_emb_arr = np.array([float(x) for x in geo_emb.strip('[]').split(',')])
        else:
            geo_emb_arr = np.array(list(geo_emb))
        if isinstance(amap_emb, str):
            amap_emb_arr = np.array([float(x) for x in amap_emb.strip('[]').split(',')])
        else:
            amap_emb_arr = np.array(list(amap_emb))

        # Manual cosine similarity
        similarity = np.dot(geo_emb_arr, amap_emb_arr) / (np.linalg.norm(geo_emb_arr) * np.linalg.norm(amap_emb_arr))

        # PostgreSQL vector distance
        cur.execute("""
            SELECT 1 - (embedding <=> %s::vector) as similarity
            FROM amap_poi_wgs84
            WHERE entity_id = %s
        """, (geo_emb, amap_id))
        pg_similarity = cur.fetchone()[0]

        print(f'  {geo_name} <-> {amap_name}')
        print(f'    distance: {dist_km:.2f} km')
        print(f'    manual similarity: {similarity:.4f}')
        print(f'    PostgreSQL similarity: {pg_similarity:.4f}')

conn.close()