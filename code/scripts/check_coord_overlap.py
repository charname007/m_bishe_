"""
检查两个数据源的坐标范围是否重叠
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
    # geo_entity_names coordinate range
    cur.execute("""
        SELECT MIN(longitude), MAX(longitude), MIN(latitude), MAX(latitude), COUNT(*)
        FROM geo_entity_names
        WHERE type = 'poi' AND longitude IS NOT NULL
    """)
    geo_range = cur.fetchone()

    # amap_poi_wgs84 coordinate range
    cur.execute("""
        SELECT MIN(longitude), MAX(longitude), MIN(latitude), MAX(latitude), COUNT(*)
        FROM amap_poi_wgs84
        WHERE longitude IS NOT NULL
    """)
    amap_range = cur.fetchone()

    print('geo_entity_names (poi) range:')
    print(f'  Count: {geo_range[4]}')
    print(f'  lon: {geo_range[0]:.4f} - {geo_range[1]:.4f}')
    print(f'  lat: {geo_range[2]:.4f} - {geo_range[3]:.4f}')

    print('amap_poi_wgs84 range:')
    print(f'  Count: {amap_range[4]}')
    print(f'  lon: {amap_range[0]:.4f} - {amap_range[1]:.4f}')
    print(f'  lat: {amap_range[2]:.4f} - {amap_range[3]:.4f}')

    # Check overlap
    lon_overlap = max(0, min(geo_range[1], amap_range[1]) - max(geo_range[0], amap_range[0]))
    lat_overlap = max(0, min(geo_range[3], amap_range[3]) - max(geo_range[2], amap_range[2]))

    print(f'\nCoordinate overlap:')
    print(f'  lon overlap: {lon_overlap:.4f} ({lon_overlap * 111:.1f} km approx)')
    print(f'  lat overlap: {lat_overlap:.4f} ({lat_overlap * 111:.1f} km approx)')

    if lon_overlap > 0 and lat_overlap > 0:
        print('  Status: OVERLAP exists')
    else:
        print('  Status: NO overlap!')

    # Sample geo POI coordinates
    cur.execute("""
        SELECT entity_id, name, longitude, latitude
        FROM geo_entity_names
        WHERE type = 'poi' AND longitude IS NOT NULL
        ORDER BY id
        LIMIT 10
    """)
    geo_samples = cur.fetchall()
    print('\ngeo_entity_names POI samples:')
    for row in geo_samples:
        print(f'  {row[1]}: ({row[2]:.4f}, {row[3]:.4f})')

    # Find nearest amap POI for each geo sample
    print('\nNearest amap POI for geo samples:')
    for geo_id, geo_name, geo_lon, geo_lat in geo_samples[:5]:
        cur.execute("""
            SELECT name, longitude, latitude,
                   sqrt(pow(longitude - %s, 2) + pow(latitude - %s, 2)) as dist_deg
            FROM amap_poi_wgs84
            ORDER BY sqrt(pow(longitude - %s, 2) + pow(latitude - %s, 2))
            LIMIT 3
        """, (geo_lon, geo_lat, geo_lon, geo_lat))
        nearest = cur.fetchall()
        print(f'  {geo_name} ({geo_lon:.4f}, {geo_lat:.4f}):')
        for n in nearest:
            dist_km = n[3] * 111  # approx conversion
            print(f'    -> {n[0]} ({n[1]:.4f}, {n[2]:.4f}) dist={dist_km:.1f}km')

conn.close()