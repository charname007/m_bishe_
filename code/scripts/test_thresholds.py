"""
测试不同阈值下的匹配数量
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

pg_config = settings.get_postgres_config()
conn = psycopg2.connect(
    host=pg_config['host'], port=pg_config['port'],
    database=pg_config['database'], user=pg_config['user'], password=pg_config['password']
)

print('不同相似度阈值下的匹配数量 (空间阈值200m):')

for threshold in [0.20, 0.25, 0.30, 0.35, 0.40]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM (
                SELECT g.entity_id, g.name,
                       1 - (g.embedding <=> a.embedding) as sim
                FROM geo_entity_names g
                CROSS JOIN LATERAL (
                    SELECT embedding, longitude, latitude
                    FROM amap_poi_wgs84
                    WHERE longitude BETWEEN g.longitude - 0.002 AND g.longitude + 0.002
                    AND latitude BETWEEN g.latitude - 0.002 AND g.latitude + 0.002
                    AND embedding IS NOT NULL
                    ORDER BY embedding <=> g.embedding
                    LIMIT 1
                ) a
                WHERE g.type = 'poi' AND g.embedding IS NOT NULL AND g.longitude IS NOT NULL
                AND 1 - (g.embedding <=> a.embedding) >= %s
            ) t
        """, (threshold,))
        count = cur.fetchone()[0]
        print(f'  阈值 {threshold:.2f}: {count} 个匹配 ({count/540*100:.1f}%)')

conn.close()