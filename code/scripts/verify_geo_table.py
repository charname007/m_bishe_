"""验证 geo_entity_names 表结构和数据"""
import sys
sys.path.insert(0, '.')

from settings import settings
from kg.postgres_client import PostgresClient

pg_config = settings.get_postgres_config()
pg = PostgresClient(**pg_config)

with pg.conn.cursor() as cur:
    # 查看表结构
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'geo_entity_names'
        ORDER BY ordinal_position
    """)
    print('表结构:')
    for row in cur.fetchall():
        print(f'  {row[0]}: {row[1]}')

    # 查看记录数
    cur.execute('SELECT COUNT(*) FROM geo_entity_names')
    print(f'\n总记录数: {cur.fetchone()[0]}')

    # 几何类型分布
    cur.execute('SELECT geom_type, COUNT(*) FROM geo_entity_names GROUP BY geom_type ORDER BY COUNT(*) DESC')
    print('\n几何类型分布:')
    for row in cur.fetchall():
        print(f'  {row[0]}: {row[1]}')

    # 有 geom 数据的记录数
    cur.execute('SELECT COUNT(*) FROM geo_entity_names WHERE geom IS NOT NULL')
    print(f'\n有 geom 数据的记录: {cur.fetchone()[0]}')

    # 空间索引
    cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'geo_entity_names'")
    print('\n索引列表:')
    for row in cur.fetchall():
        print(f'  {row[0]}')

    # 示例数据
    cur.execute("""
        SELECT entity_id, name, type, geom_type, longitude, latitude
        FROM geo_entity_names LIMIT 3
    """)
    print('\n示例数据:')
    for row in cur.fetchall():
        print(f'  {row}')

pg.close()
print('\n验证完成!')