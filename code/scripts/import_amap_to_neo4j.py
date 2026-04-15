"""
将高德POI数据导入Neo4j，并建立实体对齐关系

功能：
1. 从 amap_poi_wgs84 表读取全部37675条高德POI
2. entity_id 格式：poi_{n}，从 geo_entity_names 中 poi 数量+1 开始（poi_837）
3. 保存原始高德entity_id为 original_id 属性
4. 创建 Neo4j 节点，标签：geo_entity_node, Poi, amap
5. 根据 poi_alignment_filtered 表建立 ENTITY_ALIGNMENT 关系
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
from kg.neo4j_client import Neo4jClient
from loguru import logger


def get_poi_max_id(neo4j_client: Neo4jClient) -> int:
    """从 Neo4j 获取现有 Poi 节点的最大 entity_id 编号"""
    import re
    with neo4j_client.driver.session() as session:
        result = session.run("""
            MATCH (n:Poi)
            WHERE NOT n:amap
            AND n.entity_id STARTS WITH 'poi_'
            RETURN n.entity_id as eid
        """)
        max_id = 0
        for record in result:
            eid = record['eid']
            if eid and eid.startswith('poi_'):
                num = int(eid.replace('poi_', ''))
                max_id = max(max_id, num)
        logger.info(f"Neo4j 中原有 Poi 最大编号: {max_id}")
        return max_id


def get_poi_count(pg_conn) -> int:
    """获取 geo_entity_names 中 poi 类型的数量"""
    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM geo_entity_names WHERE type = 'poi'")
        count = cur.fetchone()[0]
        logger.info(f"geo_entity_names 中 poi 类型数量: {count}")
        return count


def fetch_amap_poi(pg_conn):
    """获取高德POI数据"""
    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM amap_poi_wgs84")
        total = cur.fetchone()[0]
        logger.info(f"高德POI总数: {total}")

        # 包含原始 entity_id
        cur.execute("""
            SELECT id, entity_id, name, type, longitude, latitude, address
            FROM amap_poi_wgs84
            ORDER BY id
        """)
        return cur.fetchall(), total


def fetch_alignment_with_internal_id(pg_conn, max_poi_id: int):
    """获取对齐匹配数据，包含 amap 内部 id"""
    with pg_conn.cursor() as cur:
        cur.execute("""
            SELECT f.geo_entity_id, f.amap_entity_id,
                   a.id as amap_internal_id,
                   f.embedding_sim, f.name_sim, f.distance_m
            FROM poi_alignment_filtered f
            JOIN amap_poi_wgs84 a ON a.entity_id = f.amap_entity_id
        """)
        alignment_data = cur.fetchall()

        relations = []
        for geo_entity_id, amap_original_id, amap_internal_id, emb_sim, name_sim, distance in alignment_data:
            # neo4j entity_id = poi_{max_poi_id + amap_internal_id}
            # amap_internal_id 从1开始，所以第一个是 poi_{835 + 1} = poi_836
            neo4j_amap_entity_id = f"poi_{max_poi_id + amap_internal_id}"
            relations.append({
                "geo_entity_id": geo_entity_id,
                "neo4j_amap_entity_id": neo4j_amap_entity_id,
                "embedding_sim": float(emb_sim),
                "name_sim": float(name_sim),
                "distance_m": float(distance)
            })

        logger.info(f"对齐匹配数: {len(relations)}")
        return relations


def create_amap_nodes(neo4j_client: Neo4jClient, pois: list, max_poi_id: int):
    """批量创建高德POI节点"""
    logger.info(f"开始创建 {len(pois)} 个高德POI节点，entity_id 从 poi_{max_poi_id + 1} 开始")

    nodes_data = []
    for i, (amap_id, original_entity_id, name, type_, lon, lat, address) in enumerate(pois):
        entity_id = f"poi_{max_poi_id + i + 1}"  # poi_836, poi_837, ...
        nodes_data.append({
            "entity_id": entity_id,
            "original_id": original_entity_id,  # 保存原始高德entity_id: amap_B0FFLCH14H
            "name": name,
            "type": type_,
            "longitude": lon,
            "latitude": lat,
            "address": address or "",
            "source": "amap"
        })

    batch_size = 500
    created = 0

    for i in range(0, len(nodes_data), batch_size):
        batch = nodes_data[i:i+batch_size]

        query = """
        UNWIND $batch AS node
        CREATE (n:geo_entity_node:Poi:amap)
        SET n.entity_id = node.entity_id,
            n.original_id = node.original_id,
            n.name = node.name,
            n.type = node.type,
            n.longitude = node.longitude,
            n.latitude = node.latitude,
            n.address = node.address,
            n.source = node.source
        RETURN count(n) as created
        """

        with neo4j_client.driver.session() as session:
            result = session.run(query, {"batch": batch})
            record = result.single()
            created += record["created"] if record else len(batch)

        if (i + batch_size) % 5000 == 0 or i + batch_size >= len(nodes_data):
            logger.info(f"已创建 {created}/{len(nodes_data)} 个节点")

    logger.success(f"完成创建 {created} 个高德POI节点")
    return nodes_data


def create_alignment_relations(neo4j_client: Neo4jClient, relations: list):
    """根据匹配表建立 ENTITY_ALIGNMENT 关系"""
    logger.info(f"开始建立 {len(relations)} 个 ENTITY_ALIGNMENT 关系")

    batch_size = 100
    created = 0

    for i in range(0, len(relations), batch_size):
        batch = relations[i:i+batch_size]

        batch_query = """
        UNWIND $batch AS rel
        MATCH (geo:geo_entity_node {entity_id: rel.geo_entity_id})
        MATCH (amap:amap {entity_id: rel.neo4j_amap_entity_id})
        MERGE (geo)-[r:ENTITY_ALIGNMENT]->(amap)
        SET r.embedding_similarity = rel.embedding_sim,
            r.name_similarity = rel.name_sim,
            r.distance_m = rel.distance_m
        RETURN count(r) as created
        """

        with neo4j_client.driver.session() as session:
            result = session.run(batch_query, {"batch": batch})
            record = result.single()
            created += record["created"] if record else 0

        if (i + batch_size) % 1000 == 0 or i + batch_size >= len(relations):
            logger.info(f"已建立 {created}/{len(relations)} 个关系")

    logger.success(f"完成创建 {created} 个 ENTITY_ALIGNMENT 关系")
    return created


def main():
    logger.info("=" * 60)
    logger.info("高德POI导入Neo4j + 实体对齐关系建立")
    logger.info("=" * 60)

    pg_config = settings.get_postgres_config()
    pg_conn = psycopg2.connect(
        host=pg_config['host'], port=pg_config['port'],
        database=pg_config['database'], user=pg_config['user'], password=pg_config['password']
    )

    neo4j_client = Neo4jClient(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )

    try:
        # 1. 从 Neo4j 获取起始ID (原有 poi 最大编号)
        max_poi_id = get_poi_max_id(neo4j_client)  # 原有最大835
        logger.info(f"原有 Poi 最大编号: {max_poi_id}, 新增从 poi_{max_poi_id + 1} 开始")

        # 2. 获取高德POI数据
        pois, total = fetch_amap_poi(pg_conn)

        # 3. 创建Neo4j节点（从 poi_{max_poi_id + 1} 开始，即 poi_836）
        amap_nodes = create_amap_nodes(neo4j_client, pois, max_poi_id)

        # 4. 获取对齐数据
        relations = fetch_alignment_with_internal_id(pg_conn, max_poi_id)

        # 5. 建立关系
        create_alignment_relations(neo4j_client, relations)

        # 6. 验证结果
        verify_query = "MATCH (n:amap) RETURN count(n) as amap_count"
        with neo4j_client.driver.session() as session:
            result = session.run(verify_query)
            record = result.single()
            amap_count = record["amap_count"] if record else 0

        rel_query = "MATCH ()-[r:ENTITY_ALIGNMENT]->() RETURN count(r) as rel_count"
        with neo4j_client.driver.session() as session:
            result = session.run(rel_query)
            record = result.single()
            rel_count = record["rel_count"] if record else 0

        logger.success("=" * 60)
        logger.success(f"导入完成！amap节点: {amap_count}, ENTITY_ALIGNMENT关系: {rel_count}")
        logger.success("=" * 60)

    finally:
        pg_conn.close()
        neo4j_client.close()


if __name__ == "__main__":
    main()