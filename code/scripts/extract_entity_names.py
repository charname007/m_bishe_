"""
从Neo4j提取所有地理实体name，保存到文件/PostgreSQL
用于后续语义相似度比对

表结构说明：
- geo_entity_names: 地理实体表
  - id, entity_id, name, type, geom_type, longitude, latitude, geom (PostGIS), embedding, created_at
  - geom 使用 PostGIS geometry(Geometry, 4326) 类型，支持 Point/LineString/Polygon/MultiLineString/MultiPolygon
  - longitude/latitude 为质心坐标（用于简化查询）
  - 自动检测并迁移现有表，添加 geom 字段和空间索引
- corpus_entity_names: 语料实体表，包含 id, name, type, category, aliases, embedding, created_at
- embedding 列维度由 settings.EMBEDDING_DIM 配置或模型实际维度决定

依赖关系：
- extract_entity_names.py 创建表结构（embedding 列初始为 NULL）
- embed_entity_names.py 计算并填充 embedding 数据

PostGIS 空间查询示例：
- 查找附近实体（基于质心）: SELECT * FROM geo_entity_names WHERE ST_DWithin(geom::geography, ST_MakePoint(lon,lat)::geography, 500);
- 空间相交查询: SELECT * FROM geo_entity_names WHERE ST_Intersects(geom, ST_MakePolygon(...));
- 计算距离: SELECT ST_Distance(geom::geography, ST_MakePoint(lon,lat)::geography) FROM geo_entity_names;
"""
import json
import sys
from pathlib import Path
from typing import List, Dict
from loguru import logger
from psycopg2.extras import execute_values

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
from kg.neo4j_client import Neo4jClient
from kg.postgres_client import PostgresClient


def _check_and_install_extension(pg_client: PostgresClient, ext_name: str, display_name: str) -> bool:
    """检查并安装 PostgreSQL 扩展"""
    has_ext = False
    try:
        with pg_client.conn.cursor() as cur:
            cur.execute("SELECT * FROM pg_extension WHERE extname = %s", (ext_name,))
            if cur.fetchone():
                has_ext = True
    except Exception:
        pass

    if not has_ext:
        try:
            with pg_client.conn.cursor() as cur:
                cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext_name}")
                pg_client.conn.commit()
                has_ext = True
                logger.info(f"{display_name}扩展已安装")
        except Exception as e:
            logger.warning(f"无法安装{display_name}扩展: {e}")
            pg_client.conn.rollback()
            has_ext = False

    return has_ext


def _create_geo_table(cur, table_name: str, has_postgis: bool, has_vector: bool, embedding_dim: int):
    """创建地理实体表"""
    # 基础字段
    base_sql = f"""
        CREATE TABLE {table_name} (
            id SERIAL PRIMARY KEY,
            entity_id VARCHAR(100) NOT NULL UNIQUE,
            name VARCHAR(200) NOT NULL,
            type VARCHAR(50),
            geom_type VARCHAR(50),
            longitude DOUBLE PRECISION,
            latitude DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT NOW()
    """

    # 添加 PostGIS geometry 字段（支持所有几何类型）
    if has_postgis:
        base_sql += ", geom geometry(Geometry, 4326)"

    # 添加 vector embedding 字段
    if has_vector:
        base_sql += f", embedding VECTOR({embedding_dim})"

    base_sql += ")"

    cur.execute(base_sql)

    # 创建索引
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_geo_entity_name ON {table_name}(name)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_geo_entity_type ON {table_name}(type)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_geo_entity_geom_type ON {table_name}(geom_type)")

    if has_postgis:
        # 创建空间索引（GiST索引）
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_geo_entity_geom ON {table_name} USING GIST (geom)")

    logger.info(f"表 {table_name} 已创建 (PostGIS: {has_postgis}, pgvector: {has_vector})")


def _migrate_geo_table(cur, table_name: str, has_postgis: bool, has_vector: bool, embedding_dim: int):
    """迁移现有表，添加缺失字段和索引"""
    # 获取现有列
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = %s
    """, (table_name,))
    existing_columns = {row[0] for row in cur.fetchall()}

    # 需要添加的字段
    columns_to_add = []

    # 基础字段（必须存在）
    if "geom_type" not in existing_columns:
        columns_to_add.append("ADD COLUMN geom_type VARCHAR(50)")
    if "longitude" not in existing_columns:
        columns_to_add.append("ADD COLUMN longitude DOUBLE PRECISION")
    if "latitude" not in existing_columns:
        columns_to_add.append("ADD COLUMN latitude DOUBLE PRECISION")

    # PostGIS geometry 字段
    if has_postgis and "geom" not in existing_columns:
        columns_to_add.append("ADD COLUMN geom geometry(Geometry, 4326)")

    # pgvector embedding 字段
    if has_vector and "embedding" not in existing_columns:
        columns_to_add.append(f"ADD COLUMN embedding VECTOR({embedding_dim})")

    # 执行 ALTER TABLE
    if columns_to_add:
        alter_sql = f"ALTER TABLE {table_name} {', '.join(columns_to_add)}"
        cur.execute(alter_sql)
        logger.info(f"表 {table_name} 已迁移，添加字段: {columns_to_add}")

        # 如果添加了 geom 字段，且表中已有 longitude/latitude 数据，则填充
        # 注意：新添加的 longitude/latitude 字段此时都是 NULL，所以不需要填充
        if has_postgis and "geom" not in existing_columns:
            if "longitude" in existing_columns and "latitude" in existing_columns:
                # 只有当这些字段原本就存在时才尝试填充
                cur.execute(f"""
                    UPDATE {table_name}
                    SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
                    WHERE longitude IS NOT NULL AND latitude IS NOT NULL AND geom IS NULL
                """)
                updated = cur.rowcount
                if updated > 0:
                    logger.info(f"已从现有坐标填充 {updated} 条记录的 geom 字段")
            else:
                logger.info("新增 longitude/latitude/geom 字段，将由后续插入数据填充")

    # 创建缺失索引
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_geo_entity_name ON {table_name}(name)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_geo_entity_type ON {table_name}(type)")

    if "geom_type" not in existing_columns:
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_geo_entity_geom_type ON {table_name}(geom_type)")

    if has_postgis:
        # 检查空间索引是否存在
        cur.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = %s AND indexname = 'idx_geo_entity_geom'
        """, (table_name,))
        if not cur.fetchone():
            cur.execute(f"CREATE INDEX idx_geo_entity_geom ON {table_name} USING GIST (geom)")
            logger.info(f"已创建空间索引 idx_geo_entity_geom")


def _insert_geo_entities(cur, entities: List[Dict], has_postgis: bool):
    """批量插入地理实体"""
    if not has_postgis:
        # 无 PostGIS，仅插入基本信息
        data = [
            (e.get("entity_id", ""), e.get("name", ""), e.get("type", ""),
             e.get("geom_type", ""), e.get("longitude"), e.get("latitude"))
            for e in entities
        ]
        execute_values(cur, """
            INSERT INTO geo_entity_names (entity_id, name, type, geom_type, longitude, latitude)
            VALUES %s
            ON CONFLICT (entity_id) DO UPDATE SET
                name = EXCLUDED.name, type = EXCLUDED.type, geom_type = EXCLUDED.geom_type,
                longitude = EXCLUDED.longitude, latitude = EXCLUDED.latitude
        """, data)
        return

    # 有 PostGIS，分类处理
    wkt_entities = []      # 有完整 WKT 数据（线、面）
    point_entities = []    # 只有坐标的点类型
    invalid_entities = []  # 无坐标数据

    for e in entities:
        if e.get("geometry_wkt"):
            wkt_entities.append(e)
        elif e.get("geom_type") == "Point" and e.get("longitude") and e.get("latitude"):
            point_entities.append(e)
        elif e.get("longitude") and e.get("latitude"):
            # 有坐标但没有 WKT，构建 Point WKT
            lon, lat = e.get("longitude"), e.get("latitude")
            wkt_entities.append({**e, "geometry_wkt": f"POINT ({lon} {lat})"})
        else:
            invalid_entities.append(e)

    if invalid_entities:
        logger.warning(f"跳过 {len(invalid_entities)} 个无坐标数据的实体")

    # 处理 WKT 实体（线、面）- 使用 UPSERT
    for e in wkt_entities:
        # 使用 ON CONFLICT 更新所有字段（包括 geom）
        # geom 使用 CASE 来处理：插入新记录时用 ST_GeomFromText，更新时也用 ST_GeomFromText
        cur.execute("""
            INSERT INTO geo_entity_names (entity_id, name, type, geom_type, longitude, latitude, geom)
            VALUES (%s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326))
            ON CONFLICT (entity_id) DO UPDATE SET
                name = EXCLUDED.name,
                type = EXCLUDED.type,
                geom_type = EXCLUDED.geom_type,
                longitude = EXCLUDED.longitude,
                latitude = EXCLUDED.latitude,
                geom = ST_GeomFromText(%s, 4326)
        """, (
            e.get("entity_id", ""), e.get("name", ""), e.get("type", ""),
            e.get("geom_type", ""), e.get("longitude"), e.get("latitude"),
            e.get("geometry_wkt"),  # 用于 INSERT
            e.get("geometry_wkt")   # 用于 UPDATE
        ))

    if wkt_entities:
        logger.info(f"已处理 {len(wkt_entities)} 条带几何数据的实体")

    # 处理 Point 类型实体 - 使用批量 UPSERT
    if point_entities:
        # 批量插入基本信息
        data = [(e.get("entity_id", ""), e.get("name", ""), e.get("type", ""),
                 e.get("geom_type", ""), e.get("longitude"), e.get("latitude"))
                for e in point_entities]
        execute_values(cur, """
            INSERT INTO geo_entity_names (entity_id, name, type, geom_type, longitude, latitude)
            VALUES %s
            ON CONFLICT (entity_id) DO UPDATE SET
                name = EXCLUDED.name, type = EXCLUDED.type, geom_type = EXCLUDED.geom_type,
                longitude = EXCLUDED.longitude, latitude = EXCLUDED.latitude
        """, data)
        # 批量更新 geom
        execute_values(cur, """
            UPDATE geo_entity_names SET geom = ST_SetSRID(ST_MakePoint(v.longitude, v.latitude), 4326)
            FROM (VALUES %s) AS v(entity_id, name, type, geom_type, longitude, latitude)
            WHERE geo_entity_names.entity_id = v.entity_id
        """, data)
        logger.info(f"已处理 {len(point_entities)} 条 Point 类型实体")

    # 处理无坐标数据的实体
    if invalid_entities:
        data_invalid = [(e.get("entity_id", ""), e.get("name", ""), e.get("type", ""), e.get("geom_type", ""))
                        for e in invalid_entities]
        execute_values(cur, """
            INSERT INTO geo_entity_names (entity_id, name, type, geom_type)
            VALUES %s
            ON CONFLICT (entity_id) DO UPDATE SET
                name = EXCLUDED.name, type = EXCLUDED.type, geom_type = EXCLUDED.geom_type
        """, data_invalid)


def extract_entity_names(neo4j_client: Neo4jClient, source: str = "geo") -> List[Dict]:
    """
    从Neo4j提取所有实体的name和相关属性

    Args:
        source: 数据来源
            - "geo": 提取地理实体 (geo_entity_node:Road/Poi/Building/Block)
            - "corpus": 提取语料实体 (Entity标签)

    Returns:
        geo实体: [{"name", "type", "entity_id", "geom_type", "longitude", "latitude", "geometry_wkt"}]
        corpus实体: [{"name", "type", "category", "aliases"}]
    """
    # 定义无效名称集合
    invalid_names = {"", "未命名", "null", "None", "nan"}

    with neo4j_client.driver.session() as session:
        if source == "geo":
            # 提取地理实体 (shp2kg.py 创建的节点)
            # 包含完整 geometry (WKT格式) 和几何类型
            result = session.run("""
                MATCH (n:geo_entity_node)
                RETURN n.name as name, n.entity_type as type, n.entity_id as entity_id,
                       n.geom_type as geom_type, n.longitude as longitude, n.latitude as latitude,
                       n.geometry as geometry_wkt
                ORDER BY n.name
            """)
            entities = []
            for record in result:
                name = record["name"]
                # 过滤空名称和无效名称
                if name and str(name).strip() not in invalid_names:
                    entities.append({
                        "name": str(name).strip(),
                        "type": record["type"],
                        "entity_id": record["entity_id"],
                        "geom_type": record["geom_type"],
                        "longitude": record["longitude"],
                        "latitude": record["latitude"],
                        "geometry_wkt": record["geometry_wkt"]  # WKT 格式的完整几何数据
                    })
        else:
            # 提取语料实体 (neo4j_client.py 创建的节点)
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.name as name, e.type as type, e.category as category, e.aliases as aliases
                ORDER BY e.name
            """)
            entities = []
            for record in result:
                name = record["name"]
                # 过滤空名称和无效名称
                if name and str(name).strip() not in invalid_names:
                    entities.append({
                        "name": str(name).strip(),
                        "type": record["type"],
                        "category": record["category"],
                        "aliases": record["aliases"] or []
                    })

        logger.info(f"提取到 {len(entities)} 个实体 (来源: {source})")
        return entities


def save_to_json(entities: List[Dict], filepath: str = "entity_names.json"):
    """保存为JSON文件"""
    output_path = Path(__file__).parent.parent / filepath
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存到 {output_path}")
    return str(output_path)


def save_to_csv(entities: List[Dict], filepath: str = "entity_names.csv", source: str = "geo"):
    """保存为CSV文件"""
    import csv
    output_path = Path(__file__).parent.parent / filepath
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if source == "geo":
            writer.writerow(["name", "type", "entity_id", "geom_type", "longitude", "latitude", "geometry_wkt"])
            for e in entities:
                writer.writerow([
                    e.get("name", ""), e.get("type", ""), e.get("entity_id", ""),
                    e.get("geom_type", ""), e.get("longitude", ""), e.get("latitude", ""),
                    e.get("geometry_wkt", "")
                ])
        else:
            writer.writerow(["name", "type", "category", "aliases"])
            for e in entities:
                writer.writerow([e.get("name", ""), e.get("type", ""), e.get("category", ""), "|".join(e.get("aliases", []))])
    logger.info(f"已保存到 {output_path}")
    return str(output_path)


def save_to_postgres(entities: List[Dict], pg_client: PostgresClient, source: str = "geo"):
    """
    保存到PostgreSQL（使用PostGIS存储地理几何）

    Args:
        entities: 实体列表
        pg_client: PostgreSQL客户端
        source: 数据来源 ("geo" 或 "corpus")

    注意：
        - geo实体使用 PostGIS geometry(Geometry, 4326) 存储完整几何（支持点/线/面）
        - longitude/latitude 为质心坐标，便于简化查询
        - 自动检测并迁移现有表，添加缺失字段和索引
        - embedding 列初始为 NULL，由 embed_entity_names.py 脚本填充
    """
    # 从配置获取嵌入维度
    embedding_dim = settings.get_embedding_config()["dim"]

    # 检查并安装 PostGIS 扩展
    has_postgis = _check_and_install_extension(pg_client, "postgis", "PostGIS")
    # 检查并安装 pgvector 扩展
    has_vector = _check_and_install_extension(pg_client, "vector", "pgvector")

    with pg_client.conn.cursor() as cur:
        if source == "geo":
            # 地理实体表
            table_name = "geo_entity_names"

            # 检查表是否存在
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables WHERE table_name = %s
                )
            """, (table_name,))
            table_exists = cur.fetchone()[0]

            if not table_exists:
                # 创建新表
                _create_geo_table(cur, table_name, has_postgis, has_vector, embedding_dim)
            else:
                # 表已存在，检查并迁移缺失字段
                _migrate_geo_table(cur, table_name, has_postgis, has_vector, embedding_dim)

            pg_client.conn.commit()

            # 批量插入地理实体
            _insert_geo_entities(cur, entities, has_postgis)

            logger.info(f"已保存 {len(entities)} 条地理实体记录 (PostGIS: {has_postgis})")
        else:
            # 语料实体表
            if has_vector:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS corpus_entity_names (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(200) NOT NULL UNIQUE,
                        type VARCHAR(50),
                        category VARCHAR(50),
                        aliases TEXT[],
                        embedding VECTOR({embedding_dim}),
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
            else:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS corpus_entity_names (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(200) NOT NULL UNIQUE,
                        type VARCHAR(50),
                        category VARCHAR(50),
                        aliases TEXT[],
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_corpus_entity_name ON corpus_entity_names(name)")

            pg_client.conn.commit()

            # 批量插入语料实体
            data = [
                (e["name"], e.get("type", ""), e.get("category", ""), e.get("aliases", []))
                for e in entities
            ]

            execute_values(cur, """
                INSERT INTO corpus_entity_names (name, type, category, aliases)
                VALUES %s
                ON CONFLICT (name) DO UPDATE SET
                    type = EXCLUDED.type,
                    category = EXCLUDED.category,
                    aliases = EXCLUDED.aliases
            """, data)

        pg_client.conn.commit()
        logger.info(f"已保存 {len(entities)} 条记录到 PostgreSQL ({source}实体表)")


def main(source: str = "geo", save_to_pg: bool = False):
    """
    主函数

    Args:
        source: 数据来源
            - "geo": 提取地理实体 (shp2kg.py 创建的节点)
            - "corpus": 提取语料实体 (neo4j_client.py 创建的节点)
        save_to_pg: 是否保存到PostgreSQL
    """
    # 连接Neo4j
    neo4j_config = settings.get_neo4j_config()
    neo4j_client = Neo4jClient(**neo4j_config)

    # 提取实体
    entities = extract_entity_names(neo4j_client, source=source)

    if not entities:
        logger.warning(f"未找到任何实体 (来源: {source})")
        neo4j_client.close()
        return

    # 保存方式选择
    filename_suffix = f"_{source}" if source != "geo" else ""
    save_to_json(entities, filepath=f"entity_names{filename_suffix}.json")
    save_to_csv(entities, filepath=f"entity_names{filename_suffix}.csv", source=source)

    # 保存到PostgreSQL（需要安装pgvector扩展）
    if save_to_pg:
        pg_config = settings.get_postgres_config()
        pg_client = PostgresClient(**pg_config)
        save_to_postgres(entities, pg_client, source=source)
        pg_client.close()

    neo4j_client.close()
    logger.success(f"提取完成，共 {len(entities)} 个实体")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从Neo4j提取地理实体并保存")
    parser.add_argument("--source", choices=["geo", "corpus"], default="geo",
                        help="数据来源 (geo: 地理实体, corpus: 语料实体)")
    parser.add_argument("--save-pg", action="store_true",
                        help="保存到 PostgreSQL (需要 PostGIS/pgvector 扩展)")
    args = parser.parse_args()
    main(source=args.source, save_to_pg=args.save_pg)