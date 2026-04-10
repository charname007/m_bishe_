"""
从Neo4j提取所有地理实体name，保存到文件/PostgreSQL
用于后续语义相似度比对
"""
import json
import sys
from pathlib import Path
from typing import List, Dict
from loguru import logger

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from neo4j_client import Neo4jClient
from postgres_client import PostgresClient


def extract_entity_names(neo4j_client: Neo4jClient, source: str = "geo") -> List[Dict]:
    """
    从Neo4j提取所有实体的name和相关属性

    Args:
        source: 数据来源
            - "geo": 提取地理实体 (Node:Road/Poi/Building/Block)
            - "corpus": 提取语料实体 (Entity标签)

    Returns:
        [{"name": "...", "type": "...", "entity_id": "..."}]
    """
    with neo4j_client.driver.session() as session:
        if source == "geo":
            # 提取地理实体 (shp2kg.py 创建的节点)
            result = session.run("""
                MATCH (n:Node)
                RETURN n.name as name, n.entity_type as type, n.entity_id as entity_id
                ORDER BY n.name
            """)
            entities = []
            for record in result:
                if record["name"]:  # 过滤空名称
                    entities.append({
                        "name": record["name"],
                        "type": record["type"],
                        "entity_id": record["entity_id"]
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
                entities.append({
                    "name": record["name"],
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
            writer.writerow(["name", "type", "entity_id"])
            for e in entities:
                writer.writerow([e.get("name", ""), e.get("type", ""), e.get("entity_id", "")])
        else:
            writer.writerow(["name", "type", "category", "aliases"])
            for e in entities:
                writer.writerow([e.get("name", ""), e.get("type", ""), e.get("category", ""), "|".join(e.get("aliases", []))])
    logger.info(f"已保存到 {output_path}")
    return str(output_path)


def save_to_postgres(entities: List[Dict], pg_client: PostgresClient):
    """
    保存到PostgreSQL的geo_entity_names表
    包含embedding字段用于语义相似度比对
    """
    with pg_client.conn.cursor() as cur:
        # 创建表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS geo_entity_names (
                id SERIAL PRIMARY KEY,
                name VARCHAR(200) NOT NULL UNIQUE,
                type VARCHAR(50),
                category VARCHAR(50),
                aliases TEXT[],
                embedding VECTOR(1536),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # 创建索引
        cur.execute("CREATE INDEX IF NOT EXISTS idx_geo_entity_name ON geo_entity_names(name)")

        pg_client.conn.commit()

        # 批量插入
        from psycopg2.extras import execute_values
        data = [
            (e["name"], e["type"], e["category"], e["aliases"])
            for e in entities
        ]

        execute_values(cur, """
            INSERT INTO geo_entity_names (name, type, category, aliases)
            VALUES %s
            ON CONFLICT (name) DO UPDATE SET
                type = EXCLUDED.type,
                category = EXCLUDED.category,
                aliases = EXCLUDED.aliases
        """, data)

        pg_client.conn.commit()
        logger.info(f"已保存 {len(entities)} 条记录到 PostgreSQL")


def main(source: str = "geo"):
    """
    主函数

    Args:
        source: 数据来源
            - "geo": 提取地理实体 (shp2kg.py 创建的节点)
            - "corpus": 提取语料实体 (neo4j_client.py 创建的节点)
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

    # 如果需要保存到PostgreSQL（需要安装pgvector扩展）
    # pg_config = settings.get_postgres_config()
    # pg_client = PostgresClient(**pg_config)
    # save_to_postgres(entities, pg_client)
    # pg_client.close()

    neo4j_client.close()
    logger.success(f"提取完成，共 {len(entities)} 个实体")


if __name__ == "__main__":
    main()