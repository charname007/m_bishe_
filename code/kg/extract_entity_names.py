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


def extract_entity_names(neo4j_client: Neo4jClient) -> List[Dict]:
    """
    从Neo4j提取所有Entity节点的name和相关属性

    Returns:
        [{"name": "...", "type": "...", "category": "...", "aliases": [...]}]
    """
    with neo4j_client.driver.session() as session:
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

        logger.info(f"提取到 {len(entities)} 个实体")
        return entities


def save_to_json(entities: List[Dict], filepath: str = "entity_names.json"):
    """保存为JSON文件"""
    output_path = Path(__file__).parent.parent / filepath
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存到 {output_path}")
    return str(output_path)


def save_to_csv(entities: List[Dict], filepath: str = "entity_names.csv"):
    """保存为CSV文件"""
    import csv
    output_path = Path(__file__).parent.parent / filepath
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "type", "category", "aliases"])
        for e in entities:
            writer.writerow([e["name"], e["type"], e["category"], "|".join(e["aliases"])])
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


def main():
    """主函数"""
    # 连接Neo4j
    neo4j_config = settings.get_neo4j_config()
    neo4j_client = Neo4jClient(**neo4j_config)

    # 提取实体
    entities = extract_entity_names(neo4j_client)

    # 保存方式选择（可通过命令行参数或直接修改）
    # 默认保存为JSON文件，更方便后续embedding处理
    save_to_json(entities)
    save_to_csv(entities)  # 同时保存CSV备份

    # 如果需要保存到PostgreSQL（需要安装pgvector扩展）
    # pg_config = settings.get_postgres_config()
    # pg_client = PostgresClient(**pg_config)
    # save_to_postgres(entities, pg_client)
    # pg_client.close()

    neo4j_client.close()
    logger.success("提取完成")


if __name__ == "__main__":
    main()