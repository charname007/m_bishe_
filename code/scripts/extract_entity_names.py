"""
从Neo4j提取所有地理实体name，保存到文件/PostgreSQL
用于后续语义相似度比对

表结构说明：
- geo_entity_names: 地理实体表，包含 id, entity_id, name, type, embedding, created_at
- corpus_entity_names: 语料实体表，包含 id, name, type, category, aliases, embedding, created_at
- embedding 列维度由 settings.EMBEDDING_DIM 配置或模型实际维度决定

依赖关系：
- extract_entity_names.py 创建表结构（embedding 列初始为 NULL）
- embed_entity_names.py 计算并填充 embedding 数据
"""
import json
import sys
from pathlib import Path
from typing import List, Dict
from loguru import logger

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
from kg.neo4j_client import Neo4jClient
from kg.postgres_client import PostgresClient


def extract_entity_names(neo4j_client: Neo4jClient, source: str = "geo") -> List[Dict]:
    """
    从Neo4j提取所有实体的name和相关属性

    Args:
        source: 数据来源
            - "geo": 提取地理实体 (geo_entity_node:Road/Poi/Building/Block)
            - "corpus": 提取语料实体 (Entity标签)

    Returns:
        [{"name": "...", "type": "...", "entity_id": "..."}]
    """
    # 定义无效名称集合
    invalid_names = {"", "未命名", "null", "None", "nan"}

    with neo4j_client.driver.session() as session:
        if source == "geo":
            # 提取地理实体 (shp2kg.py 创建的节点)
            result = session.run("""
                MATCH (n:geo_entity_node)
                RETURN n.name as name, n.entity_type as type, n.entity_id as entity_id
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
            writer.writerow(["name", "type", "entity_id"])
            for e in entities:
                writer.writerow([e.get("name", ""), e.get("type", ""), e.get("entity_id", "")])
        else:
            writer.writerow(["name", "type", "category", "aliases"])
            for e in entities:
                writer.writerow([e.get("name", ""), e.get("type", ""), e.get("category", ""), "|".join(e.get("aliases", []))])
    logger.info(f"已保存到 {output_path}")
    return str(output_path)


def save_to_postgres(entities: List[Dict], pg_client: PostgresClient, source: str = "geo"):
    """
    保存到PostgreSQL

    Args:
        entities: 实体列表
        pg_client: PostgreSQL客户端
        source: 数据来源 ("geo" 或 "corpus")

    注意：
        - embedding 列初始为 NULL，由 embed_entity_names.py 脚本填充
        - 嵌入维度由 settings.EMBEDDING_DIM 配置（默认 768）
        - 如果维度需要修改，运行 embed_entity_names.py --modify-column
    """
    from psycopg2.extras import execute_values

    # 从配置获取嵌入维度
    embedding_dim = settings.get_embedding_config()["dim"]

    # 检查 pgvector 扩展是否可用
    has_vector = False
    try:
        with pg_client.conn.cursor() as cur:
            cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
            if cur.fetchone():
                has_vector = True
    except Exception:
        pass

    if not has_vector:
        try:
            with pg_client.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                pg_client.conn.commit()
                has_vector = True
                logger.info("pgvector扩展已安装")
        except Exception as e:
            logger.warning(f"无法安装pgvector扩展，将跳过embedding字段")
            pg_client.conn.rollback()  # 回滚失败的扩展创建
            has_vector = False

    with pg_client.conn.cursor() as cur:
        if source == "geo":
            # 地理实体表
            if has_vector:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS geo_entity_names (
                        id SERIAL PRIMARY KEY,
                        entity_id VARCHAR(100) NOT NULL UNIQUE,
                        name VARCHAR(200) NOT NULL,
                        type VARCHAR(50),
                        embedding VECTOR({embedding_dim}),
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
            else:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS geo_entity_names (
                        id SERIAL PRIMARY KEY,
                        entity_id VARCHAR(100) NOT NULL UNIQUE,
                        name VARCHAR(200) NOT NULL,
                        type VARCHAR(50),
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_geo_entity_name ON geo_entity_names(name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_geo_entity_type ON geo_entity_names(type)")

            pg_client.conn.commit()

            # 批量插入地理实体
            data = [
                (e.get("entity_id", ""), e.get("name", ""), e.get("type", ""))
                for e in entities
            ]

            execute_values(cur, """
                INSERT INTO geo_entity_names (entity_id, name, type)
                VALUES %s
                ON CONFLICT (entity_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    type = EXCLUDED.type
            """, data)
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
    main()