"""
PostgreSQL 关系数据库客户端
"""
import json
import psycopg2
from psycopg2.extras import execute_values
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger


class PostgresClient:
    """PostgreSQL客户端"""

    def __init__(self, host: str, port: int, database: str,
                 user: str, password: str):
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password
        }
        self.conn = None
        self.connect()
        logger.info(f"PostgreSQL连接已建立: {host}:{port}/{database}")

    def connect(self):
        """建立连接"""
        self.conn = psycopg2.connect(**self.conn_params)

    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()
            logger.info("PostgreSQL连接已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def create_tables(self):
        """创建表结构"""
        with self.conn.cursor() as cur:
            # 批次表 - 添加 neo4j_sync 和 error_message 字段用于事务追踪
            cur.execute("""
                CREATE TABLE IF NOT EXISTS extraction_batches (
                    batch_id VARCHAR(36) PRIMARY KEY,
                    corpus_count INTEGER,
                    worker_count INTEGER,
                    status VARCHAR(20),
                    neo4j_sync BOOLEAN DEFAULT FALSE,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    completed_at TIMESTAMP
                )
            """)

            # 实体表 - 添加 attrs 列用于存储实体属性
            cur.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id SERIAL PRIMARY KEY,
                    batch_id VARCHAR(36),
                    name VARCHAR(200) NOT NULL,
                    type VARCHAR(20),
                    category VARCHAR(50),
                    aliases TEXT[],
                    occurrence_count INTEGER,
                    corpus_ids TEXT[],
                    attrs JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    FOREIGN KEY (batch_id) REFERENCES extraction_batches(batch_id)
                )
            """)

            # 三元组表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS triples (
                    id SERIAL PRIMARY KEY,
                    batch_id VARCHAR(36),
                    head_entity VARCHAR(200),
                    relation VARCHAR(50),
                    tail_entity VARCHAR(200),
                    evidence TEXT,
                    sem_score INTEGER,
                    fac_score INTEGER,
                    con_score INTEGER,
                    passed_eval BOOLEAN,
                    relation_type VARCHAR(50),
                    relation_subtype VARCHAR(50),
                    corpus_ids TEXT[],
                    created_at TIMESTAMP DEFAULT NOW(),
                    FOREIGN KEY (batch_id) REFERENCES extraction_batches(batch_id)
                )
            """)

            # 语料来源表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS corpus_sources (
                    corpus_id VARCHAR(50) PRIMARY KEY,
                    batch_id VARCHAR(36),
                    raw_text TEXT,
                    entities JSONB,
                    triples JSONB,
                    processed_at TIMESTAMP DEFAULT NOW(),
                    FOREIGN KEY (batch_id) REFERENCES extraction_batches(batch_id)
                )
            """)

            # 创建索引
            cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_batch ON entities(batch_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_triples_batch ON triples(batch_id)")

            # 添加新列（如果不存在）- 兼容旧表结构
            # triples 表新增 relation_type 和 relation_subtype 列
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'triples'
                AND table_schema = CURRENT_SCHEMA()
                AND column_name IN ('relation_type', 'relation_subtype')
            """)
            existing_columns = [row[0] for row in cur.fetchall()]
            if 'relation_type' not in existing_columns:
                cur.execute("ALTER TABLE triples ADD COLUMN relation_type VARCHAR(50)")
            if 'relation_subtype' not in existing_columns:
                cur.execute("ALTER TABLE triples ADD COLUMN relation_subtype VARCHAR(50)")

            # extraction_batches 表新增 neo4j_sync 和 error_message 列
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'extraction_batches'
                AND table_schema = CURRENT_SCHEMA()
                AND column_name IN ('neo4j_sync', 'error_message')
            """)
            batch_columns = [row[0] for row in cur.fetchall()]
            if 'neo4j_sync' not in batch_columns:
                cur.execute("ALTER TABLE extraction_batches ADD COLUMN neo4j_sync BOOLEAN DEFAULT FALSE")
            if 'error_message' not in batch_columns:
                cur.execute("ALTER TABLE extraction_batches ADD COLUMN error_message TEXT")

            # entities 表新增 attrs 列
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'entities'
                AND table_schema = CURRENT_SCHEMA()
                AND column_name = 'attrs'
            """)
            if not cur.fetchone():
                cur.execute("ALTER TABLE entities ADD COLUMN attrs JSONB")

            self.conn.commit()
            logger.debug("PostgreSQL表结构创建完成")

    def insert_batch(self, batch_id: str, corpus_count: int,
                     worker_count: int) -> bool:
        """插入批次记录"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO extraction_batches (batch_id, corpus_count, worker_count, status)
                VALUES (%s, %s, %s, %s)
            """, (batch_id, corpus_count, worker_count, "processing"))
            self.conn.commit()
            return True

    def update_batch_status(self, batch_id: str, status: str):
        """更新批次状态"""
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE extraction_batches
                SET status = %s, completed_at = %s
                WHERE batch_id = %s
            """, (status, datetime.now(), batch_id))
            self.conn.commit()

    def update_batch_status_with_error(self, batch_id: str, status: str, error_message: str = None):
        """更新批次状态并记录错误信息"""
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE extraction_batches
                SET status = %s, completed_at = %s, error_message = %s
                WHERE batch_id = %s
            """, (status, datetime.now(), error_message, batch_id))
            self.conn.commit()

    def update_neo4j_sync_status(self, batch_id: str, synced: bool):
        """更新 Neo4j 同步状态"""
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE extraction_batches
                SET neo4j_sync = %s
                WHERE batch_id = %s
            """, (synced, batch_id))
            self.conn.commit()

    def get_batch_status(self, batch_id: str) -> Optional[Dict]:
        """获取批次状态详情"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT status, neo4j_sync, error_message, created_at, completed_at
                FROM extraction_batches
                WHERE batch_id = %s
            """, (batch_id,))
            row = cur.fetchone()
            if row:
                return {
                    "status": row[0],
                    "neo4j_sync": row[1],
                    "error_message": row[2],
                    "created_at": row[3],
                    "completed_at": row[4]
                }
            return None

    def insert_entities(self, batch_id: str, entities: List[Dict]) -> int:
        """批量插入实体"""
        if not entities:
            return 0

        with self.conn.cursor() as cur:
            data = [
                (
                    batch_id,
                    e["name"],
                    e.get("type", ""),
                    e.get("category", ""),
                    e.get("aliases", []),
                    e.get("occurrence_count", 1),
                    e.get("corpus_ids", []),
                    json.dumps(e.get("attrs", {}), ensure_ascii=False)  # 新增 attrs 字段
                )
                for e in entities
            ]

            execute_values(cur, """
                INSERT INTO entities
                (batch_id, name, type, category, aliases, occurrence_count, corpus_ids, attrs)
                VALUES %s
            """, data)

            self.conn.commit()
            logger.info(f"插入 {len(entities)} 个实体")
            return len(entities)

    def insert_triples(self, batch_id: str, triples: List[Dict]) -> int:
        """批量插入三元组"""
        if not triples:
            return 0

        with self.conn.cursor() as cur:
            data = [
                (
                    batch_id,
                    t["head"],
                    t["relation"],
                    t["tail"],
                    t.get("evidence", ""),
                    t.get("sem_score", 0),
                    t.get("fac_score", 0),
                    t.get("con_score", 0),
                    t.get("passed_eval", True),
                    t.get("relation_type", ""),
                    t.get("relation_subtype", ""),
                    t.get("corpus_ids", [])
                )
                for t in triples
            ]

            execute_values(cur, """
                INSERT INTO triples
                (batch_id, head_entity, relation, tail_entity, evidence,
                 sem_score, fac_score, con_score, passed_eval, relation_type, relation_subtype, corpus_ids)
                VALUES %s
            """, data)

            self.conn.commit()
            logger.info(f"插入 {len(triples)} 个三元组")
            return len(triples)

    def insert_corpus_sources(self, batch_id: str,
                              corpus_states: List[Any]) -> int:
        """插入语料来源"""
        with self.conn.cursor() as cur:
            for state in corpus_states:
                # 使用 .get() 防止 KeyError，兼容失败语料
                cur.execute("""
                    INSERT INTO corpus_sources
                    (corpus_id, batch_id, raw_text, entities, triples)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (corpus_id) DO UPDATE SET
                        batch_id = EXCLUDED.batch_id,
                        raw_text = EXCLUDED.raw_text,
                        entities = EXCLUDED.entities,
                        triples = EXCLUDED.triples
                """, (
                    state.get("corpus_id", "unknown"),
                    batch_id,
                    state.get("raw_text", ""),
                    json.dumps(state.get("entities", {}), ensure_ascii=False),
                    json.dumps(state.get("triples", []), ensure_ascii=False)
                ))

            self.conn.commit()
            return len(corpus_states)

    def query_entities(self, batch_id: str) -> List[Dict]:
        """查询批次实体"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT name, type, category, aliases, occurrence_count
                FROM entities
                WHERE batch_id = %s
            """, (batch_id,))
            return [
                {
                    "name": row[0],
                    "type": row[1],
                    "category": row[2],
                    "aliases": row[3],
                    "occurrence_count": row[4]
                }
                for row in cur.fetchall()
            ]

    def query_triples(self, batch_id: str) -> List[Dict]:
        """查询批次三元组"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT head_entity, relation, tail_entity, evidence, passed_eval
                FROM triples
                WHERE batch_id = %s
            """, (batch_id,))
            return [
                {
                    "head": row[0],
                    "relation": row[1],
                    "tail": row[2],
                    "evidence": row[3],
                    "passed_eval": row[4]
                }
                for row in cur.fetchall()
            ]

    def get_stats(self, batch_id: str) -> Dict:
        """获取批次统计"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM entities WHERE batch_id = %s", (batch_id,))
            entity_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM triples WHERE batch_id = %s", (batch_id,))
            triple_count = cur.fetchone()[0]

            return {
                "entity_count": entity_count,
                "triple_count": triple_count
            }

    def fetch_corpus_for_extraction(
        self,
        table_name: str = "xiaohongshu_notes",
        text_column: str = "desc_cleaned",
        id_column: str = "note_id",
        limit: int = 100,
        offset: int = 0,
        where_clause: Optional[str] = None
    ) -> List[Dict]:
        """
        分页读取语料用于知识图谱抽取

        Args:
            table_name: 数据表名
            text_column: 文本列名
            id_column: ID列名
            limit: 每次读取数量
            offset: 偏移量（分页）
            where_clause: 可选的WHERE条件（不含WHERE关键字）

        Returns:
            语料列表 [{"id": str, "text": str}, ...]
        """
        with self.conn.cursor() as cur:
            base_query = f"SELECT {id_column}, {text_column} FROM {table_name}"
            if where_clause:
                base_query += f" WHERE {where_clause}"
            query = f"{base_query} ORDER BY {id_column} LIMIT %s OFFSET %s"

            cur.execute(query, (limit, offset))
            rows = cur.fetchall()

            return [
                {"id": row[0], "text": row[1] or ""}
                for row in rows
                if row[1] and len(str(row[1]).strip()) > 0  # 过滤空文本
            ]

    def count_corpus_for_extraction(
        self,
        table_name: str = "xiaohongshu_notes",
        text_column: str = "desc_cleaned",
        where_clause: Optional[str] = None
    ) -> int:
        """
        统计待处理语料总数

        Args:
            table_name: 数据表名
            text_column: 文本列名
            where_clause: 可选的WHERE条件

        Returns:
            语料总数
        """
        with self.conn.cursor() as cur:
            query = f"SELECT COUNT(*) FROM {table_name} WHERE {text_column} IS NOT NULL AND {text_column} != ''"
            if where_clause:
                query += f" AND {where_clause}"
            cur.execute(query)
            return cur.fetchone()[0]

    # ===== P12新增：实体对齐相关方法 =====

    def insert_new_geo_entity(self, entity: Dict, embedding: List[float] = None) -> Optional[str]:
        """
        将新实体插入 geo_entity_names 表（来自小红书抽取）

        Args:
            entity: {
                "entity_id": "poi_new_001",  # 可选，如果不提供会自动生成
                "name": "某某咖啡店",
                "type": "poi",
                "longitude": Optional[float],
                "latitude": Optional[float],
                "aliases": Optional[List[str]]
            }
            embedding: 实体名称的向量嵌入（可选）

        Returns:
            新创建的 entity_id，或 None
        """
        import re

        with self.conn.cursor() as cur:
            # 如果没有提供 entity_id，自动生成
            entity_id = entity.get("entity_id")
            if not entity_id:
                # 获取同类型的最大编号
                type_ = entity.get("type", "poi")
                cur.execute("""
                    SELECT entity_id FROM geo_entity_names
                    WHERE type = %s
                    ORDER BY entity_id DESC
                    LIMIT 1
                """, (type_))
                row = cur.fetchone()
                if row and row[0]:
                    match = re.search(r'(\d+)$', row[0])
                    max_num = int(match.group(1)) if match else 0
                else:
                    max_num = 0

                type_prefix = type_.lower() if type_ else "poi"
                entity_id = f"{type_prefix}_{max_num + 1}"

            # 插入记录（使用RETURNING语法，ON CONFLICT时返回NULL）
            cur.execute("""
                INSERT INTO geo_entity_names
                (entity_id, name, type, longitude, latitude, aliases, embedding, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (entity_id) DO NOTHING
                RETURNING entity_id
            """,
                entity_id,
                entity["name"],
                entity.get("type", "poi"),
                entity.get("longitude"),
                entity.get("latitude"),
                entity.get("aliases", []),
                embedding,  # embedding向量
                "xiaohongshu"  # 来源标记
            )

            row = cur.fetchone()
            if row and row[0]:
                self.conn.commit()
                logger.info(f"[Postgres] 创建新geo_entity: {entity_id} - {entity['name']}")
                return row[0]
            else:
                # entity_id 已存在，尝试生成新的
                self.conn.rollback()
                return None

    def batch_insert_new_geo_entities(self, entities: List[Dict],
                                        embeddings: List[List[float]] = None) -> Dict:
        """
        批量插入新实体到 geo_entity_names

        Args:
            entities: 实体列表
            embeddings: 对应的embedding列表（可选）

        Returns:
            {"created": count, "entity_ids": [id1, id2, ...]}
        """
        if not entities:
            return {"created": 0, "entity_ids": []}

        created_ids = []
        for i, entity in enumerate(entities):
            try:
                emb = embeddings[i] if embeddings and i < len(embeddings) else None
                entity_id = self.insert_new_geo_entity(entity, emb)
                if entity_id:
                    created_ids.append(entity_id)
            except Exception as e:
                logger.error(f"插入实体失败 {entity['name']}: {e}")

        return {"created": len(created_ids), "entity_ids": created_ids}