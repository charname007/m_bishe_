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
            # 批次表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS extraction_batches (
                    batch_id VARCHAR(36) PRIMARY KEY,
                    corpus_count INTEGER,
                    worker_count INTEGER,
                    status VARCHAR(20),
                    created_at TIMESTAMP DEFAULT NOW(),
                    completed_at TIMESTAMP
                )
            """)

            # 实体表
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
                    e.get("corpus_ids", [])
                )
                for e in entities
            ]

            execute_values(cur, """
                INSERT INTO entities
                (batch_id, name, type, category, aliases, occurrence_count, corpus_ids)
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