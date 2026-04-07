"""
Neo4j 图数据库客户端
"""
from typing import Dict, List, Optional
from neo4j import GraphDatabase
from loguru import logger


class Neo4jClient:
    """Neo4j客户端"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Neo4j连接已建立: {uri}")

    def close(self):
        """关闭连接"""
        self.driver.close()
        logger.info("Neo4j连接已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def create_indexes(self):
        """创建索引"""
        with self.driver.session() as session:
            # 实体名称索引
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)
            """)
            # 实体类型索引
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)
            """)
            logger.debug("Neo4j索引创建完成")

    def merge_entity(self, entity: Dict) -> bool:
        """
        合并实体节点

        Args:
            entity: {
                "name": "武汉大学",
                "type": "POI",
                "aliases": ["武大"],
                "category": "教育",
                "corpus_ids": ["001", "002"]
            }
        """
        with self.driver.session() as session:
            result = session.run("""
                MERGE (e:Entity {name: $name})
                ON CREATE SET
                    e.type = $type,
                    e.category = $category,
                    e.aliases = $aliases,
                    e.corpus_ids = $corpus_ids,
                    e.created_at = datetime(),
                    e.source = 'xiaohongshu'
                ON MATCH SET
                    e.aliases = CASE
                        WHEN $aliases IS NOT NULL AND size($aliases) > 0
                        THEN apoc.coll.toSet(e.aliases + $aliases)
                        ELSE e.aliases
                    END,
                    e.corpus_ids = CASE
                        WHEN $corpus_ids IS NOT NULL AND size($corpus_ids) > 0
                        THEN apoc.coll.toSet(e.corpus_ids + $corpus_ids)
                        ELSE e.corpus_ids
                    END,
                    e.updated_at = datetime()
                RETURN e
            """,
                name=entity["name"],
                type=entity.get("type", "Unknown"),
                category=entity.get("category", ""),
                aliases=entity.get("aliases", []),
                corpus_ids=entity.get("corpus_ids", [])
            )
            return result.single() is not None

    def merge_relation(self, triple: Dict) -> bool:
        """
        合并关系

        Args:
            triple: {
                "head": "武汉大学",
                "relation": "位于",
                "tail": "珞喻路",
                "evidence": "武汉大学在珞喻路上",
                "corpus_ids": ["001"],
                "relation_type": "位置关系",
                "relation_subtype": "内部"
            }
        """
        with self.driver.session() as session:
            # 先确保头尾实体存在
            session.run("""
                MERGE (:Entity {name: $head})
                MERGE (:Entity {name: $tail})
            """, head=triple["head"], tail=triple["tail"])

            # 创建关系
            result = session.run("""
                MATCH (h:Entity {name: $head})
                MATCH (t:Entity {name: $tail})
                MERGE (h)-[r:RELATION {type: $relation}]->(t)
                ON CREATE SET
                    r.evidence = $evidence,
                    r.corpus_ids = $corpus_ids,
                    r.relation_type = $relation_type,
                    r.relation_subtype = $relation_subtype,
                    r.created_at = datetime(),
                    r.source = 'xiaohongshu'
                ON MATCH SET
                    r.corpus_ids = CASE
                        WHEN $corpus_ids IS NOT NULL AND size($corpus_ids) > 0
                        THEN apoc.coll.toSet(r.corpus_ids + $corpus_ids)
                        ELSE r.corpus_ids
                    END,
                    r.relation_type = CASE
                        WHEN $relation_type IS NOT NULL AND $relation_type <> ''
                        THEN $relation_type
                        ELSE r.relation_type
                    END,
                    r.relation_subtype = CASE
                        WHEN $relation_subtype IS NOT NULL AND $relation_subtype <> ''
                        THEN $relation_subtype
                        ELSE r.relation_subtype
                    END,
                    r.updated_at = datetime()
                RETURN r
            """,
                head=triple["head"],
                relation=triple["relation"],
                tail=triple["tail"],
                evidence=triple.get("evidence", ""),
                corpus_ids=triple.get("corpus_ids", []),
                relation_type=triple.get("relation_type", ""),
                relation_subtype=triple.get("relation_subtype", "")
            )
            return result.single() is not None

    def batch_merge_entities(self, entities: List[Dict]) -> Dict:
        """批量合并实体"""
        success_count = 0
        for entity in entities:
            try:
                if self.merge_entity(entity):
                    success_count += 1
            except Exception as e:
                logger.error(f"合并实体失败 {entity['name']}: {e}")

        logger.info(f"实体合并完成: {success_count}/{len(entities)}")
        return {"merged": success_count, "total": len(entities)}

    def batch_merge_relations(self, triples: List[Dict]) -> Dict:
        """批量合并关系"""
        success_count = 0
        for triple in triples:
            try:
                if self.merge_relation(triple):
                    success_count += 1
            except Exception as e:
                logger.error(f"合并关系失败 {triple}: {e}")

        logger.info(f"关系合并完成: {success_count}/{len(triples)}")
        return {"merged": success_count, "total": len(triples)}

    def query_entity(self, name: str) -> Optional[Dict]:
        """查询实体"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: $name})
                RETURN e
            """, name=name)
            record = result.single()
            if record:
                return dict(record["e"])
            return None

    def query_relations(self, entity_name: str) -> List[Dict]:
        """查询实体相关的关系"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: $name})-[r:RELATION]-(other)
                RETURN e.name as head, r.type as relation, other.name as tail, r.evidence as evidence
            """, name=entity_name)
            return [dict(record) for record in result]

    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.driver.session() as session:
            entity_result = session.run("MATCH (e:Entity) RETURN count(e) as count").single()
            relation_result = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count").single()

            entity_count = entity_result["count"] if entity_result else 0
            relation_count = relation_result["count"] if relation_result else 0

            return {
                "entity_count": entity_count,
                "relation_count": relation_count
            }