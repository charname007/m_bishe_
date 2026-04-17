"""
Neo4j 图数据库客户端
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Any
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

    @staticmethod
    def _is_neo4j_scalar(value: Any) -> bool:
        """判断值是否为 Neo4j 支持的标量属性类型。"""
        return isinstance(value, (str, int, float, bool))

    @classmethod
    def _normalize_attrs_map(cls, attrs: Any) -> Dict[str, Any]:
        """
        将 attrs 规范化为可直接 SET n += map 的字典。
        - 标量保留
        - 标量列表保留（过滤 None）
        - 复杂结构转 JSON 字符串
        - 过滤与实体基础字段冲突的 key，避免覆盖核心属性
        """
        if not isinstance(attrs, dict):
            return {}

        reserved_keys = {
            "name",
            "type",
            "category",
            "aliases",
            "corpus_ids",
            "created_at",
            "updated_at",
            "source",
            "attrs",
        }
        normalized: Dict[str, Any] = {}

        for key, value in attrs.items():
            if not isinstance(key, str) or not key or key in reserved_keys:
                continue
            if value is None:
                continue
            if cls._is_neo4j_scalar(value):
                normalized[key] = value
                continue
            if isinstance(value, (list, tuple, set)):
                values = [v for v in value if v is not None]
                if all(cls._is_neo4j_scalar(v) for v in values):
                    normalized[key] = values
                else:
                    normalized[key] = json.dumps(value, ensure_ascii=False)
                continue
            normalized[key] = json.dumps(value, ensure_ascii=False)

        return normalized

    @classmethod
    def _normalize_attr_list_value(cls, value: Any) -> List[Any]:
        """将任意属性值归一化为“可追加”的列表值。"""
        if value is None:
            return []

        if isinstance(value, (list, tuple, set)):
            raw_values = [v for v in value if v is not None]
        else:
            raw_values = [value]

        normalized: List[Any] = []
        for item in raw_values:
            if cls._is_neo4j_scalar(item):
                normalized.append(item)
            else:
                normalized.append(json.dumps(item, ensure_ascii=False))

        # 保序去重，避免重复追加导致列表膨胀
        return list(dict.fromkeys(normalized))

    @classmethod
    def _merge_attr_maps_for_append(
        cls, existing_props: Dict[str, Any], incoming_attrs: Any
    ) -> Dict[str, Any]:
        """
        将 incoming_attrs 合并为“追加模式”属性：
        - 已存在属性：转列表后追加新值
        - 不存在属性：以列表形式写入
        """
        normalized_incoming = cls._normalize_attrs_map(incoming_attrs)
        merged: Dict[str, Any] = {}

        for key, incoming_value in normalized_incoming.items():
            existing_list = cls._normalize_attr_list_value(existing_props.get(key))
            incoming_list = cls._normalize_attr_list_value(incoming_value)
            if not incoming_list:
                continue
            merged[key] = list(dict.fromkeys(existing_list + incoming_list))

        return merged

    @staticmethod
    def _parse_json_dict(value: Any) -> Dict[str, Any]:
        """将 JSON 字符串或字典安全解析为字典。"""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    @classmethod
    def _merge_attr_dicts_for_append(
        cls, existing_attrs: Any, incoming_attrs: Any
    ) -> Dict[str, Any]:
        """
        合并两个 attrs 字典为“列表追加去重”结果。
        - 保留 existing 的全部字段
        - incoming 同名字段做追加去重
        """
        existing_norm = cls._normalize_attrs_map(existing_attrs)
        incoming_norm = cls._normalize_attrs_map(incoming_attrs)

        merged: Dict[str, Any] = {}
        for key, value in existing_norm.items():
            values = cls._normalize_attr_list_value(value)
            if values:
                merged[key] = values

        for key, value in incoming_norm.items():
            incoming_values = cls._normalize_attr_list_value(value)
            if not incoming_values:
                continue
            base_values = merged.get(key, [])
            merged[key] = list(dict.fromkeys(base_values + incoming_values))

        return merged

    @classmethod
    def _merge_relation_attrs_json(cls, existing_raw: Any, incoming_raw: Any) -> str:
        """合并关系属性并输出 JSON 字符串（字段值统一为列表）。"""
        existing_dict = cls._parse_json_dict(existing_raw)
        merged_dict = cls._merge_attr_dicts_for_append(existing_dict, incoming_raw)
        if not merged_dict:
            return ""
        return json.dumps(merged_dict, ensure_ascii=False)

    @staticmethod
    def _fetch_existing_props_map(
        session, label_expr: str, key_field: str, key_values: List[Any]
    ) -> Dict[str, Dict[str, Any]]:
        """批量读取已有节点属性，用于属性追加合并。"""
        keys = [k for k in key_values if k is not None and str(k) != ""]
        if not keys:
            return {}

        query = f"""
            UNWIND $keys AS lookup_key
            MATCH (e:{label_expr})
            WHERE e.{key_field} = lookup_key
            RETURN lookup_key, properties(e) AS props
        """
        result = session.run(query, keys=keys)
        return {
            str(record["lookup_key"]): (record["props"] or {})
            for record in result
        }

    def merge_entity(self, entity: Dict) -> bool:
        """
        合并实体节点

        Args:
            entity: {
                "name": "武汉大学",
                "type": "POI",
                "aliases": ["武大"],
                "category": "教育",
                "corpus_ids": ["001", "002"],
                "attrs": {"类别": "高校", "推荐指数": 5}  # 新增：实体属性
            }
        """
        with self.driver.session() as session:
            existing_record = session.run(
                "MATCH (e:Entity {name: $name}) RETURN properties(e) AS props",
                name=entity["name"],
            ).single()
            existing_props = existing_record["props"] if existing_record else {}
            attrs_map = self._merge_attr_maps_for_append(
                existing_props, entity.get("attrs", {})
            )

            result = session.run(
                """
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
                        THEN coll.distinct(coalesce(e.aliases, []) + $aliases)
                        ELSE coalesce(e.aliases, [])
                    END,
                    e.corpus_ids = CASE
                        WHEN $corpus_ids IS NOT NULL AND size($corpus_ids) > 0
                        THEN coll.distinct(coalesce(e.corpus_ids, []) + $corpus_ids)
                        ELSE coalesce(e.corpus_ids, [])
                    END,
                    e.updated_at = datetime()
                SET e += $attrs_map
                RETURN e
            """,
                name=entity["name"],
                type=entity.get("type", "Unknown"),
                category=entity.get("category", ""),
                aliases=entity.get("aliases", []),
                corpus_ids=entity.get("corpus_ids", []),
                attrs_map=attrs_map,
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
                "relation_subtype": "内部",
                "relation_attrs": {"距离值": "近", "方向值": "东"}  # 新增：关系属性
            }
        """
        with self.driver.session() as session:
            # 先确保头尾实体存在（设置基本属性避免空壳实体）
            session.run(
                """
                MERGE (e1:Entity {name: $head})
                ON CREATE SET
                    e1.type = 'Unknown',
                    e1.category = '',
                    e1.aliases = [],
                    e1.corpus_ids = [],
                    e1.created_at = datetime(),
                    e1.source = 'xiaohongshu'
                MERGE (e2:Entity {name: $tail})
                ON CREATE SET
                    e2.type = 'Unknown',
                    e2.category = '',
                    e2.aliases = [],
                    e2.corpus_ids = [],
                    e2.created_at = datetime(),
                    e2.source = 'xiaohongshu'
            """,
                head=triple["head"],
                tail=triple["tail"],
            )

            # 创建关系
            existing_record = session.run(
                """
                MATCH (h:Entity {name: $head})-[r:RELATION {type: $relation}]->(t:Entity {name: $tail})
                RETURN r.relation_attrs AS relation_attrs
                """,
                head=triple["head"],
                relation=triple["relation"],
                tail=triple["tail"],
            ).single()
            existing_relation_attrs = (
                existing_record["relation_attrs"] if existing_record else None
            )
            relation_attrs_json = self._merge_relation_attrs_json(
                existing_relation_attrs, triple.get("relation_attrs", {})
            )

            result = session.run(
                """
                MATCH (h:Entity {name: $head})
                MATCH (t:Entity {name: $tail})
                MERGE (h)-[r:RELATION {type: $relation}]->(t)
                ON CREATE SET
                    r.evidence = $evidence,
                    r.corpus_ids = $corpus_ids,
                    r.relation_type = $relation_type,
                    r.relation_subtype = $relation_subtype,
                    r.relation_attrs = $relation_attrs,
                    r.created_at = datetime(),
                    r.source = 'xiaohongshu'
                ON MATCH SET
                    r.corpus_ids = CASE
                        WHEN $corpus_ids IS NOT NULL AND size($corpus_ids) > 0
                        THEN coll.distinct(coalesce(r.corpus_ids, []) + $corpus_ids)
                        ELSE coalesce(r.corpus_ids, [])
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
                    r.relation_attrs = $relation_attrs,
                    r.updated_at = datetime()
                RETURN r
            """,
                head=triple["head"],
                relation=triple["relation"],
                tail=triple["tail"],
                evidence=triple.get("evidence", ""),
                corpus_ids=triple.get("corpus_ids", []),
                relation_type=triple.get("relation_type", ""),
                relation_subtype=triple.get("relation_subtype", ""),
                relation_attrs=relation_attrs_json,
            )
            return result.single() is not None

    def batch_merge_entities(self, entities: List[Dict]) -> Dict:
        """
        批量合并实体 - 使用 UNWIND 批量操作

        P15改进：根据实体类型使用不同的节点标签
        - 地理实体（道路/POI/建筑物/街区）→ geo_entity_node:Entity
        - 功能实体 → FunctionNode:Entity
        - 事件实体 → EventNode:Entity
        - 其他 → Entity

        性能优化：按类型分组批量处理
        """
        if not entities:
            return {"merged": 0, "total": 0}

        # P15改进：按实体类型分组
        GEO_TYPES = {"道路", "POI", "建筑物", "街区"}

        # P22新增：区分已对齐实体（有db_entity_id）和未对齐实体
        aligned_entities = []  # 已对齐到数据库的实体
        new_entities = []      # 新抽取的实体

        for e in entities:
            db_entity_id = e.get("db_entity_id", "")
            if db_entity_id:
                # 已对齐实体：使用数据库ID匹配
                aligned_entities.append(e)
            else:
                # 新实体：使用name匹配
                new_entities.append(e)

        # 按类型分组未对齐实体（保持原有逻辑）
        groups = {
            "geo_entity_node": [],
            "FunctionNode": [],
            "EventNode": [],
            "Entity": [],
        }

        for e in new_entities:
            entity_type = e.get("type", "Unknown")
            if entity_type in GEO_TYPES:
                groups["geo_entity_node"].append(e)
            elif entity_type == "功能":
                groups["FunctionNode"].append(e)
            elif entity_type == "事件":
                groups["EventNode"].append(e)
            else:
                groups["Entity"].append(e)

        total_merged = 0

        try:
            with self.driver.session() as session:
                # P22新增：优先处理已对齐实体（使用数据库ID匹配）
                if aligned_entities:
                    # 区分 geo_entity_names 和 amap_poi 来源
                    geo_aligned = []  # entity_id 格式
                    amap_aligned = [] # original_id 格式 (以amap_开头)

                    for e in aligned_entities:
                        db_id = e.get("db_entity_id", "")
                        if db_id.startswith("amap_"):
                            amap_aligned.append(e)
                        else:
                            geo_aligned.append(e)

                    # 处理 geo_entity_names 对齐（用 entity_id 匹配）
                    if geo_aligned:
                        geo_existing_map = self._fetch_existing_props_map(
                            session,
                            "geo_entity_node",
                            "entity_id",
                            [e.get("db_entity_id", "") for e in geo_aligned],
                        )
                        batch_data = []
                        for e in geo_aligned:
                            entity_id = e.get("db_entity_id", "")
                            attrs_map = self._merge_attr_maps_for_append(
                                geo_existing_map.get(str(entity_id), {}),
                                e.get("attrs", {}),
                            )
                            batch_data.append(
                                {
                                    "entity_id": entity_id,
                                    "name": e.get("name", ""),
                                    "type": e.get("type", "POI"),
                                    "aliases": e.get("aliases", []),
                                    "corpus_ids": e.get("corpus_ids", []),
                                    "attrs_map": attrs_map,
                                    "source": e.get("source", "geo_entity_names"),
                                }
                            )
                        result = session.run(
                            """
                            UNWIND $entities AS entity
                            MATCH (e:geo_entity_node {entity_id: entity.entity_id})
                            SET e.aliases = CASE
                                WHEN entity.aliases IS NOT NULL AND size(entity.aliases) > 0
                                THEN coll.distinct(coalesce(e.aliases, []) + entity.aliases)
                                ELSE coalesce(e.aliases, [])
                            END,
                            e.corpus_ids = CASE
                                WHEN entity.corpus_ids IS NOT NULL AND size(entity.corpus_ids) > 0
                                THEN coll.distinct(coalesce(e.corpus_ids, []) + entity.corpus_ids)
                                ELSE coalesce(e.corpus_ids, [])
                            END,
                            e.updated_at = datetime()
                            SET e += entity.attrs_map
                            RETURN count(e) as matched_count
                            """,
                            entities=batch_data,
                        )
                        record = result.single()
                        total_merged += record["matched_count"] if record else 0
                        logger.debug(f"[Neo4j] geo_aligned: {len(geo_aligned)} entities matched by entity_id")

                    # 处理 amap_poi 对齐（用 original_id 匹配）
                    if amap_aligned:
                        amap_existing_map = self._fetch_existing_props_map(
                            session,
                            "geo_entity_node",
                            "original_id",
                            [e.get("db_entity_id", "") for e in amap_aligned],
                        )
                        batch_data = []
                        for e in amap_aligned:
                            original_id = e.get("db_entity_id", "")
                            attrs_map = self._merge_attr_maps_for_append(
                                amap_existing_map.get(str(original_id), {}),
                                e.get("attrs", {}),
                            )
                            batch_data.append(
                                {
                                    "original_id": original_id,
                                    "name": e.get("name", ""),
                                    "type": e.get("type", "POI"),
                                    "aliases": e.get("aliases", []),
                                    "corpus_ids": e.get("corpus_ids", []),
                                    "attrs_map": attrs_map,
                                }
                            )
                        result = session.run(
                            """
                            UNWIND $entities AS entity
                            MATCH (e:geo_entity_node {original_id: entity.original_id})
                            SET e.aliases = CASE
                                WHEN entity.aliases IS NOT NULL AND size(entity.aliases) > 0
                                THEN coll.distinct(coalesce(e.aliases, []) + entity.aliases)
                                ELSE coalesce(e.aliases, [])
                            END,
                            e.corpus_ids = CASE
                                WHEN entity.corpus_ids IS NOT NULL AND size(entity.corpus_ids) > 0
                                THEN coll.distinct(coalesce(e.corpus_ids, []) + entity.corpus_ids)
                                ELSE coalesce(e.corpus_ids, [])
                            END,
                            e.updated_at = datetime()
                            SET e += entity.attrs_map
                            RETURN count(e) as matched_count
                            """,
                            entities=batch_data,
                        )
                        record = result.single()
                        total_merged += record["matched_count"] if record else 0
                        logger.debug(f"[Neo4j] amap_aligned: {len(amap_aligned)} entities matched by original_id")

                # 处理地理实体组（geo_entity_node:Entity 双标签）
                if groups["geo_entity_node"]:
                    geo_name_existing_map = self._fetch_existing_props_map(
                        session,
                        "geo_entity_node:Entity",
                        "name",
                        [e.get("name", "") for e in groups["geo_entity_node"]],
                    )
                    batch_data = []
                    for e in groups["geo_entity_node"]:
                        name = e.get("name", "")
                        attrs_map = self._merge_attr_maps_for_append(
                            geo_name_existing_map.get(str(name), {}),
                            e.get("attrs", {}),
                        )
                        batch_data.append(
                            {
                                "name": name,
                                "type": e.get("type", "Unknown"),
                                "category": e.get("category", ""),
                                "aliases": e.get("aliases", []),
                                "corpus_ids": e.get("corpus_ids", []),
                                "attrs_map": attrs_map,
                            }
                        )
                    result = session.run(
                        """
                        UNWIND $entities AS entity
                        MERGE (e:geo_entity_node:Entity {name: entity.name})
                        ON CREATE SET
                            e.type = entity.type,
                            e.category = entity.category,
                            e.aliases = entity.aliases,
                            e.corpus_ids = entity.corpus_ids,
                            e.created_at = datetime(),
                            e.source = 'xiaohongshu'
                        ON MATCH SET
                            e.aliases = CASE
                                WHEN entity.aliases IS NOT NULL AND size(entity.aliases) > 0
                                THEN coll.distinct(coalesce(e.aliases, []) + entity.aliases)
                                ELSE coalesce(e.aliases, [])
                            END,
                            e.corpus_ids = CASE
                                WHEN entity.corpus_ids IS NOT NULL AND size(entity.corpus_ids) > 0
                                THEN coll.distinct(coalesce(e.corpus_ids, []) + entity.corpus_ids)
                                ELSE coalesce(e.corpus_ids, [])
                            END,
                            e.updated_at = datetime()
                        SET e += entity.attrs_map
                        RETURN count(e) as merged_count
                    """,
                        entities=batch_data,
                    )
                    record = result.single()
                    total_merged += record["merged_count"] if record else 0

                # 处理功能实体组（FunctionNode:Entity 双标签）
                if groups["FunctionNode"]:
                    function_existing_map = self._fetch_existing_props_map(
                        session,
                        "FunctionNode:Entity",
                        "name",
                        [e.get("name", "") for e in groups["FunctionNode"]],
                    )
                    batch_data = []
                    for e in groups["FunctionNode"]:
                        name = e.get("name", "")
                        attrs_map = self._merge_attr_maps_for_append(
                            function_existing_map.get(str(name), {}),
                            e.get("attrs", {}),
                        )
                        batch_data.append(
                            {
                                "name": name,
                                "type": e.get("type", "功能"),
                                "category": e.get("category", ""),
                                "aliases": e.get("aliases", []),
                                "corpus_ids": e.get("corpus_ids", []),
                                "attrs_map": attrs_map,
                            }
                        )
                    result = session.run(
                        """
                        UNWIND $entities AS entity
                        MERGE (e:FunctionNode:Entity {name: entity.name})
                        ON CREATE SET
                            e.type = entity.type,
                            e.category = entity.category,
                            e.aliases = entity.aliases,
                            e.corpus_ids = entity.corpus_ids,
                            e.created_at = datetime(),
                            e.source = 'xiaohongshu'
                        ON MATCH SET
                            e.aliases = CASE
                                WHEN entity.aliases IS NOT NULL AND size(entity.aliases) > 0
                                THEN coll.distinct(coalesce(e.aliases, []) + entity.aliases)
                                ELSE coalesce(e.aliases, [])
                            END,
                            e.corpus_ids = CASE
                                WHEN entity.corpus_ids IS NOT NULL AND size(entity.corpus_ids) > 0
                                THEN coll.distinct(coalesce(e.corpus_ids, []) + entity.corpus_ids)
                                ELSE coalesce(e.corpus_ids, [])
                            END,
                            e.updated_at = datetime()
                        SET e += entity.attrs_map
                        RETURN count(e) as merged_count
                    """,
                        entities=batch_data,
                    )
                    record = result.single()
                    total_merged += record["merged_count"] if record else 0

                # 处理事件实体组（EventNode:Entity 双标签）
                if groups["EventNode"]:
                    event_existing_map = self._fetch_existing_props_map(
                        session,
                        "EventNode:Entity",
                        "name",
                        [e.get("name", "") for e in groups["EventNode"]],
                    )
                    batch_data = []
                    for e in groups["EventNode"]:
                        name = e.get("name", "")
                        attrs_map = self._merge_attr_maps_for_append(
                            event_existing_map.get(str(name), {}),
                            e.get("attrs", {}),
                        )
                        batch_data.append(
                            {
                                "name": name,
                                "type": e.get("type", "事件"),
                                "category": e.get("category", ""),
                                "aliases": e.get("aliases", []),
                                "corpus_ids": e.get("corpus_ids", []),
                                "attrs_map": attrs_map,
                            }
                        )
                    result = session.run(
                        """
                        UNWIND $entities AS entity
                        MERGE (e:EventNode:Entity {name: entity.name})
                        ON CREATE SET
                            e.type = entity.type,
                            e.category = entity.category,
                            e.aliases = entity.aliases,
                            e.corpus_ids = entity.corpus_ids,
                            e.created_at = datetime(),
                            e.source = 'xiaohongshu'
                        ON MATCH SET
                            e.aliases = CASE
                                WHEN entity.aliases IS NOT NULL AND size(entity.aliases) > 0
                                THEN coll.distinct(coalesce(e.aliases, []) + entity.aliases)
                                ELSE coalesce(e.aliases, [])
                            END,
                            e.corpus_ids = CASE
                                WHEN entity.corpus_ids IS NOT NULL AND size(entity.corpus_ids) > 0
                                THEN coll.distinct(coalesce(e.corpus_ids, []) + entity.corpus_ids)
                                ELSE coalesce(e.corpus_ids, [])
                            END,
                            e.updated_at = datetime()
                        SET e += entity.attrs_map
                        RETURN count(e) as merged_count
                    """,
                        entities=batch_data,
                    )
                    record = result.single()
                    total_merged += record["merged_count"] if record else 0

                # 处理其他实体组（仅 Entity 标签）
                if groups["Entity"]:
                    entity_existing_map = self._fetch_existing_props_map(
                        session,
                        "Entity",
                        "name",
                        [e.get("name", "") for e in groups["Entity"]],
                    )
                    batch_data = []
                    for e in groups["Entity"]:
                        name = e.get("name", "")
                        attrs_map = self._merge_attr_maps_for_append(
                            entity_existing_map.get(str(name), {}),
                            e.get("attrs", {}),
                        )
                        batch_data.append(
                            {
                                "name": name,
                                "type": e.get("type", "Unknown"),
                                "category": e.get("category", ""),
                                "aliases": e.get("aliases", []),
                                "corpus_ids": e.get("corpus_ids", []),
                                "attrs_map": attrs_map,
                            }
                        )
                    result = session.run(
                        """
                        UNWIND $entities AS entity
                        MERGE (e:Entity {name: entity.name})
                        ON CREATE SET
                            e.type = entity.type,
                            e.category = entity.category,
                            e.aliases = entity.aliases,
                            e.corpus_ids = entity.corpus_ids,
                            e.created_at = datetime(),
                            e.source = 'xiaohongshu'
                        ON MATCH SET
                            e.aliases = CASE
                                WHEN entity.aliases IS NOT NULL AND size(entity.aliases) > 0
                                THEN coll.distinct(coalesce(e.aliases, []) + entity.aliases)
                                ELSE coalesce(e.aliases, [])
                            END,
                            e.corpus_ids = CASE
                                WHEN entity.corpus_ids IS NOT NULL AND size(entity.corpus_ids) > 0
                                THEN coll.distinct(coalesce(e.corpus_ids, []) + entity.corpus_ids)
                                ELSE coalesce(e.corpus_ids, [])
                            END,
                            e.updated_at = datetime()
                        SET e += entity.attrs_map
                        RETURN count(e) as merged_count
                    """,
                        entities=batch_data,
                    )
                    record = result.single()
                    total_merged += record["merged_count"] if record else 0

                logger.info(
                    f"实体合并完成: {total_merged}/{len(entities)} (geo={len(groups['geo_entity_node'])}, func={len(groups['FunctionNode'])}, event={len(groups['EventNode'])}, other={len(groups['Entity'])})"
                )
                return {"merged": total_merged, "total": len(entities)}
        except Exception as e:
            logger.error(f"批量合并实体失败: {e}")
            # 降级为逐个处理
            success_count = 0
            for entity in entities:
                try:
                    if self.merge_entity(entity):
                        success_count += 1
                except Exception as inner_e:
                    logger.error(f"合并实体失败 {entity['name']}: {inner_e}")
            return {"merged": success_count, "total": len(entities)}

    def batch_merge_relations(self, triples: List[Dict]) -> Dict:
        """
        批量合并关系 - P15改进：按关系类型使用独立的关系标签

        P15改进：
        - 不再使用统一的 RELATION 标签 + type 属性
        - 每种关系类型使用独立的 Neo4j 关系标签（位于/包含/相对方位/具有功能/发生事件/优于/相似/劣于）
        - 按关系类型分组执行不同的 Cypher 查询
        """
        if not triples:
            return {"merged": 0, "total": 0}

        # P15改进：按关系类型分组
        # 8种标准关系类型
        relation_groups: Dict[str, List[Dict]] = defaultdict(list)
        for t in triples:
            rel_type = t.get("relation", "Unknown")
            relation_groups[rel_type].append(t)

        total_merged = 0
        group_stats = {}

        try:
            with self.driver.session() as session:
                for rel_type, group_triples in relation_groups.items():
                    # P22新增：区分可按ID对齐写入与按name回退写入的三元组
                    id_aligned_triples = []  # head/tail 都有对齐ID
                    name_merge_triples = []  # 部分或全部无对齐ID，按name合并

                    for t in group_triples:
                        has_head_aligned = bool(t.get("head_db_entity_id", ""))
                        has_tail_aligned = bool(t.get("tail_db_entity_id", ""))
                        if has_head_aligned and has_tail_aligned:
                            id_aligned_triples.append(t)
                        else:
                            name_merge_triples.append(t)

                    # P15改进：使用具体关系类型作为标签名
                    rel_label = f"`{rel_type}`"
                    rel_merged_count = 0

                    # P22新增：优先处理可按ID匹配的三元组（head/tail均有ID）
                    if id_aligned_triples:
                        existing_rel_attrs_map = {}
                        existing_result = session.run(
                            f"""
                            UNWIND $triples AS triple
                            MATCH (h:geo_entity_node)
                            WHERE h.original_id = triple.head_db_entity_id
                               OR h.entity_id = triple.head_db_entity_id
                            MATCH (t:geo_entity_node)
                            WHERE t.original_id = triple.tail_db_entity_id
                               OR t.entity_id = triple.tail_db_entity_id
                            OPTIONAL MATCH (h)-[r:{rel_label}]->(t)
                            RETURN triple.head_db_entity_id AS head_id,
                                   triple.tail_db_entity_id AS tail_id,
                                   r.relation_attrs AS relation_attrs
                            """
                            ,
                            triples=[
                                {
                                    "head_db_entity_id": t.get("head_db_entity_id", ""),
                                    "tail_db_entity_id": t.get("tail_db_entity_id", ""),
                                }
                                for t in id_aligned_triples
                            ],
                        )
                        for row in existing_result:
                            key = (
                                str(row["head_id"] or ""),
                                str(row["tail_id"] or ""),
                            )
                            existing_rel_attrs_map[key] = row["relation_attrs"]

                        batch_data = [
                            {
                                "head": t["head"],
                                "tail": t["tail"],
                                "head_db_entity_id": t.get("head_db_entity_id", ""),
                                "tail_db_entity_id": t.get("tail_db_entity_id", ""),
                                "evidence": t.get("evidence", ""),
                                "corpus_ids": t.get("corpus_ids", []),
                                "relation_type": t.get("relation_type", ""),
                                "relation_subtype": t.get("relation_subtype", ""),
                                "relation_attrs": self._merge_relation_attrs_json(
                                    existing_rel_attrs_map.get(
                                        (
                                            str(t.get("head_db_entity_id", "")),
                                            str(t.get("tail_db_entity_id", "")),
                                        )
                                    ),
                                    t.get("relation_attrs", {}),
                                ),
                            }
                            for t in id_aligned_triples
                        ]
                        query = f"""
                            UNWIND $triples AS triple
                            MATCH (h:geo_entity_node)
                            WHERE h.original_id = triple.head_db_entity_id
                               OR h.entity_id = triple.head_db_entity_id
                            MATCH (t:geo_entity_node)
                            WHERE t.original_id = triple.tail_db_entity_id
                               OR t.entity_id = triple.tail_db_entity_id
                            SET h.updated_at = datetime()
                            SET t.updated_at = datetime()
                            MERGE (h)-[r:{rel_label}]->(t)
                            ON CREATE SET
                                r.evidence = triple.evidence,
                                r.corpus_ids = triple.corpus_ids,
                                r.relation_type = triple.relation_type,
                                r.relation_subtype = triple.relation_subtype,
                                r.relation_attrs = triple.relation_attrs,
                                r.created_at = datetime(),
                                r.source = 'xiaohongshu'
                            ON MATCH SET
                                r.corpus_ids = CASE
                                    WHEN triple.corpus_ids IS NOT NULL AND size(triple.corpus_ids) > 0
                                    THEN coll.distinct(coalesce(r.corpus_ids, []) + triple.corpus_ids)
                                    ELSE coalesce(r.corpus_ids, [])
                                END,
                                r.updated_at = datetime()
                            RETURN count(r) as merged_count
                        """
                        result = session.run(query, triples=batch_data)
                        record = result.single()
                        matched_count = record["merged_count"] if record else 0
                        rel_merged_count += matched_count
                        if matched_count < len(id_aligned_triples):
                            logger.warning(
                                f"[Neo4j] {rel_type} ID对齐关系未完全写入: "
                                f"input={len(id_aligned_triples)}, merged={matched_count}"
                            )

                    # 处理按name写入的三元组（无完整ID信息时）
                    if name_merge_triples:
                        existing_rel_attrs_map = {}
                        existing_result = session.run(
                            f"""
                            UNWIND $triples AS triple
                            MATCH (h:Entity {{name: triple.head}})
                            MATCH (t:Entity {{name: triple.tail}})
                            OPTIONAL MATCH (h)-[r:{rel_label}]->(t)
                            RETURN triple.head AS head,
                                   triple.tail AS tail,
                                   r.relation_attrs AS relation_attrs
                            """,
                            triples=[
                                {"head": t["head"], "tail": t["tail"]}
                                for t in name_merge_triples
                            ],
                        )
                        for row in existing_result:
                            key = (str(row["head"] or ""), str(row["tail"] or ""))
                            existing_rel_attrs_map[key] = row["relation_attrs"]

                        batch_data = [
                            {
                                "head": t["head"],
                                "relation": t["relation"],
                                "tail": t["tail"],
                                "evidence": t.get("evidence", ""),
                                "corpus_ids": t.get("corpus_ids", []),
                                "relation_type": t.get("relation_type", ""),
                                "relation_subtype": t.get("relation_subtype", ""),
                                "relation_attrs": self._merge_relation_attrs_json(
                                    existing_rel_attrs_map.get(
                                        (str(t["head"]), str(t["tail"]))
                                    ),
                                    t.get("relation_attrs", {}),
                                ),
                            }
                            for t in name_merge_triples
                        ]

                        query = f"""
                            UNWIND $triples AS triple
                            MERGE (h:Entity {{name: triple.head}})
                            ON CREATE SET
                                h.type = 'Unknown',
                                h.category = '',
                                h.aliases = [],
                                h.corpus_ids = [],
                                h.created_at = datetime(),
                                h.source = 'xiaohongshu'
                            ON MATCH SET
                                h.updated_at = datetime()
                            MERGE (t:Entity {{name: triple.tail}})
                            ON CREATE SET
                                t.type = 'Unknown',
                                t.category = '',
                                t.aliases = [],
                                t.corpus_ids = [],
                                t.created_at = datetime(),
                                t.source = 'xiaohongshu'
                            ON MATCH SET
                                t.updated_at = datetime()
                            MERGE (h)-[r:{rel_label}]->(t)
                            ON CREATE SET
                                r.evidence = triple.evidence,
                                r.corpus_ids = triple.corpus_ids,
                                r.relation_type = triple.relation_type,
                                r.relation_subtype = triple.relation_subtype,
                                r.relation_attrs = triple.relation_attrs,
                                r.created_at = datetime(),
                                r.source = 'xiaohongshu'
                            ON MATCH SET
                                r.corpus_ids = CASE
                                    WHEN triple.corpus_ids IS NOT NULL AND size(triple.corpus_ids) > 0
                                    THEN coll.distinct(coalesce(r.corpus_ids, []) + triple.corpus_ids)
                                    ELSE coalesce(r.corpus_ids, [])
                                END,
                                r.relation_type = CASE
                                    WHEN triple.relation_type IS NOT NULL AND triple.relation_type <> ''
                                    THEN triple.relation_type
                                    ELSE r.relation_type
                                END,
                                r.relation_subtype = CASE
                                    WHEN triple.relation_subtype IS NOT NULL AND triple.relation_subtype <> ''
                                    THEN triple.relation_subtype
                                    ELSE r.relation_subtype
                                END,
                                r.relation_attrs = triple.relation_attrs,
                                r.updated_at = datetime()
                            RETURN count(r) as merged_count
                        """
                        result = session.run(query, triples=batch_data)
                        record = result.single()
                        rel_merged_count += record["merged_count"] if record else 0

                    total_merged += rel_merged_count
                    group_stats[rel_type] = rel_merged_count

                logger.info(
                    f"关系合并完成: {total_merged}/{len(triples)} ({dict(group_stats)})"
                )
                return {
                    "merged": total_merged,
                    "total": len(triples),
                    "groups": group_stats,
                }
        except Exception as e:
            logger.error(f"批量合并关系失败: {e}")
            # 降级为逐个处理
            success_count = 0
            for triple in triples:
                try:
                    if self.merge_relation(triple):
                        success_count += 1
                except Exception as inner_e:
                    logger.error(f"合并关系失败 {triple}: {inner_e}")
            return {"merged": success_count, "total": len(triples)}

    def query_entity(self, name: str) -> Optional[Dict]:
        """查询实体"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {name: $name})
                RETURN e
            """,
                name=name,
            )
            record = result.single()
            if record:
                return dict(record["e"])
            return None

    def query_relations(self, entity_name: str) -> List[Dict]:
        """查询实体相关的关系"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {name: $name})-[r]-(other)
                RETURN e.name as head, type(r) as relation, other.name as tail, r.evidence as evidence
            """,
                name=entity_name,
            )
            return [dict(record) for record in result]

    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.driver.session() as session:
            entity_result = session.run(
                "MATCH (e:Entity) RETURN count(e) as count"
            ).single()
            relation_result = session.run(
                "MATCH ()-[r]->() RETURN count(r) as count"
            ).single()

            entity_count = entity_result["count"] if entity_result else 0
            relation_count = relation_result["count"] if relation_result else 0

            return {"entity_count": entity_count, "relation_count": relation_count}

    # ===== P12新增：实体对齐相关方法 =====

    def find_entity_by_original_id(self, original_id: str) -> Optional[Dict]:
        """
        通过original_id查找实体节点（用于amap POI匹配）

        Args:
            original_id: 原始高德ID，如 'amap_B0FFLCH14H'

        Returns:
            实体信息，包含 entity_id, name, type 等，或 None
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:geo_entity_node)
                WHERE e.original_id = $original_id
                RETURN e.entity_id as entity_id,
                       e.name as name,
                       e.type as type,
                       e.longitude as longitude,
                       e.latitude as latitude,
                       e.source as source
                LIMIT 1
            """,
                original_id=original_id,
            )
            record = result.single()
            if record:
                return dict(record)
            return None

    def find_entity_by_entity_id(self, entity_id: str) -> Optional[Dict]:
        """
        通过entity_id查找实体节点

        Args:
            entity_id: neo4j实体ID，如 'poi_123' 或 'road_001'

        Returns:
            实体信息，或 None
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:geo_entity_node)
                WHERE e.entity_id = $entity_id
                RETURN e.entity_id as entity_id,
                       e.name as name,
                       e.type as type,
                       e.longitude as longitude,
                       e.latitude as latitude,
                       e.source as source
                LIMIT 1
            """,
                entity_id=entity_id,
            )
            record = result.single()
            if record:
                return dict(record)
            return None

    def create_new_geo_entity(self, entity: Dict) -> Optional[str]:
        """
        创建新的地理实体节点（来自小红书抽取）

        Args:
            entity: {
                "name": "某某咖啡店",
                "type": "POI",
                "longitude": Optional[float],
                "latitude": Optional[float],
                "aliases": Optional[List[str]],
                "source": "xiaohongshu"
            }

        Returns:
            新创建的 entity_id，或 None（如果失败）

        注意：使用原子计数器避免并发ID冲突
        """
        from datetime import datetime

        with self.driver.session() as session:
            # 获取同类型实体的前缀
            type_prefix = entity.get("type", "poi").lower()
            if type_prefix in ["poi", "建筑", "建筑物"]:
                type_prefix = "poi"
            elif type_prefix in ["道路", "路"]:
                type_prefix = "road"
            elif type_prefix in ["街区", "商圈"]:
                type_prefix = "block"
            elif type_prefix in ["行政区", "区"]:
                type_prefix = "district"
            else:
                type_prefix = "poi"  # 默认

            # 使用原子计数器获取新ID（避免并发冲突）
            # 计数器节点存储各类型实体的下一个编号
            counter_result = session.run(
                """
                MERGE (counter:EntityCounter {type: $prefix})
                ON CREATE SET counter.next_id = 1
                SET counter.next_id = counter.next_id + 1
                RETURN counter.next_id - 1 as new_id
            """,
                prefix=type_prefix,
            )

            counter_record = counter_result.single()
            if not counter_record:
                logger.error(f"[Neo4j] 获取计数器失败: {type_prefix}")
                return None

            new_id_num = counter_record["new_id"]
            new_entity_id = f"{type_prefix}_{new_id_num}"

            # 使用MERGE避免重复创建（如果ID已存在则跳过）
            create_result = session.run(
                """
                MERGE (e:geo_entity_node:Entity {entity_id: $entity_id})
                ON CREATE SET
                    e.name = $name,
                    e.type = $type,
                    e.longitude = $longitude,
                    e.latitude = $latitude,
                    e.aliases = $aliases,
                    e.source = $source,
                    e.created_at = datetime()
                RETURN e.entity_id as created_id, e.name as created_name
            """,
                entity_id=new_entity_id,
                name=entity["name"],
                type=entity.get("type", "POI"),
                longitude=entity.get("longitude"),
                latitude=entity.get("latitude"),
                aliases=entity.get("aliases", []),
                source="xiaohongshu",
            )

            created_record = create_result.single()
            if created_record and created_record["created_id"]:
                # 验证是否是新创建的（避免并发时计数器增加但节点已存在）
                if created_record["created_name"] == entity["name"]:
                    logger.info(
                        f"[Neo4j] 创建新实体: {new_entity_id} - {entity['name']}"
                    )
                    return created_record["created_id"]
                else:
                    # 节点已存在且名称不同，需要重新尝试
                    logger.warning(
                        f"[Neo4j] entity_id已存在: {new_entity_id}, 重新尝试"
                    )
                    # 递增计数器并重试（最多3次）
                    return self._retry_create_with_new_id(session, entity, type_prefix)
            return None

    def _retry_create_with_new_id(
        self, session, entity: Dict, type_prefix: str, max_retries: int = 3
    ) -> Optional[str]:
        """重试创建实体（处理并发冲突）"""
        from datetime import datetime

        for attempt in range(max_retries):
            # 再次获取计数器
            counter_result = session.run(
                """
                MERGE (counter:EntityCounter {type: $prefix})
                SET counter.next_id = counter.next_id + 1
                RETURN counter.next_id - 1 as new_id
            """,
                prefix=type_prefix,
            )

            counter_record = counter_result.single()
            if not counter_record:
                continue

            new_id_num = counter_record["new_id"]
            new_entity_id = f"{type_prefix}_{new_id_num}"

            # 尝试创建
            create_result = session.run(
                """
                MERGE (e:geo_entity_node:Entity {entity_id: $entity_id})
                ON CREATE SET
                    e.name = $name,
                    e.type = $type,
                    e.longitude = $longitude,
                    e.latitude = $latitude,
                    e.aliases = $aliases,
                    e.source = $source,
                    e.created_at = datetime()
                RETURN e.entity_id as created_id, e.name as created_name
            """,
                entity_id=new_entity_id,
                name=entity["name"],
                type=entity.get("type", "POI"),
                longitude=entity.get("longitude"),
                latitude=entity.get("latitude"),
                aliases=entity.get("aliases", []),
                source="xiaohongshu",
            )

            created_record = create_result.single()
            if created_record and created_record["created_id"]:
                if created_record["created_name"] == entity["name"]:
                    logger.info(f"[Neo4j] 重试成功: {new_entity_id} - {entity['name']}")
                    return created_record["created_id"]

        logger.error(f"[Neo4j] 重试{max_retries}次后仍失败: {entity['name']}")
        return None

    def batch_create_new_geo_entities(self, entities: List[Dict]) -> Dict:
        """
        批量创建新的地理实体节点（真正的批量操作）

        Args:
            entities: List of entity dicts

        Returns:
            {"created": count, "entity_ids": [id1, id2, ...]}

        策略：
        1. 先批量获取各类型的计数器增量
        2. 然后使用UNWIND批量创建节点
        """
        if not entities:
            return {"created": 0, "entity_ids": []}

        from datetime import datetime

        # 按类型分组
        type_groups = {}
        for entity in entities:
            type_prefix = entity.get("type", "poi").lower()
            if type_prefix in ["poi", "建筑", "建筑物"]:
                type_prefix = "poi"
            elif type_prefix in ["道路", "路"]:
                type_prefix = "road"
            elif type_prefix in ["街区", "商圈"]:
                type_prefix = "block"
            elif type_prefix in ["行政区", "区"]:
                type_prefix = "district"
            else:
                type_prefix = "poi"

            if type_prefix not in type_groups:
                type_groups[type_prefix] = []
            type_groups[type_prefix].append(entity)

        created_ids = []

        with self.driver.session() as session:
            for type_prefix, group_entities in type_groups.items():
                # 批量获取计数器增量
                counter_result = session.run(
                    """
                    MERGE (counter:EntityCounter {type: $prefix})
                    ON CREATE SET counter.next_id = 1
                    SET counter.next_id = counter.next_id + $count
                    RETURN counter.next_id - $count as start_id
                """,
                    prefix=type_prefix,
                    count=len(group_entities),
                )

                counter_record = counter_result.single()
                if not counter_record:
                    logger.error(f"[Neo4j] 批量获取计数器失败: {type_prefix}")
                    continue

                start_id = counter_record["start_id"]

                # 准备批量创建数据
                batch_data = []
                for i, entity in enumerate(group_entities):
                    entity_id = f"{type_prefix}_{start_id + i}"
                    batch_data.append(
                        {
                            "entity_id": entity_id,
                            "name": entity["name"],
                            "type": entity.get("type", "POI"),
                            "longitude": entity.get("longitude"),
                            "latitude": entity.get("latitude"),
                            "aliases": entity.get("aliases", []),
                            "source": "xiaohongshu",
                        }
                    )

                # 使用UNWIND批量创建
                create_result = session.run(
                    """
                    UNWIND $batch AS node
                    MERGE (e:geo_entity_node:Entity {entity_id: node.entity_id})
                    ON CREATE SET
                        e.name = node.name,
                        e.type = node.type,
                        e.longitude = node.longitude,
                        e.latitude = node.latitude,
                        e.aliases = node.aliases,
                        e.source = node.source,
                        e.created_at = datetime()
                    RETURN e.entity_id as created_id, e.name as created_name
                """,
                    batch=batch_data,
                )

                # 收集创建成功的ID
                for record in create_result:
                    if record["created_id"]:
                        # 验证名称匹配（避免并发时ID被占用）
                        expected_name = None
                        for bd in batch_data:
                            if bd["entity_id"] == record["created_id"]:
                                expected_name = bd["name"]
                                break

                        if record["created_name"] == expected_name:
                            created_ids.append(record["created_id"])
                        else:
                            logger.warning(
                                f"[Neo4j] entity_id已被占用: {record['created_id']}"
                            )

        logger.info(f"[Neo4j] 批量创建完成: {len(created_ids)}/{len(entities)}")
        return {"created": len(created_ids), "entity_ids": created_ids}
