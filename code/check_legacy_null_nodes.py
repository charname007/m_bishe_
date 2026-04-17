"""检查数据库中是否存在 aliases/corpus_ids 为 null 的节点"""

from neo4j import GraphDatabase
from settings import settings

config = settings.get_neo4j_config()
driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))

with driver.session() as session:
    # 检查 aliases=null 的节点
    result1 = session.run(
        "MATCH (e:Entity) WHERE e.aliases IS NULL RETURN count(e) as count, collect(e.name)[0..5] as samples"
    ).single()
    print(f"aliases=null 的节点数: {result1['count']}")
    if result1["count"] > 0:
        print(f"示例节点: {result1['samples']}")

    # 检查 corpus_ids=null 的节点
    result2 = session.run(
        "MATCH (e:Entity) WHERE e.corpus_ids IS NULL RETURN count(e) as count, collect(e.name)[0..5] as samples"
    ).single()
    print(f"corpus_ids=null 的节点数: {result2['count']}")
    if result2["count"] > 0:
        print(f"示例节点: {result2['samples']}")

    # 检查 geo_entity_node 的 null 值情况
    result3 = session.run(
        "MATCH (e:geo_entity_node) WHERE e.aliases IS NULL RETURN count(e) as count"
    ).single()
    print(f"geo_entity_node aliases=null: {result3['count']}")

    result4 = session.run(
        "MATCH (e:geo_entity_node) WHERE e.corpus_ids IS NULL RETURN count(e) as count"
    ).single()
    print(f"geo_entity_node corpus_ids=null: {result4['count']}")

driver.close()
