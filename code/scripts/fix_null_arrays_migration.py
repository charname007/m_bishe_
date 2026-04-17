"""
数据库迁移脚本：修复遗留节点的 null 数组字段

运行此脚本将所有 aliases 和 corpus_ids 为 null 的节点转换为空列表 []
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
from settings import settings
from loguru import logger


def fix_null_arrays():
    """修复所有节点的 null 数组字段"""
    config = settings.get_neo4j_config()
    driver = GraphDatabase.driver(
        config["uri"], auth=(config["user"], config["password"])
    )

    with driver.session() as session:
        # 1. Entity节点
        logger.info("=== Fixing Entity nodes ===")
        result1 = session.run("""
            MATCH (e:Entity)
            WHERE e.aliases IS NULL OR e.corpus_ids IS NULL
            SET e.aliases = coalesce(e.aliases, []),
                e.corpus_ids = coalesce(e.corpus_ids, [])
            RETURN count(e) as fixed_count
        """).single()
        logger.info(f"Fixed {result1['fixed_count']} Entity nodes")

        # 2. geo_entity_node节点（主力）
        logger.info("=== Fixing geo_entity_node nodes ===")
        result2 = session.run("""
            MATCH (e:geo_entity_node)
            WHERE e.aliases IS NULL OR e.corpus_ids IS NULL
            SET e.aliases = coalesce(e.aliases, []),
                e.corpus_ids = coalesce(e.corpus_ids, [])
            RETURN count(e) as fixed_count
        """).single()
        logger.info(f"Fixed {result2['fixed_count']} geo_entity_node nodes")

        # 3. FunctionNode节点
        logger.info("=== Fixing FunctionNode nodes ===")
        result3 = session.run("""
            MATCH (e:FunctionNode)
            WHERE e.aliases IS NULL OR e.corpus_ids IS NULL
            SET e.aliases = coalesce(e.aliases, []),
                e.corpus_ids = coalesce(e.corpus_ids, [])
            RETURN count(e) as fixed_count
        """).single()
        logger.info(f"Fixed {result3['fixed_count']} FunctionNode nodes")

        # 4. EventNode节点
        logger.info("=== Fixing EventNode nodes ===")
        result4 = session.run("""
            MATCH (e:EventNode)
            WHERE e.aliases IS NULL OR e.corpus_ids IS NULL
            SET e.aliases = coalesce(e.aliases, []),
                e.corpus_ids = coalesce(e.corpus_ids, [])
            RETURN count(e) as fixed_count
        """).single()
        logger.info(f"Fixed {result4['fixed_count']} EventNode nodes")

        # 5. 关系节点（corpus_ids）
        logger.info("=== Fixing relations ===")
        result5 = session.run("""
            MATCH ()-[r]->()
            WHERE r.corpus_ids IS NULL
            SET r.corpus_ids = []
            RETURN count(r) as fixed_count
        """).single()
        logger.info(f"Fixed {result5['fixed_count']} relations")

        total_fixed = (
            result1["fixed_count"]
            + result2["fixed_count"]
            + result3["fixed_count"]
            + result4["fixed_count"]
            + result5["fixed_count"]
        )
        logger.success(
            f"\n=== Migration complete: Fixed {total_fixed} total nodes/relations ==="
        )

    driver.close()


def verify_fix():
    """验证修复后是否还有null节点"""
    config = settings.get_neo4j_config()
    driver = GraphDatabase.driver(
        config["uri"], auth=(config["user"], config["password"])
    )

    with driver.session() as session:
        # 检查是否还有 null
        check1 = session.run(
            "MATCH (e:Entity) WHERE e.aliases IS NULL RETURN count(e) as count"
        ).single()
        check2 = session.run(
            "MATCH (e:Entity) WHERE e.corpus_ids IS NULL RETURN count(e) as count"
        ).single()
        check3 = session.run(
            "MATCH (e:geo_entity_node) WHERE e.aliases IS NULL RETURN count(e) as count"
        ).single()
        check4 = session.run(
            "MATCH (e:geo_entity_node) WHERE e.corpus_ids IS NULL RETURN count(e) as count"
        ).single()
        check5 = session.run(
            "MATCH ()-[r]->() WHERE r.corpus_ids IS NULL RETURN count(r) as count"
        ).single()

        logger.info("=== Verification ===")
        logger.info(f"Entity aliases=null: {check1['count']}")
        logger.info(f"Entity corpus_ids=null: {check2['count']}")
        logger.info(f"geo_entity_node aliases=null: {check3['count']}")
        logger.info(f"geo_entity_node corpus_ids=null: {check4['count']}")
        logger.info(f"relations corpus_ids=null: {check5['count']}")

        if all(c["count"] == 0 for c in [check1, check2, check3, check4, check5]):
            logger.success("✓ All null arrays fixed successfully!")
        else:
            logger.error("✗ Some null arrays still exist!")

    driver.close()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Database Migration: Fix null array fields")
    logger.info("=" * 60)

    # 执行迁移
    fix_null_arrays()

    # 验证修复
    verify_fix()
