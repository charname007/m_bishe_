"""
测试 coalesce 修复后的 coll.distinct 行为
验证能正确处理 aliases/corpus_ids 为 null 的遗留节点
"""

from neo4j import GraphDatabase
from settings import settings


def test_coalesce_with_null_nodes():
    """测试：coalesce 修复能否正确处理 null 值节点"""
    config = settings.get_neo4j_config()
    driver = GraphDatabase.driver(
        config["uri"], auth=(config["user"], config["password"])
    )

    with driver.session() as session:
        # 1. 创建测试节点（模拟遗留数据）
        session.run(
            "CREATE (test:TestLegacyNode {name: 'legacy_null', aliases: null, corpus_ids: null})"
        )
        session.run(
            "CREATE (test:TestLegacyNode {name: 'legacy_empty', aliases: [], corpus_ids: []})"
        )

        # 2. 测试修复后的逻辑
        # 模拟 merge_entity 的 ON MATCH SET
        result1 = session.run("""
            MATCH (e:TestLegacyNode {name: 'legacy_null'})
            SET e.aliases = CASE
                WHEN ['alias1', 'alias2'] IS NOT NULL AND size(['alias1', 'alias2']) > 0
                THEN coll.distinct(coalesce(e.aliases, []) + ['alias1', 'alias2'])
                ELSE coalesce(e.aliases, [])
            END,
            e.corpus_ids = CASE
                WHEN ['corpus1'] IS NOT NULL AND size(['corpus1']) > 0
                THEN coll.distinct(coalesce(e.corpus_ids, []) + ['corpus1'])
                ELSE coalesce(e.corpus_ids, [])
            END
            RETURN e.aliases as aliases, e.corpus_ids as corpus_ids
        """).single()

        print(f"[OK] legacy_null node after fix:")
        print(f"   aliases: {result1['aliases']} (expected: ['alias1', 'alias2'])")
        print(f"   corpus_ids: {result1['corpus_ids']} (expected: ['corpus1'])")

        assert result1["aliases"] == ["alias1", "alias2"], (
            f"aliases fix failed: {result1['aliases']}"
        )
        assert result1["corpus_ids"] == ["corpus1"], (
            f"corpus_ids fix failed: {result1['corpus_ids']}"
        )

        # 3. 第二次merge（累积去重）
        result2 = session.run("""
            MATCH (e:TestLegacyNode {name: 'legacy_null'})
            SET e.aliases = CASE
                WHEN ['alias2', 'alias3'] IS NOT NULL AND size(['alias2', 'alias3']) > 0
                THEN coll.distinct(coalesce(e.aliases, []) + ['alias2', 'alias3'])
                ELSE coalesce(e.aliases, [])
            END,
            e.corpus_ids = CASE
                WHEN ['corpus2'] IS NOT NULL AND size(['corpus2']) > 0
                THEN coll.distinct(coalesce(e.corpus_ids, []) + ['corpus2'])
                ELSE coalesce(e.corpus_ids, [])
            END
            RETURN e.aliases as aliases, e.corpus_ids as corpus_ids
        """).single()

        print(f"\n[OK] After second accumulation:")
        print(
            f"   aliases: {result2['aliases']} (expected: ['alias1', 'alias2', 'alias3'])"
        )
        print(
            f"   corpus_ids: {result2['corpus_ids']} (expected: ['corpus1', 'corpus2'])"
        )

        assert result2["aliases"] == ["alias1", "alias2", "alias3"], (
            f"aliases accumulation failed: {result2['aliases']}"
        )
        assert result2["corpus_ids"] == ["corpus1", "corpus2"], (
            f"corpus_ids accumulation failed: {result2['corpus_ids']}"
        )

        # 4. 测试空列表节点的行为
        result3 = session.run("""
            MATCH (e:TestLegacyNode {name: 'legacy_empty'})
            SET e.aliases = CASE
                WHEN ['alias1'] IS NOT NULL AND size(['alias1']) > 0
                THEN coll.distinct(coalesce(e.aliases, []) + ['alias1'])
                ELSE coalesce(e.aliases, [])
            END
            RETURN e.aliases as aliases
        """).single()

        print(f"\n[OK] legacy_empty node:")
        print(f"   aliases: {result3['aliases']} (expected: ['alias1'])")

        assert result3["aliases"] == ["alias1"], (
            f"empty node handling failed: {result3['aliases']}"
        )

        # 清理
        session.run("MATCH (n:TestLegacyNode) DELETE n")

    driver.close()
    print("\n[OK] All tests passed! coalesce fix is effective!")


if __name__ == "__main__":
    print("=" * 60)
    print("Test: coalesce fix for null value nodes")
    print("=" * 60)
    test_coalesce_with_null_nodes()
