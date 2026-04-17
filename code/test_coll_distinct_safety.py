"""
测试 coll.distinct 与 apoc.coll.toSet 的兼容性
验证修改不会引入bug
"""

from neo4j import GraphDatabase
from settings import settings
from loguru import logger


def test_list_concatenation_with_null():
    """测试：列表拼接时遇到null的情况"""
    config = settings.get_neo4j_config()
    driver = GraphDatabase.driver(
        config["uri"], auth=(config["user"], config["password"])
    )

    with driver.session() as session:
        # 测试1: null + list 的行为
        result1 = session.run("RETURN null + [1, 2] as result").single()
        print(f"null + [1, 2] = {result1['result']}")  # 预期: null

        # 测试2: [] + list 的行为
        result2 = session.run("RETURN [] + [1, 2, 2, 3] as result").single()
        print(f"[] + [1, 2, 2, 3] = {result2['result']}")  # 预期: [1, 2, 2, 3]

        # 测试3: coll.distinct 对空列表
        result3 = session.run("RETURN coll.distinct([]) as result").single()
        print(f"coll.distinct([]) = {result3['result']}")  # 预期: []

        # 测试4: coll.distinct 对有重复的列表
        result4 = session.run(
            "RETURN coll.distinct([1, 2, 2, 3, 1]) as result"
        ).single()
        print(
            f"coll.distinct([1, 2, 2, 3, 1]) = {result4['result']}"
        )  # 预期: [1, 2, 3]

        # 测试5: coll.distinct 对 null 的行为（关键测试）
        try:
            result5 = session.run("RETURN coll.distinct(null) as result").single()
            print(f"coll.distinct(null) = {result5['result']}")
        except Exception as e:
            print(f"coll.distinct(null) 报错: {e}")

        # 测试6: 模拟实际场景 - e.aliases可能是null的情况
        # 创建测试节点
        session.run(
            "CREATE (test:TestNode {name: 'test_null_aliases', aliases: null, corpus_ids: null})"
        )
        session.run(
            "CREATE (test:TestNode {name: 'test_empty_aliases', aliases: [], corpus_ids: []})"
        )

        # 测试6a: null aliases + new_aliases（应该会失败）
        try:
            result6a = session.run("""
                MATCH (e:TestNode {name: 'test_null_aliases'})
                RETURN coll.distinct(e.aliases + ['new_alias']) as result
            """).single()
            print(f"null aliases + ['new_alias'] = {result6a['result']}")
        except Exception as e:
            print(f"⚠️  null aliases + ['new_alias'] 报错: {e}")

        # 测试6b: [] aliases + new_aliases（应该成功）
        result6b = session.run("""
            MATCH (e:TestNode {name: 'test_empty_aliases'})
            RETURN coll.distinct(e.aliases + ['new_alias', 'new_alias']) as result
        """).single()
        print(
            f"[] aliases + ['new_alias', 'new_alias'] = {result6b['result']}"
        )  # 预期: ['new_alias']

        # 测试7: CASE语句的防护是否有效
        result7 = session.run("""
            MATCH (e:TestNode {name: 'test_null_aliases'})
            RETURN CASE
                WHEN ['new_alias'] IS NOT NULL AND size(['new_alias']) > 0
                THEN coll.distinct(e.aliases + ['new_alias'])
                ELSE e.aliases
            END as result
        """).single()
        print(
            f"CASE防护（aliases=null）: {result7['result']}"
        )  # 预期: null（因为 e.aliases + ... 会是 null）

        # 清理测试节点
        session.run("MATCH (n:TestNode) DELETE n")

    driver.close()


def test_merge_behavior_comparison():
    """测试：实际merge操作的行为对比"""
    config = settings.get_neo4j_config()
    driver = GraphDatabase.driver(
        config["uri"], auth=(config["user"], config["password"])
    )

    with driver.session() as session:
        # 测试场景：同一个实体被多次merge，aliases应该正确去重累积

        # 第一次merge
        session.run("""
            MERGE (e:TestMergeEntity {name: 'test_entity'})
            ON CREATE SET
                e.aliases = ['alias1', 'alias2'],
                e.corpus_ids = ['corpus1']
            ON MATCH SET
                e.aliases = CASE
                    WHEN ['alias3', 'alias2'] IS NOT NULL AND size(['alias3', 'alias2']) > 0
                    THEN coll.distinct(e.aliases + ['alias3', 'alias2'])
                    ELSE e.aliases
                END
        """)

        # 第二次merge（触发ON MATCH）
        session.run("""
            MERGE (e:TestMergeEntity {name: 'test_entity'})
            ON CREATE SET
                e.aliases = [],
                e.corpus_ids = []
            ON MATCH SET
                e.aliases = CASE
                    WHEN ['alias3', 'alias4'] IS NOT NULL AND size(['alias3', 'alias4']) > 0
                    THEN coll.distinct(e.aliases + ['alias3', 'alias4'])
                    ELSE e.aliases
                END,
                e.corpus_ids = CASE
                    WHEN ['corpus2'] IS NOT NULL AND size(['corpus2']) > 0
                    THEN coll.distinct(e.corpus_ids + ['corpus2'])
                    ELSE e.corpus_ids
                END
        """)

        # 检查结果
        result = session.run("""
            MATCH (e:TestMergeEntity {name: 'test_entity'})
            RETURN e.aliases as aliases, e.corpus_ids as corpus_ids
        """).single()

        print(f"\n最终aliases: {result['aliases']}")
        print(f"预期: ['alias1', 'alias2', 'alias3', 'alias4']（去重）")
        print(f"最终corpus_ids: {result['corpus_ids']}")
        print(f"预期: ['corpus1', 'corpus2']（去重）")

        # 验证顺序是否正确（coll.distinct保留首次出现顺序）
        assert result["aliases"] == ["alias1", "alias2", "alias3", "alias4"], (
            f"aliases去重失败: {result['aliases']}"
        )
        assert result["corpus_ids"] == ["corpus1", "corpus2"], (
            f"corpus_ids去重失败: {result['corpus_ids']}"
        )

        # 清理
        session.run("MATCH (e:TestMergeEntity) DELETE e")

    driver.close()
    print("\n✅ coll.distinct行为验证通过！")


if __name__ == "__main__":
    print("=" * 60)
    print("测试 coll.distinct 函数安全性")
    print("=" * 60)
    test_list_concatenation_with_null()

    print("\n" + "=" * 60)
    print("测试实际merge场景")
    print("=" * 60)
    test_merge_behavior_comparison()
