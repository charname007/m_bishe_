"""
数据库连接测试脚本
"""
from settings import settings
from kg.neo4j_client import Neo4jClient
from kg.postgres_client import PostgresClient
from loguru import logger


def test_neo4j():
    """测试 Neo4j 连接"""
    print("\n=== Neo4j 连接测试 ===")
    try:
        config = settings.get_neo4j_config()
        print(f"URI: {config['uri']}")
        print(f"User: {config['user']}")

        client = Neo4jClient(
            uri=config['uri'],
            user=config['user'],
            password=config['password']
        )

        # 测试查询
        stats = client.get_stats()
        print(f"[OK] 连接成功!")
        print(f"   实体数量: {stats['entity_count']}")
        print(f"   关系数量: {stats['relation_count']}")

        client.close()
        return True
    except Exception as e:
        print(f"[FAIL] 连接失败: {e}")
        return False


def test_postgres():
    """测试 PostgreSQL 连接"""
    print("\n=== PostgreSQL 连接测试 ===")
    try:
        config = settings.get_postgres_config()
        print(f"Host: {config['host']}:{config['port']}")
        print(f"Database: {config['database']}")
        print(f"User: {config['user']}")

        client = PostgresClient(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )

        # 测试查询
        with client.conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            print(f"[OK] 连接成功!")
            print(f"   PostgreSQL 版本: {version}")

            # 检查表是否存在
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in cur.fetchall()]
            print(f"   已有表: {tables if tables else '无'}")

        client.close()
        return True
    except Exception as e:
        print(f"[FAIL] 连接失败: {e}")
        return False


def main():
    print("=" * 50)
    print("数据库连接测试")
    print("=" * 50)

    neo4j_ok = test_neo4j()
    pg_ok = test_postgres()

    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print(f"  Neo4j:      {'[OK] 正常' if neo4j_ok else '[FAIL] 失败'}")
    print(f"  PostgreSQL: {'[OK] 正常' if pg_ok else '[FAIL] 失败'}")
    print("=" * 50)


if __name__ == "__main__":
    main()