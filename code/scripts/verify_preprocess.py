"""
验证预处理效果脚本
运行: python scripts/verify_preprocess.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from dotenv import load_dotenv
load_dotenv()

# 设置输出编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

DB_CONFIG = {
    'host': os.getenv('PG_HOST', 'localhost'),
    'port': int(os.getenv('PG_PORT', 5432)),
    'database': os.getenv('PG_DATABASE', 'bishe'),
    'user': os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD', 'postgres')
}

def verify():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        print("=" * 50)
        print("数据量统计:")
        print("=" * 50)

        cur.execute('SELECT COUNT(*) FROM social_media_notes_v2')
        total = cur.fetchone()[0]
        print(f"  总记录: {total}")

        # 验证秒拍/微博视频清理效果
        print("\n" + "=" * 50)
        print("秒拍/微博视频清理验证:")
        print("=" * 50)

        cur.execute('''
            SELECT COUNT(*) FROM social_media_notes_v2
            WHERE content_cleaned LIKE '%秒拍%' OR content_cleaned LIKE '%微博视频%'
        ''')
        remaining = cur.fetchone()[0]
        print(f"  残留记录数: {remaining}")

        if remaining > 0:
            print("\n  部分残留示例:")
            cur.execute('''
                SELECT note_id, content_cleaned FROM social_media_notes_v2
                WHERE content_cleaned LIKE '%秒拍%' OR content_cleaned LIKE '%微博视频%'
                LIMIT 5
            ''')
            for row in cur.fetchall():
                print(f"    {row[0]}: {row[1][:100]}...")

        # 验证私有区Unicode清理效果
        print("\n" + "=" * 50)
        print("私有区Unicode清理验证:")
        print("=" * 50)

        cur.execute('''
            SELECT COUNT(*) FROM social_media_notes_v2
            WHERE content_cleaned ~ '[\\ue000-\\uf8ff]'
        ''')
        pua_remaining = cur.fetchone()[0]
        print(f"  残留记录数: {pua_remaining}")

        # 抽样对比原文和清洗后
        print("\n" + "=" * 50)
        print("抽样对比:")
        print("=" * 50)

        test_ids = ['30d3e02d2209c440', '9cbc95aa119bf633', '2034b5a81f2f6d6c']
        for note_id in test_ids:
            cur.execute('''
                SELECT content, content_cleaned FROM social_media_notes_v2
                WHERE note_id = %s
            ''', (note_id,))
            row = cur.fetchone()
            if row:
                print(f"\n  [{note_id}]")
                print(f"    原文: {row[0][:80]}...")
                print(f"    清洗: {row[1][:80]}...")

        cur.close()
        conn.close()

        print("\n验证完成!")

    except Exception as e:
        print(f"错误: {e}")
        raise


if __name__ == '__main__':
    verify()