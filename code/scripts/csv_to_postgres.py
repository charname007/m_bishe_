"""
CSV to PostgreSQL 导入脚本（含预处理）
仅导入 type='normal' 的数据
集成文本预处理用于知识图谱实体提取
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime
import json

from utils.text_preprocessor import (
    SocialMediaTextPreprocessor,
    preprocess_tags,
    preprocess_amount_field
)


# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'your_database',
    'user': 'your_user',
    'password': 'your_password'
}

# CSV文件路径
CSV_FILE = 'data.csv'

# 表名
TABLE_NAME = 'xiaohongshu_notes'

# 预处理配置
PREPROCESSOR = SocialMediaTextPreprocessor(convert_to_simplified=True)


def create_tables(cur):
    """创建表结构（主表 + 预处理结果表）"""
    # 主表：存储原始清洗后的数据
    main_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        note_id VARCHAR(50) PRIMARY KEY,
        type VARCHAR(20),
        title TEXT,
        desc TEXT,
        desc_cleaned TEXT,
        video_url TEXT,
        time BIGINT,
        last_update_time BIGINT,
        user_id VARCHAR(50),
        nickname VARCHAR(100),
        avatar TEXT,
        liked_count FLOAT,
        collected_count FLOAT,
        comment_count INTEGER,
        share_count INTEGER,
        ip_location VARCHAR(50),
        image_list TEXT,
        tag_list TEXT,
        tags JSONB,
        last_modify_ts BIGINT,
        note_url TEXT,
        source_keyword VARCHAR(100),
        xsec_token TEXT,
        -- 预处理提取的信息
        topics JSONB,
        amounts JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    cur.execute(main_table_sql)

    # 创建索引
    index_sqls = [
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_type ON {TABLE_NAME}(type);",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_user_id ON {TABLE_NAME}(user_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_ip_location ON {TABLE_NAME}(ip_location);",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_tags ON {TABLE_NAME} USING GIN(tags);",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_topics ON {TABLE_NAME} USING GIN(topics);",
    ]
    for sql in index_sqls:
        cur.execute(sql)

    print(f"表 {TABLE_NAME} 及索引创建完成")


def clean_value(value):
    """基础数据清理"""
    if value is None or value == '' or value == 'NULL':
        return None
    return value.strip() if isinstance(value, str) else value


def parse_int_safe(value):
    """安全转换为整数"""
    if not value or value == '':
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def preprocess_note(row: dict) -> dict:
    """
    对单条笔记进行完整预处理
    返回处理后的数据字典
    """
    # 处理标题
    title_result = PREPROCESSOR.preprocess(row.get('title', ''))

    # 处理描述（主要内容）
    desc_result = PREPROCESSOR.preprocess(row.get('desc', ''))

    # 处理标签列表
    tags = preprocess_tags(row.get('tag_list', ''))

    # 处理数值字段
    liked_count = preprocess_amount_field(row.get('liked_count'))
    collected_count = preprocess_amount_field(row.get('collected_count'))
    comment_count = parse_int_safe(row.get('comment_count'))
    share_count = parse_int_safe(row.get('share_count'))

    return {
        # 原始清洗字段
        'note_id': clean_value(row.get('note_id')),
        'type': clean_value(row.get('type')),
        'title': title_result['cleaned_text'] or title_result['original'],
        'desc': desc_result['original'],  # 保留原始描述
        'desc_cleaned': desc_result['cleaned_text'],  # 清洗后的描述

        # 其他原始字段
        'video_url': clean_value(row.get('video_url')),
        'time': parse_int_safe(row.get('time')),
        'last_update_time': parse_int_safe(row.get('last_update_time')),
        'user_id': clean_value(row.get('user_id')),
        'nickname': clean_value(row.get('nickname')),
        'avatar': clean_value(row.get('avatar')),
        'ip_location': clean_value(row.get('ip_location')),
        'image_list': clean_value(row.get('image_list')),
        'tag_list': clean_value(row.get('tag_list')),
        'last_modify_ts': parse_int_safe(row.get('last_modify_ts')),
        'note_url': clean_value(row.get('note_url')),
        'source_keyword': clean_value(row.get('source_keyword')),
        'xsec_token': clean_value(row.get('xsec_token')),

        # 预处理提取的结构化信息
        'tags': tags,
        'topics': title_result['topics'] + desc_result['topics'],  # 合并标题和描述中的话题
        'amounts': title_result['amounts'] + desc_result['amounts'],

        # 转换后的数值
        'liked_count': liked_count,
        'collected_count': collected_count,
        'comment_count': comment_count,
        'share_count': share_count,
    }


def import_csv_to_postgres():
    """主导入函数"""
    conn = None
    cur = None

    try:
        # 连接数据库
        print("正在连接数据库...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # 创建表
        create_tables(cur)
        conn.commit()

        # 读取CSV并筛选type='normal'的数据
        print(f"正在读取 {CSV_FILE}...")
        normal_records = []
        total_count = 0
        preprocess_errors = []

        with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_count += 1
                if row.get('type', '').strip() == 'normal':
                    try:
                        processed = preprocess_note(row)
                        normal_records.append(processed)
                    except Exception as e:
                        preprocess_errors.append({
                            'note_id': row.get('note_id'),
                            'error': str(e)
                        })

        print(f"总记录数: {total_count}")
        print(f"type=normal 记录数: {len(normal_records)}")
        print(f"预处理错误数: {len(preprocess_errors)}")

        if preprocess_errors:
            print("\n预处理错误详情:")
            for err in preprocess_errors[:5]:  # 只显示前5个
                print(f"  note_id={err['note_id']}: {err['error']}")

        if not normal_records:
            print("没有type=normal的数据需要导入")
            return

        # 批量插入数据
        insert_sql = f"""
        INSERT INTO {TABLE_NAME} (
            note_id, type, title, desc, desc_cleaned, video_url, time,
            last_update_time, user_id, nickname, avatar, liked_count,
            collected_count, comment_count, share_count, ip_location,
            image_list, tag_list, tags, last_modify_ts, note_url,
            source_keyword, xsec_token, topics, amounts
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb
        ) ON CONFLICT (note_id) DO UPDATE SET
            title = EXCLUDED.title,
            desc = EXCLUDED.desc,
            desc_cleaned = EXCLUDED.desc_cleaned,
            liked_count = EXCLUDED.liked_count,
            collected_count = EXCLUDED.collected_count,
            tags = EXCLUDED.tags,
            topics = EXCLUDED.topics,
            amounts = EXCLUDED.amounts,
            last_update_time = EXCLUDED.last_update_time
        """

        # 准备批量数据
        batch_data = []
        for rec in normal_records:
            data = (
                rec['note_id'],
                rec['type'],
                rec['title'],
                rec['desc'],
                rec['desc_cleaned'],
                rec['video_url'],
                rec['time'],
                rec['last_update_time'],
                rec['user_id'],
                rec['nickname'],
                rec['avatar'],
                rec['liked_count'],
                rec['collected_count'],
                rec['comment_count'],
                rec['share_count'],
                rec['ip_location'],
                rec['image_list'],
                rec['tag_list'],
                json.dumps(rec['tags']) if rec['tags'] else None,
                rec['last_modify_ts'],
                rec['note_url'],
                rec['source_keyword'],
                rec['xsec_token'],
                json.dumps(rec['topics']) if rec['topics'] else None,
                json.dumps(rec['amounts']) if rec['amounts'] else None,
            )
            batch_data.append(data)

        # 执行批量插入
        print("正在导入数据...")
        batch_size = 100
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            execute_batch(cur, insert_sql, batch)
            if (i + batch_size) % 500 == 0 or i + batch_size >= len(batch_data):
                print(f"  已处理 {min(i + batch_size, len(batch_data))}/{len(batch_data)} 条")
                conn.commit()

        conn.commit()
        print(f"\n成功导入 {len(batch_data)} 条记录")

        # 统计信息
        print("\n" + "=" * 50)
        print("导入统计:")
        print("=" * 50)

        # 基础统计
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE type = 'normal'")
        count = cur.fetchone()[0]
        print(f"  type=normal 总记录数: {count}")

        # IP地域分布
        cur.execute(f"""
            SELECT ip_location, COUNT(*) as cnt
            FROM {TABLE_NAME}
            WHERE ip_location IS NOT NULL AND ip_location != ''
            GROUP BY ip_location
            ORDER BY cnt DESC
            LIMIT 10
        """)
        print("\n  TOP10 IP地域分布:")
        for loc, cnt in cur.fetchall():
            print(f"    {loc}: {cnt}")

        # 话题统计
        cur.execute(f"""
            SELECT jsonb_array_elements_text(topics) as topic, COUNT(*) as cnt
            FROM {TABLE_NAME}
            WHERE topics IS NOT NULL AND jsonb_array_length(topics) > 0
            GROUP BY topic
            ORDER BY cnt DESC
            LIMIT 10
        """)
        print("\n  TOP10 话题:")
        for topic, cnt in cur.fetchall():
            print(f"    {topic}: {cnt}")

        # 标签统计
        cur.execute(f"""
            SELECT jsonb_array_elements_text(tags) as tag, COUNT(*) as cnt
            FROM {TABLE_NAME}
            WHERE tags IS NOT NULL AND jsonb_array_length(tags) > 0
            GROUP BY tag
            ORDER BY cnt DESC
            LIMIT 10
        """)
        print("\n  TOP10 标签:")
        for tag, cnt in cur.fetchall():
            print(f"    {tag}: {cnt}")

    except Exception as e:
        print(f"错误: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        print("\n数据库连接已关闭")


if __name__ == '__main__':
    import_csv_to_postgres()