"""
JSON to PostgreSQL 导入脚本（含预处理）
支持单文件或批量文件夹导入
集成文本预处理用于知识图谱实体提取
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime
from glob import glob

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

# JSON文件路径（支持文件夹或单文件）
JSON_PATH = 'data.json'  # 或 'data/*.json'

# 表名
TABLE_NAME = 'xiaohongshu_notes'

# 预处理器
PREPROCESSOR = SocialMediaTextPreprocessor(convert_to_simplified=True)


def create_tables(cur):
    """创建表结构"""
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
        topics JSONB,
        amounts JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    cur.execute(main_table_sql)

    # 索引
    indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_type ON {TABLE_NAME}(type);",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_user_id ON {TABLE_NAME}(user_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_ip_location ON {TABLE_NAME}(ip_location);",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_source_keyword ON {TABLE_NAME}(source_keyword);",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_tags ON {TABLE_NAME} USING GIN(tags);",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_topics ON {TABLE_NAME} USING GIN(topics);",
    ]
    for sql in indexes:
        cur.execute(sql)

    print(f"表 {TABLE_NAME} 创建完成")


def clean_value(value):
    """基础清理"""
    if value is None or value == '' or value == 'NULL':
        return None
    return str(value).strip() if isinstance(value, str) else value


def parse_int_safe(value):
    """安全转整数"""
    if value is None or value == '':
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def preprocess_note(note: dict) -> dict:
    """单条笔记预处理"""
    # 处理标题
    title_result = PREPROCESSOR.preprocess(note.get('title', ''))

    # 处理描述
    desc_result = PREPROCESSOR.preprocess(note.get('desc', ''))

    # 处理标签
    tags = preprocess_tags(note.get('tag_list', ''))

    # 处理数值
    liked_count = preprocess_amount_field(note.get('liked_count'))
    collected_count = preprocess_amount_field(note.get('collected_count'))
    comment_count = parse_int_safe(note.get('comment_count'))
    share_count = parse_int_safe(note.get('share_count'))

    return {
        'note_id': clean_value(note.get('note_id')),
        'type': clean_value(note.get('type')),
        'title': title_result['cleaned_text'] or title_result['original'],
        'desc': desc_result['original'],
        'desc_cleaned': desc_result['cleaned_text'],
        'video_url': clean_value(note.get('video_url')),
        'time': parse_int_safe(note.get('time')),
        'last_update_time': parse_int_safe(note.get('last_update_time')),
        'user_id': clean_value(note.get('user_id')),
        'nickname': clean_value(note.get('nickname')),
        'avatar': clean_value(note.get('avatar')),
        'liked_count': liked_count,
        'collected_count': collected_count,
        'comment_count': comment_count,
        'share_count': share_count,
        'ip_location': clean_value(note.get('ip_location')),
        'image_list': clean_value(note.get('image_list')),
        'tag_list': clean_value(note.get('tag_list')),
        'tags': tags,
        'last_modify_ts': parse_int_safe(note.get('last_modify_ts')),
        'note_url': clean_value(note.get('note_url')),
        'source_keyword': clean_value(note.get('source_keyword')),
        'xsec_token': clean_value(note.get('xsec_token')),
        'topics': title_result['topics'] + desc_result['topics'],
        'amounts': title_result['amounts'] + desc_result['amounts'],
    }


def load_json_files(path: str) -> list:
    """加载JSON文件"""
    notes = []

    # 判断是文件还是目录
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = glob(os.path.join(path, '*.json'))
    else:
        # 支持glob模式
        files = glob(path)

    print(f"找到 {len(files)} 个JSON文件")

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 支持多种JSON结构
                if isinstance(data, list):
                    notes.extend(data)
                elif isinstance(data, dict):
                    # 可能是 {"data": [...]} 或单条记录
                    if 'data' in data:
                        notes.extend(data['data'])
                    elif 'notes' in data:
                        notes.extend(data['notes'])
                    elif 'note_id' in data:
                        notes.append(data)
                    else:
                        print(f"  文件 {file_path} 格式未知，跳过")
        except Exception as e:
            print(f"  加载 {file_path} 失败: {e}")

    return notes


def import_json_to_postgres(json_path: str = JSON_PATH, filter_type: str = 'normal'):
    """主导入函数"""
    conn = None
    cur = None

    try:
        # 连接数据库
        print("连接数据库...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        create_tables(cur)
        conn.commit()

        # 加载JSON
        print(f"加载JSON数据...")
        all_notes = load_json_files(json_path)
        print(f"总记录数: {len(all_notes)}")

        # 筛选并预处理
        processed_records = []
        errors = []

        for note in all_notes:
            note_type = str(note.get('type', '')).strip()
            if note_type != filter_type:
                continue

            try:
                processed = preprocess_note(note)
                processed_records.append(processed)
            except Exception as e:
                errors.append({
                    'note_id': note.get('note_id'),
                    'error': str(e)
                })

        print(f"type={filter_type} 记录数: {len(processed_records)}")
        print(f"预处理错误: {len(errors)}")

        if not processed_records:
            print("无数据导入")
            return

        # 批量插入
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

        # 准备数据
        batch_data = []
        for rec in processed_records:
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

        # 执行
        print("导入数据...")
        batch_size = 100
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            execute_batch(cur, insert_sql, batch)
            if (i + batch_size) % 500 == 0 or i + batch_size >= len(batch_data):
                print(f"  {min(i + batch_size, len(batch_data))}/{len(batch_data)}")
                conn.commit()

        conn.commit()
        print(f"\n导入完成: {len(batch_data)} 条")

        # 统计
        print_stats(cur)

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


def print_stats(cur):
    """打印统计信息"""
    print("\n" + "=" * 50)
    print("数据统计:")
    print("=" * 50)

    cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    print(f"  总记录: {cur.fetchone()[0]}")

    cur.execute(f"""
        SELECT source_keyword, COUNT(*) as cnt
        FROM {TABLE_NAME}
        WHERE source_keyword IS NOT NULL AND source_keyword != ''
        GROUP BY source_keyword
        ORDER BY cnt DESC
        LIMIT 10
    """)
    print("\n  TOP10 搜索关键词:")
    for kw, cnt in cur.fetchall():
        print(f"    {kw}: {cnt}")

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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='JSON导入PostgreSQL')
    parser.add_argument('--path', default=JSON_PATH, help='JSON文件路径')
    parser.add_argument('--type', default='normal', help='筛选type类型')
    args = parser.parse_args()

    import_json_to_postgres(args.path, args.type)