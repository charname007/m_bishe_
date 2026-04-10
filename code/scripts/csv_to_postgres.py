"""
CSV to PostgreSQL 导入脚本
仅导入 type='normal' 的数据
"""

import csv
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime


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


def create_table(cur):
    """创建表结构"""
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        note_id VARCHAR(50) PRIMARY KEY,
        type VARCHAR(20),
        title TEXT,
        desc TEXT,
        video_url TEXT,
        time BIGINT,
        last_update_time BIGINT,
        user_id VARCHAR(50),
        nickname VARCHAR(100),
        avatar TEXT,
        liked_count VARCHAR(50),
        collected_count VARCHAR(50),
        comment_count VARCHAR(50),
        share_count VARCHAR(50),
        ip_location VARCHAR(50),
        image_list TEXT,
        tag_list TEXT,
        last_modify_ts BIGINT,
        note_url TEXT,
        source_keyword VARCHAR(100),
        xsec_token TEXT
    );
    """
    cur.execute(create_sql)
    print(f"表 {TABLE_NAME} 创建完成")


def clean_value(value):
    """清理数据值"""
    if value is None or value == '' or value == 'NULL':
        return None
    # 处理数字字段（可能是中文如"1.4万"）
    if any(kw in str(value) for kw in ['万', '亿']):
        return value  # 保留原始中文数值
    return value.strip() if isinstance(value, str) else value


def parse_int_safe(value):
    """安全转换为整数"""
    if not value or value == '':
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


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
        create_table(cur)
        conn.commit()

        # 读取CSV并筛选type='normal'的数据
        print(f"正在读取 {CSV_FILE}...")
        normal_records = []
        total_count = 0

        with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_count += 1
                if row.get('type', '').strip() == 'normal':
                    normal_records.append(row)

        print(f"总记录数: {total_count}, type=normal 记录数: {len(normal_records)}")

        if not normal_records:
            print("没有type=normal的数据需要导入")
            return

        # 批量插入数据
        insert_sql = f"""
        INSERT INTO {TABLE_NAME} (
            note_id, type, title, desc, video_url, time, last_update_time,
            user_id, nickname, avatar, liked_count, collected_count,
            comment_count, share_count, ip_location, image_list, tag_list,
            last_modify_ts, note_url, source_keyword, xsec_token
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (note_id) DO UPDATE SET
            title = EXCLUDED.title,
            desc = EXCLUDED.desc,
            liked_count = EXCLUDED.liked_count,
            collected_count = EXCLUDED.collected_count,
            last_update_time = EXCLUDED.last_update_time
        """

        # 准备数据
        batch_data = []
        for row in normal_records:
            data = (
                clean_value(row.get('note_id')),
                clean_value(row.get('type')),
                clean_value(row.get('title')),
                clean_value(row.get('desc')),
                clean_value(row.get('video_url')),
                parse_int_safe(row.get('time')),
                parse_int_safe(row.get('last_update_time')),
                clean_value(row.get('user_id')),
                clean_value(row.get('nickname')),
                clean_value(row.get('avatar')),
                clean_value(row.get('liked_count')),
                clean_value(row.get('collected_count')),
                clean_value(row.get('comment_count')),
                clean_value(row.get('share_count')),
                clean_value(row.get('ip_location')),
                clean_value(row.get('image_list')),
                clean_value(row.get('tag_list')),
                parse_int_safe(row.get('last_modify_ts')),
                clean_value(row.get('note_url')),
                clean_value(row.get('source_keyword')),
                clean_value(row.get('xsec_token'))
            )
            batch_data.append(data)

        # 批量执行
        print("正在导入数据...")
        execute_batch(cur, insert_sql, batch_data, page_size=100)
        conn.commit()

        print(f"成功导入 {len(batch_data)} 条记录")

        # 验证
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE type = 'normal'")
        count = cur.fetchone()[0]
        print(f"数据库中 type=normal 记录数: {count}")

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
        print("数据库连接已关闭")


if __name__ == '__main__':
    import_csv_to_postgres()