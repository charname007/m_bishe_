"""
语料分层抽样脚本
从武汉相关数据中按话题分层抽样至目标数量
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from dotenv import load_dotenv
import random

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('PG_HOST'),
    'port': int(os.getenv('PG_PORT')),
    'database': os.getenv('PG_DATABASE'),
    'user': os.getenv('PG_USER'),
    'password': os.getenv('PG_PASSWORD')
}

# 武汉关键词
WUHAN_KEYWORDS = [
    '武汉', '洪山', '光谷', '华科', '华师', '地大', '武理',
    '街道口', '虎泉', '卓刀泉', '白沙洲', '广埠屯', '鲁磨路',
    '华农', '财大', '武体', '关山大道', '珞狮路', '雄楚大道'
]

# 上海关键词（排除）
SHANGHAI_KEYWORDS = [
    '上海', '沪', '浦东', '静安', '徐汇', '黄浦', '长宁',
    '普陀', '虹口', '杨浦', '闵行', '宝山', '嘉定', '松江',
    '青浦', '奉贤', '金山'
]

# 分层话题关键词
LAYER_KEYWORDS = {
    'rent': ['武汉租房', '光谷租房', '租房', '洪山租房', '华科租房', '华师租房'],
    'university': ['华科', '华中科技大学', '华师', '华中师范大学', '地大', '中国地质大学',
                   '武理', '武汉理工大学', '华农', '华中农业大学', '财大', '中南财经政法大学',
                   '武体', '武汉体育学院', '湖北工业大学', '武汉工程大学'],
    'food': ['武汉美食', '光谷美食', '洪山美食', '武汉吃喝', '光谷吃喝'],
    'life': ['日常', '生活', 'vlog', '日记', '打卡'],
}

TARGET_COUNT = 30000


def classify_layer(row_topics: list, row_keyword: str, row_content: str) -> str:
    """根据话题、关键词、内容分类到层级"""
    # 合并所有文本来源
    all_text = f"{row_keyword} {row_content}"
    topics_str = ' '.join(row_topics) if row_topics else ''

    # 检查各层关键词
    for layer, keywords in LAYER_KEYWORDS.items():
        for kw in keywords:
            if kw in topics_str or kw in all_text:
                return layer

    return 'other'


def get_wuhan_data(cur):
    """筛选武汉数据（不含上海关键词）"""
    wuhan_cond = ' OR '.join([
        f"(source_keyword LIKE '%{kw}%' OR content_cleaned LIKE '%{kw}%')"
        for kw in WUHAN_KEYWORDS
    ])
    shanghai_cond = ' OR '.join([
        f"(source_keyword LIKE '%{kw}%' OR content_cleaned LIKE '%{kw}%')"
        for kw in SHANGHAI_KEYWORDS
    ])

    sql = f"""
    SELECT note_id, source_keyword, content_cleaned, topics
    FROM social_media_notes
    WHERE ({wuhan_cond}) AND NOT ({shanghai_cond})
    """
    cur.execute(sql)
    return cur.fetchall()


def stratified_sample(data: list, target: int) -> list:
    """分层抽样"""
    # 分类
    layers = {}
    for row in data:
        note_id, keyword, content, topics = row
        layer = classify_layer(topics or [], keyword or '', content or '')
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(note_id)

    # 打印各层统计
    total = sum(len(v) for v in layers.values())
    print(f"\n原始分布: 总计 {total} 条")
    for layer, ids in sorted(layers.items()):
        print(f"  {layer}: {len(ids)} 条")

    # 计算抽样率
    sample_rate = target / total

    # 分层抽样
    sampled_ids = []
    for layer, ids in layers.items():
        layer_target = int(len(ids) * sample_rate)
        sampled = random.sample(ids, min(layer_target, len(ids)))
        sampled_ids.extend(sampled)
        print(f"  {layer} 抽样: {len(sampled)} 条")

    return sampled_ids


def export_sampled_data(cur, sampled_ids: list):
    """导出抽样数据到新表"""
    # 创建新表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS social_media_notes_sampled (
            LIKE social_media_notes INCLUDING ALL
        )
    """)

    # 清空表
    cur.execute("TRUNCATE TABLE social_media_notes_sampled")

    # 复制数据
    ids_str = ','.join([f"'{id}'" for id in sampled_ids])
    cur.execute(f"""
        INSERT INTO social_media_notes_sampled
        SELECT * FROM social_media_notes
        WHERE note_id IN ({ids_str})
    """)


def main():
    print("连接数据库...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print("筛选武汉数据...")
    wuhan_data = get_wuhan_data(cur)
    print(f"武汉数据: {len(wuhan_data)} 条")

    print("分层抽样...")
    sampled_ids = stratified_sample(wuhan_data, TARGET_COUNT)
    print(f"抽样结果: {len(sampled_ids)} 条")

    print("导出数据...")
    export_sampled_data(cur, sampled_ids)
    conn.commit()

    # 验证
    cur.execute("SELECT COUNT(*) FROM social_media_notes_sampled")
    final_count = cur.fetchone()[0]
    print(f"\n完成! 导入 {final_count} 条数据到 social_media_notes_sampled 表")

    cur.close()
    conn.close()


if __name__ == '__main__':
    main()