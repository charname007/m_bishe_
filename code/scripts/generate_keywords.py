"""
洪山区社交媒体关键词生成器（微博版）

用于采集微博平台的语料，生成搜索关键词组合。

微博搜索特点：
    - 空格分隔：如 "光谷 美食"
    - 引号精确匹配：如 "光谷美食"
    - 话题格式：如 #光谷美食#

运行方式：
    python scripts/generate_keywords.py

输出：
    - hongshan_keywords.txt: 每行一个关键词（多种格式混合）
    - hongshan_keywords.json: 带分类的JSON格式
"""

import json
from pathlib import Path
from datetime import datetime


# ============================================================
# 地点池定义
# ============================================================

# 行政区与核心商圈
DISTRICTS = [
    "洪山区", "光谷", "街道口", "南湖", "虎泉", "卓刀泉", "白沙洲",
    "关山大道", "珞狮路", "鲁巷", "广埠屯", "鲁磨路", "民族大道",
    "雄楚大道", "珞瑜路", "杨家湾",
]

# 热门商场与地标
LANDMARKS = [
    "光谷步行街", "光谷大悦城", "银泰创意城", "群光广场", "光谷天地",
    "泛悦城", "K11Select", "维佳体验城", "光谷广场", "世界城广场",
]

# 高校
UNIVERSITIES = [
    "华科", "华中科技大学", "华师", "华中师范大学", "武理", "武汉理工大学",
    "地大", "中国地质大学", "华农", "华中农业大学", "财大", "中南财经政法大学",
    "武体", "武汉体育学院", "湖北工业大学", "武汉工程大学",
]

# 园区与景区
PARKS = [
    "光谷软件园", "光谷金融港", "马鞍山森林公园", "欢乐谷", "东湖绿道",
]

# 热门小区（租房高频）
RESIDENTIAL = [
    "光谷青年城", "万科城市花园", "保利华都", "当代国际花园",
    "光谷理想城", "光谷8号", "清江山水", "锦绣龙城",
]

# 地铁站
METRO_STATIONS = [
    "光谷广场站", "华中科技大学站", "杨家湾站", "街道口站",
    "广埠屯站", "虎泉站", "卓刀泉站", "宝通寺站",
]


# ============================================================
# 行为池定义（微博高频词为主）
# ============================================================

# 餐饮美食
EAT_ACTIONS = [
    "美食", "好吃的", "餐厅", "烧烤", "火锅", "宵夜",
    "甜品", "奶茶", "面包店", "咖啡店", "夜市", "小吃",
    "自助餐", "日料", "韩料", "川菜", "湘菜", "外卖",
]

# 休闲娱乐
PLAY_ACTIONS = [
    "探店", "打卡", "拍照", "周末", "好玩", "约会",
    "酒吧", "KTV", "电影院", "遛娃", "公园", "游玩",
]

# 生活服务
LIFE_ACTIONS = [
    "租房", "合租", "找室友", "二手房", "搬家", "停车",
    "健身房", "瑜伽", "游泳", "自习室", "图书馆",
    "美甲", "理发", "染发", "医美", "牙科",
    "宠物医院", "驾校",
]

# 求职工作
WORK_ACTIONS = [
    "求职", "招聘", "兼职", "实习", "面试", "简历",
    "写字楼", "创业", "加班", "通勤",
]

# 购物消费
SHOP_ACTIONS = [
    "买衣服", "逛街", "商场", "超市", "菜市场",
    "二手", "闲置", "数码", "手机",
]

# 信息查询
INFO_ACTIONS = [
    "攻略", "推荐", "避雷", "吐槽", "测评", "合集",
    "排名", "哪个好", "多少钱", "怎么去", "体验",
]

# 微博高频词（核心特色）
WEIBO高频词 = [
    "吐槽", "避雷", "日常", "vlog", "分享", "记录",
    "真实", "踩坑", "求助", "爆料", "建议", "感受",
    "怎么样", "有没有", "怎么样", "好不好", "能不能",
]

# 微博情绪词
WEIBO情绪词 = [
    "无语", "崩溃", "气死", "开心", "惊喜", "失望",
    "后悔", "推荐", "不推荐", "值得", "不值得",
]

# 季节性/时效性
SEASONAL = [
    "樱花", "赏花", "毕业季", "开学", "暑假", "寒假",
    "圣诞节", "跨年", "情人节", "五一", "十一",
]


# ============================================================
# 组合策略（微博格式：空格分隔、引号精确、#话题#）
# ============================================================

def generate_keywords():
    """生成关键词列表，返回列表和字典（带分类）"""
    keywords = []
    keywords_with_category = {}

    # -------- 策略1：核心区域 × 生活行为 --------
    category = "核心区域生活"
    core_locations = ["洪山区", "光谷", "街道口", "南湖", "虎泉", "卓刀泉", "白沙洲", "关山大道"]
    core_actions = [
        "美食", "好吃的", "烧烤", "火锅", "宵夜", "甜品", "奶茶", "咖啡店", "夜市", "小吃",
        "探店", "打卡", "拍照", "周末", "约会", "好玩",
        "租房", "合租", "健身房", "美甲", "理发", "驾校", "宠物医院",
        "吐槽", "避雷", "攻略", "推荐", "日常", "vlog", "求助",
    ]

    for loc in core_locations:
        for action in core_actions:
            kw = f"{loc} {action}"
            keywords.append(kw)
            keywords_with_category.setdefault(category, []).append(kw)

    # -------- 策略2：商场地标 × 消费行为 --------
    category = "商场地标消费"
    consume_actions = [
        "美食", "好吃的", "火锅", "烧烤", "甜品", "奶茶", "咖啡店",
        "探店", "打卡", "拍照", "约会", "逛街", "买衣服",
        "推荐", "避雷", "吐槽", "攻略",
    ]

    for loc in LANDMARKS:
        for action in consume_actions:
            kw = f"{loc} {action}"
            keywords.append(kw)
            keywords_with_category.setdefault(category, []).append(kw)

    # -------- 策略3：高校 × 校园生活 --------
    category = "高校校园生活"
    universities_short = ["华科", "华师", "武理", "地大", "华农", "财大", "武体", "湖北工业大学"]
    campus_actions = [
        "食堂", "周边美食", "图书馆", "自习室", "租房", "兼职",
        "快递", "健身房", "美甲", "理发", "驾校",
        "毕业", "开学", "考试", "社团", "宿舍", "吐槽", "避雷",
    ]

    for loc in universities_short:
        for action in campus_actions:
            kw = f"{loc} {action}"
            keywords.append(kw)
            keywords_with_category.setdefault(category, []).append(kw)

    # -------- 策略4：园区 × 上班族 --------
    category = "园区职场"
    work_actions = ["美食", "外卖", "咖啡店", "健身房", "租房", "停车", "通勤", "加班", "吐槽", "避雷"]

    for loc in PARKS:
        for action in work_actions:
            kw = f"{loc} {action}"
            keywords.append(kw)
            keywords_with_category.setdefault(category, []).append(kw)

    # -------- 策略5：小区租房 --------
    category = "小区租房"
    rental_actions = ["租房", "合租", "找室友", "房价", "物业", "交通", "避雷", "吐槽"]

    for loc in RESIDENTIAL:
        for action in rental_actions:
            kw = f"{loc} {action}"
            keywords.append(kw)
            keywords_with_category.setdefault(category, []).append(kw)

    # -------- 策略6：地铁站 --------
    category = "地铁站周边"
    station_actions = ["美食", "停车", "租房", "换乘"]

    for loc in METRO_STATIONS:
        for action in station_actions:
            kw = f"{loc} {action}"
            keywords.append(kw)
            keywords_with_category.setdefault(category, []).append(kw)

    # -------- 策略7：季节性 --------
    category = "季节时效"
    for loc in ["洪山区", "光谷", "华科", "街道口"]:
        for s in ["樱花", "赏花", "毕业季", "开学", "暑假", "圣诞节", "跨年"]:
            kw = f"{loc} {s}"
            keywords.append(kw)
            keywords_with_category.setdefault(category, []).append(kw)

    # -------- 策略8：微博话题 --------
    category = "微博话题"
    for loc in ["洪山区", "光谷", "街道口", "华科", "南湖", "虎泉"]:
        kw = f"#{loc}#"
        keywords.append(kw)
        keywords_with_category.setdefault(category, []).append(kw)
        for word in ["美食", "租房", "探店", "吐槽", "避雷", "攻略", "日常", "vlog", "打卡"]:
            kw = f"#{loc}{word}#"
            keywords.append(kw)
            keywords_with_category.setdefault(category, []).append(kw)

    # -------- 策略9：长尾精准词 --------
    category = "长尾精准"
    long_tail = [
        "洪山区 政务服务中心", "洪山区 人民医院", "洪山区 图书馆",
        "光谷 网红打卡", "光谷 相亲", "光谷 夜跑", "光谷 周末 带娃",
        "华科 哪个食堂好吃", "华科 女生", "武理 男女比例", "地大 珠宝学院", "华农 赏花",
        "街道口 美甲 推荐", "南湖 健身房 推荐",
        "光谷 到机场怎么走", "洪山区 地铁站",
        "光谷 有没有 好吃的", "街道口 能不能 停车", "华科 怎么样", "南湖 租房 建议",
    ]
    keywords.extend(long_tail)
    keywords_with_category[category] = long_tail

    # 去重并排序
    keywords = sorted(set(keywords))

    return keywords, keywords_with_category


def save_to_txt(keywords: list, output_path: Path):
    """保存为TXT文件，每行一个关键词"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# 洪山区社交媒体关键词列表\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 关键词总数: {len(keywords)}\n")
        f.write(f"# 用途: 小红书、微博等社交媒体语料采集\n")
        f.write("#" + "="*50 + "\n\n")

        for kw in keywords:
            f.write(kw + "\n")

    print(f"已保存 TXT 文件: {output_path}")
    print(f"关键词总数: {len(keywords)}")


def save_to_json(keywords_with_category: dict, output_path: Path):
    """保存为JSON文件，带分类信息"""
    result = {
        "meta": {
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_keywords": sum(len(v) for v in keywords_with_category.values()),
            "categories": list(keywords_with_category.keys()),
        },
        "categories": {}
    }

    for category, kws in keywords_with_category.items():
        result["categories"][category] = {
            "count": len(kws),
            "keywords": sorted(set(kws))
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已保存 JSON 文件: {output_path}")

    # 打印分类统计
    print("\n分类统计:")
    print("-" * 40)
    for cat, data in result["categories"].items():
        print(f"  {cat}: {data['count']} 个")


def main():
    """主函数"""
    # 项目根目录
    project_root = Path(__file__).parent.parent

    # 生成关键词
    print("正在生成关键词...")
    keywords, keywords_with_category = generate_keywords()

    # 保存文件
    save_to_txt(keywords, project_root / "hongshan_keywords.txt")
    save_to_json(keywords_with_category, project_root / "hongshan_keywords.json")

    print("\n生成完成!")


if __name__ == "__main__":
    main()