"""
测试50条语料 - 验证属性保存
"""
import asyncio
import sys
import os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env')

from langchain_openai import ChatOpenAI
from agent.agents.workflow import build_distributed_workflow
from agent.agents.config import ExtractionConfig
from collections import defaultdict


# 50条测试语料
TEST_CORPUS = [
    {'id': 'test_001', 'text': '武大的樱花开了，很多学生在行政楼前拍照打卡'},
    {'id': 'test_002', 'text': '珞喻路上的群光广场适合逛街购物，周末人很多'},
    {'id': 'test_003', 'text': '光谷步行街很热闹，有很多小吃店和奶茶店'},
    {'id': 'test_004', 'text': '汉街的万达广场就在楚河汉街旁边，购物方便'},
    {'id': 'test_005', 'text': '江汉路步行街是武汉最繁华的商业街'},
    {'id': 'test_006', 'text': '黄鹤楼在武昌蛇山，是武汉地标建筑'},
    {'id': 'test_007', 'text': '东湖绿道适合骑行，风景很好'},
    {'id': 'test_008', 'text': '华中科技大学的校园很大，在西园食堂吃饭'},
    {'id': 'test_009', 'text': '街道口地铁站附近有广埠屯电脑城'},
    {'id': 'test_010', 'text': '司门口户部巷有很多小吃，热干面好吃'},
    {'id': 'test_011', 'text': '武汉天地是年轻人聚会的好地方'},
    {'id': 'test_012', 'text': '归元寺在汉阳，每年春节很多人去祈福'},
    {'id': 'test_013', 'text': '长江大桥连接武昌和汉口'},
    {'id': 'test_014', 'text': '武汉大学的老图书馆是民国建筑'},
    {'id': 'test_015', 'text': '汉口江滩晚上很多人散步'},
    {'id': 'test_016', 'text': '中山公园是武汉市中心的公园'},
    {'id': 'test_017', 'text': '解放大道上有武汉展览馆'},
    {'id': 'test_018', 'text': '武昌火车站旁边有很多酒店'},
    {'id': 'test_019', 'text': '汉正街是小商品批发市场'},
    {'id': 'test_020', 'text': '宝通寺在洪山区，是个古寺'},
    {'id': 'test_021', 'text': '光谷国际广场在光谷中心'},
    {'id': 'test_022', 'text': '卓刀泉路有华中师范大学'},
    {'id': 'test_023', 'text': '解放公园在汉口，适合遛娃'},
    {'id': 'test_024', 'text': '月湖桥连接汉阳和汉口'},
    {'id': 'test_025', 'text': '南湖那边有很多大学'},
    {'id': 'test_026', 'text': '武汉体育中心在沌口'},
    {'id': 'test_027', 'text': '航空路有同济医院'},
    {'id': 'test_028', 'text': '常青花园是大型居住区'},
    {'id': 'test_029', 'text': '金银湖是汉口的湖泊'},
    {'id': 'test_030', 'text': '沙湖公园在武昌'},
    {'id': 'test_031', 'text': '徐东大街有徐东平价广场'},
    {'id': 'test_032', 'text': '建设大道上有建设银行大楼'},
    {'id': 'test_033', 'text': '欢乐谷在华侨城，适合周末游玩'},
    {'id': 'test_034', 'text': '武汉植物园在东湖边'},
    {'id': 'test_035', 'text': '首义广场在武昌首义路'},
    {'id': 'test_036', 'text': '昙华林是文艺街区，有很多咖啡馆'},
    {'id': 'test_037', 'text': '洪山广场在武昌洪山路'},
    {'id': 'test_038', 'text': '青年路有青年公园'},
    {'id': 'test_039', 'text': '武汉博物馆在汉口青年路'},
    {'id': 'test_040', 'text': '武昌江滩比汉口江滩安静'},
    {'id': 'test_041', 'text': '后湖是汉口的新区'},
    {'id': 'test_042', 'text': '古田是汉口的工业区'},
    {'id': 'test_043', 'text': '东西湖有金银潭医院'},
    {'id': 'test_044', 'text': '沌口开发区有东风汽车'},
    {'id': 'test_045', 'text': '盘龙城是汉口北部的新区'},
    {'id': 'test_046', 'text': '江夏区有很多工厂'},
    {'id': 'test_047', 'text': '蔡甸区在中法生态城'},
    {'id': 'test_048', 'text': '新洲区比较偏远'},
    {'id': 'test_049', 'text': '黄陂区有木兰山'},
    {'id': 'test_050', 'text': '汉南区是武汉开发区'},
]


async def test_50_corpus():
    print('='*60)
    print('Testing 50 corpus with full workflow')
    print('='*60)

    # 创建LLM
    api_key = os.getenv('DEEPSEEK_API_KEY')
    base_url = os.getenv('DEEPSEEK_API_BASE_URL')
    model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)

    # 创建配置
    config = ExtractionConfig.from_env()
    config.enable_entity_alignment = True  # 启用实体对齐节点
    config.enable_batch_llm = True  # P15修复：启用批量模式测试

    # 构建分布式工作流
    workflow = build_distributed_workflow(llm, config)

    print(f'Running workflow with {len(TEST_CORPUS)} corpus...')

    initial_state = {
        'batch_id': 'test_batch_50',
        'corpus_list': TEST_CORPUS,
        'worker_count': 5,
        'total_count': len(TEST_CORPUS),
    }

    thread_config = {'configurable': {'thread_id': 'test_50_corpus'}}

    result = await workflow.ainvoke(initial_state, thread_config)

    # 分析结果
    print('\n' + '-'*40)
    print('RESULTS SUMMARY')
    print('-'*40)

    aggregated_entities = result.get('aggregated_entities', [])
    aggregated_triples = result.get('aggregated_triples', [])

    print(f'Total entities extracted: {len(aggregated_entities)}')
    print(f'Total triples extracted: {len(aggregated_triples)}')

    # 统计各类型实体数量
    entity_type_counts = defaultdict(int)
    for e in aggregated_entities:
        entity_type_counts[e.get('type', 'Unknown')] += 1

    print('\nEntity type distribution:')
    for t, c in sorted(entity_type_counts.items(), key=lambda x: -x[1]):
        print(f'  {t}: {c}')

    # 统计关系类型数量
    relation_type_counts = defaultdict(int)
    for t in aggregated_triples:
        relation_type_counts[t.get('relation', 'Unknown')] += 1

    print('\nRelation type distribution:')
    for r, c in sorted(relation_type_counts.items(), key=lambda x: -x[1]):
        print(f'  {r}: {c}')

    # 统计有category的实体
    entities_with_category = 0
    entities_without_category = 0
    for e in aggregated_entities:
        if e.get('category') and e.get('category') != '':
            entities_with_category += 1
        else:
            entities_without_category += 1

    print(f'\nEntities with category: {entities_with_category}')
    print(f'Entities without category: {entities_without_category}')

    # 查看功能实体和事件实体的属性
    func_entities = [e for e in aggregated_entities if e.get('type') == '功能']
    event_entities = [e for e in aggregated_entities if e.get('type') == '事件']

    print(f'\nFunction entities ({len(func_entities)}):')
    for e in func_entities[:5]:
        attrs = e.get('attrs', {})
        print(f'  - {e.get("name")}: func_type={attrs.get("功能类型")}, category={e.get("category")}')

    print(f'\nEvent entities ({len(event_entities)}):')
    for e in event_entities[:5]:
        attrs = e.get('attrs', {})
        print(f'  - {e.get("name")}: event_category={attrs.get("事件类别")}, category={e.get("category")}')

    # 查看地理实体的属性
    geo_entities = [e for e in aggregated_entities if e.get('type') in ['POI', '道路', '建筑物', '街区']]
    print(f'\nGeo entities sample ({len(geo_entities)} total):')
    for e in geo_entities[:10]:
        print(f'  - {e.get("name")}: type={e.get("type")}, category={e.get("category")}')

    return result


if __name__ == '__main__':
    asyncio.run(test_50_corpus())