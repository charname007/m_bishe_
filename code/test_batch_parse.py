"""测试 BatchExtractionResult 解析错误"""
import json
import sys
sys.path.insert(0, '.')
from agent.agents.schemas import (
    BatchExtractionResult, BatchCorpusResult, JointEntity, JointTriple,
    TripleAttributes, FunctionEntityAttributes, EntityTypeEnum, FunctionEnum
)

# 直接使用日志中的 JSON 数据（第一段完整数据）
test_json = """
{"results": [{"corpus_id": "003622a343071b8a", "entities": {"道路": [], "POI": ["华科", "篮球场", "图书馆"], "建筑物": [], "街区": [], "功能": ["运动", "学习"], "事件": []}, "full_entities": [{"name": "华科", "type": "POI", "category": "大学", "aliases": [], "evidence": "帮我去华科逛了逛", "function_attrs": null, "event_attrs": null}, {"name": "篮球场", "type": "POI", "category": "运动场所", "aliases": [], "evidence": "去了篮球场🏀", "function_attrs": null, "event_attrs": null}, {"name": "图书馆", "type": "POI", "category": "文化场所", "aliases": [], "evidence": "图书馆还有好多地方", "function_attrs": null, "event_attrs": null}, {"name": "运动", "type": "功能", "category": "休闲", "aliases": [], "evidence": "去了篮球场🏀", "function_attrs": {"功能类型": "休闲", "功能细分": "篮球运动", "适合时段": null, "适合人群": null, "具有限制": null, "情感倾向": "正面", "推荐指数": null, "evidence": "去了篮球场🏀"}, "event_attrs": null}, {"name": "学习", "type": "功能", "category": "文化", "aliases": [], "evidence": "图书馆还有好多地方", "function_attrs": {"功能类型": "文化", "功能细分": "阅读学习", "适合时段": null, "适合人群": null, "具有限制": null, "情感倾向": "正面", "推荐指数": null, "evidence": "图书馆还有好多地方"}, "event_attrs": null}], "triples": [{"head": "篮球场", "relation": "位于", "tail": "华科", "evidence": "去了篮球场🏀图书馆还有好多地方", "confidence": "high", "attributes": null}, {"head": "图书馆", "relation": "位于", "tail": "华科", "evidence": "去了篮球场🏀图书馆还有好多地方", "confidence": "high", "attributes": null}, {"head": "华科", "relation": "具有功能", "tail": "运动", "evidence": "去了篮球场🏀", "confidence": "high", "attributes": {"情感倾向": "正面"}}, {"head": "华科", "relation": "具有功能", "tail": "学习", "evidence": "图书馆还有好多地方", "confidence": "high", "attributes": {"情感倾向": "正面"}}], "confidence": "high", "has_geo_info": true, "skip_reason": null}], "cross_corpus_aliases": [], "cross_corpus_relations": [], "overall_confidence": "high", "batch_size": 5, "extraction_strategy": "batch_joint"}
"""

print("=== 测试逐层验证 ===\n")

try:
    data = json.loads(test_json)
    print("✓ JSON 解析成功")

    # 逐个测试嵌套模型
    print("\n1. 测试 JointEntity 验证:")
    for i, entity_data in enumerate(data["results"][0]["full_entities"]):
        try:
            entity = JointEntity.model_validate(entity_data)
            print(f"  ✓ Entity {i}: {entity.name} (type={entity.type})")
            if entity.function_attrs:
                print(f"    function_attrs 功能类型: {entity.function_attrs.功能类型}")
        except Exception as e:
            print(f"  ✗ Entity {i} 失败: {entity_data.get('name', 'unknown')}")
            print(f"    错误: {e}")
            if hasattr(e, 'errors'):
                for err in e.errors():
                    print(f"      - {err.get('loc', [])}: {err.get('msg', '')}")

    print("\n2. 测试 JointTriple 验证:")
    for i, triple_data in enumerate(data["results"][0]["triples"]):
        try:
            triple = JointTriple.model_validate(triple_data)
            print(f"  ✓ Triple {i}: {triple.head} -> {triple.relation} -> {triple.tail}")
        except Exception as e:
            print(f"  ✗ Triple {i} 失败: {triple_data}")
            print(f"    错误: {e}")
            if hasattr(e, 'errors'):
                for err in e.errors():
                    print(f"      - {err.get('loc', [])}: {err.get('msg', '')}")

    print("\n3. 测试 BatchCorpusResult 验证:")
    try:
        corpus = BatchCorpusResult.model_validate(data["results"][0])
        print(f"  ✓ BatchCorpusResult: {corpus.corpus_id}")
    except Exception as e:
        print(f"  ✗ BatchCorpusResult 失败")
        print(f"    错误: {e}")
        if hasattr(e, 'errors'):
            for err in e.errors():
                print(f"      - {err.get('loc', [])}: {err.get('msg', '')}")

    print("\n4. 测试 BatchExtractionResult 验证:")
    try:
        result = BatchExtractionResult.model_validate(data)
        print('  ✓ BatchExtractionResult 解析成功')
        print(f"    batch_size: {result.batch_size}")
        print(f"    overall_confidence: {result.overall_confidence}")
    except Exception as e:
        print(f'  ✗ BatchExtractionResult 失败')
        print(f'    错误类型: {type(e).__name__}')
        print(f'    详细信息: {e}')
        if hasattr(e, 'errors'):
            for err in e.errors():
                print(f"      - 字段: {err.get('loc', [])}")
                print(f"        类型: {err.get('type', '')}")
                print(f"        消息: {err.get('msg', '')}")
                print(f"        输入: {err.get('input', '')}")

except json.JSONDecodeError as e:
    print(f'JSON解析错误: {e}')
except Exception as e:
    print(f'未知错误: {type(e).__name__}: {e}')