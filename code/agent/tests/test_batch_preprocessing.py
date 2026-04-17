"""
测试批量预处理节点和流程
"""
import asyncio
import sys
import os
sys.path.insert(0, "e:\\study\\毕设\\code")

# 设置编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from agent.agents import (
    ExtractionConfig,
    create_batch_filter_node,
    create_batch_normalize_node,
    create_batch_qa_scaffold_node,
    process_batch_preprocessing,
    BATCH_FILTER_PROMPT,
    BATCH_NORMALIZE_PROMPT,
    BATCH_QA_SCAFFOLD_PROMPT,
    BatchFilterItemResult,
    BatchNormalizeItemResult,
    BatchQAScaffoldItemResult,
)


import pytest


@pytest.mark.asyncio
async def test_batch_preprocessing_without_llm():
    """测试批量预处理节点的创建（不实际调用LLM）"""

    # 创建测试语料
    test_corpus = [
        {"id": "test_1", "text": "武汉大学位于珞珈山下，附近有广八路美食街"},
        {"id": "test_2", "text": "今天去了光谷步行街，感觉很热闹"},
        {"id": "test_3", "text": "华科的学生经常去东湖绿道骑行"},
    ]

    # 测试配置
    config = ExtractionConfig.from_env()

    print("=" * 50)
    print("测试批量预处理节点创建")
    print("=" * 50)

    # 测试节点创建函数
    print("1. 创建 batch_filter_node...")
    try:
        # 模拟LLM（不实际调用）
        class MockLLM:
            async def ainvoke(self, prompt):
                # 返回模拟的JSON响应（使用新的 schema 结构）
                return type('obj', (object,), {'content': '''
{
    "results": [
        {"corpus_id": "test_1", "is_valid": true, "skip_reason": null, "confidence": "high", "has_geo_entity": true, "has_spatial_relation": true, "is_non_wuhan_region": false},
        {"corpus_id": "test_2", "is_valid": true, "skip_reason": null, "confidence": "medium", "has_geo_entity": true, "has_spatial_relation": false, "is_non_wuhan_region": false},
        {"corpus_id": "test_3", "is_valid": true, "skip_reason": null, "confidence": "high", "has_geo_entity": true, "has_spatial_relation": false, "is_non_wuhan_region": false}
    ],
    "overall_confidence": "high",
    "batch_size": 3
}
'''})()

        mock_llm = MockLLM()
        filter_node = create_batch_filter_node(mock_llm)
        print("   [OK] create_batch_filter_node 成功")

        # 为归一化节点创建另一个mock
        class MockLLM2:
            async def ainvoke(self, prompt):
                return type('obj', (object,), {'content': '''
{
    "results": [
        {"corpus_id": "test_1", "normalized_text": "武汉大学位于珞珈山下，附近有广八路美食街", "aliases": {"武大": "武汉大学"}, "confidence": "high"},
        {"corpus_id": "test_2", "normalized_text": "今天去了光谷步行街，感觉很热闹", "aliases": {}, "confidence": "medium"},
        {"corpus_id": "test_3", "normalized_text": "华中科技大学的学生经常去东湖绿道骑行", "aliases": {"华科": "华中科技大学"}, "confidence": "high"}
    ],
    "overall_confidence": "high",
    "batch_size": 3
}
'''})()

        normalize_node = create_batch_normalize_node(MockLLM2())
        print("   [OK] create_batch_normalize_node 成功")

        # 为QA脚手架节点创建另一个mock
        class MockLLM3:
            async def ainvoke(self, prompt):
                return type('obj', (object,), {'content': '''
{
    "results": [
        {"corpus_id": "test_1", "qa_pairs": [{"question": "武汉大学在哪里?", "answer": "珞珈山"}], "entity_hints": ["武汉大学", "珞珈山", "广八路"], "relation_hints": ["位于", "附近"], "context_dependencies": [], "overall_confidence": "high"},
        {"corpus_id": "test_2", "qa_pairs": [], "entity_hints": ["光谷步行街"], "relation_hints": [], "context_dependencies": [], "overall_confidence": "medium"},
        {"corpus_id": "test_3", "qa_pairs": [], "entity_hints": ["华中科技大学", "东湖绿道"], "relation_hints": ["骑行"], "context_dependencies": [], "overall_confidence": "high"}
    ],
    "overall_confidence": "high",
    "batch_size": 3
}
'''})()

        qa_node = create_batch_qa_scaffold_node(MockLLM3())
        print("   [OK] create_batch_qa_scaffold_node 成功")

    except Exception as e:
        print(f"   [FAIL] 节点创建失败: {e}")
        return False

    print("\n2. 测试批量预处理流程...")

    # 模拟StreamWriter
    def mock_writer(event):
        pass

    try:
        # 测试筛选节点
        result = await filter_node(test_corpus, mock_writer)
        print(f"   筛选结果: {len(result['processed_corpus'])} 条保留, {len(result['skipped_corpus'])} 条跳过")

        # 测试归一化节点
        if result['processed_corpus']:
            norm_result = await normalize_node(result['processed_corpus'], mock_writer)
            print(f"   归一化结果: {len(norm_result['normalized_corpus'])} 条归一化")

            # 测试QA脚手架节点
            if norm_result['normalized_corpus']:
                qa_result = await qa_node(norm_result['normalized_corpus'], mock_writer)
                print(f"   QA脚手架结果: {len(qa_result['qa_corpus'])} 条脚手架构建")

        print("\n   [OK] 批量预处理流程测试成功")

    except Exception as e:
        print(f"   [FAIL] 流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)

    return True


def test_prompts():
    """测试批量预处理提示词"""
    print("\n" + "=" * 50)
    print("测试批量预处理提示词")
    print("=" * 50)

    test_corpus_str = """
【语料1】(ID: test_1)
武汉大学位于珞珈山下，附近有广八路美食街

【语料2】(ID: test_2)
今天去了光谷步行街，感觉很热闹

【语料3】(ID: test_3)
华科的学生经常去东湖绿道骑行
"""

    try:
        # 测试 BATCH_FILTER_PROMPT
        filter_prompt = BATCH_FILTER_PROMPT.invoke({"batch_size": 3, "corpus_list": test_corpus_str})
        print("1. BATCH_FILTER_PROMPT [OK]")
        print(f"   System: {filter_prompt.messages[0].content[:100]}...")

        # 测试 BATCH_NORMALIZE_PROMPT
        norm_prompt = BATCH_NORMALIZE_PROMPT.invoke({"batch_size": 3, "corpus_list": test_corpus_str})
        print("2. BATCH_NORMALIZE_PROMPT [OK]")

        # 测试 BATCH_QA_SCAFFOLD_PROMPT
        qa_prompt = BATCH_QA_SCAFFOLD_PROMPT.invoke({"batch_size": 3, "corpus_list": test_corpus_str})
        print("3. BATCH_QA_SCAFFOLD_PROMPT [OK]")

        print("\n   所有提示词测试通过")

    except Exception as e:
        print(f"   [FAIL] 提示词测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("开始测试批量预处理节点...")

    # 测试提示词
    test_prompts()

    # 测试节点创建和流程
    asyncio.run(test_batch_preprocessing_without_llm())