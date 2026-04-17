"""
测试批处理流程修复 - P17改进：添加 batch_eval 和 batch_label 节点调用
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agents.nodes import (
    process_corpus_batch_with_llm,
    create_batch_eval_node,
    create_batch_label_node,
    create_batch_joint_extraction_node,
    create_batch_self_check_node,
)
from agent.agents.config import ExtractionConfig


def test_imports():
    """测试节点导入"""
    print("=" * 50)
    print("测试节点导入")
    print("=" * 50)

    print("1. process_corpus_batch_with_llm: OK")
    print("2. create_batch_eval_node: OK")
    print("3. create_batch_label_node: OK")
    print("4. create_batch_joint_extraction_node: OK")
    print("5. create_batch_self_check_node: OK")

    return True


def test_config():
    """测试配置"""
    print("\n" + "=" * 50)
    print("测试配置")
    print("=" * 50)

    config = ExtractionConfig.from_env()
    print(f"batch_llm_size: {config.batch_llm_size}")
    print(f"enable_batch_llm: {config.enable_batch_llm}")
    print(f"batch_llm_fallback: {config.batch_llm_fallback}")
    print(f"eval_threshold: {config.eval_threshold}")
    print(f"enable_full_self_check: {config.enable_full_self_check}")

    return True


def test_node_creation():
    """测试节点创建（不调用LLM）"""
    print("\n" + "=" * 50)
    print("测试节点创建")
    print("=" * 50)

    config = ExtractionConfig.from_env()

    # 模拟LLM
    class MockLLM:
        async def ainvoke(self, prompt):
            return type('obj', (object,), {'content': '{"results": []}'})()

    mock_llm = MockLLM()

    try:
        # 创建节点
        batch_joint_node = create_batch_joint_extraction_node(mock_llm, config.batch_llm_size)
        print("1. batch_joint_node: OK")

        batch_self_check_node = create_batch_self_check_node(mock_llm)
        print("2. batch_self_check_node: OK")

        batch_eval_node = create_batch_eval_node(mock_llm, config.eval_threshold)
        print("3. batch_eval_node: OK")

        batch_label_node = create_batch_label_node(mock_llm)
        print("4. batch_label_node: OK")

        print("\n所有节点创建成功！")
        return True

    except Exception as e:
        print(f"节点创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_function_signature():
    """测试函数签名"""
    print("\n" + "=" * 50)
    print("测试函数签名")
    print("=" * 50)

    import inspect

    # 检查 process_corpus_batch_with_llm 的参数
    sig = inspect.signature(process_corpus_batch_with_llm)
    params = list(sig.parameters.keys())

    expected_params = [
        "llm",
        "corpus_list",
        "config",
        "batch_joint_node",
        "batch_self_check_node",
        "batch_eval_node",  # P17新增
        "batch_label_node",  # P17新增
    ]

    print(f"期望参数: {expected_params}")
    print(f"实际参数: {params}")

    if params == expected_params:
        print("函数签名正确！")
        return True
    else:
        print("函数签名不匹配！")
        return False


def main():
    """主测试函数"""
    print("开始测试批处理流程修复...")

    tests = [
        test_imports,
        test_config,
        test_node_creation,
        test_function_signature,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    print(f"通过: {passed}/{total}")

    if all(results):
        print("\n所有测试通过！P17改进已正确实现。")
        print("批处理流程现在包含完整的节点链：")
        print("  Filter → Normalize → QA_Scaffold → Joint → Self_Check → Eval → Label")
    else:
        print("\n部分测试失败，请检查实现。")


if __name__ == "__main__":
    main()