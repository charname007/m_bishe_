#!/usr/bin/env python3
"""
验证 workflow 参数传递修复

测试目标：
1. 确认 build_corpus_workflow 正确接收所有参数
2. 确认日志输出反映正确的节点启用状态
3. 确认工作流图包含所有预期的节点
"""
import asyncio
import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
# 配置日志输出到控制台
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time} | {level} | {message}")

from agent.agents import ExtractionConfig
from agent.agents.workflow import build_corpus_workflow, build_distributed_workflow


def test_config_propagation():
    """测试配置参数传递"""
    print("=" * 60)
    print("Phase 3.1: 配置参数传递测试")
    print("=" * 60)

    # 创建完整配置（模拟 process_social_media_corpus.py）
    config = ExtractionConfig.from_env()
    config.enable_filter = True
    config.enable_normalize = True
    config.enable_qa_scaffold = True
    config.enable_full_self_check = True
    config.enable_entity_alignment = True
    config.enable_batch_llm = True

    print("配置状态:")
    print(f"  enable_filter: {config.enable_filter}")
    print(f"  enable_normalize: {config.enable_normalize}")
    print(f"  enable_qa_scaffold: {config.enable_qa_scaffold}")
    print(f"  enable_full_self_check: {config.enable_full_self_check}")
    print(f"  enable_entity_alignment: {config.enable_entity_alignment}")

    return config


def test_corpus_workflow_nodes(config):
    """测试 corpus_workflow 包含所有节点"""
    print("\n" + "=" * 60)
    print("Phase 3.2: corpus_workflow 节点验证")
    print("=" * 60)

    # 需要LLM实例来构建workflow
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE_URL")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)

    print("构建 corpus_workflow...")

    # 构建工作流（会输出日志）
    workflow = build_corpus_workflow(
        llm,
        use_simplified_eval=config.use_simplified_eval,
        enable_self_check=config.enable_self_check,
        enable_filter=config.enable_filter,
        enable_normalize=config.enable_normalize,
        enable_qa_scaffold=config.enable_qa_scaffold,
        enable_full_self_check=config.enable_full_self_check,
        enable_entity_alignment=config.enable_entity_alignment,
        enable_self_check_filter=config.enable_self_check_filter,
        enable_self_check_normalize=config.enable_self_check_normalize,
        config=config,
        max_retries=config.self_check_max_retries,
        prompt_version=config.prompt_version,
    )

    # 获取工作流节点列表
    nodes = workflow.nodes
    print(f"\n工作流节点数量: {len(nodes)}")
    print("节点列表:")
    for name in sorted(nodes.keys()):
        print(f"  - {name}")

    # 验证关键节点是否存在
    expected_nodes = [
        "filter",
        "normalize",
        "qa_scaffold",
        "joint_ner_re",
        "self_check_qa",
        "self_check_joint",
        "eval",
        "self_check_eval",
        "label",
        "self_check_label",
        "entity_alignment",
    ]

    print("\n关键节点验证:")
    missing_nodes = []
    for node in expected_nodes:
        if node in nodes:
            print(f"  [OK] {node}")
        else:
            print(f"  [MISSING] {node}")
            missing_nodes.append(node)

    return workflow, missing_nodes


def test_distributed_workflow_propagation(config):
    """测试 distributed_workflow 参数传递"""
    print("\n" + "=" * 60)
    print("Phase 3.3: distributed_workflow 参数传递验证")
    print("=" * 60)

    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE_URL")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)

    print("构建 distributed_workflow...")

    workflow = build_distributed_workflow(llm, config)

    print("distributed_workflow 节点:")
    for name in sorted(workflow.nodes.keys()):
        print(f"  - {name}")

    return workflow


def main():
    print("=" * 60)
    print("Workflow 参数传递修复验证")
    print("=" * 60)

    # Phase 3.1: 配置参数
    config = test_config_propagation()

    # Phase 3.2: corpus_workflow 节点验证
    corpus_wf, missing = test_corpus_workflow_nodes(config)

    # Phase 3.3: distributed_workflow 验证
    distributed_wf = test_distributed_workflow_propagation(config)

    # 最终结果
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)

    if not missing:
        print("[PASS] 所有预期节点已启用")
        print("[PASS] 参数传递修复有效")
        return True
    else:
        print(f"[FAIL] 缺失节点: {missing}")
        print("[FAIL] 参数传递存在问题")
        return False


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)