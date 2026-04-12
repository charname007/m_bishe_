"""
测试智能体工作流 - 不保存到数据库
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 加载环境变量
load_dotenv(os.path.join(project_root, '.env'))

from langchain_openai import ChatOpenAI
from agent.agents.workflow import build_corpus_workflow
from agent.agents.state import CorpusState, StepEnum, DEFAULT_MAX_RETRIES
from agent.agents.config import ExtractionConfig


def create_llm():
    """创建 LLM 实例（使用 DeepSeek API）"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE_URL")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 未设置，请检查 .env 文件")

    print(f"[LLM] 使用模型: {model}")
    print(f"[LLM] API Base URL: {base_url}")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


def create_initial_state(corpus_id: str, raw_text: str) -> CorpusState:
    """创建初始状态"""
    return {
        "corpus_id": corpus_id,
        "raw_text": raw_text,
        # P5: Filter 筛选初始状态
        "filter_result": {},
        # P6: Normalize 归一化初始状态
        "normalize_result": {},
        "normalized_text": "",
        # Step 1: NER
        "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
        # Step 2: RE
        "triples": [],
        # Step 3: Eval
        "eval_scores": [],
        "eval_passed": False,
        "corrected_triples": [],
        # Step 3.5: Self-Check
        "self_check_ner_result": {},
        "self_check_re_result": {},
        "final_entities": [],
        "final_triples": [],
        "verification_confidence": "medium",
        # 反思循环控制
        "retry_count": 0,
        "max_retries": DEFAULT_MAX_RETRIES,
        "retry_reason": "",
        "problem_entities": [],
        "problem_triples": [],
        "needs_review": False,
        # Step 4: Label
        "entity_attrs": {},
        "relation_attrs": {},
        # 状态控制
        "current_step": StepEnum.NER,
        "error": None,
    }


async def test_single_corpus(enable_filter=False, enable_normalize=False):
    """测试单条语料处理"""
    print("\n" + "=" * 60)
    print(f"[测试] 单条语料处理")
    print(f"[配置] enable_filter={enable_filter}, enable_normalize={enable_normalize}")
    print("=" * 60)

    llm = create_llm()

    workflow = build_corpus_workflow(
        llm,
        use_simplified_eval=True,
        enable_filter=enable_filter,
        enable_normalize=enable_normalize,
        enable_self_check=False,
    )

    # 测试语料
    corpus_id = "test_001"
    raw_text = "武大的樱花开了，很多人在行政楼前拍照打卡"

    initial_state = create_initial_state(corpus_id, raw_text)

    print(f"\n[输入] 语料ID: {corpus_id}")
    print(f"[输入] 原文: {raw_text}")

    thread_config = {"configurable": {"thread_id": f"test_{corpus_id}_{os.getpid()}"}}

    try:
        result = await workflow.ainvoke(initial_state, thread_config)

        print("\n" + "-" * 40)
        print("[输出结果]")
        print("-" * 40)

        # Filter 结果
        if enable_filter and result.get("filter_result"):
            fr = result["filter_result"]
            print(f"\n[Filter] is_valid: {fr.get('is_valid')}")
            print(f"[Filter] confidence: {fr.get('confidence')}")
            print(f"[Filter] skip_reason: {fr.get('skip_reason')}")
            print(f"[Filter] has_geo_entity: {fr.get('has_geo_entity')}")

        # Normalize 结果
        if enable_normalize and result.get("normalize_result"):
            nr = result["normalize_result"]
            print(f"\n[Normalize] normalized_text: {nr.get('normalized_text')}")
            print(f"[Normalize] has_changes: {nr.get('has_changes')}")
            print(f"[Normalize] confidence: {nr.get('confidence')}")
            if nr.get("normalizations"):
                print("[Normalize] 归一化记录:")
                for n in nr["normalizations"]:
                    print(f"  - {n.get('raw')} → {n.get('normalized')} ({n.get('type')})")

        # NER 结果
        entities = result.get("entities", {})
        print(f"\n[NER] 实体:")
        for entity_type, names in entities.items():
            if names:
                print(f"  - {entity_type}: {names}")

        # RE 结果
        triples = result.get("triples", [])
        print(f"\n[RE] 三元组 ({len(triples)} 条):")
        for t in triples:
            print(f"  - <{t['head']}, {t['relation']}, {t['tail']}>")

        # Eval 结果
        corrected_triples = result.get("corrected_triples", [])
        eval_passed = result.get("eval_passed", False)
        print(f"\n[Eval] 评估通过: {eval_passed}")
        print(f"[Eval] 修正后三元组 ({len(corrected_triples)} 条):")
        for t in corrected_triples:
            scores = f"SEM={t.get('sem_score', 0)}, FAC={t.get('fac_score', 0)}, CON={t.get('con_score', 0)}"
            passed = t.get("passed_eval", False)
            print(f"  - <{t['head']}, {t['relation']}, {t['tail']}> [{scores}] passed={passed}")

        # Label 结果
        entity_attrs = result.get("entity_attrs", {})
        relation_attrs = result.get("relation_attrs", {})
        print(f"\n[Label] 实体属性 ({len(entity_attrs)} 个):")
        for name, attrs in entity_attrs.items():
            print(f"  - {name}: {attrs}")
        print(f"[Label] 关系属性 ({len(relation_attrs)} 个):")
        for key, attrs in relation_attrs.items():
            print(f"  - {key}: {attrs}")

        # 错误检查
        if result.get("error"):
            print(f"\n[错误] {result['error']}")

        print("\n[测试] 完成 ✅")
        return result

    except Exception as e:
        print(f"\n[错误] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_invalid_corpus():
    """测试无效语料（应被 Filter 节点过滤）"""
    print("\n" + "=" * 60)
    print("[测试] 无效语料处理（Filter 应跳过）")
    print("=" * 60)

    llm = create_llm()

    workflow = build_corpus_workflow(
        llm,
        use_simplified_eval=True,
        enable_filter=True,
        enable_normalize=False,
        enable_self_check=False,
    )

    # 无效语料（无地理信息）
    corpus_id = "test_invalid"
    raw_text = "今天心情不好，感觉有点累"

    initial_state = create_initial_state(corpus_id, raw_text)

    print(f"\n[输入] 语料ID: {corpus_id}")
    print(f"[输入] 原文: {raw_text}")

    thread_config = {"configurable": {"thread_id": f"test_{corpus_id}_{os.getpid()}"}}

    try:
        result = await workflow.ainvoke(initial_state, thread_config)

        print("\n" + "-" * 40)
        print("[输出结果]")
        print("-" * 40)

        fr = result.get("filter_result", {})
        print(f"\n[Filter] is_valid: {fr.get('is_valid')}")
        print(f"[Filter] skip_reason: {fr.get('skip_reason')}")
        print(f"[Filter] confidence: {fr.get('confidence')}")

        # 应该没有后续处理结果
        entities = result.get("entities", {})
        has_entities = any(names for names in entities.values())
        triples = result.get("triples", [])
        print(f"\n[验证] 后续处理是否执行: {has_entities or len(triples) > 0}")

        if not fr.get("is_valid") and not has_entities and len(triples) == 0:
            print("\n[测试] Filter 正确跳过无效语料 ✅")
        else:
            print("\n[测试] Filter 未正确跳过无效语料 ❌")

        return result

    except Exception as e:
        print(f"\n[错误] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("智能体工作流测试")
    print("=" * 60)

    # 测试 1: 基础模式（无 Filter，无 Normalize）
    await test_single_corpus(enable_filter=False, enable_normalize=False)

    # 测试 2: 启用 Filter
    await test_single_corpus(enable_filter=True, enable_normalize=False)

    # 测试 3: 启用 Normalize
    await test_single_corpus(enable_filter=False, enable_normalize=True)

    # 测试 4: 启用 Filter + Normalize
    await test_single_corpus(enable_filter=True, enable_normalize=True)

    # 测试 5: 无效语料测试
    await test_invalid_corpus()

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())