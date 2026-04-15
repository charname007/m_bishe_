"""
测试智能体工作流 - 不保存到数据库
P9改进：添加联合抽取 + Reflexion + 全Self-Check测试
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
    """创建初始状态（含P9新增字段）"""
    return {
        "corpus_id": corpus_id,
        "raw_text": raw_text,
        # P9新增：配置标记字段（用于路由函数判断后续节点是否启用）
        "_config_enable_normalize": False,
        "_config_enable_qa_scaffold": False,
        # P5: Filter 筛选初始状态
        "filter_result": {},
        # P6: Normalize 归一化初始状态
        "normalize_result": {},
        "normalized_text": "",
        # P8: QA Scaffold 脚手架初始状态
        "qa_scaffold_result": {},
        "semantic_summary": "",
        "qa_entity_hints": [],
        "qa_relation_hints": [],
        "qa_context_dependencies": [],
        # P9新增：联合抽取初始状态
        "joint_extraction_result": {},
        "extraction_strategy": "",
        # P9新增：Self-Check初始状态
        "self_check_filter_result": {},
        "self_check_normalize_result": {},
        "self_check_qa_result": {},
        "self_check_joint_result": {},
        "self_check_eval_result": {},
        "self_check_label_result": {},
        "reflection_text": "",
        "improvement_strategy": "",
        "reflection_history": [],
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
        "retry_suggested": False,
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


async def test_single_corpus(enable_filter=False, enable_normalize=False, enable_qa_scaffold=False):
    """测试单条语料处理"""
    print("\n" + "=" * 60)
    print(f"[测试] 单条语料处理")
    print(f"[配置] enable_filter={enable_filter}, enable_normalize={enable_normalize}, enable_qa_scaffold={enable_qa_scaffold}")
    print("=" * 60)

    llm = create_llm()

    workflow = build_corpus_workflow(
        llm,
        use_simplified_eval=True,
        enable_filter=enable_filter,
        enable_normalize=enable_normalize,
        enable_qa_scaffold=enable_qa_scaffold,
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
            print(f"[Filter] is_non_wuhan_region: {fr.get('is_non_wuhan_region')}")
            print(f"[Filter] region_hint: {fr.get('region_hint')}")

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

        # QA Scaffold 结果（P8新增）
        if enable_qa_scaffold and result.get("qa_scaffold_result"):
            qr = result["qa_scaffold_result"]
            print(f"\n[QA_Scaffold] semantic_summary: {qr.get('semantic_summary')}")
            print(f"[QA_Scaffold] overall_confidence: {qr.get('overall_confidence')}")
            print(f"[QA_Scaffold] should_skip: {qr.get('should_skip_detailed_extraction')}")
            if qr.get("entity_hints"):
                print(f"[QA_Scaffold] entity_hints: {qr.get('entity_hints')}")
            if qr.get("relation_hints"):
                print(f"[QA_Scaffold] relation_hints: {qr.get('relation_hints')}")
            if qr.get("qa_pairs"):
                print("[QA_Scaffold] 问答对:")
                for qa in qr["qa_pairs"]:
                    print(f"  - [{qa.get('dimension')}] Q: {qa.get('question')}")
                    print(f"    A: {qa.get('answer')}")

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

        print("\n[测试] 完成 [OK]")
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
        print(f"[Filter] is_non_wuhan_region: {fr.get('is_non_wuhan_region')}")
        print(f"[Filter] region_hint: {fr.get('region_hint')}")

        # 应该没有后续处理结果
        entities = result.get("entities", {})
        has_entities = any(names for names in entities.values())
        triples = result.get("triples", [])
        print(f"\n[验证] 后续处理是否执行: {has_entities or len(triples) > 0}")

        if not fr.get("is_valid") and not has_entities and len(triples) == 0:
            print("\n[测试] Filter 正确跳过无效语料 [OK]")
        else:
            print("\n[测试] Filter 未正确跳过无效语料 [FAIL]")

        return result

    except Exception as e:
        print(f"\n[错误] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_joint_extraction_with_full_self_check():
    """
    P9新增测试：联合抽取 + Reflexion + 全Self-Check模式

    这是P9改进的核心测试，验证：
    1. 联合抽取节点（Joint_NER_RE）一次性抽取实体和关系
    2. Self-Check-Joint节点生成反思建议
    3. 全Self-Check节点（QA、Eval、Label）的二次检查
    4. Reflexion驱动的重试机制
    """
    print("\n" + "=" * 60)
    print("[测试P9] 联合抽取 + Reflexion + 全Self-Check模式")
    print("=" * 60)

    llm = create_llm()

    workflow = build_corpus_workflow(
        llm,
        use_simplified_eval=True,
        enable_filter=True,
        enable_normalize=True,
        enable_qa_scaffold=True,
        enable_self_check=False,  # 兼容参数
        use_joint_extraction=True,  # P9: 联合抽取模式
        enable_full_self_check=True,  # P9: 全Self-Check
        max_retries=3,
    )

    # 测试语料
    corpus_id = "test_p9_joint"
    raw_text = "群光广场就在珞喻路上，比街道口更热闹，周末适合带娃逛街"

    initial_state = create_initial_state(corpus_id, raw_text)

    print(f"\n[输入] 语料ID: {corpus_id}")
    print(f"[输入] 原文: {raw_text}")
    print(f"[配置] use_joint_extraction=True, enable_full_self_check=True")

    thread_config = {"configurable": {"thread_id": f"test_{corpus_id}_{os.getpid()}"}}

    try:
        result = await workflow.ainvoke(initial_state, thread_config)

        print("\n" + "-" * 40)
        print("[输出结果]")
        print("-" * 40)

        # Filter 结果
        fr = result.get("filter_result", {})
        print(f"\n[Filter] is_valid: {fr.get('is_valid')}")

        # Normalize 结果
        nr = result.get("normalize_result", {})
        print(f"\n[Normalize] normalized_text: {nr.get('normalized_text')}")

        # QA Scaffold 结果
        qr = result.get("qa_scaffold_result", {})
        print(f"\n[QA_Scaffold] semantic_summary: {qr.get('semantic_summary')}")
        print(f"\n[QA_Scaffold] entity_hints: {qr.get('entity_hints')}")
        print(f"\n[QA_Scaffold] relation_hints: {qr.get('relation_hints')}")

        # P9: Self-Check-QA 结果
        sc_qa = result.get("self_check_qa_result", {})
        if sc_qa:
            print(f"\n[Self-Check-QA] confidence: {sc_qa.get('overall_confidence')}")
            print(f"\n[Self-Check-QA] retry_suggested: {sc_qa.get('retry_suggested')}")

        # P9: 联合抽取结果
        jer = result.get("joint_extraction_result", {})
        extraction_strategy = result.get("extraction_strategy", "")
        print(f"\n[Joint_NER_RE] extraction_strategy: {extraction_strategy}")
        print(f"\n[Joint_NER_RE] entities ({len(jer.get('entities', []))} 个):")
        for e in jer.get("entities", []):
            print(f"  - {e.get('name')} [{e.get('type')}] 类别:{e.get('category')}")
        print(f"\n[Joint_NER_RE] triples ({len(jer.get('triples', []))} 条):")
        for t in jer.get("triples", []):
            attrs = t.get("attributes", {})
            attr_str = f" [{', '.join(f'{k}={v}' for k, v in attrs.items())}]" if attrs else ""
            print(f"  - <{t.get('head')}, {t.get('relation')}, {t.get('tail')}>{attr_str}")

        # P9: Self-Check-Joint 结果（含Reflexion）
        sc_joint = result.get("self_check_joint_result", {})
        if sc_joint:
            print(f"\n[Self-Check-Joint] confidence: {sc_joint.get('overall_confidence')}")
            print(f"\n[Self-Check-Joint] retry_suggested: {sc_joint.get('retry_suggested')}")
            reflection = result.get("reflection_text", "")
            if reflection:
                print(f"\n[Reflexion] 反思建议: {reflection[:200]}...")
            improvement = result.get("improvement_strategy", "")
            if improvement:
                print(f"\n[Reflexion] 改进策略: {improvement[:200]}...")

        # Eval 结果
        corrected_triples = result.get("corrected_triples", [])
        print(f"\n[Eval] 修正后三元组 ({len(corrected_triples)} 条)")

        # P9: Self-Check-Eval 结果
        sc_eval = result.get("self_check_eval_result", {})
        if sc_eval:
            print(f"\n[Self-Check-Eval] confidence: {sc_eval.get('overall_confidence')}")

        # Label 结果
        entity_attrs = result.get("entity_attrs", {})
        print(f"\n[Label] 实体属性 ({len(entity_attrs)} 个)")

        # P9: Self-Check-Label 结果
        sc_label = result.get("self_check_label_result", {})
        if sc_label:
            print(f"\n[Self-Check-Label] confidence: {sc_label.get('overall_confidence')}")

        # 重试历史
        retry_count = result.get("retry_count", 0)
        reflection_history = result.get("reflection_history", [])
        print(f"\n[Retry] 总重试次数: {retry_count}")
        print(f"\n[Reflexion History] {len(reflection_history)} 轮反思")

        # 验证
        entities = jer.get("entities", [])
        triples = jer.get("triples", [])
        if len(entities) >= 2 and len(triples) >= 1:
            print("\n[测试P9] 联合抽取成功 [OK]")
        else:
            print("\n[测试P9] 联合抽取结果不完整 [WARN]")

        if extraction_strategy == "joint":
            print("\n[测试P9] 抽取策略正确 [OK]")
        else:
            print("\n[测试P9] 抽取策略异常 [WARN]")

        return result

    except Exception as e:
        print(f"\n[错误] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_qa_mentor_mode():
    """
    P10新增测试：QA导师模式

    测试：
    1. QA导师节点使用Reasoner模型进行深度分析
    2. 后续节点（Joint_NER_RE）使用Chat模型抽取
    3. QA审批节点审批结果，决定是否需要修改
    4. 修改循环机制
    """
    print("\n" + "=" * 60)
    print("[测试P10] QA导师模式")
    print("=" * 60)

    from agent.agents.workflow import build_qa_mentor_workflow

    # 创建两个LLM实例
    qa_llm = create_llm()  # 用于导师节点（实际部署时使用deepseek-reasoner）
    worker_llm = create_llm()  # 用于工作节点（实际部署时使用deepseek-chat）

    config = ExtractionConfig(
        enable_qa_mentor=True,
        enable_qa_scaffold=True,
        enable_filter=True,
        enable_normalize=True,
        max_revision_cycles=3,
        qa_llm_model="deepseek-reasoner",
        worker_llm_model="deepseek-chat",
    )

    workflow = build_qa_mentor_workflow(qa_llm, worker_llm, config)

    # 测试语料
    corpus_id = "test_p10_mentor"
    raw_text = "光谷步行街很热闹，有很多人在那里逛街吃饭，旁边就是光谷广场地铁站"

    initial_state = create_initial_state(corpus_id, raw_text)

    # 添加P10导师模式所需的初始字段
    initial_state.update({
        "mentor_guidance": {},
        "qa_approval_result": {},
        "integrated_semantic_summary": "",
        "revision_feedbacks": [],
        "revision_cycle_count": 0,
        "max_revision_cycles": 3,
        "pending_approval_nodes": [],
        "qa_llm_model": "deepseek-reasoner",
        "worker_llm_model": "deepseek-chat",
    })

    print(f"\n[输入] 语料ID: {corpus_id}")
    print(f"[输入] 原文: {raw_text}")
    print(f"[配置] enable_qa_mentor=True, max_revision_cycles=3")

    thread_config = {"configurable": {"thread_id": f"test_{corpus_id}_{os.getpid()}"}}

    try:
        result = await workflow.ainvoke(initial_state, thread_config)

        print("\n" + "-" * 40)
        print("[输出结果]")
        print("-" * 40)

        # Filter 结果
        fr = result.get("filter_result", {})
        print(f"\n[Filter] is_valid: {fr.get('is_valid')}")

        # Normalize 结果
        nr = result.get("normalize_result", {})
        print(f"\n[Normalize] normalized_text: {nr.get('normalized_text')}")

        # QA导师脚手架结果
        mentor_result = result.get("qa_scaffold_result", {})  # 导师模式复用qa_scaffold_result字段
        print(f"\n[QA_Mentor] semantic_summary: {mentor_result.get('semantic_summary')}")
        print(f"\n[QA_Mentor] overall_confidence: {mentor_result.get('overall_confidence')}")
        if mentor_result.get("entity_hints"):
            print(f"\n[QA_Mentor] entity_hints: {mentor_result.get('entity_hints')}")
        if mentor_result.get("relation_hints"):
            print(f"\n[QA_Mentor] relation_hints: {mentor_result.get('relation_hints')}")

        # 导师指导信息
        mg = result.get("mentor_guidance", {})
        if mg:
            print(f"\n[MentorGuidance] semantic_focus: {mg.get('semantic_focus')}")
            print(f"\n[MentorGuidance] entity_priorities: {mg.get('entity_priorities')}")
            print(f"\n[MentorGuidance] quality_standards: {mg.get('quality_standards')}")

        # 联合抽取结果
        jer = result.get("joint_extraction_result", {})
        print(f"\n[Joint_NER_RE] extraction_strategy: {result.get('extraction_strategy')}")
        print(f"\n[Joint_NER_RE] entities ({len(jer.get('entities', []))} 个):")
        for e in jer.get("entities", []):
            print(f"  - {e.get('name')} [{e.get('type')}]")
        print(f"\n[Joint_NER_RE] triples ({len(jer.get('triples', []))} 条):")
        for t in jer.get("triples", []):
            print(f"  - <{t.get('head')}, {t.get('relation')}, {t.get('tail')}>")

        # QA审批结果
        qa_approval = result.get("qa_approval_result", {})
        if qa_approval:
            print(f"\n[QA_Approval] overall_status: {qa_approval.get('overall_status')}")
            print(f"\n[QA_Approval] overall_confidence: {qa_approval.get('overall_confidence')}")
            joint_approval = qa_approval.get("joint_approval", {})
            if joint_approval:
                print(f"\n[QA_Approval-Joint] approval_status: {joint_approval.get('approval_status')}")
                if joint_approval.get("feedbacks"):
                    print(f"\n[QA_Approval-Joint] feedbacks ({len(joint_approval.get('feedbacks'))} 条)")
                    for fb in joint_approval.get("feedbacks"):
                        print(f"  - [{fb.get('severity')}] {fb.get('description')}: {fb.get('suggestion')}")

        # 修改循环计数
        revision_count = result.get("revision_cycle_count", 0)
        print(f"\n[Revision] 循环次数: {revision_count}")

        # 验证
        if result.get("mentor_guidance"):
            print("\n[测试P10] 导师指导生成成功 [OK]")
        else:
            print("\n[测试P10] 导师指导未生成 [WARN]")

        if qa_approval.get("overall_status") in ["approved", "needs_revision", "rejected"]:
            print("\n[测试P10] QA审批流程正常 [OK]")
        else:
            print("\n[测试P10] QA审批结果异常 [WARN]")

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
    await test_single_corpus(enable_filter=False, enable_normalize=False, enable_qa_scaffold=False)

    # 测试 2: 启用 Filter
    await test_single_corpus(enable_filter=True, enable_normalize=False, enable_qa_scaffold=False)

    # 测试 3: 启用 Normalize
    await test_single_corpus(enable_filter=False, enable_normalize=True, enable_qa_scaffold=False)

    # 测试 4: 启用 Filter + Normalize
    await test_single_corpus(enable_filter=True, enable_normalize=True, enable_qa_scaffold=False)

    # 测试 5: 启用 Filter + Normalize + QA Scaffold（完整流程）
    await test_single_corpus(enable_filter=True, enable_normalize=True, enable_qa_scaffold=True)

    # 测试 6: 无效语料测试
    await test_invalid_corpus()

    # 测试 7: P9联合抽取 + Reflexion + 全Self-Check模式
    await test_joint_extraction_with_full_self_check()

    # 测试 8: P10 QA导师模式
    await test_qa_mentor_mode()

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())