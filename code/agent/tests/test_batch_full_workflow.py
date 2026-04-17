"""
P15批量节点完整流程测试 - TDD开发
测试批量Self-Check-QA、Eval、Label、Entity_Alignment节点
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List


class TestBatchSelfCheckQA:
    """Batch_Self_Check_QA 测试类"""

    @pytest.mark.asyncio
    async def test_batch_self_check_qa_returns_verified_and_rejected_results(self):
        """
        测试：Batch_Self_Check_QA 应该返回 verified_results 和 rejected_results

        输入：
            batch_qa_results: {corpus_id: {qa_pairs, entity_hints, relation_hints}}
            corpus_texts: {corpus_id: text}

        输出：
            {
                "verified_results": [{corpus_id, qa_pairs, confidence}],
                "rejected_results": [{corpus_id, reason}],
                "overall_confidence": "high/medium/low"
            }
        """
        # 准备输入数据
        batch_qa_results = {
            "corpus_001": {
                "qa_pairs": [
                    {"question": "谁发布了这条内容？", "answer": "张三"},
                    {"question": "在哪里发布？", "answer": "北京"},
                ],
                "entity_hints": [{"name": "张三", "type": "PERSON"}],
                "relation_hints": [{"relation": "发布"}],
                "confidence": "medium",
            },
            "corpus_002": {
                "qa_pairs": [],
                "entity_hints": [],
                "relation_hints": [],
                "confidence": "low",
            },
        }
        corpus_texts = {
            "corpus_001": "张三在北京发布了一条关于旅游的内容",
            "corpus_002": "这是没有实体的短文本",
        }

        # 模拟LLM响应
        mock_llm_response = """
        {
            "verified_results": [
                {
                    "corpus_id": "corpus_001",
                    "verified_qa_pairs": [{"question": "谁发布了这条内容？", "answer": "张三"}],
                    "entity_coverage": "high",
                    "relation_coverage": "medium",
                    "confidence": "medium"
                }
            ],
            "rejected_results": [
                {
                    "corpus_id": "corpus_002",
                    "reason": "QA问答为空，无法校验"
                }
            ],
            "overall_confidence": "medium",
            "retry_suggested": false
        }
        """

        # 创建模拟LLM
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=mock_llm_response))

        # 创建节点并执行
        from agent.agents.nodes import create_batch_self_check_qa_node

        node = create_batch_self_check_qa_node(mock_llm)

        def dummy_writer(event):
            pass

        result = await node(batch_qa_results, corpus_texts, dummy_writer)

        # 验证输出结构
        assert "verified_results" in result
        assert "rejected_results" in result
        assert len(result["verified_results"]) == 1
        assert len(result["rejected_results"]) == 1
        assert result["verified_results"][0]["corpus_id"] == "corpus_001"
        assert result["rejected_results"][0]["corpus_id"] == "corpus_002"


    @pytest.mark.asyncio
    async def test_batch_self_check_qa_handles_empty_input(self):
        """
        测试：Batch_Self_Check_QA 应该正确处理空输入

        输入：空字典
        输出：verified_results=[], rejected_results=[]
        """
        batch_qa_results = {}
        corpus_texts = {}

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock()

        from agent.agents.nodes import create_batch_self_check_qa_node

        node = create_batch_self_check_qa_node(mock_llm)

        def dummy_writer(event):
            pass

        result = await node(batch_qa_results, corpus_texts, dummy_writer)

        assert result["verified_results"] == []
        assert result["rejected_results"] == []


class TestBatchEval:
    """Batch_Eval 测试类"""

    @pytest.mark.asyncio
    async def test_batch_eval_returns_scores_for_each_corpus(self):
        """
        测试：Batch_Eval 应该为每条语料的三元组返回评分

        输入：
            batch_extraction_results: {corpus_id: {entities, triples}}
            corpus_texts: {corpus_id: text}

        输出：
            {
                "batch_eval_results": {corpus_id: {scores, eval_passed, corrected_triples}},
                "overall_confidence": "high/medium/low"
            }
        """
        # 准备输入数据
        batch_extraction_results = {
            "corpus_001": {
                "entities": {"PERSON": ["张三"], "LOCATION": ["北京"]},
                "triples": [
                    {"head": "张三", "relation": "发布", "tail": "内容"},
                    {"head": "北京", "relation": "位于", "tail": "中国"},
                ],
                "confidence": "medium",
            },
            "corpus_002": {
                "entities": {"LOCATION": ["上海"]},
                "triples": [
                    {"head": "上海", "relation": "位于", "tail": "中国"},
                ],
                "confidence": "medium",
            },
        }
        corpus_texts = {
            "corpus_001": "张三在北京发布了一条关于旅游的内容",
            "corpus_002": "上海位于中国东部",
        }

        # 模拟LLM响应
        mock_llm_response = """
        {
            "batch_eval_results": [
                {
                    "corpus_id": "corpus_001",
                    "scores": [
                        {"triple": {"head": "张三", "relation": "发布", "tail": "内容"}, "SEM": 4, "FAC": 4, "CON": 5},
                        {"triple": {"head": "北京", "relation": "位于", "tail": "中国"}, "SEM": 3, "FAC": 3, "CON": 4}
                    ],
                    "eval_passed": true,
                    "corrected_triples": [],
                    "confidence": "medium"
                },
                {
                    "corpus_id": "corpus_002",
                    "scores": [
                        {"triple": {"head": "上海", "relation": "位于", "tail": "中国"}, "SEM": 5, "FAC": 5, "CON": 5}
                    ],
                    "eval_passed": true,
                    "corrected_triples": [],
                    "confidence": "high"
                }
            ],
            "overall_confidence": "medium"
        }
        """

        # 创建模拟LLM
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=mock_llm_response))

        # 创建节点并执行
        from agent.agents.nodes import create_batch_eval_node

        node = create_batch_eval_node(mock_llm)

        def dummy_writer(event):
            pass

        result = await node(batch_extraction_results, corpus_texts, dummy_writer)

        # 验证输出结构
        assert "batch_eval_results" in result
        assert len(result["batch_eval_results"]) == 2
        # 验证第一条语料
        eval_001 = result["batch_eval_results"]["corpus_001"]
        assert "scores" in eval_001
        assert "eval_passed" in eval_001
        assert len(eval_001["scores"]) == 2
        # 验证第二条语料
        eval_002 = result["batch_eval_results"]["corpus_002"]
        assert eval_002["eval_passed"] == True


    @pytest.mark.asyncio
    async def test_batch_eval_handles_empty_triples(self):
        """
        测试：Batch_Eval 应该正确处理无三元组的语料

        输入：包含无三元组的语料
        输出：eval_passed=True, scores=[]
        """
        batch_extraction_results = {
            "corpus_001": {
                "entities": {"PERSON": ["张三"]},
                "triples": [],
                "confidence": "low",
            },
        }
        corpus_texts = {
            "corpus_001": "张三",
        }

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock()

        from agent.agents.nodes import create_batch_eval_node

        node = create_batch_eval_node(mock_llm)

        def dummy_writer(event):
            pass

        result = await node(batch_extraction_results, corpus_texts, dummy_writer)

        # 无三元组时，应标记为通过
        eval_001 = result["batch_eval_results"]["corpus_001"]
        assert eval_001["eval_passed"] == True
        assert eval_001["scores"] == []


class TestBatchSelfCheckEval:
    """Batch_Self_Check_Eval 测试类"""

    @pytest.mark.asyncio
    async def test_batch_self_check_eval_verifies_eval_results(self):
        """
        测试：Batch_Self_Check_Eval 应该校验批量评估结果

        输入：
            batch_eval_results: {corpus_id: {scores, eval_passed, corrected_triples}}
            corpus_texts: {corpus_id: text}

        输出：
            {
                "verified_results": [{corpus_id, verified_triples}],
                "rejected_results": [{corpus_id, reason}],
                "retry_suggested": False
            }
        """
        # 准备输入数据
        batch_eval_results = {
            "corpus_001": {
                "scores": [
                    {"triple": {"head": "张三", "relation": "发布", "tail": "内容"}, "SEM": 4, "FAC": 4, "CON": 5}
                ],
                "eval_passed": True,
                "corrected_triples": [],
                "confidence": "high",
            },
            "corpus_002": {
                "scores": [
                    {"triple": {"head": "幻觉", "relation": "虚构", "tail": "内容"}, "SEM": 1, "FAC": 1, "CON": 1}
                ],
                "eval_passed": False,
                "corrected_triples": [],
                "confidence": "low",
            },
        }
        corpus_texts = {
            "corpus_001": "张三在北京发布了一条关于旅游的内容",
            "corpus_002": "这个语料不应该有幻觉内容",
        }

        # 模拟LLM响应
        mock_llm_response = """
        {
            "verified_results": [
                {
                    "corpus_id": "corpus_001",
                    "verified_triples": [{"head": "张三", "relation": "发布", "tail": "内容"}],
                    "score_consistency": "high",
                    "confidence": "high"
                }
            ],
            "rejected_results": [
                {
                    "corpus_id": "corpus_002",
                    "reason": "三元组评分过低，可能存在幻觉"
                }
            ],
            "overall_confidence": "medium",
            "retry_suggested": false
        }
        """

        # 创建模拟LLM
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=mock_llm_response))

        # 创建节点并执行
        from agent.agents.nodes import create_batch_self_check_eval_node

        node = create_batch_self_check_eval_node(mock_llm)

        def dummy_writer(event):
            pass

        result = await node(batch_eval_results, corpus_texts, dummy_writer)

        # 验证输出结构
        assert "verified_results" in result
        assert "rejected_results" in result
        assert len(result["verified_results"]) == 1
        assert len(result["rejected_results"]) == 1


class TestBatchNodesIntegration:
    """集成测试：验证批量节点串联工作"""

    @pytest.mark.asyncio
    async def test_batch_nodes_can_chain_correctly(self):
        """
        测试：批量节点应该能够正确串联处理数据流

        流程：
        1. Batch_QA_Scaffold (已有) → Batch_Self_Check_QA
        2. Batch_Joint (已有) → Batch_Eval
        3. Batch_Eval → Batch_Self_Check_Eval

        验证：数据格式在各节点间正确传递
        """
        # 模拟语料输入
        corpus_list = [
            {"id": "corpus_001", "text": "张三在北京发布了一条关于旅游的内容"},
            {"id": "corpus_002", "text": "上海位于中国东部"},
        ]
        corpus_texts = {c["id"]: c["text"] for c in corpus_list}

        # ===== Step 1: 模拟 Batch_QA_Scaffold 输出 =====
        batch_qa_results = {
            "corpus_001": {
                "qa_pairs": [{"question": "谁发布了这条内容？", "answer": "张三"}],
                "entity_hints": [{"name": "张三", "type": "PERSON"}],
                "relation_hints": [{"relation": "发布"}],
                "confidence": "medium",
            },
            "corpus_002": {
                "qa_pairs": [{"question": "上海位于哪里？", "answer": "中国东部"}],
                "entity_hints": [{"name": "上海", "type": "LOCATION"}],
                "relation_hints": [{"relation": "位于"}],
                "confidence": "medium",
            },
        }

        # 模拟 Batch_Self_Check_QA
        mock_llm_qa = MagicMock()
        mock_llm_qa.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="""
                {
                    "verified_results": [
                        {"corpus_id": "corpus_001", "verified_qa_pairs": [], "entity_coverage": "high", "relation_coverage": "medium", "confidence": "medium"},
                        {"corpus_id": "corpus_002", "verified_qa_pairs": [], "entity_coverage": "high", "relation_coverage": "high", "confidence": "medium"}
                    ],
                    "rejected_results": [],
                    "overall_confidence": "medium",
                    "retry_suggested": false
                }
                """
            )
        )

        from agent.agents.nodes import create_batch_self_check_qa_node

        qa_check_node = create_batch_self_check_qa_node(mock_llm_qa)
        qa_check_result = await qa_check_node(batch_qa_results, corpus_texts, lambda e: None)

        # 验证 Batch_Self_Check_QA 输出格式
        assert "verified_results" in qa_check_result
        assert len(qa_check_result["verified_results"]) == 2

        # ===== Step 2: 模拟 Batch_Joint 输出 =====
        batch_extraction_results = {
            "corpus_001": {
                "entities": {"PERSON": ["张三"], "LOCATION": ["北京"]},
                "triples": [{"head": "张三", "relation": "发布", "tail": "内容"}],
                "confidence": "medium",
            },
            "corpus_002": {
                "entities": {"LOCATION": ["上海"]},
                "triples": [{"head": "上海", "relation": "位于", "tail": "中国"}],
                "confidence": "medium",
            },
        }

        # 模拟 Batch_Eval
        mock_llm_eval = MagicMock()
        mock_llm_eval.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="""
                {
                    "batch_eval_results": [
                        {"corpus_id": "corpus_001", "scores": [{"triple": {"head": "张三", "relation": "发布", "tail": "内容"}, "SEM": 4, "FAC": 4, "CON": 5}], "eval_passed": true, "corrected_triples": [], "confidence": "medium"},
                        {"corpus_id": "corpus_002", "scores": [{"triple": {"head": "上海", "relation": "位于", "tail": "中国"}, "SEM": 5, "FAC": 5, "CON": 5}], "eval_passed": true, "corrected_triples": [], "confidence": "high"}
                    ],
                    "overall_confidence": "medium"
                }
                """
            )
        )

        from agent.agents.nodes import create_batch_eval_node

        eval_node = create_batch_eval_node(mock_llm_eval)
        eval_result = await eval_node(batch_extraction_results, corpus_texts, lambda e: None)

        # 验证 Batch_Eval 输出格式
        assert "batch_eval_results" in eval_result
        assert len(eval_result["batch_eval_results"]) == 2
        assert eval_result["batch_eval_results"]["corpus_001"]["eval_passed"] == True

        # ===== Step 3: Batch_Eval 输出 → Batch_Self_Check_Eval =====
        # 使用 Batch_Eval 的输出作为 Batch_Self_Check_Eval 的输入
        mock_llm_eval_check = MagicMock()
        mock_llm_eval_check.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="""
                {
                    "verified_results": [
                        {"corpus_id": "corpus_001", "verified_triples": [{"head": "张三", "relation": "发布", "tail": "内容"}], "score_consistency": "high", "confidence": "medium"},
                        {"corpus_id": "corpus_002", "verified_triples": [{"head": "上海", "relation": "位于", "tail": "中国"}], "score_consistency": "high", "confidence": "high"}
                    ],
                    "rejected_results": [],
                    "overall_confidence": "medium",
                    "retry_suggested": false
                }
                """
            )
        )

        from agent.agents.nodes import create_batch_self_check_eval_node

        eval_check_node = create_batch_self_check_eval_node(mock_llm_eval_check)
        eval_check_result = await eval_check_node(
            eval_result["batch_eval_results"], corpus_texts, lambda e: None
        )

        # 验证 Batch_Self_Check_Eval 输出格式
        assert "verified_results" in eval_check_result
        assert len(eval_check_result["verified_results"]) == 2

        # ===== 验证完整流程数据传递正确 =====
        print("\n===== 集成测试验证 =====")
        print(f"Step 1: Batch_Self_Check_QA 通过 {len(qa_check_result['verified_results'])} 条")
        print(f"Step 2: Batch_Eval 评估 {len(eval_result['batch_eval_results'])} 条")
        print(f"Step 3: Batch_Self_Check_Eval 通过 {len(eval_check_result['verified_results'])} 条")
        print("===== 所有节点串联验证成功 =====")


    @pytest.mark.asyncio
    async def test_batch_nodes_handle_errors_gracefully(self):
        """
        测试：批量节点应该在出错时优雅降级

        验证：LLM调用失败时，节点应返回保守结果（低置信度通过）
        """
        # 准备输入
        batch_extraction_results = {
            "corpus_001": {
                "entities": {"PERSON": ["张三"]},
                "triples": [{"head": "张三", "relation": "发布", "tail": "内容"}],
                "confidence": "medium",
            },
        }
        corpus_texts = {"corpus_001": "张三在北京发布了一条关于旅游的内容"}

        # 模拟 LLM 调用失败
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API连接失败"))

        from agent.agents.nodes import create_batch_eval_node

        eval_node = create_batch_eval_node(mock_llm)
        result = await eval_node(batch_extraction_results, corpus_texts, lambda e: None)

        # 验证：失败时应返回保守结果，而不是抛出异常
        assert "batch_eval_results" in result
        assert "corpus_001" in result["batch_eval_results"]
        assert result["overall_confidence"] == "low"  # 低置信度
        assert result["batch_eval_results"]["corpus_001"]["eval_passed"] == True  # 保守通过


class TestBatchLabel:
    """Batch_Label 测试类"""

    @pytest.mark.asyncio
    async def test_batch_label_adds_attributes_to_entities(self):
        """
        测试：Batch_Label 应该为批量实体添加属性标注

        输入：
            batch_eval_results: {corpus_id: {entities, triples, corrected_triples}}
            corpus_texts: {corpus_id: text}

        输出：
            {
                "batch_label_results": {corpus_id: {entity_attrs, relation_attrs}},
                "overall_confidence": "high/medium/low"
            }
        """
        # 准备输入数据（来自 Batch_Eval 输出）
        batch_eval_results = {
            "corpus_001": {
                "entities": {"PERSON": ["张三"], "LOCATION": ["北京"]},
                "triples": [{"head": "张三", "relation": "发布", "tail": "内容"}],
                "corrected_triples": [{"head": "张三", "relation": "发布", "tail": "内容"}],
                "eval_passed": True,
                "confidence": "medium",
            },
            "corpus_002": {
                "entities": {"LOCATION": ["上海", "中国"]},
                "triples": [{"head": "上海", "relation": "位于", "tail": "中国"}],
                "corrected_triples": [{"head": "上海", "relation": "位于", "tail": "中国"}],
                "eval_passed": True,
                "confidence": "high",
            },
        }
        corpus_texts = {
            "corpus_001": "张三在北京发布了一条关于旅游的内容",
            "corpus_002": "上海位于中国东部",
        }

        # 模拟LLM响应
        mock_llm_response = """
        {
            "batch_label_results": [
                {
                    "corpus_id": "corpus_001",
                    "entity_attrs": {
                        "张三": {"type": "POI", "confidence": "medium"},
                        "北京": {"type": "街区", "confidence": "medium"}
                    },
                    "relation_attrs": {
                        "<张三, 发布, 内容>": {"confidence": "medium"}
                    },
                    "confidence": "medium"
                },
                {
                    "corpus_id": "corpus_002",
                    "entity_attrs": {
                        "上海": {"type": "街区", "confidence": "high"},
                        "中国": {"type": "街区", "confidence": "high"}
                    },
                    "relation_attrs": {
                        "<上海, 位于, 中国>": {"confidence": "high"}
                    },
                    "confidence": "high"
                }
            ],
            "overall_confidence": "medium"
        }
        """

        # 创建模拟LLM
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=mock_llm_response))

        # 创建节点并执行
        from agent.agents.nodes import create_batch_label_node

        node = create_batch_label_node(mock_llm)

        def dummy_writer(event):
            pass

        result = await node(batch_eval_results, corpus_texts, dummy_writer)

        # 验证输出结构
        assert "batch_label_results" in result
        assert len(result["batch_label_results"]) == 2
        # 验证第一条语料的实体属性
        label_001 = result["batch_label_results"]["corpus_001"]
        assert "entity_attrs" in label_001
        assert "张三" in label_001["entity_attrs"]
        assert "relation_attrs" in label_001


    @pytest.mark.asyncio
    async def test_batch_label_handles_no_entities(self):
        """
        测试：Batch_Label 应该正确处理无实体的语料

        输入：包含无实体语料的批次
        输出：entity_attrs={}, relation_attrs={}
        """
        batch_eval_results = {
            "corpus_001": {
                "entities": {},
                "triples": [],
                "corrected_triples": [],
                "eval_passed": True,
                "confidence": "low",
            },
        }
        corpus_texts = {"corpus_001": "这是一段没有实体的文本"}

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock()

        from agent.agents.nodes import create_batch_label_node

        node = create_batch_label_node(mock_llm)

        result = await node(batch_eval_results, corpus_texts, lambda e: None)

        # 无实体时，应返回空属性
        label_001 = result["batch_label_results"]["corpus_001"]
        assert label_001["entity_attrs"] == {}
        assert label_001["relation_attrs"] == {}


class TestBatchSelfCheckLabel:
    """Batch_Self_Check_Label 测试类"""

    @pytest.mark.asyncio
    async def test_batch_self_check_label_verifies_label_results(self):
        """
        测试：Batch_Self_Check_Label 应该校验批量标注结果

        输入：
            batch_label_results: {corpus_id: {entity_attrs, relation_attrs}}
            corpus_texts: {corpus_id: text}

        输出：
            {
                "verified_results": [{corpus_id, verified_entity_attrs}],
                "rejected_results": [{corpus_id, reason}],
                "retry_suggested": False
            }
        """
        # 准备输入数据（来自 Batch_Label 输出）
        batch_label_results = {
            "corpus_001": {
                "entity_attrs": {
                    "张三": {"type": "POI", "confidence": "medium"},
                    "北京": {"type": "街区", "confidence": "medium"},
                },
                "relation_attrs": {"<张三, 发布, 内容>": {"confidence": "medium"}},
                "confidence": "medium",
            },
            "corpus_002": {
                "entity_attrs": {},
                "relation_attrs": {},
                "confidence": "low",
            },
        }
        corpus_texts = {
            "corpus_001": "张三在北京发布了一条关于旅游的内容",
            "corpus_002": "无实体文本",
        }

        # 模拟LLM响应
        mock_llm_response = """
        {
            "verified_results": [
                {
                    "corpus_id": "corpus_001",
                    "verified_entity_attrs": {"张三": {"type": "POI"}},
                    "verified_relation_attrs": {"<张三, 发布, 内容>": {}},
                    "attr_completeness": "medium",
                    "confidence": "medium"
                }
            ],
            "rejected_results": [
                {
                    "corpus_id": "corpus_002",
                    "reason": "无实体属性标注"
                }
            ],
            "overall_confidence": "medium",
            "retry_suggested": false
        }
        """

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=mock_llm_response))

        from agent.agents.nodes import create_batch_self_check_label_node

        node = create_batch_self_check_label_node(mock_llm)

        result = await node(batch_label_results, corpus_texts, lambda e: None)

        # 验证输出结构
        assert "verified_results" in result
        assert "rejected_results" in result
        assert len(result["verified_results"]) == 1
        assert len(result["rejected_results"]) == 1


class TestBatchEntityAlignment:
    """Batch_Entity_Alignment 测试类"""

    @pytest.mark.asyncio
    async def test_batch_entity_alignment_matches_with_database(self):
        """
        测试：Batch_Entity_Alignment 应该将实体与数据库已有实体对齐

        输入：
            batch_label_results: {corpus_id: {entity_attrs}}
            existing_entities: 数据库已有实体列表（模拟）

        输出：
            {
                "aligned_results": {corpus_id: {aligned_entity_attrs}},
                "new_entities": [待新增实体列表]
            }
        """
        batch_label_results = {
            "corpus_001": {
                "entity_attrs": {
                    "武汉大学": {"type": "POI"},
                    "樱花": {"type": "功能"},
                },
                "confidence": "medium",
            },
        }
        corpus_texts = {"corpus_001": "武汉大学樱花很美"}

        # 模拟数据库已有实体
        existing_entities = ["武汉大学", "武大"]  # 已存在于数据库

        mock_llm_response = """
        {
            "aligned_results": [
                {
                    "corpus_id": "corpus_001",
                    "aligned_entity_attrs": {
                        "武汉大学": {"type": "POI", "aligned_to": "武汉大学", "confidence": "high"},
                        "樱花": {"type": "功能", "aligned_to": null, "confidence": "medium"}
                    },
                    "new_entities": ["樱花"]
                }
            ],
            "overall_confidence": "medium"
        }
        """

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content=mock_llm_response))

        from agent.agents.nodes import create_batch_entity_alignment_node

        node = create_batch_entity_alignment_node(mock_llm)

        result = await node(batch_label_results, corpus_texts, existing_entities, lambda e: None)

        # 验证输出结构
        assert "aligned_results" in result
        assert "corpus_001" in result["aligned_results"]