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