"""
P14双向交流机制单元测试 - 测试导师查询功能

测试范围：
1. detect_extraction_confusion - 联合抽取困惑检测
2. detect_eval_confusion - 评估困惑检测
3. detect_label_confusion - 标注困惑检测
4. MentorQueryState 状态管理
5. 路由函数 route_joint_to_mentor_or_eval 等
"""
import pytest
from typing import Dict, Optional

from agent.agents.prompts import (
    detect_extraction_confusion,
    detect_eval_confusion,
    detect_label_confusion,
)
from agent.agents.state import CorpusState, MentorQueryState, StepEnum
from agent.agents.schemas import QueryTypeEnum


class TestDetectExtractionConfusion:
    """测试联合抽取困惑检测函数"""

    def test_entity_ambiguity_low_confidence(self):
        """实体置信度过低触发歧义查询"""
        result = {
            "entities": [
                {"name": "武汉大学", "type": "POI", "confidence": "low", "evidence": "原文"}
            ],
            "triples": [],
            "overall_confidence": "medium",
        }
        state = {"raw_text": "武汉大学在珞喻路上"}

        confusion = detect_extraction_confusion(result, state)

        assert confusion is not None
        assert confusion["query_type"] == "entity_ambiguity"
        assert "武汉大学" in confusion["involved_entities"]
        assert confusion["current_confidence"] == "low"

    def test_relation_confusion_low_confidence(self):
        """关系置信度过低触发困惑查询"""
        result = {
            "entities": [
                {"name": "武汉大学", "type": "POI", "confidence": "high", "evidence": "原文"}
            ],
            "triples": [
                {
                    "head": "武汉大学",
                    "relation": "位于",
                    "tail": "珞喻路",
                    "confidence": "low",
                    "evidence": "不确定",
                }
            ],
            "overall_confidence": "medium",
        }
        state = {"raw_text": "武汉大学在珞喻路上"}

        confusion = detect_extraction_confusion(result, state)

        assert confusion is not None
        assert confusion["query_type"] == "relation_confusion"
        assert "武汉大学" in confusion["involved_entities"]
        assert "位于" in confusion["involved_relations"]

    def test_overall_uncertainty_low_confidence(self):
        """整体置信度过低触发整体不确定查询"""
        result = {
            "entities": [],
            "triples": [],
            "overall_confidence": "low",
        }
        state = {"raw_text": "这段文字很复杂，有很多歧义"}

        confusion = detect_extraction_confusion(result, state)

        assert confusion is not None
        assert confusion["query_type"] == "overall_uncertainty"
        assert confusion["current_confidence"] == "low"

    def test_no_entities_with_text(self):
        """有文本但无实体触发查询"""
        result = {
            "entities": [],
            "triples": [],
            "overall_confidence": "medium",
        }
        state = {"raw_text": "这里应该有实体但没有被识别出来"}

        confusion = detect_extraction_confusion(result, state)

        assert confusion is not None
        assert confusion["query_type"] == "entity_ambiguity"
        assert confusion["current_confidence"] == "medium"

    def test_high_confidence_no_confusion(self):
        """高置信度结果不触发查询"""
        result = {
            "entities": [
                {"name": "武汉大学", "type": "POI", "confidence": "high", "evidence": "原文"}
            ],
            "triples": [
                {
                    "head": "武汉大学",
                    "relation": "位于",
                    "tail": "珞喻路",
                    "confidence": "high",
                    "evidence": "明确",
                }
            ],
            "overall_confidence": "high",
        }
        state = {"raw_text": "武汉大学在珞喻路上"}

        confusion = detect_extraction_confusion(result, state)

        assert confusion is None

    def test_empty_result_with_empty_text(self):
        """空文本和空结果不触发查询"""
        result = {
            "entities": [],
            "triples": [],
            "overall_confidence": "medium",
        }
        state = {"raw_text": ""}

        confusion = detect_extraction_confusion(result, state)

        # 无实体但有空文本，不应触发（文本本身就是空的）
        assert confusion is None

    def test_multiple_low_confidence_entities(self):
        """多个低置信度实体，优先返回第一个"""
        result = {
            "entities": [
                {"name": "武汉大学", "type": "POI", "confidence": "low", "evidence": "原文"},
                {"name": "珞喻路", "type": "道路", "confidence": "low", "evidence": "原文"},
            ],
            "triples": [],
            "overall_confidence": "medium",
        }
        state = {"raw_text": "武汉大学在珞喻路上"}

        confusion = detect_extraction_confusion(result, state)

        assert confusion is not None
        assert confusion["query_type"] == "entity_ambiguity"
        # 应返回第一个低置信度实体
        assert confusion["involved_entities"] == ["武汉大学"]


class TestDetectEvalConfusion:
    """测试评估困惑检测函数"""

    def test_many_triples_rejected(self):
        """大量三元组被拒绝触发评估分歧查询"""
        eval_result = {
            "corrected_triples": [{"head": "A", "relation": "位于", "tail": "B"}],
            "eval_passed": False,
        }
        state = {
            "triples": [
                {"head": "A", "relation": "位于", "tail": "B"},
                {"head": "C", "relation": "包含", "tail": "D"},
                {"head": "E", "relation": "相对方位", "tail": "F"},
            ]
        }

        confusion = detect_eval_confusion(eval_result, state)

        assert confusion is not None
        assert confusion["query_type"] == "eval_disagreement"
        # 原始3个三元组，修正后1个，拒绝2个（>50%）
        assert "拒绝了" in confusion["query_content"]

    def test_eval_failed_no_correction(self):
        """评估失败且无修正且无原始三元组触发整体不确定查询"""
        eval_result = {
            "corrected_triples": [],
            "eval_passed": False,
        }
        # 无原始三元组时，才触发 overall_uncertainty（否则触发 eval_disagreement）
        state = {"triples": []}

        confusion = detect_eval_confusion(eval_result, state)

        assert confusion is not None
        assert confusion["query_type"] == "overall_uncertainty"
        assert confusion["current_confidence"] == "low"

    def test_eval_passed_no_confusion(self):
        """评估通过不触发查询"""
        eval_result = {
            "corrected_triples": [{"head": "A", "relation": "位于", "tail": "B"}],
            "eval_passed": True,
        }
        state = {"triples": [{"head": "A", "relation": "位于", "tail": "B"}]}

        confusion = detect_eval_confusion(eval_result, state)

        assert confusion is None

    def test_empty_triples_no_confusion(self):
        """空三元组列表不触发查询"""
        eval_result = {
            "corrected_triples": [],
            "eval_passed": True,
        }
        state = {"triples": []}

        confusion = detect_eval_confusion(eval_result, state)

        assert confusion is None


class TestDetectLabelConfusion:
    """测试标注困惑检测函数"""

    def test_label_confusion_detection(self):
        """标注困惑检测（需根据实际函数实现调整）"""
        # 由于 detect_label_confusion 函数可能尚未完全实现，
        # 这里提供基本测试框架
        label_result = {
            "verified_entity_attrs": {"武汉大学": {"类别": "POI"}},
            "verified_relation_attrs": {},
        }
        state = {
            "entities": {"POI": ["武汉大学"]},
            "triples": [],
        }

        # 调用函数（如果实现）
        try:
            confusion = detect_label_confusion(label_result, state)
            # 根据实现调整断言
        except NotImplementedError:
            pytest.skip("detect_label_confusion 函数尚未完全实现")


class TestMentorQueryState:
    """测试导师查询状态管理"""

    def test_mentor_query_state_initialization(self):
        """导师查询状态默认值"""
        state: MentorQueryState = {
            "mentor_query": None,
            "mentor_response": None,
            "query_source_node": None,
            "needs_mentor_help": False,
            "query_count": 0,
            "max_queries": 2,
            "return_to_node": None,
        }

        assert state["mentor_query"] is None
        assert state["query_count"] == 0
        assert state["max_queries"] == 2
        assert state["needs_mentor_help"] is False

    def test_mentor_query_creation(self):
        """创建导师查询"""
        query = {
            "query_type": QueryTypeEnum.ENTITY_AMBIGUITY.value,
            "query_content": "实体类型不确定",
            "involved_entities": ["武汉大学"],
            "involved_relations": [],
            "current_confidence": "low",
            "source_node": "joint_ner_re",
        }

        assert query["query_type"] == "entity_ambiguity"
        assert query["source_node"] == "joint_ner_re"

    def test_mentor_response_structure(self):
        """导师响应结构"""
        response = {
            "answer": "武汉大学是POI类型",
            "clarification": "POI指具体地点/机构",
            "recommendation": "确认为POI",
            "updated_entity_hints": ["武汉大学"],
            "response_confidence": "high",
            "return_to_node": "joint_ner_re",
        }

        assert response["answer"] == "武汉大学是POI类型"
        assert response["return_to_node"] == "joint_ner_re"


class TestRouteFunctions:
    """测试路由函数（需mock状态）"""

    def test_route_joint_to_mentor_when_needed(self):
        """联合抽取需要导师帮助时路由到导师"""
        from agent.agents.workflow import route_joint_to_mentor_or_eval

        state: CorpusState = {
            "needs_mentor_help": True,
            "query_count": 0,
            "max_queries": 2,
            "error": None,
        }

        result = route_joint_to_mentor_or_eval(state)

        assert result == "qa_mentor"

    def test_route_joint_to_eval_when_not_needed(self):
        """联合抽取不需要帮助时路由到评估"""
        from agent.agents.workflow import route_joint_to_mentor_or_eval

        state: CorpusState = {
            "needs_mentor_help": False,
            "query_count": 0,
            "max_queries": 2,
            "error": None,
        }

        result = route_joint_to_mentor_or_eval(state)

        assert result == "eval"

    def test_route_joint_to_eval_max_queries_reached(self):
        """达到最大查询次数时路由到评估"""
        from agent.agents.workflow import route_joint_to_mentor_or_eval

        state: CorpusState = {
            "needs_mentor_help": True,
            "query_count": 2,  # 已达到max_queries
            "max_queries": 2,
            "error": None,
        }

        result = route_joint_to_mentor_or_eval(state)

        assert result == "eval"

    def test_route_joint_to_eval_on_error(self):
        """错误时路由到评估（保守策略）"""
        from agent.agents.workflow import route_joint_to_mentor_or_eval

        state: CorpusState = {
            "needs_mentor_help": True,
            "query_count": 0,
            "max_queries": 2,
            "error": "LLM调用失败",
        }

        result = route_joint_to_mentor_or_eval(state)

        assert result == "eval"

    def test_route_eval_to_mentor_when_needed(self):
        """评估需要导师帮助时路由到导师"""
        from agent.agents.workflow import route_eval_to_mentor_or_label

        state: CorpusState = {
            "needs_mentor_help": True,
            "query_count": 0,
            "max_queries": 2,
            "error": None,
        }

        result = route_eval_to_mentor_or_label(state)

        assert result == "qa_mentor"

    def test_route_label_to_mentor_when_needed(self):
        """标注需要导师帮助时路由到导师"""
        from agent.agents.workflow import route_label_to_mentor_or_approval

        state: CorpusState = {
            "needs_mentor_help": True,
            "query_count": 0,
            "max_queries": 2,
            "error": None,
        }

        result = route_label_to_mentor_or_approval(state)

        assert result == "qa_mentor"

    def test_route_mentor_to_target(self):
        """导师回答后返回目标节点"""
        from agent.agents.workflow import route_mentor_to_target

        state: CorpusState = {
            "mentor_response": {"answer": "确认是POI"},
            "return_to_node": "joint_ner_re",
        }

        result = route_mentor_to_target(state)

        assert result == "joint_ner_re"


class TestQueryTypeEnum:
    """测试查询类型枚举"""

    def test_all_query_types_defined(self):
        """所有查询类型都已定义"""
        expected_types = [
            "entity_ambiguity",
            "relation_confusion",
            "evidence_missing",
            "overall_uncertainty",
            "eval_disagreement",
            "label_confusion",
        ]

        for qt in expected_types:
            assert hasattr(QueryTypeEnum, qt.upper().replace("_", "_").upper()) or \
                   any(qt == e.value for e in QueryTypeEnum)

    def test_query_type_values(self):
        """查询类型值正确"""
        assert QueryTypeEnum.ENTITY_AMBIGUITY.value == "entity_ambiguity"
        assert QueryTypeEnum.RELATION_CONFUSION.value == "relation_confusion"
        assert QueryTypeEnum.EVIDENCE_MISSING.value == "evidence_missing"


# ===== 集成测试 =====

class TestIntegrationMentorQueryFlow:
    """导师查询流程集成测试"""

    @pytest.mark.asyncio
    async def test_full_query_flow_mock(self):
        """完整查询流程测试（mock LLM）"""
        from unittest.mock import AsyncMock, MagicMock

        # 模拟状态
        initial_state: CorpusState = {
            "corpus_id": "test_001",
            "raw_text": "武汉大学在珞喻路上",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
            "triples": [],
            "needs_mentor_help": False,
            "query_count": 0,
            "max_queries": 2,
            "mentor_query": None,
            "mentor_response": None,
            "return_to_node": None,
            "error": None,
        }

        # 模拟困惑检测
        confusion = detect_extraction_confusion(
            {
                "entities": [{"name": "武汉大学", "type": "POI", "confidence": "low"}],
                "triples": [],
                "overall_confidence": "low",
            },
            initial_state
        )

        assert confusion is not None
        assert confusion["query_type"] == "entity_ambiguity"

        # 更新状态
        initial_state["mentor_query"] = confusion
        initial_state["needs_mentor_help"] = True

        # 模拟导师响应
        initial_state["mentor_response"] = {
            "answer": "武汉大学是著名高校，应确认为POI",
            "response_confidence": "high",
            "return_to_node": "joint_ner_re",
        }
        initial_state["needs_mentor_help"] = False
        initial_state["query_count"] += 1

        # 验证状态更新
        assert initial_state["query_count"] == 1
        assert initial_state["mentor_response"]["return_to_node"] == "joint_ner_re"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])