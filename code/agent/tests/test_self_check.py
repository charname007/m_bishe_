"""
Self-Check + 反思循环单元测试
"""
import pytest
from typing import Dict, List

from agent.agents.schemas import (
    SelfCheckNERResult,
    SelfCheckREResult,
    VerifiedEntity,
    MissingEntity,
    EntityNormalization,
    VerifiedTriple,
    RejectedTriple,
    TripleCorrectionForSelfCheck,
)
from agent.agents.state import StepEnum, DEFAULT_MAX_RETRIES, CorpusState
from agent.agents.workflow import route_after_self_check_ner, route_after_self_check_re
from agent.agents.config import ExtractionConfig


# ===== Schema 测试 =====

class TestSelfCheckNERResult:
    """Self-Check-NER 结果模型测试"""

    def test_default_values(self):
        """测试默认值"""
        result = SelfCheckNERResult()
        assert result.verified_entities == []
        assert result.missing_entities == []
        assert result.entity_normalizations == []
        assert result.removed_entities == []
        assert result.overall_confidence == "medium"

    def test_with_verified_entities(self):
        """测试包含验证实体"""
        result = SelfCheckNERResult(
            verified_entities=[
                VerifiedEntity(
                    name="武汉大学",
                    type="POI",
                    confidence="high",
                    aliases=["武大"]
                )
            ],
            overall_confidence="high"
        )
        assert len(result.verified_entities) == 1
        assert result.verified_entities[0].name == "武汉大学"
        assert "武大" in result.verified_entities[0].aliases

    def test_with_missing_entities(self):
        """测试包含遗漏实体"""
        result = SelfCheckNERResult(
            missing_entities=[
                MissingEntity(
                    name="珞珈山",
                    suggested_type="街区?",
                    reason="原文提及但未抽取"
                )
            ],
            overall_confidence="medium"
        )
        assert len(result.missing_entities) == 1
        assert result.missing_entities[0].name == "珞珈山"


class TestSelfCheckREResult:
    """Self-Check-RE 结果模型测试"""

    def test_default_values(self):
        """测试默认值"""
        result = SelfCheckREResult()
        assert result.verified_triples == []
        assert result.rejected_triples == []
        assert result.corrected_triples == []
        assert result.overall_confidence == "medium"
        assert result.retry_suggested == False

    def test_with_rejected_triples(self):
        """测试包含拒绝三元组"""
        result = SelfCheckREResult(
            rejected_triples=[
                RejectedTriple(
                    head="武汉大学",
                    relation="位于",
                    tail="珞喻路",
                    reason="幻觉：原文未提及珞喻路",
                    suggested_fix="删除"
                )
            ],
            overall_confidence="low",
            retry_suggested=True
        )
        assert len(result.rejected_triples) == 1
        assert result.overall_confidence == "low"
        assert result.retry_suggested == True


# ===== 路由函数测试 =====

class TestRouteAfterSelfCheckNER:
    """Self-Check-NER 路由测试"""

    def test_routes_to_re_on_success(self):
        """高置信度时路由到 re"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "test",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": True,
            "corrected_triples": [],
            "self_check_ner_result": {"overall_confidence": "high", "missing_entities": []},
            "self_check_re_result": {},
            "final_entities": [],
            "final_triples": [],
            "verification_confidence": "high",
            "retry_count": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
            "retry_reason": "",
            "problem_entities": [],
            "problem_triples": [],
            "needs_review": False,
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.RE,
            "error": None,
        }
        result = route_after_self_check_ner(state)
        assert result == "re"

    def test_routes_to_ner_on_missing_entities(self):
        """NER 遗漏过多时路由回 ner"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "test",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "self_check_ner_result": {
                "overall_confidence": "low",
                "missing_entities": [
                    {"name": "珞珈山"},
                    {"name": "樱花"},
                    {"name": "行政楼"},
                ]
            },
            "self_check_re_result": {},
            "final_entities": [],
            "final_triples": [],
            "verification_confidence": "low",
            "retry_count": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
            "retry_reason": "",
            "problem_entities": ["珞珈山", "樱花", "行政楼"],
            "problem_triples": [],
            "needs_review": False,
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.RE,
            "error": None,
        }
        result = route_after_self_check_ner(state)
        assert result == "ner"

    def test_routes_to_re_on_max_retries(self):
        """达到最大重试次数时强制路由到 re"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "test",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "self_check_ner_result": {"overall_confidence": "low", "missing_entities": [{"name": "A"}, {"name": "B"}, {"name": "C"}]},
            "self_check_re_result": {},
            "final_entities": [],
            "final_triples": [],
            "verification_confidence": "low",
            "retry_count": 3,  # 达到最大重试次数
            "max_retries": 3,
            "retry_reason": "",
            "problem_entities": ["A", "B", "C"],
            "problem_triples": [],
            "needs_review": True,
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.RE,
            "error": None,
        }
        result = route_after_self_check_ner(state)
        assert result == "re"


class TestRouteAfterSelfCheckRE:
    """Self-Check-RE 路由测试"""

    def test_routes_to_eval_on_success(self):
        """高置信度时路由到 eval"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "test",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": True,
            "corrected_triples": [],
            "self_check_ner_result": {"overall_confidence": "high", "missing_entities": []},
            "self_check_re_result": {"overall_confidence": "high", "rejected_triples": [], "retry_suggested": False},
            "final_entities": [],
            "final_triples": [],
            "verification_confidence": "high",
            "retry_count": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
            "retry_reason": "",
            "problem_entities": [],
            "problem_triples": [],
            "needs_review": False,
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.EVAL,
            "error": None,
        }
        result = route_after_self_check_re(state)
        assert result == "eval"

    def test_routes_to_re_on_hallucination(self):
        """RE 幻觉过多时路由回 re"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "test",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "self_check_ner_result": {"overall_confidence": "medium", "missing_entities": []},
            "self_check_re_result": {
                "overall_confidence": "low",
                "rejected_triples": [
                    {"head": "A", "relation": "B", "tail": "C"},
                    {"head": "D", "relation": "E", "tail": "F"},
                    {"head": "G", "relation": "H", "tail": "I"},
                ],
                "retry_suggested": False
            },
            "final_entities": [],
            "final_triples": [],
            "verification_confidence": "low",
            "retry_count": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
            "retry_reason": "",
            "problem_entities": [],
            "problem_triples": [],
            "needs_review": False,
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.EVAL,
            "error": None,
        }
        result = route_after_self_check_re(state)
        assert result == "re"

    def test_routes_to_ner_on_entity_issue(self):
        """Self-Check-RE 建议回退 NER"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "test",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "self_check_ner_result": {"overall_confidence": "medium", "missing_entities": []},
            "self_check_re_result": {
                "overall_confidence": "low",
                "rejected_triples": [],
                "retry_suggested": True,
                "retry_target": "ner",
                "retry_reason": "实体缺失导致三元组问题"
            },
            "final_entities": [],
            "final_triples": [],
            "verification_confidence": "low",
            "retry_count": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
            "retry_reason": "",
            "problem_entities": [],
            "problem_triples": [],
            "needs_review": False,
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.EVAL,
            "error": None,
        }
        result = route_after_self_check_re(state)
        assert result == "ner"

    def test_routes_to_eval_on_max_retries(self):
        """达到最大重试次数时强制路由到 eval"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "test",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "self_check_ner_result": {},
            "self_check_re_result": {
                "overall_confidence": "low",
                "rejected_triples": [{"head": "A", "relation": "B", "tail": "C"}],
                "retry_suggested": True,
                "retry_target": "re"
            },
            "final_entities": [],
            "final_triples": [],
            "verification_confidence": "low",
            "retry_count": 3,  # 达到最大重试次数
            "max_retries": 3,
            "retry_reason": "",
            "problem_entities": [],
            "problem_triples": [],
            "needs_review": True,
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.EVAL,
            "error": None,
        }
        result = route_after_self_check_re(state)
        assert result == "eval"

    def test_routes_to_end_on_error(self):
        """有错误时路由到 END"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "test",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "self_check_ner_result": {},
            "self_check_re_result": {},
            "final_entities": [],
            "final_triples": [],
            "verification_confidence": "low",
            "retry_count": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
            "retry_reason": "",
            "problem_entities": [],
            "problem_triples": [],
            "needs_review": True,
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.DONE,
            "error": "Something went wrong",
        }
        result = route_after_self_check_re(state)
        assert result == "__end__"


# ===== 配置测试 =====

class TestSelfCheckConfig:
    """Self-Check 配置测试"""

    def test_default_config(self):
        """默认配置测试"""
        config = ExtractionConfig()
        assert config.enable_self_check == False
        assert config.self_check_max_retries == 3
        assert config.self_check_ner_low_threshold == 2
        assert config.self_check_re_low_threshold == 2

    def test_enable_self_check(self):
        """启用 Self-Check 配置"""
        config = ExtractionConfig(enable_self_check=True, self_check_max_retries=5)
        assert config.enable_self_check == True
        assert config.self_check_max_retries == 5

    def test_from_dict_with_self_check(self):
        """从字典加载 Self-Check 配置"""
        config = ExtractionConfig.from_dict({
            "enable_self_check": True,
            "self_check_max_retries": 4,
            "self_check_ner_low_threshold": 3,
        })
        assert config.enable_self_check == True
        assert config.self_check_max_retries == 4
        assert config.self_check_ner_low_threshold == 3

    def test_to_dict_includes_self_check(self):
        """转换为字典包含 Self-Check 配置"""
        config = ExtractionConfig(enable_self_check=True)
        d = config.to_dict()
        assert "enable_self_check" in d
        assert "self_check_max_retries" in d
        assert d["enable_self_check"] == True


# ===== 辅助函数测试 =====

class TestSelfCheckHelpers:
    """Self-Check 辅助函数测试"""

    def test_calculate_overall_confidence_high(self):
        """高置信度计算"""
        from agent.agents.nodes import _calculate_overall_confidence
        result = _calculate_overall_confidence(
            {"overall_confidence": "high"},
            {"overall_confidence": "high"}
        )
        assert result == "high"

    def test_calculate_overall_confidence_low(self):
        """低置信度计算"""
        from agent.agents.nodes import _calculate_overall_confidence
        result = _calculate_overall_confidence(
            {"overall_confidence": "low"},
            {"overall_confidence": "low"}
        )
        assert result == "low"

    def test_calculate_overall_confidence_mixed(self):
        """混合置信度计算"""
        from agent.agents.nodes import _calculate_overall_confidence
        result = _calculate_overall_confidence(
            {"overall_confidence": "high"},
            {"overall_confidence": "low"}
        )
        assert result == "medium"