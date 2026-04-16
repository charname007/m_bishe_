"""
Normalize 归一化节点单元测试
"""
import pytest
from typing import Dict, List

from agent.agents.schemas import NormalizeResult, NormalizationRecord
from agent.agents.state import StepEnum, DEFAULT_MAX_RETRIES, CorpusState
from agent.agents.workflow import route_after_normalize, route_after_filter_to_normalize
from agent.agents.config import ExtractionConfig


# ===== Schema 测试 =====

class TestNormalizationRecord:
    """NormalizationRecord 模型测试"""

    def test_default_values(self):
        """测试默认值"""
        record = NormalizationRecord(raw="武大", normalized="武汉大学", type="alias")
        assert record.raw == "武大"
        assert record.normalized == "武汉大学"
        assert record.type == "alias"
        assert record.confidence == "high"

    def test_with_low_confidence(self):
        """测试低置信度"""
        record = NormalizationRecord(
            raw="这里",
            normalized="武汉大学",
            type="reference",
            confidence="low"
        )
        assert record.confidence == "low"


class TestNormalizeResult:
    """NormalizeResult 模型测试"""

    def test_default_values(self):
        """测试默认值"""
        result = NormalizeResult(normalized_text="测试文本")
        assert result.normalized_text == "测试文本"
        assert result.normalizations == []
        assert result.confidence == "medium"
        assert result.preserved_semantics == True
        assert result.has_changes == False

    def test_with_normalizations(self):
        """测试包含归一化记录"""
        result = NormalizeResult(
            normalized_text="武汉大学的樱花开放了",
            normalizations=[
                NormalizationRecord(raw="武大", normalized="武汉大学", type="alias"),
                NormalizationRecord(raw="开了", normalized="开放了", type="other"),
            ],
            confidence="high",
            has_changes=True
        )
        assert len(result.normalizations) == 2
        assert result.has_changes == True
        assert result.confidence == "high"

    def test_no_changes(self):
        """测试无改动"""
        result = NormalizeResult(
            normalized_text="武汉大学在珞喻路上",
            normalizations=[],
            confidence="low",
            has_changes=False
        )
        assert result.has_changes == False


# ===== 路由函数测试 =====

class TestRouteAfterNormalize:
    """Normalize 后路由测试"""

    def test_routes_to_ner_on_success(self):
        """成功时路由到 NER"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "武大的樱花开了",
            "filter_result": {},
            "normalize_result": {
                "normalized_text": "武汉大学的樱花开放了",
                "normalizations": [{"raw": "武大", "normalized": "武汉大学", "type": "alias"}],
                "confidence": "high",
                "preserved_semantics": True,
                "has_changes": True
            },
            "normalized_text": "武汉大学的樱花开放了",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "self_check_ner_result": {},
            "self_check_re_result": {},
            "final_entities": [],
            "final_triples": [],
            "verification_confidence": "medium",
            "retry_count": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
            "retry_reason": "",
            "problem_entities": [],
            "problem_triples": [],
            "needs_review": False,
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.NER,
            "error": None,
        }
        result = route_after_normalize(state)
        assert result == "ner"

    def test_routes_to_ner_on_error(self):
        """有错误时仍然路由到 NER（使用原文）"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "测试文本",
            "filter_result": {},
            "normalize_result": {},
            "normalized_text": "",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "self_check_ner_result": {},
            "self_check_re_result": {},
            "final_entities": [],
            "final_triples": [],
            "verification_confidence": "medium",
            "retry_count": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
            "retry_reason": "",
            "problem_entities": [],
            "problem_triples": [],
            "needs_review": False,
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.NER,
            "error": "LLM调用失败",
        }
        result = route_after_normalize(state)
        assert result == "ner"


class TestRouteAfterFilterToNormalize:
    """Filter → Normalize 路由测试"""

    def test_routes_to_normalize_on_valid(self):
        """有效文本路由到 Normalize"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "武汉大学在珞喻路上",
            "filter_result": {
                "is_valid": True,
                "confidence": "high",
                "skip_reason": None,
                "has_geo_entity": True,
                "has_spatial_relation": True,
            },
            "normalize_result": {},
            "normalized_text": "",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "self_check_ner_result": {},
            "self_check_re_result": {},
            "final_entities": [],
            "final_triples": [],
            "verification_confidence": "medium",
            "retry_count": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
            "retry_reason": "",
            "problem_entities": [],
            "problem_triples": [],
            "needs_review": False,
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.NORMALIZE,
            "error": None,
        }
        result = route_after_filter_to_normalize(state)
        assert result == "normalize"

    def test_routes_to_end_on_invalid(self):
        """无效文本路由到 END"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "今天心情不好",
            "filter_result": {
                "is_valid": False,
                "skip_reason": "无地理信息",
                "confidence": "high",
                "has_geo_entity": False,
                "has_spatial_relation": False,
            },
            "normalize_result": {},
            "normalized_text": "",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": [], "功能": [], "事件": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "self_check_ner_result": {},
            "self_check_re_result": {},
            "final_entities": [],
            "final_triples": [],
            "verification_confidence": "medium",
            "retry_count": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
            "retry_reason": "",
            "problem_entities": [],
            "problem_triples": [],
            "needs_review": False,
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.DONE,
            "error": None,
        }
        result = route_after_filter_to_normalize(state)
        assert result == "__end__"


# ===== 配置测试 =====

class TestNormalizeConfig:
    """Normalize 配置测试"""

    def test_default_config(self):
        """默认配置测试"""
        config = ExtractionConfig()
        assert config.enable_normalize == False

    def test_enable_normalize(self):
        """启用 Normalize 配置"""
        config = ExtractionConfig(enable_normalize=True)
        assert config.enable_normalize == True

    def test_from_dict_with_normalize(self):
        """从字典加载 Normalize 配置"""
        config = ExtractionConfig.from_dict({
            "enable_normalize": True,
        })
        assert config.enable_normalize == True

    def test_to_dict_includes_normalize(self):
        """转换为字典包含 Normalize 配置"""
        config = ExtractionConfig(enable_normalize=True)
        d = config.to_dict()
        assert "enable_normalize" in d
        assert d["enable_normalize"] == True

    def test_combined_config(self):
        """组合配置测试"""
        config = ExtractionConfig(
            enable_filter=True,
            enable_normalize=True,
            enable_self_check=True,
        )
        assert config.enable_filter == True
        assert config.enable_normalize == True
        assert config.enable_self_check == True


# ===== StepEnum 测试 =====

class TestStepEnumNormalize:
    """StepEnum.NORMALIZE 测试"""

    def test_normalize_step_exists(self):
        """测试 NORMALIZE 步骤存在"""
        assert hasattr(StepEnum, 'NORMALIZE')
        assert StepEnum.NORMALIZE == "normalize"

    def test_step_order(self):
        """测试步骤顺序"""
        steps = [StepEnum.FILTER, StepEnum.NORMALIZE, StepEnum.NER]
        assert steps[0] == StepEnum.FILTER
        assert steps[1] == StepEnum.NORMALIZE
        assert steps[2] == StepEnum.NER