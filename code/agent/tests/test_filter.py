"""
Filter 筛选节点单元测试
"""
import pytest
from typing import Dict, List

from agent.agents.schemas import FilterResult
from agent.agents.state import StepEnum, DEFAULT_MAX_RETRIES, CorpusState
from agent.agents.workflow import route_after_filter
from agent.agents.config import ExtractionConfig


# ===== Schema 测试 =====

class TestFilterResult:
    """FilterResult 模型测试"""

    def test_default_values(self):
        """测试默认值"""
        result = FilterResult()
        assert result.is_valid == True
        assert result.skip_reason is None
        assert result.confidence == "medium"
        assert result.has_geo_entity == False
        assert result.has_spatial_relation == False
        assert result.geo_entity_hint is None

    def test_with_valid_result(self):
        """测试有效文本结果"""
        result = FilterResult(
            is_valid=True,
            confidence="high",
            has_geo_entity=True,
            has_spatial_relation=True,
            geo_entity_hint="武汉大学、珞喻路"
        )
        assert result.is_valid == True
        assert result.confidence == "high"
        assert result.has_geo_entity == True
        assert result.has_spatial_relation == True
        assert result.geo_entity_hint == "武汉大学、珞喻路"

    def test_with_invalid_result(self):
        """测试无效文本结果"""
        result = FilterResult(
            is_valid=False,
            skip_reason="无地理信息，纯情感表达",
            confidence="high",
            has_geo_entity=False,
            has_spatial_relation=False
        )
        assert result.is_valid == False
        assert result.skip_reason == "无地理信息，纯情感表达"
        assert result.confidence == "high"

    def test_with_low_confidence(self):
        """测试低置信度边界情况"""
        result = FilterResult(
            is_valid=True,
            confidence="low",
            has_geo_entity=False,
            has_spatial_relation=True,
            geo_entity_hint="这里（模糊地点指代）"
        )
        assert result.is_valid == True
        assert result.confidence == "low"
        assert result.has_geo_entity == False
        assert result.has_spatial_relation == True


# ===== 路由函数测试 =====

class TestRouteAfterFilter:
    """Filter 后路由测试"""

    def test_routes_to_ner_on_valid(self):
        """有效文本路由到 NER"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "武汉大学在珞喻路上",
            "filter_result": {
                "is_valid": True,
                "confidence": "high",
                "skip_reason": None,
                "has_geo_entity": True,
                "has_spatial_relation": True,
                "geo_entity_hint": "武汉大学、珞喻路"
            },
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
        result = route_after_filter(state)
        assert result == "ner"

    def test_routes_to_end_on_invalid(self):
        """无效文本路由到 END"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "今天心情不好",
            "filter_result": {
                "is_valid": False,
                "skip_reason": "无地理信息，纯情感表达",
                "confidence": "high",
                "has_geo_entity": False,
                "has_spatial_relation": False,
                "geo_entity_hint": None
            },
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
        result = route_after_filter(state)
        assert result == "__end__"

    def test_routes_to_ner_on_missing_filter_result(self):
        """缺少 filter_result 时默认路由到 NER（保守策略）"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "测试文本",
            "filter_result": {},  # 空的 filter_result
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
        result = route_after_filter(state)
        assert result == "ner"  # 默认继续处理

    def test_routes_to_ner_on_low_confidence_valid(self):
        """低置信度但有效时路由到 NER"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "这里挺好",
            "filter_result": {
                "is_valid": True,
                "skip_reason": None,
                "confidence": "low",
                "has_geo_entity": False,
                "has_spatial_relation": True,
                "geo_entity_hint": "这里（模糊地点指代）"
            },
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
        result = route_after_filter(state)
        assert result == "ner"


# ===== 配置测试 =====

class TestFilterConfig:
    """Filter 配置测试"""

    def test_default_config(self):
        """默认配置测试"""
        config = ExtractionConfig()
        assert config.enable_filter == False

    def test_enable_filter(self):
        """启用 Filter 配置"""
        config = ExtractionConfig(enable_filter=True)
        assert config.enable_filter == True

    def test_from_dict_with_filter(self):
        """从字典加载 Filter 配置"""
        config = ExtractionConfig.from_dict({
            "enable_filter": True,
        })
        assert config.enable_filter == True

    def test_to_dict_includes_filter(self):
        """转换为字典包含 Filter 配置"""
        config = ExtractionConfig(enable_filter=True)
        d = config.to_dict()
        assert "enable_filter" in d
        assert d["enable_filter"] == True

    def test_combined_config(self):
        """组合配置测试"""
        config = ExtractionConfig(
            enable_filter=True,
            enable_self_check=True,
            use_simplified_eval=True,
        )
        assert config.enable_filter == True
        assert config.enable_self_check == True
        assert config.use_simplified_eval == True