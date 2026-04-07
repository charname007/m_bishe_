"""
工作流单元测试
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.agents.workflow import (
    WorkflowConfig,
    _validate_corpus_text,
    _validate_corpus_id,
    _get_database_config,
    route_after_ner,
)
from agent.agents.state import CorpusState, StepEnum


class TestWorkflowConfig:
    """测试工作流配置"""

    def test_config_values(self):
        """配置值应该合理"""
        assert WorkflowConfig.CORPUS_PER_WORKER > 0
        assert WorkflowConfig.MAX_WORKERS > 0
        assert 0 < WorkflowConfig.EVAL_PASSED_THRESHOLD <= 5
        assert 0 < WorkflowConfig.DEFAULT_SIMILARITY_THRESHOLD <= 1


class TestValidateCorpusText:
    """测试语料文本验证"""

    def test_valid_text(self):
        """有效文本"""
        result = _validate_corpus_text("这是一条测试文本")
        assert result == "这是一条测试文本"

    def test_strips_whitespace(self):
        """去除首尾空白"""
        result = _validate_corpus_text("  测试文本  ")
        assert result == "测试文本"

    def test_empty_text_raises(self):
        """空文本抛出异常"""
        with pytest.raises(ValueError, match="不能为空"):
            _validate_corpus_text("")

    def test_whitespace_only_raises(self):
        """仅空白字符抛出异常"""
        with pytest.raises(ValueError, match="不能为空"):
            _validate_corpus_text("   ")

    def test_truncates_long_text(self):
        """长文本截断"""
        long_text = "测试" * 10000  # 20000字符
        result = _validate_corpus_text(long_text)
        assert len(result) == WorkflowConfig.MAX_TEXT_LENGTH


class TestValidateCorpusId:
    """测试语料ID验证"""

    def test_valid_id(self):
        """有效ID"""
        result = _validate_corpus_id("corpus_001")
        assert result == "corpus_001"

    def test_none_id_generates(self):
        """None ID生成新ID"""
        result = _validate_corpus_id(None)
        assert result.startswith("auto_")

    def test_empty_string_generates(self):
        """空字符串生成新ID"""
        result = _validate_corpus_id("")
        assert result.startswith("auto_")

    def test_truncates_long_id(self):
        """长ID截断"""
        long_id = "a" * 200
        result = _validate_corpus_id(long_id)
        assert len(result) == 100


class TestGetDatabaseConfig:
    """测试数据库配置获取"""

    def test_missing_password_raises(self):
        """缺少密码抛出异常"""
        with patch.dict('os.environ', {
            'NEO4J_PASSWORD': '',
            'NEO4J_PASS': '',
            'NEO4J_PWD': '',
            'PG_PASSWORD': ''
        }, clear=True):
            with pytest.raises(ValueError, match="Neo4j密码未设置"):
                _get_database_config()

    @patch.dict('os.environ', {
        'NEO4J_PASSWORD': 'test_n4j_pass',
        'PG_PASSWORD': 'test_pg_pass',
        'NEO4J_URI': 'bolt://test:7687',
        'PG_HOST': 'testhost'
    })
    def test_returns_config(self):
        """返回正确配置"""
        config = _get_database_config()
        assert config["neo4j_password"] == "test_n4j_pass"
        assert config["pg_password"] == "test_pg_pass"
        assert config["neo4j_uri"] == "bolt://test:7687"
        assert config["pg_host"] == "testhost"


class TestRouteAfterNer:
    """测试NER后路由"""

    def test_routes_to_re_on_success(self):
        """成功时路由到RE"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "test",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.RE,
            "error": None,
        }
        result = route_after_ner(state)
        assert result == "re"

    def test_routes_to_end_on_error(self):
        """错误时路由到END"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "test",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.DONE,
            "error": "Something went wrong",
        }
        result = route_after_ner(state)
        assert result == "END"

    def test_routes_to_end_on_done(self):
        """DONE状态路由到END"""
        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "test",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.DONE,
            "error": None,
        }
        result = route_after_ner(state)
        assert result == "END"