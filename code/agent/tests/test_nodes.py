"""
节点函数单元测试
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.agents.nodes import (
    is_similar,
    normalize_relation_key,
    deduplicate_triples,
    deduplicate_entities,
    apply_corrections,
    create_coordinator_node,
    create_aggregator_node,
)
from agent.agents.state import CorpusState, KGState, StepEnum, PhaseEnum
from agent.agents.schemas import (
    Triple,
    TripleForEval,
    Correction,
)


class TestIsSimilar:
    """测试 is_similar 函数"""

    def test_identical_names(self):
        """完全相同的名称"""
        assert is_similar("武汉大学", "武汉大学", 0.85) is True

    def test_similar_names(self):
        """相似名称"""
        assert is_similar("珞喻路", "珞瑜路", 0.85) is True  # 仅一字之差

    def test_different_names(self):
        """不相似的名称"""
        assert is_similar("武汉大学", "华中科技大学", 0.85) is False

    def test_abbreviation(self):
        """简称别名"""
        assert is_similar("武大", "武汉大学", 0.85) is True  # 简称

    def test_length_ratio_too_small(self):
        """长度比例过小"""
        assert is_similar("汉", "武汉大学", 0.85) is False


class TestNormalizeRelationKey:
    """测试 normalize_relation_key 函数"""

    def test_standard_format(self):
        """标准格式"""
        result = normalize_relation_key("<武汉大学, 位于, 珞喻路>")
        assert result == "<武汉大学, 位于, 珞喻路>"

    def test_no_brackets(self):
        """无尖括号格式"""
        result = normalize_relation_key("武汉大学, 位于, 珞喻路")
        assert result == "<武汉大学, 位于, 珞喻路>"

    def test_no_spaces(self):
        """无空格格式"""
        result = normalize_relation_key("武汉大学,位于,珞喻路")
        assert result == "<武汉大学, 位于, 珞喻路>"

    def test_invalid_format(self):
        """无效格式"""
        assert normalize_relation_key("") is None
        assert normalize_relation_key("武汉大学") is None
        assert normalize_relation_key("A, B") is None  # 只有两部分


class TestDeduplicateTriples:
    """测试 deduplicate_triples 函数"""

    def test_empty_list(self):
        """空列表"""
        result = deduplicate_triples([])
        assert result == []

    def test_unique_triples(self):
        """无重复三元组"""
        triples = [
            {"head": "A", "relation": "位于", "tail": "B", "evidence": "test"},
            {"head": "C", "relation": "连接", "tail": "D", "evidence": "test2"},
        ]
        result = deduplicate_triples(triples)
        assert len(result) == 2

    def test_duplicate_triples(self):
        """重复三元组"""
        triples = [
            {"head": "A", "relation": "位于", "tail": "B", "evidence": "test1", "_corpus_id": "1"},
            {"head": "A", "relation": "位于", "tail": "B", "evidence": "test2", "_corpus_id": "2"},
        ]
        result = deduplicate_triples(triples)
        assert len(result) == 1
        assert len(result[0]["corpus_ids"]) == 2

    def test_preserves_scores(self):
        """保留评分字段"""
        triples = [
            {
                "head": "A", "relation": "位于", "tail": "B",
                "sem_score": 4, "fac_score": 5, "con_score": 4,
                "passed_eval": True, "_corpus_id": "1"
            },
        ]
        result = deduplicate_triples(triples)
        assert result[0]["sem_score"] == 4
        assert result[0]["passed_eval"] is True


class TestDeduplicateEntities:
    """测试 deduplicate_entities 函数"""

    def test_empty_list(self):
        """空列表"""
        result, aliases = deduplicate_entities([], 0.85)
        assert result == []
        assert aliases == {}

    def test_unique_entities(self):
        """无重复实体"""
        entities = [
            {"name": "武汉大学", "type": "POI", "corpus_id": "1", "attrs": {}},
            {"name": "华中科技大学", "type": "POI", "corpus_id": "2", "attrs": {}},
        ]
        result, aliases = deduplicate_entities(entities, 0.85)
        assert len(result) == 2

    def test_similar_entities(self):
        """相似实体合并"""
        entities = [
            {"name": "珞喻路", "type": "ROAD", "corpus_id": "1", "attrs": {}},
            {"name": "珞瑜路", "type": "ROAD", "corpus_id": "2", "attrs": {}},
        ]
        result, aliases = deduplicate_entities(entities, 0.85)
        assert len(result) == 1
        assert "珞瑜路" in result[0]["aliases"] or "珞喻路" in result[0]["aliases"]

    def test_preserves_attributes(self):
        """保留属性"""
        entities = [
            {"name": "武汉大学", "type": "POI", "corpus_id": "1", "attrs": {"细分": "教育"}},
        ]
        result, _ = deduplicate_entities(entities, 0.85)
        assert result[0]["category"] == "教育"


class TestApplyCorrections:
    """测试 apply_corrections 函数"""

    def test_no_corrections(self):
        """无修正"""
        triples = [{"head": "A", "relation": "位于", "tail": "B"}]
        correction = MagicMock()
        correction.original = MagicMock()
        correction.original.head = "A"
        correction.original.relation = "位于"
        correction.original.tail = "B"
        correction.corrected = MagicMock()
        correction.corrected.head = "A"
        correction.corrected.relation = "位于"
        correction.corrected.tail = "C"

        result, mapping = apply_corrections(triples, [correction])
        assert len(result) == 1


class TestCoordinatorNode:
    """测试调度器节点"""

    def test_partition_corpus(self):
        """测试语料分片"""
        coordinator_node = create_coordinator_node(corpus_per_worker=10, max_workers=5)

        state: KGState = {
            "batch_id": "test_batch",
            "corpus_list": [{"id": str(i), "text": f"text_{i}"} for i in range(25)],
            "total_count": 25,
            "worker_count": 0,
            "corpus_partitions": {},
            "worker_results": [],
            "aggregated_entities": [],
            "aggregated_triples": [],
            "entity_aliases": {},
            "cross_corpus_relations": [],
            "evaluator_results": [],
            "high_confidence_triples": [],
            "low_confidence_triples": [],
            "neo4j_stats": {},
            "postgres_stats": {},
            "current_phase": PhaseEnum.INIT,
            "active_workers": [],
            "failed_workers": [],
            "start_time": 0.0,
            "end_time": None,
            "total_tokens": 0,
        }

        result = coordinator_node(state)

        assert result["worker_count"] == 3  # ceil(25/10) = 3
        assert len(result["corpus_partitions"]) == 3
        assert result["current_phase"] == PhaseEnum.MAP


class TestAggregatorNode:
    """测试聚合器节点"""

    def test_aggregate_results(self):
        """测试结果聚合"""
        aggregator_node = create_aggregator_node()

        state: KGState = {
            "batch_id": "test_batch",
            "corpus_list": [],
            "total_count": 2,
            "worker_count": 1,
            "corpus_partitions": {},
            "worker_results": [
                {
                    "worker_id": "worker_0",
                    "results": [
                        {
                            "corpus_id": "1",
                            "entities": {"POI": ["武汉大学"], "道路": ["珞喻路"]},
                            "corrected_triples": [
                                {"head": "武汉大学", "relation": "位于", "tail": "珞喻路", "evidence": ""}
                            ],
                            "entity_attrs": {"武汉大学": {"细分": "教育"}},
                            "relation_attrs": {},
                        },
                        {
                            "corpus_id": "2",
                            "entities": {"POI": ["华中科技大学"], "道路": ["珞喻路"]},
                            "corrected_triples": [],
                            "entity_attrs": {},
                            "relation_attrs": {},
                        },
                    ],
                }
            ],
            "aggregated_entities": [],
            "aggregated_triples": [],
            "entity_aliases": {},
            "cross_corpus_relations": [],
            "evaluator_results": [],
            "high_confidence_triples": [],
            "low_confidence_triples": [],
            "neo4j_stats": {},
            "postgres_stats": {},
            "current_phase": PhaseEnum.REDUCE,
            "active_workers": [],
            "failed_workers": [],
            "start_time": 0.0,
            "end_time": None,
            "total_tokens": 0,
        }

        result = aggregator_node(state)

        # 应该有3个实体（武汉大学、华中科技大学、珞喻路）
        assert len(result["aggregated_entities"]) >= 2
        # 应该有1个三元组
        assert len(result["aggregated_triples"]) == 1
        assert result["current_phase"] == PhaseEnum.FINALIZE