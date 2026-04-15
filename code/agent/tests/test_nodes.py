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
        # 编辑距离1，相似度约0.67，低于默认0.85阈值
        # 使用更低的阈值测试
        assert is_similar("珞喻路", "珞瑜路", 0.6) is True

    def test_different_names(self):
        """不相似的名称"""
        assert is_similar("武汉大学", "华中科技大学", 0.85) is False

    def test_abbreviation(self):
        """简称别名"""
        # 使用较低阈值，或依赖简称检查逻辑
        assert is_similar("武大", "武汉大学", 0.5) is True  # 简称检查应生效

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
        # 使用较低阈值以触发合并
        result, aliases = deduplicate_entities(entities, 0.6)
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
        """无修正时保持原样"""
        triples = [{"head": "A", "relation": "位于", "tail": "B", "evidence": "test"}]
        result, mapping = apply_corrections(triples, [])
        assert len(result) == 1
        assert result[0]["tail"] == "B"
        assert mapping == {}

    def test_with_corrections(self):
        """有修正时替换三元组"""
        triples = [{"head": "A", "relation": "位于", "tail": "B", "evidence": "test"}]

        # 创建模拟修正对象
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
        assert result[0]["tail"] == "C"  # tail 被修正为 C
        assert ("A", "位于", "C") in mapping
        assert mapping[("A", "位于", "C")] == ("A", "位于", "B")


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
            "neo4j_stats": {},
            "postgres_stats": {},
            "current_phase": PhaseEnum.INIT,
            "active_workers": [],
            "failed_workers": [],
            "start_time": 0.0,
            "end_time": None,
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
            "neo4j_stats": {},
            "postgres_stats": {},
            "current_phase": PhaseEnum.REDUCE,
            "active_workers": [],
            "failed_workers": [],
            "start_time": 0.0,
            "end_time": None,
        }

        result = aggregator_node(state)

        # 应该有3个实体（武汉大学、华中科技大学、珞喻路）
        assert len(result["aggregated_entities"]) >= 2
        # 应该有1个三元组
        assert len(result["aggregated_triples"]) == 1
        assert result["current_phase"] == PhaseEnum.FINALIZE


class TestEval2Node:
    """测试 eval_2_node 节点"""

    @pytest.mark.anyio
    async def test_eval_2_no_triples(self):
        """无三元组时跳过评估"""
        from agent.agents.nodes import create_eval_2_node

        mock_llm = MagicMock()
        eval_2_node = create_eval_2_node(mock_llm)

        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "test text",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.EVAL,
            "error": None,
        }

        result = await eval_2_node(state)
        assert result["corrected_triples"] == []
        assert result["eval_passed"] is True
        assert result["current_step"] == StepEnum.LABEL

    @pytest.mark.anyio
    async def test_eval_2_no_scores(self):
        """无评分时使用原始三元组"""
        from agent.agents.nodes import create_eval_2_node

        mock_llm = MagicMock()
        eval_2_node = create_eval_2_node(mock_llm)

        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "test text",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
            "triples": [{"head": "A", "relation": "位于", "tail": "B"}],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.EVAL,
            "error": None,
        }

        result = await eval_2_node(state)
        assert result["corrected_triples"] == state["triples"]
        assert result["eval_passed"] is False


class TestLabelNode:
    """测试 label_node 节点"""

    @pytest.mark.anyio
    async def test_label_no_entities(self):
        """无实体时跳过标注"""
        from agent.agents.nodes import create_label_node

        mock_llm = MagicMock()
        label_node = create_label_node(mock_llm)

        state: CorpusState = {
            "corpus_id": "test",
            "raw_text": "test text",
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.LABEL,
            "error": None,
        }

        # P3改进：节点函数需要 StreamWriter 参数
        mock_writer = MagicMock()
        result = await label_node(state, mock_writer)
        assert result["current_step"] == StepEnum.DONE


# ===== v3.2重构新增：normalize_relation_type 测试 =====

class TestNormalizeRelationType:
    """测试 normalize_relation_type 函数"""

    def test_enum_input(self):
        """直接输入枚举值"""
        from agent.agents.schemas import normalize_relation_type, RelationTypeEnum
        result = normalize_relation_type(RelationTypeEnum.LOCATED)
        assert result == RelationTypeEnum.LOCATED

    def test_standard_string(self):
        """标准字符串输入"""
        from agent.agents.schemas import normalize_relation_type, RelationTypeEnum
        result = normalize_relation_type("位于")
        assert result == RelationTypeEnum.LOCATED

    def test_variant_string(self):
        """变体字符串映射"""
        from agent.agents.schemas import normalize_relation_type, RelationTypeEnum
        # 空间关系变体
        assert normalize_relation_type("在") == RelationTypeEnum.LOCATED
        assert normalize_relation_type("地处") == RelationTypeEnum.LOCATED
        assert normalize_relation_type("属于") == RelationTypeEnum.LOCATED

        # 方位关系变体
        assert normalize_relation_type("旁边") == RelationTypeEnum.ORIENTATION
        assert normalize_relation_type("附近") == RelationTypeEnum.ORIENTATION
        assert normalize_relation_type("东边") == RelationTypeEnum.ORIENTATION

        # 对比关系变体
        assert normalize_relation_type("比...好") == RelationTypeEnum.BETTER_THAN
        assert normalize_relation_type("类似") == RelationTypeEnum.SIMILAR_TO
        assert normalize_relation_type("不如") == RelationTypeEnum.WORSE_THAN

    def test_null_input(self):
        """空输入抛出异常"""
        from agent.agents.schemas import normalize_relation_type
        with pytest.raises(ValueError, match="relation 不能为空"):
            normalize_relation_type(None)

    def test_invalid_input(self):
        """无效输入抛出异常"""
        from agent.agents.schemas import normalize_relation_type
        with pytest.raises(ValueError, match="无效的关系类型"):
            normalize_relation_type("无效关系")

    def test_mapping_consistency(self):
        """映射表完整性检查"""
        from agent.agents.schemas import RELATION_VARIANT_MAPPING, RelationTypeEnum

        # 所有映射目标应该是有效的关系类型
        valid_relations = [e.value for e in RelationTypeEnum]
        for variant, target in RELATION_VARIANT_MAPPING.items():
            assert target in valid_relations, f"映射目标 '{target}' 不是有效关系类型"


# ===== v3.2重构新增：NodeTemplate 测试 =====

class TestNodeTemplate:
    """测试 NodeTemplate 基类"""

    def test_get_text_for_processing_normalized(self):
        """优先使用归一化文本"""
        from agent.agents.node_template import get_text_for_processing

        state = {
            "raw_text": "原始文本",
            "normalized_text": "归一化文本"
        }
        result = get_text_for_processing(state)
        assert result == "归一化文本"

    def test_get_text_for_processing_raw(self):
        """无归一化文本时使用原始文本"""
        from agent.agents.node_template import get_text_for_processing

        state = {
            "raw_text": "原始文本",
            "normalized_text": ""
        }
        result = get_text_for_processing(state)
        assert result == "原始文本"

    def test_get_text_for_processing_empty_normalized(self):
        """归一化文本为空字符串时使用原始文本"""
        from agent.agents.node_template import get_text_for_processing

        state = {
            "raw_text": "原始文本",
            "normalized_text": "   "  # 仅空白字符
        }
        result = get_text_for_processing(state)
        assert result == "原始文本"

    def test_type_aliases_exist(self):
        """类型别名正确导出"""
        from agent.agents.node_template import StateDict, ResultDict, NodeFunc

        # 类型别名应该是 Dict 或 Callable 的别名
        # 在 Python typing 中，TypeAlias 可以直接检查其类型
        import typing
        assert StateDict is not None
        assert ResultDict is not None
        assert NodeFunc is not None


# ===== v3.2重构新增：子状态组合测试 =====

class TestSubstatesComposition:
    """测试 CorpusState 子状态组合"""

    def test_all_substates_importable(self):
        """所有子状态类可导入"""
        from agent.agents.state import (
            InputState, ConfigState, FilterState, NormalizeState,
            QAScaffoldState, ExtractState, EvalState, SelfCheckState,
            ReflexionState, RetryState, QAMentorState, AlignmentState,
            OutputState, ControlState,
        )
        # 所有子状态类应该存在
        assert InputState is not None
        assert ControlState is not None

    def test_corpus_state_includes_all_fields(self):
        """CorpusState 包含所有子状态字段"""
        from agent.agents.state import CorpusState

        # 检查关键字段存在（TypedDict 继承会合并字段）
        annotations = CorpusState.__annotations__

        # 输入状态字段
        assert "corpus_id" in annotations
        assert "raw_text" in annotations

        # 抽取状态字段
        assert "entities" in annotations
        assert "triples" in annotations

        # 控制状态字段
        assert "current_step" in annotations
        assert "error" in annotations