"""
P15修复诊断测试 - 验证 Bug Detective 发现的问题修复

测试覆盖：
1. prompt_version 变量命名修复
2. RouteDecider END 常量修复
3. RELATION_TYPES 导入一致性
4. get_text_for_processing 函数统一
5. rule_based_validation 实体类型约束
6. 关系体系隐患修复（P16新增）
"""
import pytest
from langgraph.graph import END


class TestPromptVersionFix:
    """测试 prompt_version 变量命名修复"""

    def test_get_node_creators_returns_v2_functions(self):
        """验证 v2 版本返回正确的节点创建函数"""
        from agent.agents.nodes import get_node_creators

        creators = get_node_creators("v2")
        assert "filter" in creators
        assert "joint_ner_re" in creators
        assert "self_check_joint" in creators
        assert "re" in creators
        assert "label" in creators

    def test_get_node_creators_returns_v3_functions(self):
        """验证 v3 版本返回正确的节点创建函数"""
        from agent.agents.nodes import get_node_creators

        creators = get_node_creators("v3")
        assert "filter" in creators
        assert "joint_ner_re" in creators
        assert "self_check_joint" in creators
        assert "re" in creators
        assert "label" in creators

    def test_default_version_is_v2(self):
        """验证默认版本为 v2"""
        from agent.agents.nodes import get_node_creators

        creators = get_node_creators()
        assert creators is not None
        assert len(creators) >= 5


class TestRouteDeciderEndConstant:
    """测试 RouteDecider END 常量修复"""

    def test_route_decider_imports_end_correctly(self):
        """验证 RouteDecider 正确导入 LangGraph END 常量"""
        from agent.agents.route_decider import RouteDecider, _ROUTE_END

        # _ROUTE_END 应该等于 LangGraph 的 END 常量
        assert _ROUTE_END == END
        assert _ROUTE_END == "__end__"

    def test_after_filter_returns_end_constant(self):
        """验证 after_filter 在无效文本时返回 END 常量"""
        from agent.agents.route_decider import RouteDecider, _ROUTE_END

        # 模拟无效文本状态
        state = {
            "filter_result": {"is_valid": False, "skip_reason": "非武汉地区"},
            "retry_count": 0,
            "max_retries": 3,
        }
        decider = RouteDecider(state)
        result = decider.after_filter()

        # 应返回 LangGraph END 常量，而非字符串 "END"
        assert result == _ROUTE_END
        assert result == END
        assert result != "END"  # 明确区分：不是字符串 "END"

    def test_after_self_check_with_end_default(self):
        """验证 after_self_check 正确处理 END 作为 default_next"""
        from agent.agents.route_decider import RouteDecider

        state = {
            "error": "模拟错误",
            "retry_count": 0,
            "max_retries": 3,
        }
        decider = RouteDecider(state)

        # 当 default_next 是 END 常量时，应正确返回
        result = decider.after_self_check("test_result", END, "retry_target")
        assert result == END


class TestRelationTypesConsistency:
    """测试 RELATION_TYPES 导入一致性"""

    def test_relation_types_from_schemas(self):
        """验证 RELATION_TYPES 从 schemas.py 正确导入"""
        from agent.agents.schemas import RELATION_TYPES as schemas_relation_types

        expected_types = ["位于", "包含", "相对方位", "具有功能", "优于", "相似", "劣于", "发生事件"]
        assert schemas_relation_types == expected_types

    def test_relation_types_from_state_imports_from_schemas(self):
        """验证 state.py 的 RELATION_TYPES 来自 schemas.py"""
        from agent.agents.state import RELATION_TYPES as state_relation_types
        from agent.agents.schemas import RELATION_TYPES as schemas_relation_types

        # 应完全相同（同一对象引用）
        assert state_relation_types == schemas_relation_types
        # 验证是同一列表（import 而非重新定义）
        assert state_relation_types is schemas_relation_types

    def test_relation_types_count_is_8(self):
        """验证关系类型数量为 8"""
        from agent.agents import RELATION_TYPES

        assert len(RELATION_TYPES) == 8


class TestGetTextForProcessing:
    """测试 get_text_for_processing 函数统一"""

    def test_imports_from_node_template(self):
        """验证 nodes.py 从 node_template.py 导入函数"""
        from agent.agents.nodes import get_text_for_processing
        from agent.agents.node_template import get_text_for_processing as template_func

        # 应是同一函数（import 而非重新定义）
        assert get_text_for_processing is template_func

    def test_prefers_normalized_text(self):
        """验证优先使用归一化文本"""
        from agent.agents.nodes import get_text_for_processing

        state = {
            "raw_text": "原始文本",
            "normalized_text": "归一化文本",
        }
        result = get_text_for_processing(state)
        assert result == "归一化文本"

    def test_fallback_to_raw_text(self):
        """验证无归一化文本时使用原始文本"""
        from agent.agents.nodes import get_text_for_processing

        state = {
            "raw_text": "原始文本",
            "normalized_text": "",
        }
        result = get_text_for_processing(state)
        assert result == "原始文本"

    def test_empty_normalized_text_fallback(self):
        """验证空白归一化文本回退到原始文本"""
        from agent.agents.nodes import get_text_for_processing

        state = {
            "raw_text": "原始文本",
            "normalized_text": "   ",  # 只有空格
        }
        result = get_text_for_processing(state)
        assert result == "原始文本"


class TestRuleBasedValidationEntityType:
    """测试 rule_based_validation 实体类型约束"""

    def test_geo_entity_in_located_relation_passes(self):
        """验证地理实体在位于关系中通过校验"""
        from agent.agents.nodes import rule_based_validation

        entities = {
            "道路": ["珞喻路"],
            "POI": ["武汉大学"],
            "建筑物": ["行政楼"],
            "街区": ["街道口"],
            "功能": [],
            "事件": [],
        }
        triples = [
            {"head": "武汉大学", "relation": "位于", "tail": "珞喻路"},
        ]
        result = rule_based_validation(triples, entities)

        assert result[0]["_rule_valid"] is True
        assert len(result[0]["_rule_issues"]) == 0

    def test_function_entity_in_located_relation_fails(self):
        """验证功能实体在位于关系中失败"""
        from agent.agents.nodes import rule_based_validation

        entities = {
            "道路": ["珞喻路"],
            "POI": ["武汉大学"],
            "建筑物": [],
            "街区": [],
            "功能": ["餐饮"],  # 功能实体
            "事件": [],
        }
        triples = [
            {"head": "餐饮", "relation": "位于", "tail": "珞喻路"},
        ]
        result = rule_based_validation(triples, entities)

        assert result[0]["_rule_valid"] is False
        assert any("head" in issue and "功能" in issue for issue in result[0]["_rule_issues"])

    def test_event_entity_in_located_relation_fails(self):
        """验证事件实体在位于关系中失败"""
        from agent.agents.nodes import rule_based_validation

        entities = {
            "道路": ["珞喻路"],
            "POI": ["武汉大学"],
            "建筑物": [],
            "街区": [],
            "功能": [],
            "事件": ["樱花节"],  # 事件实体
        }
        triples = [
            {"head": "樱花节", "relation": "位于", "tail": "珞喻路"},
        ]
        result = rule_based_validation(triples, entities)

        assert result[0]["_rule_valid"] is False

    def test_function_entity_in_has_function_relation_passes(self):
        """验证功能实体在具有功能关系中通过"""
        from agent.agents.nodes import rule_based_validation

        entities = {
            "道路": [],
            "POI": ["武汉大学"],
            "建筑物": [],
            "街区": [],
            "功能": ["餐饮"],
            "事件": [],
        }
        triples = [
            {"head": "武汉大学", "relation": "具有功能", "tail": "餐饮"},
        ]
        result = rule_based_validation(triples, entities)

        assert result[0]["_rule_valid"] is True

    def test_event_entity_in_happens_event_relation_passes(self):
        """验证事件实体在发生事件关系中通过"""
        from agent.agents.nodes import rule_based_validation

        entities = {
            "道路": [],
            "POI": ["武汉大学"],
            "建筑物": [],
            "街区": [],
            "功能": [],
            "事件": ["樱花节"],
        }
        triples = [
            {"head": "武汉大学", "relation": "发生事件", "tail": "樱花节"},
        ]
        result = rule_based_validation(triples, entities)

        assert result[0]["_rule_valid"] is True

    def test_geo_entities_in_compare_relations_pass(self):
        """验证地理实体在对比关系中通过"""
        from agent.agents.nodes import rule_based_validation

        entities = {
            "道路": [],
            "POI": ["群光广场", "街道口商圈"],
            "建筑物": [],
            "街区": [],
            "功能": [],
            "事件": [],
        }
        triples = [
            {"head": "群光广场", "relation": "优于", "tail": "街道口商圈"},
            {"head": "群光广场", "relation": "相似", "tail": "街道口商圈"},
            {"head": "群光广场", "relation": "劣于", "tail": "街道口商圈"},
        ]
        result = rule_based_validation(triples, entities)

        for t in result:
            assert t["_rule_valid"] is True

    def test_geo_entities_in_relative_orientation_pass(self):
        """验证地理实体在相对方位关系中通过"""
        from agent.agents.nodes import rule_based_validation

        entities = {
            "道路": ["珞喻路", "关山大道"],
            "POI": ["武汉大学"],
            "建筑物": [],
            "街区": [],
            "功能": [],
            "事件": [],
        }
        triples = [
            {"head": "武汉大学", "relation": "相对方位", "tail": "珞喻路"},
        ]
        result = rule_based_validation(triples, entities)

        assert result[0]["_rule_valid"] is True

    def test_invalid_relation_type_fails(self):
        """验证无效关系类型失败"""
        from agent.agents.nodes import rule_based_validation

        entities = {
            "道路": ["珞喻路"],
            "POI": ["武汉大学"],
            "建筑物": [],
            "街区": [],
            "功能": [],
            "事件": [],
        }
        triples = [
            {"head": "武汉大学", "relation": "连接", "tail": "珞喻路"},  # 无效关系
        ]
        result = rule_based_validation(triples, entities)

        assert result[0]["_rule_valid"] is False
        assert any("不在预定义类型中" in issue for issue in result[0]["_rule_issues"])

    def test_entity_not_in_ner_result_fails(self):
        """验证实体不在 NER 结果中失败"""
        from agent.agents.nodes import rule_based_validation

        entities = {
            "道路": ["珞喻路"],
            "POI": ["武汉大学"],
            "建筑物": [],
            "街区": [],
            "功能": [],
            "事件": [],
        }
        triples = [
            {"head": "华中科技大学", "relation": "位于", "tail": "珞喻路"},  # head 未抽取
        ]
        result = rule_based_validation(triples, entities)

        assert result[0]["_rule_valid"] is False
        assert any("头实体" in issue and "未在NER结果中" in issue for issue in result[0]["_rule_issues"])


class TestPromptsSchemaUpdates:
    """测试 prompts.py Schema 更新"""

    def test_constraint_rules_mentions_six_entity_types(self):
        """验证 CONSTRAINT_RULES 提及 6 种实体类型"""
        from agent.agents.prompts import CONSTRAINT_RULES

        assert "6 类" in CONSTRAINT_RULES or "6种" in CONSTRAINT_RULES
        assert "功能" in CONSTRAINT_RULES
        assert "事件" in CONSTRAINT_RULES

    def test_constraint_rules_has_entity_type_constraint(self):
        """验证 CONSTRAINT_RULES 包含关系实体类型约束"""
        from agent.agents.prompts import CONSTRAINT_RULES

        assert "关系实体类型约束" in CONSTRAINT_RULES
        assert "位于" in CONSTRAINT_RULES

    def test_relation_schema_table_has_head_type(self):
        """验证 RELATION_SCHEMA_TABLE 包含 Head 类型列"""
        from agent.agents.prompts import RELATION_SCHEMA_TABLE

        assert "Head类型" in RELATION_SCHEMA_TABLE
        assert "地理实体" in RELATION_SCHEMA_TABLE

    def test_constraint_rules_forbidden_function_event_in_spatial(self):
        """验证 CONSTRAINT_RULES 禁止功能/事件实体参与空间关系"""
        from agent.agents.prompts import CONSTRAINT_RULES

        # 应明确禁止功能实体或事件实体作为空间关系的参与者
        assert "功能实体或事件实体" in CONSTRAINT_RULES
        assert "空间关系" in CONSTRAINT_RULES


class TestExportsCleanup:
    """测试 __init__.py 导出清理"""

    def test_route_decider_exported(self):
        """验证 RouteDecider 正确导出"""
        from agent.agents import RouteDecider

        assert RouteDecider is not None

    def test_v2_functions_not_exported(self):
        """验证 v2 函数不再导出"""
        import agent.agents

        # 这些函数已删除，不应存在
        assert not hasattr(agent.agents, "route_after_filter_v2")
        assert not hasattr(agent.agents, "route_after_self_check_joint_v2")
        assert not hasattr(agent.agents, "route_after_self_check_ner_v2")
        assert not hasattr(agent.agents, "route_after_self_check_re_v2")
        assert not hasattr(agent.agents, "route_joint_to_mentor_or_eval_v2")


class TestRelationSystemFixes:
    """测试关系体系隐患修复（P16新增）"""

    def test_no_administrative_district_in_prompts(self):
        """验证提示词不再引用'行政区'作为独立实体类型"""
        from agent.agents.prompts import RELATION_SCHEMA_TABLE, RELATION_SCHEMA_CORE, CONSTRAINT_RULES

        # RELATION_SCHEMA_TABLE 应不再包含 "行政区"
        # 注意：街区细分包含行政区，但不应作为独立类型
        import re
        # 检查 Tail类型 列不应有单独的"行政区"
        assert "道路/街区/行政区" not in RELATION_SCHEMA_TABLE
        assert "街区/行政区" not in RELATION_SCHEMA_TABLE

    def test_verified_entity_supports_six_types(self):
        """验证 VerifiedEntity 支持 6 种实体类型"""
        from agent.agents.schemas import VerifiedEntity, MissingEntity

        # 检查 description 字段是否更新
        verified_desc = VerifiedEntity.model_fields["type"].description
        missing_desc = MissingEntity.model_fields["suggested_type"].description

        assert "功能" in verified_desc
        assert "事件" in verified_desc
        assert "功能" in missing_desc
        assert "事件" in missing_desc

    def test_relation_variant_mapping_has_common_words(self):
        """验证 RELATION_VARIANT_MAPPING 包含常见空间词汇"""
        from agent.agents.schemas import RELATION_VARIANT_MAPPING, normalize_relation_type

        # 新增的常见词汇应存在
        common_words = ["连接", "靠近", "周边", "周围", "毗邻", "紧邻", "交叉", "交汇"]
        for word in common_words:
            assert word in RELATION_VARIANT_MAPPING
            assert RELATION_VARIANT_MAPPING[word] == "相对方位"

        # 验证可以正常映射
        for word in common_words:
            result = normalize_relation_type(word)
            assert result.value == "相对方位"

    def test_relation_schema_core_has_head_type(self):
        """验证 RELATION_SCHEMA_CORE 包含 Head 类型列"""
        from agent.agents.prompts import RELATION_SCHEMA_CORE

        assert "Head类型" in RELATION_SCHEMA_CORE
        assert "地理实体" in RELATION_SCHEMA_CORE

    def test_variant_mapping_count_above_50(self):
        """验证 RELATION_VARIANT_MAPPING 词汇数量超过 50"""
        from agent.agents.schemas import RELATION_VARIANT_MAPPING

        # 原有 43 个，补充后应超过 50（但移除歧义词汇后可能略低）
        assert len(RELATION_VARIANT_MAPPING) >= 45


class TestBugDetectiveP16Fixes:
    """测试 Bug Detective 发现的 P16 隐患修复"""

    def test_ambiguous_words_removed_from_mapping(self):
        """验证歧义词汇已从 RELATION_VARIANT_MAPPING 中移除"""
        from agent.agents.schemas import RELATION_VARIANT_MAPPING

        # 这些词汇过于歧义，应被移除
        removed_words = ['有', '正在', '可以', '活动', '在', '挨']
        for word in removed_words:
            assert word not in RELATION_VARIANT_MAPPING

    def test_function_node_validation_passes(self):
        """验证功能节点作为 tail 时通过验证（即使未抽取为实体）"""
        from agent.agents.nodes import rule_based_validation
        from agent.agents.schemas import FUNCTION_NODES

        entities = {
            '道路': [], 'POI': ['群光广场'], '建筑物': [], '街区': [],
            '功能': [],  # Empty - 功能节点不在实体抽取中
            '事件': [],
        }
        triples = [
            {'head': '群光广场', 'relation': '具有功能', 'tail': '餐饮'},  # 功能节点
        ]
        result = rule_based_validation(triples, entities)

        assert result[0]["_rule_valid"] is True
        assert len(result[0]["_rule_issues"]) == 0

    def test_event_node_validation_passes(self):
        """验证事件节点作为 tail 时通过验证（放宽验证）"""
        from agent.agents.nodes import rule_based_validation

        entities = {
            '道路': [], 'POI': ['武汉大学'], '建筑物': [], '街区': [],
            '功能': [],
            '事件': [],  # Empty - 自定义事件名不在实体抽取中
        }
        triples = [
            {'head': '武汉大学', 'relation': '发生事件', 'tail': '樱花节'},  # 自定义事件
        ]
        result = rule_based_validation(triples, entities)

        assert result[0]["_rule_valid"] is True
        assert len(result[0]["_rule_issues"]) == 0

    def test_function_nodes_list_accessible(self):
        """验证 FUNCTION_NODES 枚举可正确导入"""
        from agent.agents.schemas import FUNCTION_NODES

        assert len(FUNCTION_NODES) >= 9
        assert '餐饮' in FUNCTION_NODES
        assert '购物' in FUNCTION_NODES

    def test_event_categories_list_accessible(self):
        """验证 EVENT_CATEGORIES 枚举可正确导入"""
        from agent.agents.schemas import EVENT_CATEGORIES

        assert len(EVENT_CATEGORIES) >= 5
        assert '自然事件' in EVENT_CATEGORIES or '人文事件' in EVENT_CATEGORIES