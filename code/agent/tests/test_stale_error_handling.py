"""
回归测试：防止全局 error 残留导致路由误判与聚合误跳过。
"""

from agent.agents.nodes import create_aggregator_node
from agent.agents.state import PhaseEnum
from agent.agents.workflow import (
    route_after_joint_extraction,
    route_after_self_check_eval,
    route_after_self_check_label,
    route_after_self_check_qa,
)


def test_route_after_joint_extraction_ignores_stale_error_when_joint_result_exists():
    """联合抽取已有结果时，不应因历史 error 跳过 self_check_joint。"""
    state = {
        "error": "历史错误",
        "joint_extraction_result": {"entities": [{"name": "武汉大学"}]},
        "triples": [{"head": "武汉大学", "relation": "位于", "tail": "珞喻路"}],
    }

    result = route_after_joint_extraction(state)
    assert result == "self_check_joint"


def test_route_after_self_check_qa_uses_check_result_even_with_stale_error():
    """Self-Check-QA 已返回建议重试时，应走 qa_scaffold。"""
    state = {
        "error": "历史错误",
        "self_check_qa_result": {"retry_suggested": True},
        "retry_count": 0,
        "max_retries": 3,
    }

    result = route_after_self_check_qa(state)
    assert result == "qa_scaffold"


def test_route_after_self_check_eval_uses_check_result_even_with_stale_error():
    """Self-Check-Eval 已返回建议重试时，应走 eval。"""
    state = {
        "error": "历史错误",
        "self_check_eval_result": {"retry_suggested": True},
        "retry_count": 0,
        "max_retries": 3,
    }

    result = route_after_self_check_eval(state)
    assert result == "eval"


def test_route_after_self_check_label_uses_check_result_even_with_stale_error():
    """Self-Check-Label 已返回建议重试时，应走 label。"""
    state = {
        "error": "历史错误",
        "self_check_label_result": {"retry_suggested": True},
        "retry_count": 0,
        "max_retries": 3,
        "_config_enable_entity_alignment": False,
    }

    result = route_after_self_check_label(state)
    assert result == "label"


def test_aggregator_keeps_usable_corpus_with_stale_error():
    """即使有历史 error，只要有有效抽取结果就不应被聚合器跳过。"""
    aggregator_node = create_aggregator_node()
    state = {
        "batch_id": "test_batch",
        "corpus_list": [],
        "total_count": 1,
        "worker_count": 1,
        "corpus_partitions": {},
        "worker_results": [
            {
                "worker_id": "worker_0",
                "results": [
                    {
                        "corpus_id": "c1",
                        "error": "历史错误",
                        "entities": {"POI": ["武汉大学"], "道路": ["珞喻路"]},
                        "corrected_triples": [
                            {
                                "head": "武汉大学",
                                "relation": "位于",
                                "tail": "珞喻路",
                                "evidence": "武汉大学在珞喻路上",
                            }
                        ],
                        "entity_attrs": {},
                        "relation_attrs": {},
                    }
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
    assert len(result["aggregated_entities"]) >= 2
    assert len(result["aggregated_triples"]) == 1
