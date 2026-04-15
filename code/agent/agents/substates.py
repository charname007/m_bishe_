"""
子状态定义 - 用于 CorpusState 的多重继承组合

通过 TypedDict 多重继承，将 CorpusState 拆分为逻辑清晰的子状态，
便于维护和理解，同时保持 LangGraph 兼容性。

注意：Annotated reducer 函数在 state.py 中定义，这里仅提供类型结构。
实际使用时，应从 state.py 导入带 reducer 的子状态。
"""

# ===== 子状态分组说明 =====
#
# CorpusState 拆分为以下逻辑分组：
#
# 1. InputState        - 输入：corpus_id, raw_text
# 2. ConfigState       - 配置标记：_config_enable_normalize, _config_enable_qa_scaffold
# 3. FilterState       - 筛选结果：filter_result
# 4. NormalizeState    - 归一化结果：normalize_result, normalized_text
# 5. QAScaffoldState   - QA脚手架：qa_scaffold_result, semantic_summary, qa_entity_hints, qa_relation_hints, qa_context_dependencies
# 6. ExtractState      - 抽取结果：entities, triples, joint_extraction_result, extraction_strategy, entity_attrs, relation_attrs
# 7. EvalState         - 评估结果：eval_scores, eval_passed, corrected_triples
# 8. SelfCheckState    - 校验结果：8个 self_check_xxx_result 字段
# 9. ReflexionState    - 反思机制：reflection_text, improvement_strategy, reflection_history
# 10. RetryState       - 重试控制：retry_count, max_retries, retry_reason, retry_suggested, problem_entities, problem_triples, needs_review
# 11. QAMentorState    - QA导师：mentor_guidance, qa_approval_result, integrated_semantic_summary, revision_feedbacks, revision_cycle_count, max_revision_cycles, pending_approval_nodes, reasoning_trace
# 12. AlignmentState   - 实体对齐：entity_alignment_result, aligned_entity_ids, new_entity_names
# 13. OutputState      - 最终输出：final_entities, final_triples, verification_confidence
# 14. ControlState     - 流程控制：current_step, error
#
# ===== 使用方式 =====
#
# 在 state.py 中：
#   from .substates_groups import (
#       InputState, ConfigState, FilterState, NormalizeState, QAScaffoldState,
#       ExtractState, EvalState, SelfCheckState, ReflexionState, RetryState,
#       QAMentorState, AlignmentState, OutputState, ControlState,
#       merge_list, merge_dict, replace_value,
#   )
#
#   class CorpusState(
#       InputState, ConfigState, FilterState, NormalizeState, QAScaffoldState,
#       ExtractState, EvalState, SelfCheckState, ReflexionState, RetryState,
#       QAMentorState, AlignmentState, OutputState, ControlState
#   ):
#       pass
#
# ===== 字段清单（供参考） =====

SUBSTATE_FIELDS = {
    "InputState": ["corpus_id", "raw_text"],
    "ConfigState": ["_config_enable_normalize", "_config_enable_qa_scaffold"],
    "FilterState": ["filter_result"],
    "NormalizeState": ["normalize_result", "normalized_text"],
    "QAScaffoldState": ["qa_scaffold_result", "semantic_summary", "qa_entity_hints", "qa_relation_hints", "qa_context_dependencies"],
    "ExtractState": ["entities", "triples", "joint_extraction_result", "extraction_strategy", "entity_attrs", "relation_attrs"],
    "EvalState": ["eval_scores", "eval_passed", "corrected_triples"],
    "SelfCheckState": [
        "self_check_ner_result", "self_check_re_result",
        "self_check_filter_result", "self_check_normalize_result",
        "self_check_qa_result", "self_check_joint_result",
        "self_check_eval_result", "self_check_label_result",
    ],
    "ReflexionState": ["reflection_text", "improvement_strategy", "reflection_history"],
    "RetryState": ["retry_count", "max_retries", "retry_reason", "retry_suggested", "problem_entities", "problem_triples", "needs_review"],
    "QAMentorState": [
        "mentor_guidance", "qa_approval_result", "integrated_semantic_summary",
        "revision_feedbacks", "revision_cycle_count", "max_revision_cycles",
        "pending_approval_nodes", "reasoning_trace",
    ],
    "AlignmentState": ["entity_alignment_result", "aligned_entity_ids", "new_entity_names"],
    "OutputState": ["final_entities", "final_triples", "verification_confidence"],
    "ControlState": ["current_step", "error"],
}

# ===== 总字段统计 =====
TOTAL_FIELDS = sum(len(fields) for fields in SUBSTATE_FIELDS.values())
# TOTAL_FIELDS = 52 个字段