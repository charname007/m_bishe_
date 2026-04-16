"""
路由决策器 - 集中管理工作流路由逻辑（P15新增）

解决问题：
- 原有20+个路由函数分散在 workflow.py 中
- 每个路由函数都有相似的模式（检查错误 → 检查重试次数 → 判断置信度 → 决定）
- 代码重复度高，维护负担重

改进：
- 创建 RouteDecider 类封装路由决策逻辑
- 使用方法代替独立函数，共享状态访问
- 提供清晰的决策流程和可配置的策略
"""
from typing import Dict, Any, Optional, Callable
from loguru import logger

# P15修复：导入 LangGraph END 常量，避免使用字符串 "END"
from langgraph.graph import END

from .state import CorpusState, DEFAULT_MAX_RETRIES

# 定义内部常量，避免直接使用字符串
_ROUTE_END = END  # LangGraph 结束标记（值为 "__end__"）


class RouteDecider:
    """
    路由决策器 - 集中化路由决策逻辑

    使用方法：
    >>> decider = RouteDecider(state)
    >>> next_node = decider.after_self_check("self_check_joint_result", "eval")

    Attributes:
        state: 当前语料状态
        retry_count: 当前重试次数
        max_retries: 最大重试次数
        has_error: 是否有错误
    """

    def __init__(self, state: CorpusState):
        """
        初始化路由决策器

        Args:
            state: 当前 CorpusState 状态字典
        """
        self.state = state
        self.retry_count: int = state.get("retry_count", 0)
        self.max_retries: int = state.get("max_retries", DEFAULT_MAX_RETRIES)
        self.has_error: bool = bool(state.get("error"))

    def should_retry(self) -> bool:
        """检查是否应该重试（未达到最大重试次数）"""
        return self.retry_count < self.max_retries

    def get_retry_info(self) -> Dict[str, Any]:
        """获取重试信息摘要"""
        return {
            "current": self.retry_count,
            "max": self.max_retries,
            "can_retry": self.should_retry(),
        }

    def log_decision(self, from_node: str, to_node: str, reason: str) -> None:
        """记录路由决策日志"""
        logger.info(f"[{from_node}-Route] → {to_node}, 原因: {reason}")

    def log_warning(self, from_node: str, message: str) -> None:
        """记录警告日志"""
        logger.warning(f"[{from_node}-Route] {message}")

    # ===== 基础路由决策 =====

    def on_error_goto(self, default_target: str, error_target: str = "END") -> str:
        """
        错误时的默认路由

        Args:
            default_target: 默认目标节点
            error_target: 错误时的目标节点（默认END）

        Returns:
            目标节点名称
        """
        if self.has_error:
            return error_target
        return default_target

    def on_retry_limit_goto(self, retry_target: str, continue_target: str) -> str:
        """
        达到重试上限时的路由

        Args:
            retry_target: 需要重试时的目标节点
            continue_target: 继续处理的目标节点

        Returns:
            目标节点名称
        """
        if not self.should_retry():
            self.log_warning("Route", f"达到最大重试 {self.retry_count}/{self.max_retries}")
            return continue_target
        return retry_target

    # ===== Self-Check 通用路由 =====

    def after_self_check(
        self,
        result_key: str,
        default_next: str,
        retry_target: Optional[str] = None,
        confidence_threshold: str = "low"
    ) -> str:
        """
        Self-Check 节点后的通用路由决策

        决策逻辑：
        1. 检查错误 → END 或 default_next
        2. 检查重试次数 → 达到上限则 default_next
        3. 检查 retry_suggested → retry_target
        4. 检查置信度 → confidence_threshold 时 retry_target
        5. 默认 → default_next

        Args:
            result_key: Self-Check 结果的键名（如 "self_check_joint_result"）
            default_next: 默认的下一个节点
            retry_target: 需要重试时的目标节点（如 "joint_ner_re"）
            confidence_threshold: 触发重试的置信度阈值（默认 "low"）

        Returns:
            下一个节点名称
        """
        from_node = result_key.replace("_result", "").replace("_", "-").title()

        # 1. 错误处理
        if self.has_error:
            self.log_warning(from_node, "有错误")
            return default_next if default_next != "END" else "END"

        # 2. 获取检查结果
        check_result = self.state.get(result_key, {})
        retry_suggested = check_result.get("retry_suggested", False)
        confidence = check_result.get("overall_confidence", "medium")

        # 3. 达到最大重试次数
        if not self.should_retry():
            self.log_warning(from_node, f"达到最大重试，强制通过，置信度: {confidence}")
            return default_next

        # 4. 检查是否建议重试
        if retry_suggested and retry_target:
            self.log_decision(from_node, retry_target, "Self-Check建议重试")
            return retry_target

        # 5. 检查置信度
        if confidence == confidence_threshold and retry_target:
            self.log_decision(from_node, retry_target, f"置信度过低: {confidence}")
            return retry_target

        # 6. 通过，继续
        self.log_decision(from_node, default_next, f"通过，置信度: {confidence}")
        return default_next

    # ===== 特定节点路由 =====

    def after_filter(
        self,
        valid_target: str = "ner",
        enable_normalize: bool = False,
        enable_qa_scaffold: bool = False
    ) -> str:
        """
        Filter 后的路由决策

        Args:
            valid_target: 有效文本的目标节点（默认NER）
            enable_normalize: 是否启用Normalize节点
            enable_qa_scaffold: 是否启用QA Scaffold节点

        Returns:
            下一个节点名称
        """
        if self.has_error and not self.state.get("filter_result"):
            self.log_warning("Filter", "筛选失败但有错误，继续处理")
            return valid_target

        filter_result = self.state.get("filter_result", {})
        is_valid = filter_result.get("is_valid", True)
        confidence = filter_result.get("confidence", "medium")

        if is_valid:
            # 根据配置决定下一个节点
            if enable_normalize:
                target = "normalize"
            elif enable_qa_scaffold:
                target = "qa_scaffold"
            else:
                target = valid_target
            self.log_decision("Filter", target, f"文本有效，置信度: {confidence}")
            return target
        else:
            skip_reason = filter_result.get("skip_reason", "未指定原因")
            self.log_decision("Filter", "END", f"文本无效，原因: {skip_reason}")
            return _ROUTE_END  # P15修复：使用 LangGraph END 常量

    def after_joint_extraction(self, enable_self_check: bool = True) -> str:
        """
        Joint_NER_RE 后的路由决策

        Args:
            enable_self_check: 是否启用Self-Check

        Returns:
            下一个节点名称
        """
        if self.has_error:
            self.log_warning("Joint", "有错误，跳转到 Eval")
            return "eval"

        if enable_self_check:
            return "self_check_joint"
        return "eval"

    def after_normalize(self, enable_qa_scaffold: bool = False) -> str:
        """
        Normalize 后的路由决策

        Args:
            enable_qa_scaffold: 是否启用QA Scaffold

        Returns:
            下一个节点名称
        """
        normalize_result = self.state.get("normalize_result", {})
        confidence = normalize_result.get("confidence", "medium")
        has_changes = normalize_result.get("has_changes", False)

        if self.has_error and not normalize_result:
            self.log_warning("Normalize", "归一化失败但有错误，使用原文继续")
        else:
            self.log_decision("Normalize", "ner" if not enable_qa_scaffold else "qa_scaffold",
                            f"置信度: {confidence}, 有改动: {has_changes}")

        return "qa_scaffold" if enable_qa_scaffold else "ner"

    def after_qa_scaffold(self, joint_mode: bool = True) -> str:
        """
        QA Scaffold 后的路由决策

        Args:
            joint_mode: 是否使用联合抽取模式

        Returns:
            下一个节点名称
        """
        qa_result = self.state.get("qa_scaffold_result", {})
        should_skip = qa_result.get("should_skip_detailed_extraction", False)
        confidence = qa_result.get("overall_confidence", "medium")

        if should_skip:
            self.log_decision("QA_Scaffold", "END", "建议跳过详细抽取")
            return _ROUTE_END  # P15修复：使用 LangGraph END 常量

        target = "joint_ner_re" if joint_mode else "ner"
        self.log_decision("QA_Scaffold", target, f"置信度: {confidence}")
        return target

    # ===== P14导师模式路由 =====

    def needs_mentor_help(self, query_count: int = 0, max_queries: int = 2) -> bool:
        """
        检查是否需要导师帮助

        Args:
            query_count: 当前查询次数
            max_queries: 最大查询次数

        Returns:
            是否需要导师帮助
        """
        needs_help = self.state.get("needs_mentor_help", False)
        return needs_help and query_count < max_queries

    def after_joint_with_mentor(
        self,
        enable_bidirectional: bool = True,
        query_count: int = 0,
        max_queries: int = 2
    ) -> str:
        """
        Joint_NER_RE 后的路由（含导师求助）

        Args:
            enable_bidirectional: 是否启用双向交流
            query_count: 当前查询次数
            max_queries: 最大查询次数

        Returns:
            下一个节点名称
        """
        if self.has_error:
            self.log_warning("Joint", "有错误，跳转到 Eval")
            return "eval"

        if enable_bidirectional and self.needs_mentor_help(query_count, max_queries):
            self.log_decision("Joint", "qa_mentor", f"需要导师帮助 (查询: {query_count}/{max_queries})")
            return "qa_mentor"

        return "eval"

    def after_eval_with_mentor(
        self,
        enable_bidirectional: bool = True,
        query_count: int = 0,
        max_queries: int = 2
    ) -> str:
        """
        Eval 后的路由（含导师求助）

        Args:
            enable_bidirectional: 是否启用双向交流
            query_count: 当前查询次数
            max_queries: 最大查询次数

        Returns:
            下一个节点名称
        """
        if self.has_error:
            self.log_warning("Eval", "有错误，跳转到 Label")
            return "label"

        if enable_bidirectional and self.needs_mentor_help(query_count, max_queries):
            self.log_decision("Eval", "qa_mentor", f"需要导师帮助 (查询: {query_count}/{max_queries})")
            return "qa_mentor"

        return "label"

    def after_label_with_mentor(
        self,
        enable_bidirectional: bool = True,
        query_count: int = 0,
        max_queries: int = 2
    ) -> str:
        """
        Label 后的路由（含导师求助）

        Args:
            enable_bidirectional: 是否启用双向交流
            query_count: 当前查询次数
            max_queries: 最大查询次数

        Returns:
            下一个节点名称
        """
        if self.has_error:
            self.log_warning("Label", "有错误，跳转到 QA_Approval")
            return "qa_approval"

        if enable_bidirectional and self.needs_mentor_help(query_count, max_queries):
            self.log_decision("Label", "qa_mentor", f"需要导师帮助 (查询: {query_count}/{max_queries})")
            return "qa_mentor"

        return "qa_approval"

    def after_mentor_response(self) -> str:
        """
        QA_Mentor 后的路由（根据返回目标）

        Returns:
            下一个节点名称
        """
        mentor_response = self.state.get("mentor_response")
        return_to_node = self.state.get("return_to_node")

        if mentor_response and return_to_node:
            self.log_decision("Mentor", return_to_node, "返回到发起查询的节点")
            return return_to_node

        # 检查是否建议跳过详细抽取
        qa_scaffold_result = self.state.get("qa_scaffold_result", {})
        if qa_scaffold_result.get("should_skip_detailed_extraction"):
            self.log_decision("Mentor", "END", "建议跳过详细抽取")
            return _ROUTE_END  # P15修复：使用 LangGraph END 常量

        return "joint_ner_re"

    def after_qa_approval(
        self,
        max_revision_cycles: int = 3
    ) -> str:
        """
        QA_Approval 后的路由（修改循环）

        Args:
            max_revision_cycles: 最大修改轮次

        Returns:
            下一个节点名称
        """
        if self.has_error:
            return _ROUTE_END  # P15修复：使用 LangGraph END 常量

        revision_cycle_count = self.state.get("revision_cycle_count", 0)
        approval_result = self.state.get("qa_approval_result", {})
        overall_status = approval_result.get("overall_status", "approved")
        retry_suggested = approval_result.get("retry_suggested", False)
        retry_target_nodes = approval_result.get("retry_target_nodes", [])

        # 达到最大修改轮次
        if revision_cycle_count >= max_revision_cycles:
            self.log_warning("QA_Approval", f"达到最大修改轮次 {revision_cycle_count}/{max_revision_cycles}")
            return _ROUTE_END  # P15修复：使用 LangGraph END 常量

        # 审批通过
        if overall_status == "approved" and not retry_suggested:
            self.log_decision("QA_Approval", "END", "审批通过")
            return _ROUTE_END  # P15修复：使用 LangGraph END 常量

        # 需要修改
        if retry_suggested and retry_target_nodes:
            target_node = retry_target_nodes[0]
            self.log_decision("QA_Approval", f"revision_joint", f"需要修改: {target_node}")

            if target_node == "joint_ner_re":
                return "revision_joint"
            elif target_node == "eval":
                return "eval"
            elif target_node == "label":
                return "label"

        return _ROUTE_END  # P15修复：使用 LangGraph END 常量


# ===== 兼容性函数包装器 =====
# 提供与原有路由函数兼容的接口，便于渐进式迁移

def create_route_function(decider_method: Callable) -> Callable:
    """
    创建兼容性路由函数

    Args:
        decider_method: RouteDecider 的方法

    Returns:
        路由函数（接收 state 参数）
    """
    def route_function(state: CorpusState) -> str:
        decider = RouteDecider(state)
        return decider_method(decider)
    return route_function


# 预定义的路由函数（使用 RouteDecider）
def route_after_filter_v2(state: CorpusState) -> str:
    """Filter 后路由（使用 RouteDecider）"""
    decider = RouteDecider(state)
    enable_normalize = state.get("_config_enable_normalize", False)
    enable_qa_scaffold = state.get("_config_enable_qa_scaffold", False)
    return decider.after_filter(enable_normalize=enable_normalize, enable_qa_scaffold=enable_qa_scaffold)


def route_after_self_check_joint_v2(state: CorpusState) -> str:
    """Self-Check-Joint 后路由（使用 RouteDecider）"""
    decider = RouteDecider(state)
    return decider.after_self_check("self_check_joint_result", "eval", "joint_ner_re")


def route_after_self_check_ner_v2(state: CorpusState) -> str:
    """Self-Check-NER 后路由（使用 RouteDecider）"""
    decider = RouteDecider(state)
    return decider.after_self_check("self_check_ner_result", "re", "ner")


def route_after_self_check_re_v2(state: CorpusState) -> str:
    """Self-Check-RE 后路由（使用 RouteDecider）"""
    decider = RouteDecider(state)
    return decider.after_self_check("self_check_re_result", "eval", "re")


def route_joint_to_mentor_or_eval_v2(state: CorpusState) -> str:
    """Joint 后路由（含导师求助，使用 RouteDecider）"""
    decider = RouteDecider(state)
    query_count = state.get("query_count", 0)
    max_queries = state.get("max_queries", 2)
    return decider.after_joint_with_mentor(True, query_count, max_queries)