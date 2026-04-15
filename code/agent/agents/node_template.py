"""
节点模板基类 - 封装节点工厂函数的公共模式

通过抽象基类封装：
- PydanticOutputParser 创建
- StreamWriter 进度事件
- Try/Except 异常处理
- 日志记录
- 返回值结构

使用方式：
1. 继承 NodeTemplate 基类
2. 定义类属性：step_name, result_key, pydantic_model, prompt_template, next_step, error_next_step
3. 实现 build_prompt_args() 和 process_result() 方法
4. 在 nodes.py 中调用 template.create_node() 创建节点函数
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional, List, Callable, Awaitable, ClassVar
from pydantic import BaseModel

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import StreamWriter

from loguru import logger

from .state import CorpusState, StepEnum


# ===== 类型别名 =====

StateDict = Dict[str, Any]
ResultDict = Dict[str, Any]
NodeFunc = Callable[[StateDict, StreamWriter], Awaitable[ResultDict]]


def get_text_for_processing(state: StateDict) -> str:
    """
    获取用于处理的文本，优先使用归一化文本

    Args:
        state: 当前语料状态

    Returns:
        用于 NER/RE/Eval 等后续处理的文本
        - 如果有归一化文本（normalized_text），优先使用
        - 否则使用原始文本（raw_text）
    """
    normalized = state.get("normalized_text", "")
    if normalized and normalized.strip():
        return normalized
    return state.get("raw_text", "")


class NodeTemplate(ABC):
    """
    节点模板基类 - 封装 LLM 节点的公共模式

    子类需要定义的属性：
    - step_name: 步骤名称（如 "filter", "ner", "re"）
    - result_key: 结果存储键（如 "filter_result", "entities"）
    - pydantic_model: Pydantic 输出模型类
    - prompt_template: Prompt 模板对象
    - next_step: 成功时的下一步（StepEnum）
    - error_next_step: 失败时的下一步（StepEnum）

    子类需要实现的方法：
    - build_prompt_args(): 构建 prompt 参数
    - process_result(): 处理 LLM 返回结果
    """

    # 子类必须定义的类属性（使用 ClassVar 明确标识）
    step_name: ClassVar[str]
    result_key: ClassVar[str]
    pydantic_model: ClassVar[Type[BaseModel]]
    prompt_template: ClassVar[ChatPromptTemplate]
    next_step: ClassVar[StepEnum]
    error_next_step: ClassVar[StepEnum]

    # 可选覆盖的类属性
    use_normalized_text: ClassVar[bool] = True  # 是否使用归一化文本
    skip_if_no_text: ClassVar[bool] = False      # 无文本时是否跳过

    # 实例属性类型声明
    llm: Any
    parser: PydanticOutputParser

    def __init__(self, llm: Any) -> None:
        """
        初始化节点模板

        Args:
            llm: LangChain LLM 实例
        """
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=self.pydantic_model)

    def get_start_message(self) -> str:
        """开始事件的消息，可被子类覆盖"""
        return f"开始{self.step_name}处理"

    def should_skip(self, state: StateDict) -> Optional[str]:
        """
        检查是否应该跳过节点

        Args:
            state: 当前状态

        Returns:
            跳过原因字符串，或 None 表示不跳过
        """
        if self.skip_if_no_text:
            text = get_text_for_processing(state)
            if not text or not text.strip():
                return "无有效文本"
        return None

    @abstractmethod
    def build_prompt_args(self, state: StateDict) -> Dict[str, Any]:
        """
        构建 prompt 参数

        Args:
            state: 当前状态

        Returns:
            Prompt 模板的参数字典
        """
        pass

    @abstractmethod
    def process_result(self, result: BaseModel, state: StateDict) -> Dict[str, Any]:
        """
        处理 LLM 返回结果

        Args:
            result: Pydantic 解析后的结果对象
            state: 当前状态

        Returns:
            包含 result_data 和 writer_fields 的字典：
            {
                "result_data": 存储到 state 的数据,
                "writer_fields": 发送到 writer 的额外字段,
            }
        """
        pass

    def get_default_error_result(self, state: StateDict) -> ResultDict:
        """
        失败时的默认结果，可被子类覆盖

        Args:
            state: 当前状态

        Returns:
            默认结果字典（保守策略）
        """
        return {}

    def get_skip_return(self, state: StateDict) -> ResultDict:
        """
        跳过时的返回值，可被子类覆盖

        Args:
            state: 当前状态

        Returns:
            跳过时的返回字典
        """
        return {
            self.result_key: {},
            "current_step": self.next_step,
        }

    async def execute(self, state: StateDict, writer: StreamWriter) -> ResultDict:
        """
        标准执行流程 - 封装 try/except/writer 模式

        Args:
            state: 当前状态
            writer: StreamWriter 实例

        Returns:
            更新状态的字典
        """
        corpus_id = state.get('corpus_id', 'unknown')
        logger.info(f"[{self.step_name.upper()}] 处理语料: {corpus_id}")

        # 发送开始事件
        writer({
            "step": self.step_name,
            "corpus_id": corpus_id,
            "status": "started",
            "message": self.get_start_message(),
        })

        # 检查跳过条件
        skip_reason = self.should_skip(state)
        if skip_reason:
            logger.debug(f"[{self.step_name.upper()}] 跳过: {skip_reason}")
            writer({
                "step": self.step_name,
                "corpus_id": corpus_id,
                "status": "skipped",
                "reason": skip_reason,
            })
            return self.get_skip_return(state)

        try:
            # 获取处理文本
            text = get_text_for_processing(state) if self.use_normalized_text else state.get("raw_text", "")

            # 构建 prompt
            prompt_args = self.build_prompt_args(state)
            prompt_text = self.prompt_template.invoke(prompt_args)

            # 添加格式指令并调用 LLM
            full_prompt = f"{prompt_text.messages[1].content}\n\n{self.parser.get_format_instructions()}"
            response = await self.llm.ainvoke(full_prompt)
            result = self.parser.parse(response.content)

            # 处理结果
            processed = self.process_result(result, state)
            result_data = processed.get("result_data", {})
            writer_fields = processed.get("writer_fields", {})

            # 发送完成事件
            writer({
                "step": self.step_name,
                "corpus_id": corpus_id,
                "status": "completed",
                **writer_fields,
            })

            logger.info(f"[{self.step_name.upper()}] 完成: {self._get_result_summary(result_data)}")

            return {
                self.result_key: result_data,
                "current_step": self.next_step,
            }

        except Exception as e:
            logger.error(f"[{self.step_name.upper()}] 失败: {e}")
            writer({
                "step": self.step_name,
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e),
            })
            return {
                self.result_key: self.get_default_error_result(state),
                "error": str(e),
                "current_step": self.error_next_step,
            }

    def _get_result_summary(self, result_data: ResultDict) -> str:
        """获取结果摘要用于日志，可被子类覆盖"""
        if isinstance(result_data, dict):
            if "entities" in result_data:
                return f"{len(result_data.get('entities', {}))}个实体"
            if "triples" in result_data:
                return f"{len(result_data.get('triples', []))}个三元组"
        return "完成"

    def create_node(self) -> NodeFunc:
        """
        创建节点函数 - 返回可用于 LangGraph 的 async 函数

        Returns:
            async def node(state, writer) -> Dict
        """
        async def node(state: StateDict, writer: StreamWriter) -> ResultDict:
            return await self.execute(state, writer)
        return node


# ===== 特殊节点模板（覆盖部分方法） =====

class RawTextNodeTemplate(NodeTemplate):
    """使用原始文本（而非归一化文本）的节点模板"""

    use_normalized_text: ClassVar[bool] = False


class NoLLMNodeTemplate(ABC):
    """
    无 LLM 调用的节点模板 - 用于 coordinator, aggregator 等非 LLM 节点
    """

    step_name: ClassVar[str]
    result_key: ClassVar[str]
    next_step: ClassVar[StepEnum]
    error_next_step: ClassVar[StepEnum]

    def get_start_message(self) -> str:
        return f"开始{self.step_name}处理"

    @abstractmethod
    def execute_logic(self, state: StateDict) -> ResultDict:
        """执行节点逻辑（无 LLM）"""
        pass

    async def execute(self, state: StateDict, writer: StreamWriter) -> ResultDict:
        corpus_id = state.get('corpus_id', 'unknown')
        logger.info(f"[{self.step_name.upper()}] 处理语料: {corpus_id}")

        writer({
            "step": self.step_name,
            "corpus_id": corpus_id,
            "status": "started",
            "message": self.get_start_message(),
        })

        try:
            result_data = self.execute_logic(state)

            writer({
                "step": self.step_name,
                "corpus_id": corpus_id,
                "status": "completed",
            })

            return {
                self.result_key: result_data,
                "current_step": self.next_step,
            }

        except Exception as e:
            logger.error(f"[{self.step_name.upper()}] 失败: {e}")
            writer({
                "step": self.step_name,
                "corpus_id": corpus_id,
                "status": "error",
                "error": str(e),
            })
            return {
                self.result_key: {},
                "error": str(e),
                "current_step": self.error_next_step,
            }

    def create_node(self) -> NodeFunc:
        async def node(state: StateDict, writer: StreamWriter) -> ResultDict:
            return await self.execute(state, writer)
        return node


__all__ = [
    "NodeTemplate",
    "RawTextNodeTemplate",
    "NoLLMNodeTemplate",
    "get_text_for_processing",
    "StateDict",
    "ResultDict",
    "NodeFunc",
]