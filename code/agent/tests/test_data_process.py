"""
数据处理框架单元测试
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from agent.kg.data_process import (
    validate_identifier,
    sanitize_error_message,
    ProcessContext,
    Pipeline,
    ProcessStep,
    AsyncProcessStep,
    ValidateStep,
    TransformStep,
    FilterStep,
    SkipStep,
    LogStep,
    CallbackStep,
    RecordToTextStep,
    BatchTextCollector,
)


# ==================== 安全验证工具测试 ====================

class TestValidateIdentifier:
    """测试 validate_identifier 函数"""

    def test_valid_identifier(self):
        """有效的标识符"""
        assert validate_identifier("table_name") == "table_name"
        assert validate_identifier("triples") == "triples"
        assert validate_identifier("_private") == "_private"
        assert validate_identifier("table123") == "table123"

    def test_empty_identifier(self):
        """空标识符"""
        with pytest.raises(ValueError, match="标识符不能为空"):
            validate_identifier("")

    def test_starts_with_digit(self):
        """以数字开头的标识符"""
        with pytest.raises(ValueError, match="不能以数字开头"):
            validate_identifier("123table")

    def test_contains_special_chars(self):
        """包含特殊字符的标识符"""
        with pytest.raises(ValueError, match="非法"):
            validate_identifier("table-name")
        with pytest.raises(ValueError, match="非法"):
            validate_identifier("table.name")
        with pytest.raises(ValueError, match="非法"):
            validate_identifier("table name")

    def test_too_long_identifier(self):
        """过长的标识符"""
        long_name = "a" * 100
        with pytest.raises(ValueError, match="过长"):
            validate_identifier(long_name)

    def test_max_length_identifier(self):
        """最大长度的标识符"""
        max_name = "a" * 63
        assert validate_identifier(max_name) == max_name


class TestSanitizeErrorMessage:
    """测试 sanitize_error_message 函数"""

    def test_normal_error(self):
        """普通错误信息"""
        error = "数据库连接失败"
        assert sanitize_error_message(error) == error

    def test_control_chars_removed(self):
        """移除控制字符"""
        error = "错误\x00\x01\x02信息"
        result = sanitize_error_message(error)
        assert "\x00" not in result
        assert "\x01" not in result
        assert result == "错误信息"

    def test_truncation(self):
        """截断过长信息"""
        error = "错误信息" * 200  # 超过 500 字符
        result = sanitize_error_message(error)
        assert len(result) == 500

    def test_empty_error(self):
        """空错误信息"""
        assert sanitize_error_message("") == ""
        assert sanitize_error_message(None) == ""


# ==================== ProcessContext 测试 ====================

class TestProcessContext:
    """测试 ProcessContext 类"""

    def test_initialization(self):
        """初始化"""
        ctx = ProcessContext(row={"id": 1, "name": "test"})
        assert ctx.row["id"] == 1
        assert ctx.success is True
        assert ctx.error == ""
        assert ctx.skip is False

    def test_set_and_get_result(self):
        """设置和获取结果"""
        ctx = ProcessContext(row={})
        ctx.set_result("key1", "value1")
        assert ctx.get_result("key1") == "value1"
        assert ctx.get_result("not_exist") is None
        assert ctx.get_result("not_exist", "default") == "default"

    def test_fail(self):
        """标记失败"""
        ctx = ProcessContext(row={})
        ctx.fail("出错了")
        assert ctx.success is False
        assert ctx.error == "出错了"

    def test_skip_rest(self):
        """跳过后续"""
        ctx = ProcessContext(row={})
        ctx.skip_rest()
        assert ctx.skip is True


# ==================== Pipeline 测试 ====================

class TestPipeline:
    """测试 Pipeline 类"""

    def test_empty_pipeline(self):
        """空管道"""
        pipeline = Pipeline()
        ctx = pipeline.run({"id": 1})
        assert ctx.success is True

    def test_single_step(self):
        """单个步骤"""
        class AddFieldStep(ProcessStep):
            def process(self, ctx):
                ctx.set_result("added", True)

        pipeline = Pipeline()
        pipeline.add_step(AddFieldStep())
        ctx = pipeline.run({"id": 1})
        assert ctx.get_result("added") is True

    def test_multiple_steps(self):
        """多个步骤"""
        class Step1(ProcessStep):
            def process(self, ctx):
                ctx.set_result("step1", 1)

        class Step2(ProcessStep):
            def process(self, ctx):
                ctx.set_result("step2", ctx.get_result("step1") + 1)

        pipeline = Pipeline()
        pipeline.add_step(Step1())
        pipeline.add_step(Step2())
        ctx = pipeline.run({"id": 1})
        assert ctx.get_result("step1") == 1
        assert ctx.get_result("step2") == 2

    def test_step_failure_stops_pipeline(self):
        """步骤失败停止管道"""
        class FailStep(ProcessStep):
            def process(self, ctx):
                ctx.fail("失败了")

        class NeverRunStep(ProcessStep):
            def process(self, ctx):
                ctx.set_result("never", True)

        pipeline = Pipeline()
        pipeline.add_step(FailStep())
        pipeline.add_step(NeverRunStep())
        ctx = pipeline.run({"id": 1})
        assert ctx.success is False
        assert ctx.get_result("never") is None

    def test_skip_stops_pipeline(self):
        """跳过停止管道"""
        class SkipStep(ProcessStep):
            def process(self, ctx):
                ctx.skip_rest()

        class NeverRunStep(ProcessStep):
            def process(self, ctx):
                ctx.set_result("never", True)

        pipeline = Pipeline()
        pipeline.add_step(SkipStep())
        pipeline.add_step(NeverRunStep())
        ctx = pipeline.run({"id": 1})
        assert ctx.skip is True
        assert ctx.get_result("never") is None

    def test_filter_step(self):
        """过滤步骤"""
        class OnlyEvenStep(ProcessStep):
            def should_process(self, ctx):
                return ctx.row.get("id", 0) % 2 == 0

            def process(self, ctx):
                ctx.set_result("even", True)

        pipeline = Pipeline()
        pipeline.add_step(OnlyEvenStep())

        ctx1 = pipeline.run({"id": 1})
        assert ctx1.get_result("even") is None

        ctx2 = pipeline.run({"id": 2})
        assert ctx2.get_result("even") is True

    def test_run_batch(self):
        """批量运行"""
        class CountStep(ProcessStep):
            def process(self, ctx):
                ctx.set_result("id", ctx.row["id"])

        pipeline = Pipeline()
        pipeline.add_step(CountStep())
        contexts = pipeline.run_batch([{"id": 1}, {"id": 2}, {"id": 3}])
        assert len(contexts) == 3
        assert [c.get_result("id") for c in contexts] == [1, 2, 3]


# ==================== 内置步骤测试 ====================

class TestValidateStep:
    """测试 ValidateStep"""

    def test_validation_pass(self):
        """验证通过"""
        step = ValidateStep(lambda r: r.get("value", 0) > 0)
        ctx = ProcessContext(row={"value": 10})
        step.process(ctx)
        assert ctx.success is True

    def test_validation_fail(self):
        """验证失败"""
        step = ValidateStep(lambda r: r.get("value", 0) > 0, "值必须大于0")
        ctx = ProcessContext(row={"value": -1})
        step.process(ctx)
        assert ctx.success is False
        assert ctx.error == "值必须大于0"
        assert ctx.skip is True


class TestTransformStep:
    """测试 TransformStep"""

    def test_transform_success(self):
        """转换成功"""
        step = TransformStep(lambda r: {"upper": r["name"].upper()})
        ctx = ProcessContext(row={"name": "test"})
        step.process(ctx)
        assert ctx.get_result("transformed") == {"upper": "TEST"}

    def test_transform_with_custom_key(self):
        """自定义结果键"""
        step = TransformStep(lambda r: r["value"] * 2, result_key="doubled")
        ctx = ProcessContext(row={"value": 5})
        step.process(ctx)
        assert ctx.get_result("doubled") == 10

    def test_transform_exception(self):
        """转换异常"""
        step = TransformStep(lambda r: r["missing_key"])
        ctx = ProcessContext(row={})
        step.process(ctx)
        assert ctx.success is False


class TestFilterStep:
    """测试 FilterStep"""

    def test_filter_true(self):
        """过滤条件为真"""
        step = FilterStep(lambda r: r.get("active", False))
        ctx = ProcessContext(row={"active": True})
        assert step.should_process(ctx) is True

    def test_filter_false(self):
        """过滤条件为假"""
        step = FilterStep(lambda r: r.get("active", False))
        ctx = ProcessContext(row={"active": False})
        assert step.should_process(ctx) is False


class TestSkipStep:
    """测试 SkipStep"""

    def test_skip_condition_true(self):
        """跳过条件为真"""
        step = SkipStep(lambda r: r.get("skip_me", False))
        ctx = ProcessContext(row={"skip_me": True})
        step.process(ctx)
        assert ctx.skip is True

    def test_skip_condition_false(self):
        """跳过条件为假"""
        step = SkipStep(lambda r: r.get("skip_me", False))
        ctx = ProcessContext(row={"skip_me": False})
        step.process(ctx)
        assert ctx.skip is False


class TestLogStep:
    """测试 LogStep"""

    def test_log_with_pk_column(self):
        """自定义主键列"""
        step = LogStep("处理中", pk_column="uuid")
        ctx = ProcessContext(row={"uuid": "abc123"})
        step.process(ctx)
        # 无异常即为成功

    def test_log_include_row(self):
        """包含行信息"""
        step = LogStep("测试", include_row=True, pk_column="id")
        ctx = ProcessContext(row={"id": 42})
        step.process(ctx)
        # 无异常即为成功


class TestCallbackStep:
    """测试 CallbackStep"""

    def test_callback_execution(self):
        """回调执行"""
        called = []

        def my_callback(ctx):
            called.append(ctx.row["id"])

        step = CallbackStep(my_callback)
        ctx = ProcessContext(row={"id": 1})
        step.process(ctx)
        assert called == [1]


class TestRecordToTextStep:
    """测试 RecordToTextStep"""

    def test_line_format(self):
        """行格式"""
        step = RecordToTextStep(format_style="line")
        ctx = ProcessContext(row={"name": "张三", "age": 25})
        step.process(ctx)
        text = ctx.get_result("text")
        assert "name: 张三" in text
        assert "age: 25" in text

    def test_compact_format(self):
        """紧凑格式"""
        step = RecordToTextStep(format_style="compact")
        ctx = ProcessContext(row={"name": "张三", "age": 25})
        step.process(ctx)
        text = ctx.get_result("text")
        assert text == "{name: 张三, age: 25}"

    def test_specific_fields(self):
        """指定字段"""
        step = RecordToTextStep(fields=["name"])
        ctx = ProcessContext(row={"name": "张三", "age": 25, "secret": "hidden"})
        step.process(ctx)
        text = ctx.get_result("text")
        assert "name: 张三" in text
        assert "secret" not in text

    def test_exclude_fields(self):
        """排除字段"""
        step = RecordToTextStep(exclude_fields=["secret"])
        ctx = ProcessContext(row={"name": "张三", "secret": "hidden"})
        step.process(ctx)
        text = ctx.get_result("text")
        assert "secret" not in text


class TestBatchTextCollector:
    """测试 BatchTextCollector"""

    def test_collect_single_text(self):
        """收集单个文本"""
        contexts = [
            ProcessContext(row={"id": 1}),
            ProcessContext(row={"id": 2}),
        ]
        contexts[0].set_result("text", "文本1")
        contexts[1].set_result("text", "文本2")

        texts = BatchTextCollector.collect(contexts)
        assert texts == ["文本1", "文本2"]

    def test_collect_list_text(self):
        """收集列表文本"""
        contexts = [ProcessContext(row={"id": 1})]
        contexts[0].set_result("_texts", ["文本1", "文本2"])

        texts = BatchTextCollector.collect(contexts)
        assert texts == ["文本1", "文本2"]

    def test_collect_failed_context(self):
        """跳过失败的上下文"""
        contexts = [
            ProcessContext(row={"id": 1}),
            ProcessContext(row={"id": 2}),
        ]
        contexts[0].set_result("text", "成功")
        contexts[1].fail("失败")

        texts = BatchTextCollector.collect(contexts)
        assert texts == ["成功"]

    def test_merge(self):
        """合并文本"""
        texts = ["文本1", "文本2", "文本3"]
        result = BatchTextCollector.merge(texts, separator="\n---\n")
        assert result == "文本1\n---\n文本2\n---\n文本3"


# ==================== 异步测试 ====================

class TestAsyncProcessStep:
    """测试异步处理步骤"""

    @pytest.mark.anyio
    async def test_async_step(self):
        """异步步骤"""
        class MyAsyncStep(AsyncProcessStep):
            async def async_process(self, ctx):
                ctx.set_result("async", True)

        step = MyAsyncStep()
        ctx = ProcessContext(row={})
        await step.async_process(ctx)
        assert ctx.get_result("async") is True

    @pytest.mark.anyio
    async def test_pipeline_run_batch_async(self):
        """管道异步批量运行"""
        class AsyncAddStep(AsyncProcessStep):
            async def async_process(self, ctx):
                ctx.set_result("async_id", ctx.row["id"])

        pipeline = Pipeline()
        pipeline.add_step(AsyncAddStep())
        contexts = await pipeline.run_batch_async([{"id": 1}, {"id": 2}])
        assert len(contexts) == 2
        assert contexts[0].get_result("async_id") == 1
        assert contexts[1].get_result("async_id") == 2