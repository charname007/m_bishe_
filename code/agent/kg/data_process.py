"""
可插拔数据处理框架
支持管道式处理流程、失败重试、并发处理、超时检测

设计：
- 单表 + 状态字段（无数据搬运）
- FOR UPDATE SKIP LOCKED（并发安全）
- 管道式处理步骤（可插拔）

安全：
- 使用 psycopg2.sql.Identifier 安全引用标识符
- 错误信息 sanitization
- 事务边界保证

注意事项：
- 处理函数应设计为幂等的，因为进程崩溃后可能重复处理
- 外部副作用（写文件、调用API）需要自行保证幂等性
"""
import re
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass, field

from loguru import logger
from psycopg2 import sql


# ==================== 安全验证工具 ====================

def validate_identifier(identifier: str, max_length: int = 63) -> str:
    """
    验证标识符合法性（表名、列名等）

    Args:
        identifier: 标识符
        max_length: 最大长度（PostgreSQL 默认 63）

    Returns:
        验证后的标识符

    Raises:
        ValueError: 标识符非法
    """
    if not identifier:
        raise ValueError("标识符不能为空")

    # 只允许字母、数字、下划线，且不能以数字开头
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
        raise ValueError(
            f"标识符 '{identifier}' 非法，只允许字母、数字、下划线，且不能以数字开头"
        )

    # 限制长度
    if len(identifier) > max_length:
        raise ValueError(f"标识符 '{identifier}' 过长，最大 {max_length} 字符")

    return identifier


def sanitize_error_message(error: str, max_length: int = 500) -> str:
    """
    清理错误信息，移除危险字符

    Args:
        error: 原始错误信息
        max_length: 最大长度

    Returns:
        清理后的错误信息
    """
    if not error:
        return ""

    # 移除控制字符和潜在的敏感信息标记
    safe_error = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', error)

    # 截断到指定长度
    return safe_error[:max_length]


# ==================== 核心数据结构 ====================

@dataclass
class ProcessContext:
    """
    处理上下文，贯穿整个处理流程
    每条数据对应一个上下文对象
    """
    row: Dict[str, Any]                          # 原始数据行
    result: Dict[str, Any] = field(default_factory=dict)  # 处理结果
    success: bool = True                         # 是否成功
    error: str = ""                              # 错误信息
    skip: bool = False                           # 是否跳过后续步骤

    def set_result(self, key: str, value: Any) -> None:
        """设置处理结果"""
        self.result[key] = value

    def get_result(self, key: str, default: Any = None) -> Any:
        """获取处理结果"""
        return self.result.get(key, default)

    def fail(self, error: str) -> None:
        """标记处理失败"""
        self.success = False
        self.error = error

    def skip_rest(self) -> None:
        """跳过后续处理步骤"""
        self.skip = True


# ==================== 处理步骤基类 ====================

class ProcessStep(ABC):
    """
    处理步骤基类
    自定义步骤继承此类，实现 process 方法
    """

    @property
    def name(self) -> str:
        """步骤名称"""
        return self.__class__.__name__

    @abstractmethod
    def process(self, ctx: ProcessContext) -> None:
        """
        处理单条数据

        Args:
            ctx: 处理上下文
        """
        pass

    def should_process(self, ctx: ProcessContext) -> bool:
        """
        是否需要处理此步骤（可用于过滤）

        Args:
            ctx: 处理上下文

        Returns:
            True 则执行 process，False 则跳过此步骤
        """
        return True


class AsyncProcessStep(ProcessStep):
    """
    异步处理步骤基类
    支持异步处理逻辑
    """

    @abstractmethod
    async def async_process(self, ctx: ProcessContext) -> None:
        """
        异步处理单条数据

        Args:
            ctx: 处理上下文
        """
        pass

    def process(self, ctx: ProcessContext) -> None:
        """同步处理（异步步骤默认不实现）"""
        raise NotImplementedError("请使用 async_process")


# ==================== 处理管道 ====================

class Pipeline:
    """
    处理管道
    管理一系列处理步骤，顺序执行
    """

    def __init__(self, steps: Optional[List[ProcessStep]] = None):
        self.steps: List[ProcessStep] = steps or []

    def add_step(self, step: ProcessStep) -> "Pipeline":
        """
        添加处理步骤

        Args:
            step: 处理步骤实例

        Returns:
            返回自身，支持链式调用
        """
        self.steps.append(step)
        return self

    def run(self, row: Dict[str, Any]) -> ProcessContext:
        """
        运行管道处理单条数据

        Args:
            row: 数据行

        Returns:
            处理上下文（包含结果和状态）
        """
        ctx = ProcessContext(row=row)

        for step in self.steps:
            # 检查是否跳过后续
            if ctx.skip:
                break

            # 检查是否需要处理此步骤
            if not step.should_process(ctx):
                continue

            # 检查是否已经失败
            if not ctx.success:
                break

            try:
                step.process(ctx)
            except Exception as e:
                ctx.fail(str(e))
                logger.error(f"步骤 [{step.name}] 处理失败: {e}")
                break

        return ctx

    def run_batch(self, rows: List[Dict[str, Any]]) -> List[ProcessContext]:
        """
        批量运行管道

        Args:
            rows: 数据行列表

        Returns:
            处理上下文列表
        """
        return [self.run(row) for row in rows]

    async def run_batch_async(self, rows: List[Dict[str, Any]]) -> List[ProcessContext]:
        """
        异步批量运行管道

        Args:
            rows: 数据行列表

        Returns:
            处理上下文列表
        """
        contexts = []
        for row in rows:
            ctx = ProcessContext(row=row)

            for step in self.steps:
                if ctx.skip or not ctx.success:
                    break

                if not step.should_process(ctx):
                    continue

                try:
                    # 支持异步步骤
                    if isinstance(step, AsyncProcessStep):
                        await step.async_process(ctx)
                    else:
                        step.process(ctx)
                except Exception as e:
                    ctx.fail(str(e))
                    logger.error(f"步骤 [{step.name}] 处理失败: {e}")
                    break

            contexts.append(ctx)

        return contexts


# ==================== 数据处理引擎 ====================

class DataProcessor:
    """
    数据处理引擎（同步）

    状态字段设计：
    - process_status: pending/processing/processed/failed
    - process_retry_count: 重试次数
    - process_error: 错误信息
    - processed_at: 处理完成时间
    - locked_at: 锁定时间（用于超时检测）
    - locked_by: 锁定者标识（worker_id）

    安全特性：
    - 使用 psycopg2.sql.Identifier 安全引用标识符
    - 事务边界保证 FOR UPDATE SKIP LOCKED 的原子性
    - 主键字段名可配置

    注意事项：
    - 处理函数应设计为幂等的，因为进程崩溃后可能重复处理
    - 外部副作用（写文件、调用API）需要自行保证幂等性
    """

    # 状态常量
    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_PROCESSED = "processed"
    STATUS_FAILED = "failed"

    def __init__(
        self,
        pg_client,
        table_name: str,
        batch_size: int = 100,
        max_retries: int = 3,
        timeout_minutes: int = 10,
        pk_column: str = "id"
    ):
        """
        初始化数据处理引擎

        Args:
            pg_client: PostgresClient 实例
            table_name: 要处理的表名
            batch_size: 单次处理数量
            max_retries: 最大重试次数
            timeout_minutes: 超时时间（分钟）
            pk_column: 主键字段名，默认为 "id"

        Raises:
            ValueError: 表名或主键名非法
        """
        self.pg = pg_client
        self.table_name = validate_identifier(table_name)  # 安全验证
        self.pk_column = validate_identifier(pk_column)    # 主键字段名验证
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout_minutes = timeout_minutes
        self.pipeline: Optional[Pipeline] = None

        # 预编译 SQL 标识符，避免重复创建
        self._table_id = sql.Identifier(self.table_name)
        self._pk_id = sql.Identifier(self.pk_column)
        self._status_id = sql.Identifier("process_status")
        self._retry_count_id = sql.Identifier("process_retry_count")
        self._error_id = sql.Identifier("process_error")
        self._processed_at_id = sql.Identifier("processed_at")
        self._locked_at_id = sql.Identifier("locked_at")
        self._locked_by_id = sql.Identifier("locked_by")

        # 确保表有状态字段
        self._ensure_status_columns()

    def set_pipeline(self, pipeline: Pipeline) -> "DataProcessor":
        """
        设置处理管道

        Args:
            pipeline: Pipeline 实例

        Returns:
            返回自身，支持链式调用
        """
        self.pipeline = pipeline
        return self

    def _ensure_status_columns(self) -> None:
        """
        确保表有处理状态相关字段
        自动添加缺失的字段和索引
        使用 psycopg2.sql 安全构建 SQL
        """
        columns_to_add = {
            'process_status': "VARCHAR(20) DEFAULT 'pending'",
            'process_retry_count': "INTEGER DEFAULT 0",
            'process_error': "TEXT",
            'processed_at': "TIMESTAMP",
            'locked_at': "TIMESTAMP",
            'locked_by': "VARCHAR(50)"
        }

        with self.pg.conn.cursor() as cur:
            # 检查现有字段
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = %s AND table_schema = CURRENT_SCHEMA()
                AND column_name IN %s
            """, (self.table_name, tuple(columns_to_add.keys())))
            existing = {row[0] for row in cur.fetchall()}

            # 添加缺失字段 - 使用安全标识符
            for col, definition in columns_to_add.items():
                if col not in existing:
                    col_id = sql.Identifier(col)
                    query = sql.SQL("ALTER TABLE {} ADD COLUMN {} {}").format(
                        self._table_id,
                        col_id,
                        sql.SQL(definition)
                    )
                    cur.execute(query)
                    logger.debug(f"表 {self.table_name} 添加字段 {col}")

            # 创建索引 - 使用安全标识符
            # PostgreSQL 索引名最大 63 字符，需要截断处理
            idx_base = f"idx_{self.table_name[:50]}_status"
            query = sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}({})").format(
                sql.Identifier(idx_base),
                self._table_id,
                self._status_id
            )
            cur.execute(query)

            # 创建复合索引优化查询性能
            # 1. (process_status, process_retry_count) 用于失败重试查询
            idx_retry = f"idx_{self.table_name[:45]}_retry"
            query = sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}({}, {})").format(
                sql.Identifier(idx_retry),
                self._table_id,
                self._status_id,
                self._retry_count_id
            )
            cur.execute(query)

            # 2. (process_status, locked_at) 用于超时检测查询
            idx_timeout = f"idx_{self.table_name[:45]}_timeout"
            query = sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}({}, {})").format(
                sql.Identifier(idx_timeout),
                self._table_id,
                self._status_id,
                self._locked_at_id
            )
            cur.execute(query)

            self.pg.conn.commit()
            logger.info(f"表 {self.table_name} 状态字段检查完成")

    def fetch_pending(
        self,
        limit: Optional[int] = None,
        worker_id: Optional[str] = None
    ) -> List[Dict]:
        """
        获取待处理数据

        使用 FOR UPDATE SKIP LOCKED 保证并发安全：
        - 多个 worker 可以并发获取数据
        - 同一条数据只会被一个 worker 获取
        - 超时的 processing 状态可重新处理（受 max_retries 限制）

        事务边界：
        - SELECT ... FOR UPDATE SKIP LOCKED 和 UPDATE 必须在同一事务中
        - 锁在事务结束时释放，确保原子性

        Args:
            limit: 获取数量，默认使用 batch_size
            worker_id: worker 标识，用于锁定追踪

        Returns:
            待处理数据列表
        """
        limit = limit or self.batch_size
        worker_id = worker_id or f"worker_{datetime.now().strftime('%H%M%S%f')}"

        # 使用显式事务确保 SELECT 和 UPDATE 的原子性
        # 注意：psycopg2 默认 autocommit=False，事务在第一个 SQL 执行时开始
        with self.pg.conn.cursor() as cur:
            # 构建安全的 SQL 查询
            # 超时记录也受 max_retries 限制，避免绕过重试次数
            query = sql.SQL("""
                SELECT * FROM {}
                WHERE {} = %s
                   OR ({} = %s AND {} < %s)
                   OR ({} = %s AND {} < NOW() - INTERVAL '%s minutes' AND {} < %s)
                ORDER BY {}
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            """).format(
                self._table_id,
                self._status_id,
                self._status_id, self._retry_count_id,
                self._status_id, self._locked_at_id, self._retry_count_id,
                self._pk_id
            )

            cur.execute(query, (
                self.STATUS_PENDING,
                self.STATUS_FAILED, self.max_retries,
                self.STATUS_PROCESSING, self.timeout_minutes, self.max_retries,  # 超时也受限制
                limit
            ))

            columns = [desc[0] for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]

            if rows:
                ids = [row[self.pk_column] for row in rows]
                # 标记为 processing 并记录锁定信息 - 同一事务内
                update_query = sql.SQL("""
                    UPDATE {}
                    SET {} = %s,
                        {} = %s,
                        {} = %s
                    WHERE {} = ANY(%s)
                """).format(
                    self._table_id,
                    self._status_id,
                    self._locked_at_id,
                    self._locked_by_id,
                    self._pk_id
                )
                cur.execute(update_query, (self.STATUS_PROCESSING, datetime.now(), worker_id, ids))
                self.pg.conn.commit()  # 提交事务，释放锁

            logger.debug(f"[{worker_id}] 获取 {len(rows)} 条待处理数据")
            return rows

    def _process_batch(self, rows: List[Dict]) -> tuple:
        """
        处理一批数据

        Args:
            rows: 数据列表

        Returns:
            (contexts, success_ids, failed_items)
        """
        if not self.pipeline:
            raise ValueError("未设置处理管道，请调用 set_pipeline()")

        contexts = self.pipeline.run_batch(rows)

        success_ids = [ctx.row[self.pk_column] for ctx in contexts if ctx.success]
        failed_items = [(ctx.row[self.pk_column], ctx.error) for ctx in contexts if not ctx.success]

        return contexts, success_ids, failed_items

    def _mark_processed(self, ids: List[Any]) -> None:
        """标记为已处理"""
        if not ids:
            return

        with self.pg.conn.cursor() as cur:
            query = sql.SQL("""
                UPDATE {}
                SET {} = %s,
                    {} = %s,
                    {} = NULL,
                    {} = NULL
                WHERE {} = ANY(%s)
            """).format(
                self._table_id,
                self._status_id,
                self._processed_at_id,
                self._locked_at_id,
                self._locked_by_id,
                self._pk_id
            )
            cur.execute(query, (self.STATUS_PROCESSED, datetime.now(), ids))
            self.pg.conn.commit()
            logger.debug(f"标记 {len(ids)} 条数据为已处理")

    def _mark_failed(self, items: List[tuple]) -> None:
        """
        标记为失败（批量更新优化）

        Args:
            items: [(id, error_msg), ...]
        """
        if not items:
            return

        with self.pg.conn.cursor() as cur:
            # 使用批量 UPDATE 优化
            data = [
                (self.STATUS_FAILED, sanitize_error_message(error), id_)
                for id_, error in items
            ]

            # 使用安全的 SQL 标识符
            query = sql.SQL("""
                UPDATE {}
                SET {} = %s,
                    {} = {} + 1,
                    {} = %s,
                    {} = NULL,
                    {} = NULL
                WHERE {} = %s
            """).format(
                self._table_id,
                self._status_id,
                self._retry_count_id, self._retry_count_id,
                self._error_id,
                self._locked_at_id,
                self._locked_by_id,
                self._pk_id
            )

            # 使用 executemany 批量更新
            cur.executemany(query, data)

            self.pg.conn.commit()
            logger.warning(f"标记 {len(items)} 条数据为失败")

    def run(self, worker_id: Optional[str] = None) -> Dict[str, int]:
        """
        运行处理（处理所有待处理数据）

        Args:
            worker_id: worker 标识

        Returns:
            {"success": count, "failed": count}
        """
        if not self.pipeline:
            raise ValueError("未设置处理管道")

        worker_id = worker_id or "main_worker"
        total_success = 0
        total_failed = 0

        while True:
            rows = self.fetch_pending(worker_id=worker_id)
            if not rows:
                break

            contexts, success_ids, failed_items = self._process_batch(rows)

            self._mark_processed(success_ids)
            self._mark_failed(failed_items)

            total_success += len(success_ids)
            total_failed += len(failed_items)

            logger.info(
                f"[{worker_id}] 批次完成: 成功 {len(success_ids)}, 失败 {len(failed_items)}"
            )

        logger.info(f"[{worker_id}] 处理完成: 成功 {total_success}, 失败 {total_failed}")
        return {"success": total_success, "failed": total_failed}

    def get_stats(self) -> Dict[str, int]:
        """
        获取处理统计

        Returns:
            各状态的数据数量
        """
        with self.pg.conn.cursor() as cur:
            query = sql.SQL("""
                SELECT {}, COUNT(*) FROM {}
                GROUP BY {}
            """).format(self._status_id, self._table_id, self._status_id)
            cur.execute(query)
            stats = {row[0]: row[1] for row in cur.fetchall()}

            # 检查超时的锁定
            query = sql.SQL("""
                SELECT COUNT(*) FROM {}
                WHERE {} = %s
                AND {} < NOW() - INTERVAL '%s minutes'
            """).format(self._table_id, self._status_id, self._locked_at_id)
            cur.execute(query, (self.STATUS_PROCESSING, self.timeout_minutes))
            timed_out = cur.fetchone()[0]

            return {
                "pending": stats.get(self.STATUS_PENDING, 0),
                "processing": stats.get(self.STATUS_PROCESSING, 0),
                "processed": stats.get(self.STATUS_PROCESSED, 0),
                "failed": stats.get(self.STATUS_FAILED, 0),
                "timed_out": timed_out
            }

    def reset_failed(self, max_attempts: Optional[int] = None) -> int:
        """
        重置失败状态，允许重新处理

        Args:
            max_attempts: 只重置重试次数小于此值的数据

        Returns:
            重置的数据条数
        """
        with self.pg.conn.cursor() as cur:
            if max_attempts:
                query = sql.SQL("""
                    UPDATE {}
                    SET {} = %s, {} = 0, {} = NULL
                    WHERE {} = %s AND {} < %s
                """).format(
                    self._table_id,
                    self._status_id, self._retry_count_id, self._error_id,
                    self._status_id, self._retry_count_id
                )
                cur.execute(query, (self.STATUS_PENDING, self.STATUS_FAILED, max_attempts))
            else:
                query = sql.SQL("""
                    UPDATE {}
                    SET {} = %s, {} = 0, {} = NULL
                    WHERE {} = %s
                """).format(
                    self._table_id,
                    self._status_id, self._retry_count_id, self._error_id,
                    self._status_id
                )
                cur.execute(query, (self.STATUS_PENDING, self.STATUS_FAILED))

            count = cur.rowcount
            self.pg.conn.commit()
            logger.info(f"重置 {count} 条失败数据为待处理状态")
            return count


class AsyncDataProcessor:
    """
    异步数据处理引擎
    支持并发处理
    """

    def __init__(
        self,
        pg_client,
        table_name: str,
        batch_size: int = 100,
        max_retries: int = 3,
        timeout_minutes: int = 10,
        pk_column: str = "id"
    ):
        self.processor = DataProcessor(
            pg_client, table_name, batch_size, max_retries, timeout_minutes, pk_column
        )

    def set_pipeline(self, pipeline: Pipeline) -> "AsyncDataProcessor":
        """设置处理管道"""
        self.processor.set_pipeline(pipeline)
        return self

    async def _run_sync(self, func: Callable, *args) -> Any:
        """在线程池运行同步函数"""
        # Python 3.9+ 推荐使用 to_thread
        return await asyncio.to_thread(func, *args)

    async def run(self) -> Dict[str, int]:
        """异步运行处理"""
        return await self._run_sync(self.processor.run)

    async def run_concurrent(self, workers: int = 3) -> Dict[str, int]:
        """
        并发处理

        Args:
            workers: 并发 worker 数量

        Returns:
            {"success": count, "failed": count}
        """
        if not self.processor.pipeline:
            raise ValueError("未设置处理管道")

        # 使用线程安全的列表收集结果
        worker_results: List[Dict[str, int]] = []
        active_workers = set()

        async def worker(worker_idx: int):
            worker_id = f"async_worker_{worker_idx}"
            active_workers.add(worker_id)

            # 本地计数器，避免竞态条件
            local_success = 0
            local_failed = 0

            try:
                while True:
                    # 获取待处理数据
                    rows = await self._run_sync(
                        self.processor.fetch_pending,
                        None, worker_id
                    )

                    if not rows:
                        break

                    # 异步运行管道
                    contexts = await self.processor.pipeline.run_batch_async(rows)

                    # 分类结果
                    pk_col = self.processor.pk_column
                    success_ids = [ctx.row[pk_col] for ctx in contexts if ctx.success]
                    failed_items = [(ctx.row[pk_col], ctx.error) for ctx in contexts if not ctx.success]

                    # 更新状态
                    await self._run_sync(self.processor._mark_processed, success_ids)
                    await self._run_sync(self.processor._mark_failed, failed_items)

                    # 更新本地计数器
                    local_success += len(success_ids)
                    local_failed += len(failed_items)

                    logger.debug(
                        f"[{worker_id}] 批次完成: 成功 {len(success_ids)}, 失败 {len(failed_items)}"
                    )
            finally:
                active_workers.discard(worker_id)
                # 返回本地计数结果
                worker_results.append({
                    "success": local_success,
                    "failed": local_failed
                })

        # 启动所有 worker
        tasks = [worker(i) for i in range(workers)]
        await asyncio.gather(*tasks)

        # 汇总所有 worker 的结果（无竞态条件）
        total_success = sum(r["success"] for r in worker_results)
        total_failed = sum(r["failed"] for r in worker_results)

        logger.info(f"并发处理完成: 成功 {total_success}, 失败 {total_failed}")
        return {"success": total_success, "failed": total_failed}

    def get_stats(self) -> Dict[str, int]:
        """获取处理统计"""
        return self.processor.get_stats()

    def reset_failed(self, max_attempts: Optional[int] = None) -> int:
        """重置失败状态"""
        return self.processor.reset_failed(max_attempts)


# ==================== 内置处理步骤 ====================

class ValidateStep(ProcessStep):
    """
    数据验证步骤
    验证失败则跳过后续处理
    """

    def __init__(
        self,
        validator: Callable[[Dict], bool],
        error_msg: str = "数据验证失败"
    ):
        """
        Args:
            validator: 验证函数，返回 True 表示通过
            error_msg: 验证失败时的错误信息
        """
        self.validator = validator
        self.error_msg = error_msg

    def process(self, ctx: ProcessContext) -> None:
        if not self.validator(ctx.row):
            ctx.fail(self.error_msg)
            ctx.skip_rest()


class TransformStep(ProcessStep):
    """
    数据转换步骤
    将转换结果存入 context
    """

    def __init__(
        self,
        transformer: Callable[[Dict], Any],
        result_key: str = "transformed"
    ):
        """
        Args:
            transformer: 转换函数
            result_key: 结果存储的键名
        """
        self.transformer = transformer
        self.result_key = result_key

    def process(self, ctx: ProcessContext) -> None:
        try:
            result = self.transformer(ctx.row)
            ctx.set_result(self.result_key, result)
        except Exception as e:
            ctx.fail(str(e))


class FilterStep(ProcessStep):
    """
    数据过滤步骤
    不满足条件的数据跳过此步骤（但不影响后续步骤）
    """

    def __init__(self, predicate: Callable[[Dict], bool]):
        """
        Args:
            predicate: 过滤函数，返回 True 则执行此步骤的后续处理
        """
        self.predicate = predicate

    def should_process(self, ctx: ProcessContext) -> bool:
        return self.predicate(ctx.row)

    def process(self, ctx: ProcessContext) -> None:
        """过滤步骤本身不执行任何操作，仅通过 should_process 控制流程"""
        pass


class SkipStep(ProcessStep):
    """
    跳过步骤
    满足条件的数据跳过后续所有处理
    """

    def __init__(self, condition: Callable[[Dict], bool]):
        """
        Args:
            condition: 条件函数，返回 True 则跳过后续所有处理
        """
        self.condition = condition

    def process(self, ctx: ProcessContext) -> None:
        if self.condition(ctx.row):
            ctx.skip_rest()


class LogStep(ProcessStep):
    """
    日志记录步骤
    """

    def __init__(
        self,
        message: str = "",
        level: str = "info",
        include_row: bool = False,
        pk_column: str = "id"
    ):
        """
        Args:
            message: 日志消息
            level: 日志级别 (debug/info/warning/error)
            include_row: 是否包含数据行信息
            pk_column: 主键字段名，默认 "id"
        """
        self.message = message
        self.level = level
        self.include_row = include_row
        self.pk_column = pk_column

    def process(self, ctx: ProcessContext) -> None:
        if self.include_row:
            pk_value = ctx.row.get(self.pk_column, "unknown")
            msg = f"{self.message} | {self.pk_column}={pk_value}"
        else:
            pk_value = ctx.row.get(self.pk_column, "unknown")
            msg = self.message or f"处理数据 {self.pk_column}={pk_value}"

        # 使用 loguru 的方法而非 logger.log()
        level_map = {
            "debug": logger.debug,
            "info": logger.info,
            "warning": logger.warning,
            "error": logger.error,
            "critical": logger.critical,
        }
        log_func = level_map.get(self.level.lower(), logger.info)
        log_func(msg)


class CallbackStep(ProcessStep):
    """
    回调步骤
    插入任意处理函数
    """

    def __init__(
        self,
        callback: Callable[[ProcessContext], None],
        name: Optional[str] = None
    ):
        """
        Args:
            callback: 回调函数，接收 ProcessContext
            name: 步骤名称
        """
        self.callback = callback
        self._name = name or callback.__name__

    @property
    def name(self) -> str:
        return self._name

    def process(self, ctx: ProcessContext) -> None:
        self.callback(ctx)


class AsyncCallbackStep(AsyncProcessStep):
    """
    异步回调步骤
    """

    def __init__(
        self,
        callback: Callable[[ProcessContext], Any],
        name: Optional[str] = None
    ):
        self.callback = callback
        self._name = name or callback.__name__

    @property
    def name(self) -> str:
        return self._name

    async def async_process(self, ctx: ProcessContext) -> None:
        await self.callback(ctx)


class RecordToTextStep(ProcessStep):
    """
    记录转文本步骤
    将单条记录转换为 {字段: 内容} 文本格式
    """

    def __init__(
        self,
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        result_key: str = "text",
        format_style: str = "line"
    ):
        """
        Args:
            fields: 要转换的字段列表，None 则转换所有字段
            exclude_fields: 要排除的字段列表（如 process_status 等状态字段）
            result_key: 结果存储的键名
            format_style: 格式风格
                - "line": 每个字段一行 "字段: 内容"
                - "compact": 紧凑格式 "{字段: 内容, 字段2: 内容2}"
        """
        self.fields = fields
        self.exclude_fields = exclude_fields or [
            'process_status', 'process_retry_count', 'process_error',
            'processed_at', 'locked_at', 'locked_by'
        ]
        self.result_key = result_key
        self.format_style = format_style

    def _format_value(self, value: Any) -> str:
        """格式化值"""
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        if isinstance(value, dict):
            return str(value)
        return str(value)

    def _record_to_text(self, row: Dict) -> str:
        """单条记录转文本"""
        # 确定要转换的字段
        if self.fields:
            target_fields = self.fields
        else:
            target_fields = [k for k in row.keys() if k not in self.exclude_fields]

        if self.format_style == "compact":
            # 紧凑格式: {字段: 内容, 字段2: 内容2}
            parts = []
            for field in target_fields:
                if field in row:
                    value = self._format_value(row[field])
                    parts.append(f"{field}: {value}")
            return "{" + ", ".join(parts) + "}"
        else:
            # 行格式: 每个字段一行
            lines = []
            for field in target_fields:
                if field in row:
                    value = self._format_value(row[field])
                    lines.append(f"{field}: {value}")
            return "\n".join(lines)

    def process(self, ctx: ProcessContext) -> None:
        try:
            text = self._record_to_text(ctx.row)
            ctx.set_result(self.result_key, text)
        except Exception as e:
            ctx.fail(str(e))


class BatchRecordsToTextStep(ProcessStep):
    """
    批量记录转文本步骤
    将多条记录合并为一个文本列表
    注意：此步骤需要配合 Pipeline 的批量模式使用
    """

    def __init__(
        self,
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        result_key: str = "texts",
        format_style: str = "line",
        separator: str = "\n---\n"
    ):
        """
        Args:
            fields: 要转换的字段列表
            exclude_fields: 要排除的字段列表
            result_key: 结果存储的键名
            format_style: 格式风格 ("line" 或 "compact")
            separator: 多条记录之间的分隔符
        """
        self.converter = RecordToTextStep(
            fields=fields,
            exclude_fields=exclude_fields,
            format_style=format_style
        )
        self.result_key = result_key
        self.separator = separator

    def process(self, ctx: ProcessContext) -> None:
        """处理单条记录，存储文本到临时结果"""
        try:
            text = self.converter._record_to_text(ctx.row)
            # 将文本存入临时列表
            texts = ctx.get_result("_texts", [])
            texts.append(text)
            ctx.set_result("_texts", texts)
        except Exception as e:
            ctx.fail(str(e))


class BatchTextCollector:
    """
    批量文本收集器
    用于收集 Pipeline 处理后的所有文本结果
    """

    @staticmethod
    def collect(contexts: List[ProcessContext]) -> List[str]:
        """
        从多个 context 中收集文本结果

        Args:
            contexts: ProcessContext 列表

        Returns:
            文本列表
        """
        texts = []
        for ctx in contexts:
            if ctx.success:
                text = ctx.get_result("text") or ctx.get_result("_texts", [])
                if isinstance(text, list):
                    texts.extend(text)
                else:
                    texts.append(text)
        return texts

    @staticmethod
    def merge(texts: List[str], separator: str = "\n---\n") -> str:
        """
        将多个文本合并为一个

        Args:
            texts: 文本列表
            separator: 分隔符

        Returns:
            合合后的文本
        """
        return separator.join(texts)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    from agent.kg.postgres_client import PostgresClient

    # 连接数据库
    pg = PostgresClient(
        host="localhost",
        port=5432,
        database="your_db",
        user="your_user",
        password="your_password"
    )

    # ==================== 示例1: 记录转文本 ====================
    # 将每条记录转换为 {字段: 内容} 文本格式

    # 单条记录转文本
    pipeline1 = Pipeline()
    pipeline1.add_step(RecordToTextStep(
        fields=["head_entity", "relation", "tail_entity", "evidence"],
        format_style="line"  # 每个字段一行
    ))

    # 处理并获取文本结果
    processor1 = DataProcessor(pg, "triples", batch_size=50)
    processor1.set_pipeline(pipeline1)

    # 模拟处理数据
    rows = processor1.fetch_pending(limit=10)
    if rows:
        contexts, success_ids, failed = processor1._process_batch(rows)
        # 收集文本结果
        texts = BatchTextCollector.collect(contexts)
        print(f"生成了 {len(texts)} 条文本")
        for i, text in enumerate(texts[:3]):  # 打印前3条
            print(f"\n--- 记录 {i+1} ---")
            print(text)

        # 合并为单个文本
        merged_text = BatchTextCollector.merge(texts, separator="\n\n===\n\n")
        print(f"\n合并文本长度: {len(merged_text)} 字符")

    # ==================== 示例2: 自定义处理 ====================
    class NormalizeEntityStep(ProcessStep):
        """实体名称标准化"""

        def process(self, ctx: ProcessContext) -> None:
            head = ctx.row.get("head_entity", "")
            tail = ctx.row.get("tail_entity", "")

            ctx.set_result("normalized_head", head.strip().upper())
            ctx.set_result("normalized_tail", tail.strip().upper())

    class ValidateTripleStep(ProcessStep):
        """三元组验证"""

        def process(self, ctx: ProcessContext) -> None:
            if not ctx.row.get("head_entity"):
                ctx.fail("缺少 head_entity")
            elif not ctx.row.get("relation"):
                ctx.fail("缺少 relation")
            elif not ctx.row.get("tail_entity"):
                ctx.fail("缺少 tail_entity")

    # 构建处理管道
    pipeline2 = Pipeline()
    pipeline2.add_step(ValidateTripleStep())                    # 验证
    pipeline2.add_step(NormalizeEntityStep())                   # 标准化
    pipeline2.add_step(RecordToTextStep(format_style="compact"))# 转文本（紧凑格式）
    pipeline2.add_step(LogStep("处理完成", level="debug"))      # 日志

    # 创建处理器
    processor2 = DataProcessor(pg, "triples", batch_size=50, max_retries=3)
    processor2.set_pipeline(pipeline2)

    # 查看统计
    print("当前状态:", processor2.get_stats())

    # 运行处理
    result = processor2.run()
    print("处理结果:", result)

    # ==================== 示例3: 异步并发处理 ====================
    # async_processor = AsyncDataProcessor(pg, "triples", batch_size=50)
    # async_processor.set_pipeline(pipeline2)
    # result = await async_processor.run_concurrent(workers=3)

    pg.close()