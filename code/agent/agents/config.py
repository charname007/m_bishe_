"""
知识图谱抽取配置 - 支持运行时配置和环境变量覆盖
P2改进：解决硬编码问题，支持灵活配置
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractionConfig:
    """
    知识图谱抽取的可配置参数

    支持三种方式配置：
    1. 直接初始化：ExtractionConfig(eval_threshold=4.0)
    2. 环境变量：EVAL_THRESHOLD=4.0
    3. 配置文件：通过 load_from_dict() 加载
    """
    # ===== 评估配置 =====
    eval_threshold: float = 3.5
    """评估通过阈值 (1-5分，平均分 >= 此值视为通过)"""

    # ===== 实体去重配置 =====
    similarity_threshold: float = 0.85
    """实体名称相似度阈值 (0-1，用于判断是否为同一实体)"""

    # ===== 分布式处理配置 =====
    corpus_per_worker: int = 10
    """每个 Worker 处理的语料数量"""

    max_workers: int = 10
    """最大 Worker 数量"""

    max_concurrent_corpus: int = 50
    """最大并发处理语料数（Semaphore限制，防止API限流）"""

    batch_size: int = 100
    """每批次从数据库读取的语料数量"""

    # ===== 文本验证配置 =====
    max_text_length: int = 10000
    """最大文本长度（超出会被截断）"""

    min_text_length: int = 1
    """最小文本长度"""

    # ===== 评估模式配置 =====
    use_simplified_eval: bool = True
    """是否使用简化评估（单次评估+规则校验）"""

    # ===== Retry 配置 =====
    retry_initial_interval: float = 1.0
    """Retry 初始等待时间（秒）"""

    retry_backoff_factor: float = 2.0
    """Retry 退避因子"""

    retry_max_interval: float = 30.0
    """Retry 最大等待时间（秒）"""

    retry_max_attempts: int = 3
    """Retry 最大重试次数"""

    # ===== Self-Check + 反思循环配置（P4新增） =====
    enable_self_check: bool = False
    """是否启用 Self-Check + 反思循环模式"""

    self_check_max_retries: int = 3
    """反思循环最大重试次数"""

    self_check_ner_low_threshold: int = 2
    """NER 遗漏实体数量阈值（超过此值触发重抽）"""

    self_check_re_low_threshold: int = 2
    """RE 幻觉三元组数量阈值（超过此值触发重抽）"""

    # ===== Filter 筛选配置（P5新增） =====
    enable_filter: bool = False
    """是否启用 Filter 筛选节点，提前过滤无效文本"""

    # ===== Normalize 归一化配置（P6新增） =====
    enable_normalize: bool = False
    """是否启用 Normalize 归一化节点，消解指代和归一化别名"""

    @classmethod
    def from_env(cls) -> "ExtractionConfig":
        """
        从环境变量加载配置

        环境变量命名规则：
        - EVAL_THRESHOLD
        - SIMILARITY_THRESHOLD
        - CORPUS_PER_WORKER
        - MAX_WORKERS
        - MAX_TEXT_LENGTH
        - MIN_TEXT_LENGTH
        - USE_SIMPLIFIED_EVAL (true/false)
        - RETRY_INITIAL_INTERVAL
        - RETRY_BACKOFF_FACTOR
        - RETRY_MAX_INTERVAL
        - RETRY_MAX_ATTEMPTS
        """
        def get_float(key: str, default: float) -> float:
            val = os.getenv(key)
            if val:
                try:
                    return float(val)
                except ValueError:
                    pass
            return default

        def get_int(key: str, default: int) -> int:
            val = os.getenv(key)
            if val:
                try:
                    return int(val)
                except ValueError:
                    pass
            return default

        def get_bool(key: str, default: bool) -> bool:
            val = os.getenv(key)
            if val:
                return val.lower() in ("true", "1", "yes", "on")
            return default

        return cls(
            eval_threshold=get_float("EVAL_THRESHOLD", 3.5),
            similarity_threshold=get_float("SIMILARITY_THRESHOLD", 0.85),
            corpus_per_worker=get_int("CORPUS_PER_WORKER", 10),
            max_workers=get_int("MAX_WORKERS", 10),
            max_concurrent_corpus=get_int("MAX_CONCURRENT_CORPUS", 50),
            batch_size=get_int("BATCH_SIZE", 100),
            max_text_length=get_int("MAX_TEXT_LENGTH", 10000),
            min_text_length=get_int("MIN_TEXT_LENGTH", 1),
            use_simplified_eval=get_bool("USE_SIMPLIFIED_EVAL", True),
            retry_initial_interval=get_float("RETRY_INITIAL_INTERVAL", 1.0),
            retry_backoff_factor=get_float("RETRY_BACKOFF_FACTOR", 2.0),
            retry_max_interval=get_float("RETRY_MAX_INTERVAL", 30.0),
            retry_max_attempts=get_int("RETRY_MAX_ATTEMPTS", 3),
            enable_self_check=get_bool("ENABLE_SELF_CHECK", False),
            self_check_max_retries=get_int("SELF_CHECK_MAX_RETRIES", 3),
            self_check_ner_low_threshold=get_int("SELF_CHECK_NER_LOW_THRESHOLD", 2),
            self_check_re_low_threshold=get_int("SELF_CHECK_RE_LOW_THRESHOLD", 2),
            enable_filter=get_bool("ENABLE_FILTER", False),
            enable_normalize=get_bool("ENABLE_NORMALIZE", False),
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ExtractionConfig":
        """从字典加载配置"""
        return cls(
            eval_threshold=config_dict.get("eval_threshold", 3.5),
            similarity_threshold=config_dict.get("similarity_threshold", 0.85),
            corpus_per_worker=config_dict.get("corpus_per_worker", 10),
            max_workers=config_dict.get("max_workers", 10),
            max_concurrent_corpus=config_dict.get("max_concurrent_corpus", 50),
            batch_size=config_dict.get("batch_size", 100),
            max_text_length=config_dict.get("max_text_length", 10000),
            min_text_length=config_dict.get("min_text_length", 1),
            use_simplified_eval=config_dict.get("use_simplified_eval", True),
            retry_initial_interval=config_dict.get("retry_initial_interval", 1.0),
            retry_backoff_factor=config_dict.get("retry_backoff_factor", 2.0),
            retry_max_interval=config_dict.get("retry_max_interval", 30.0),
            retry_max_attempts=config_dict.get("retry_max_attempts", 3),
            enable_self_check=config_dict.get("enable_self_check", False),
            self_check_max_retries=config_dict.get("self_check_max_retries", 3),
            self_check_ner_low_threshold=config_dict.get("self_check_ner_low_threshold", 2),
            self_check_re_low_threshold=config_dict.get("self_check_re_low_threshold", 2),
            enable_filter=config_dict.get("enable_filter", False),
            enable_normalize=config_dict.get("enable_normalize", False),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "eval_threshold": self.eval_threshold,
            "similarity_threshold": self.similarity_threshold,
            "corpus_per_worker": self.corpus_per_worker,
            "max_workers": self.max_workers,
            "max_concurrent_corpus": self.max_concurrent_corpus,
            "batch_size": self.batch_size,
            "max_text_length": self.max_text_length,
            "min_text_length": self.min_text_length,
            "use_simplified_eval": self.use_simplified_eval,
            "retry_initial_interval": self.retry_initial_interval,
            "retry_backoff_factor": self.retry_backoff_factor,
            "retry_max_interval": self.retry_max_interval,
            "retry_max_attempts": self.retry_max_attempts,
            "enable_self_check": self.enable_self_check,
            "self_check_max_retries": self.self_check_max_retries,
            "self_check_ner_low_threshold": self.self_check_ner_low_threshold,
            "self_check_re_low_threshold": self.self_check_re_low_threshold,
            "enable_filter": self.enable_filter,
            "enable_normalize": self.enable_normalize,
        }


# 默认配置实例
DEFAULT_CONFIG = ExtractionConfig.from_env()