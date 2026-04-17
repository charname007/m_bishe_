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
    corpus_per_worker: int = 5
    """每个 Worker 处理的语料数量（与 batch_llm_size 对齐，实现 Worker 级并发）"""

    max_workers: int = 10
    """最大 Worker 数量"""

    max_concurrent_corpus: int = 50
    """最大并发处理语料数（Semaphore限制，防止API限流）"""

    batch_size: int = 100
    """每批次从数据库读取的语料数量"""

    # ===== P10新增：批量LLM调用配置 =====
    batch_llm_size: int = 5
    """每次LLM调用处理的语料数量（批量输入模式）"""

    enable_batch_llm: bool = True
    """是否启用批量LLM调用模式（一次调用处理多条语料）"""

    batch_llm_fallback: bool = True
    """批量处理失败时是否自动退化为单条处理"""

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

    self_check_max_retries: int = 1
    """反思循环最大重试次数（P16优化：从3改为1，减少LLM调用成本）"""

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

    # ===== QA Scaffold 配置（P8新增） =====
    enable_qa_scaffold: bool = False
    """是否启用苏格拉底式QA引导节点，构建语义脚手架"""

    qa_scaffold_min_text_length: int = 20
    """启用QA脚手架的最小文本长度（过短文本跳过）"""

    # ===== 联合抽取配置（P9新增） =====
    use_joint_extraction: bool = True
    """是否使用联合抽取模式（默认True，一次推理同时抽取实体和关系）"""

    # ===== 二次检查配置（P9新增） =====
    enable_full_self_check: bool = True
    """是否启用所有节点二次检查（QA、Joint、Eval、Label）- P15修改：默认启用"""

    enable_reflexion: bool = True
    """是否启用Reflexion反思机制（仅在联合抽取模式下有效）"""

    reflexion_max_retries: int = 1
    """Reflexion反思循环最大重试次数（P16优化：减少LLM调用成本）"""

    # ===== Filter/Normalize二次检查配置（P9新增，可选） =====
    enable_self_check_filter: bool = False
    """是否启用Filter筛选二次检查（可选，默认False）"""

    enable_self_check_normalize: bool = False
    """是否启用Normalize归一化二次检查（可选，默认False）"""

    # ===== QA导师模式配置（P10新增） =====
    enable_qa_mentor: bool = False
    """是否启用QA导师模式（QA作为中心审批节点）"""

    qa_approval_enabled: bool = False
    """是否启用QA审批流程"""

    max_revision_cycles: int = 3
    """最大修改循环轮次"""

    # ===== 多LLM模型配置（P10新增） =====
    qa_llm_model: str = "deepseek-reasoner"
    """QA节点使用的LLM模型（导师模式用强模型）"""

    worker_llm_model: str = "deepseek-chat"
    """后续节点（Joint、Eval、Label）使用的LLM模型"""

    qa_llm_temperature: float = 0.7
    """QA节点LLM温度参数（导师模式需要一定灵活性）"""

    worker_llm_temperature: float = 0.0
    """工作节点LLM温度参数（确定性输出）"""

    enable_qa_reasoning_trace: bool = True
    """是否保存QA的推理过程（Reasoner模型输出）"""

    # ===== 导师困惑触发阈值配置（降token优化） =====
    mentor_query_min_confidence: str = "low"
    """触发导师查询的最低困惑级别: low / medium。默认 low（更省token）"""

    mentor_extraction_low_item_threshold: int = 2
    """联合抽取中低置信实体/关系数量阈值（达到才触发困惑）"""

    mentor_eval_reject_ratio_threshold: float = 0.8
    """Eval困惑触发阈值：拒绝比例超过该值才触发（默认0.8，较保守）"""

    mentor_label_missing_ratio_threshold: float = 0.8
    """Label困惑触发阈值：缺属性实体比例超过该值才触发（默认0.8，较保守）"""

    mentor_label_min_missing_attrs: int = 3
    """Label困惑触发阈值：至少缺失多少实体属性才触发"""

    # ===== 实体对齐配置（P11新增） =====
    enable_entity_alignment: bool = False
    """是否启用实体对齐节点，将抽取实体与数据库已有实体匹配"""

    alignment_similarity_threshold: float = 0.75
    """实体对齐相似度阈值（低于此值直接跳过，不交给LLM判断）"""

    alignment_top_k: int = 5
    """对齐时检索的候选数量（交给LLM判断的候选数）"""

    alignment_high_confidence_threshold: float = 0.90
    """高置信度阈值（超过此值直接确认匹配，无需LLM判断）"""

    alignment_embedding_model: str = "shibing624/text2vec-base-chinese"
    """实体嵌入模型名称"""

    alignment_use_llm_decision: bool = True
    """是否使用LLM对候选进行最终判断"""

    # ===== 提示词版本配置（P13新增） =====
    prompt_version: str = "v2"
    """提示词版本：v2（原版）或 v3（RISEN优化版）

    v2: 原版提示词（完整Schema定义，约4000 Token）
    v3: RISEN/CARE/TIDD-EC优化版（表格化Schema，约1500 Token，节省60%）

    推荐使用场景：
    - 生产环境（追求稳定性）: v2
    - 批量处理（追求成本效率）: v3
    - 调试测试（追求指令清晰）: v3
    """

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
            corpus_per_worker=get_int("CORPUS_PER_WORKER", 5),
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
            self_check_max_retries=get_int("SELF_CHECK_MAX_RETRIES", 1),
            self_check_ner_low_threshold=get_int("SELF_CHECK_NER_LOW_THRESHOLD", 2),
            self_check_re_low_threshold=get_int("SELF_CHECK_RE_LOW_THRESHOLD", 2),
            enable_filter=get_bool("ENABLE_FILTER", False),
            enable_normalize=get_bool("ENABLE_NORMALIZE", False),
            enable_qa_scaffold=get_bool("ENABLE_QA_SCAFFOLD", False),
            qa_scaffold_min_text_length=get_int("QA_SCAFFOLD_MIN_TEXT_LENGTH", 20),
            # P9新增参数
            use_joint_extraction=get_bool("USE_JOINT_EXTRACTION", True),
            enable_full_self_check=get_bool("ENABLE_FULL_SELF_CHECK", True),  # P15修改：默认启用
            enable_reflexion=get_bool("ENABLE_REFLEXION", True),
            reflexion_max_retries=get_int("REFLEXION_MAX_RETRIES", 1),
            # P9新增：Filter/Normalize二次检查（可选）
            enable_self_check_filter=get_bool("ENABLE_SELF_CHECK_FILTER", False),
            enable_self_check_normalize=get_bool("ENABLE_SELF_CHECK_NORMALIZE", False),
            # P10新增：批量LLM调用参数
            batch_llm_size=get_int("BATCH_LLM_SIZE", 5),
            enable_batch_llm=get_bool("ENABLE_BATCH_LLM", True),
            batch_llm_fallback=get_bool("BATCH_LLM_FALLBACK", True),
            # P10新增：QA导师模式参数
            enable_qa_mentor=get_bool("ENABLE_QA_MENTOR", False),
            qa_approval_enabled=get_bool("QA_APPROVAL_ENABLED", False),
            max_revision_cycles=get_int("MAX_REVISION_CYCLES", 3),
            qa_llm_model=os.getenv("QA_LLM_MODEL", "deepseek-reasoner"),
            worker_llm_model=os.getenv("WORKER_LLM_MODEL", "deepseek-chat"),
            qa_llm_temperature=get_float("QA_LLM_TEMPERATURE", 0.7),
            worker_llm_temperature=get_float("WORKER_LLM_TEMPERATURE", 0.0),
            enable_qa_reasoning_trace=get_bool("ENABLE_QA_REASONING_TRACE", True),
            mentor_query_min_confidence=os.getenv("MENTOR_QUERY_MIN_CONFIDENCE", "low"),
            mentor_extraction_low_item_threshold=get_int("MENTOR_EXTRACTION_LOW_ITEM_THRESHOLD", 2),
            mentor_eval_reject_ratio_threshold=get_float("MENTOR_EVAL_REJECT_RATIO_THRESHOLD", 0.8),
            mentor_label_missing_ratio_threshold=get_float("MENTOR_LABEL_MISSING_RATIO_THRESHOLD", 0.8),
            mentor_label_min_missing_attrs=get_int("MENTOR_LABEL_MIN_MISSING_ATTRS", 3),
            # P11新增：实体对齐参数
            enable_entity_alignment=get_bool("ENABLE_ENTITY_ALIGNMENT", False),
            alignment_similarity_threshold=get_float("ALIGNMENT_SIMILARITY_THRESHOLD", 0.75),
            alignment_top_k=get_int("ALIGNMENT_TOP_K", 5),
            alignment_high_confidence_threshold=get_float("ALIGNMENT_HIGH_CONFIDENCE_THRESHOLD", 0.90),
            alignment_embedding_model=os.getenv("ALIGNMENT_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"),
            alignment_use_llm_decision=get_bool("ALIGNMENT_USE_LLM_DECISION", True),
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ExtractionConfig":
        """从字典加载配置"""
        return cls(
            eval_threshold=config_dict.get("eval_threshold", 3.5),
            similarity_threshold=config_dict.get("similarity_threshold", 0.85),
            corpus_per_worker=config_dict.get("corpus_per_worker", 5),
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
            self_check_max_retries=config_dict.get("self_check_max_retries", 1),
            self_check_ner_low_threshold=config_dict.get("self_check_ner_low_threshold", 2),
            self_check_re_low_threshold=config_dict.get("self_check_re_low_threshold", 2),
            enable_filter=config_dict.get("enable_filter", False),
            enable_normalize=config_dict.get("enable_normalize", False),
            enable_qa_scaffold=config_dict.get("enable_qa_scaffold", False),
            qa_scaffold_min_text_length=config_dict.get("qa_scaffold_min_text_length", 20),
            # P9新增参数
            use_joint_extraction=config_dict.get("use_joint_extraction", True),
            enable_full_self_check=config_dict.get("enable_full_self_check", True),  # P15修改：默认启用
            enable_reflexion=config_dict.get("enable_reflexion", True),
            reflexion_max_retries=config_dict.get("reflexion_max_retries", 1),
            # P9新增：Filter/Normalize二次检查（可选）
            enable_self_check_filter=config_dict.get("enable_self_check_filter", False),
            enable_self_check_normalize=config_dict.get("enable_self_check_normalize", False),
            # P10新增：批量LLM调用参数
            batch_llm_size=config_dict.get("batch_llm_size", 5),
            enable_batch_llm=config_dict.get("enable_batch_llm", True),
            batch_llm_fallback=config_dict.get("batch_llm_fallback", True),
            # P10新增：QA导师模式参数
            enable_qa_mentor=config_dict.get("enable_qa_mentor", False),
            qa_approval_enabled=config_dict.get("qa_approval_enabled", False),
            max_revision_cycles=config_dict.get("max_revision_cycles", 3),
            qa_llm_model=config_dict.get("qa_llm_model", "deepseek-reasoner"),
            worker_llm_model=config_dict.get("worker_llm_model", "deepseek-chat"),
            qa_llm_temperature=config_dict.get("qa_llm_temperature", 0.7),
            worker_llm_temperature=config_dict.get("worker_llm_temperature", 0.0),
            enable_qa_reasoning_trace=config_dict.get("enable_qa_reasoning_trace", True),
            mentor_query_min_confidence=config_dict.get("mentor_query_min_confidence", "low"),
            mentor_extraction_low_item_threshold=config_dict.get("mentor_extraction_low_item_threshold", 2),
            mentor_eval_reject_ratio_threshold=config_dict.get("mentor_eval_reject_ratio_threshold", 0.8),
            mentor_label_missing_ratio_threshold=config_dict.get("mentor_label_missing_ratio_threshold", 0.8),
            mentor_label_min_missing_attrs=config_dict.get("mentor_label_min_missing_attrs", 3),
            # P11新增：实体对齐参数
            enable_entity_alignment=config_dict.get("enable_entity_alignment", False),
            alignment_similarity_threshold=config_dict.get("alignment_similarity_threshold", 0.75),
            alignment_top_k=config_dict.get("alignment_top_k", 5),
            alignment_high_confidence_threshold=config_dict.get("alignment_high_confidence_threshold", 0.90),
            alignment_embedding_model=config_dict.get("alignment_embedding_model", "shibing624/text2vec-base-chinese"),
            alignment_use_llm_decision=config_dict.get("alignment_use_llm_decision", True),
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
            "enable_qa_scaffold": self.enable_qa_scaffold,
            "qa_scaffold_min_text_length": self.qa_scaffold_min_text_length,
            # P9新增参数
            "use_joint_extraction": self.use_joint_extraction,
            "enable_full_self_check": self.enable_full_self_check,
            "enable_reflexion": self.enable_reflexion,
            "reflexion_max_retries": self.reflexion_max_retries,
            # P9新增：Filter/Normalize二次检查（可选）
            "enable_self_check_filter": self.enable_self_check_filter,
            "enable_self_check_normalize": self.enable_self_check_normalize,
            # P10新增：批量LLM调用参数
            "batch_llm_size": self.batch_llm_size,
            "enable_batch_llm": self.enable_batch_llm,
            "batch_llm_fallback": self.batch_llm_fallback,
            # P10新增：QA导师模式参数
            "enable_qa_mentor": self.enable_qa_mentor,
            "qa_approval_enabled": self.qa_approval_enabled,
            "max_revision_cycles": self.max_revision_cycles,
            "qa_llm_model": self.qa_llm_model,
            "worker_llm_model": self.worker_llm_model,
            "qa_llm_temperature": self.qa_llm_temperature,
            "worker_llm_temperature": self.worker_llm_temperature,
            "enable_qa_reasoning_trace": self.enable_qa_reasoning_trace,
            "mentor_query_min_confidence": self.mentor_query_min_confidence,
            "mentor_extraction_low_item_threshold": self.mentor_extraction_low_item_threshold,
            "mentor_eval_reject_ratio_threshold": self.mentor_eval_reject_ratio_threshold,
            "mentor_label_missing_ratio_threshold": self.mentor_label_missing_ratio_threshold,
            "mentor_label_min_missing_attrs": self.mentor_label_min_missing_attrs,
            # P11新增：实体对齐参数
            "enable_entity_alignment": self.enable_entity_alignment,
            "alignment_similarity_threshold": self.alignment_similarity_threshold,
            "alignment_top_k": self.alignment_top_k,
            "alignment_high_confidence_threshold": self.alignment_high_confidence_threshold,
            "alignment_embedding_model": self.alignment_embedding_model,
            "alignment_use_llm_decision": self.alignment_use_llm_decision,
        }


# 默认配置实例
DEFAULT_CONFIG = ExtractionConfig.from_env()
