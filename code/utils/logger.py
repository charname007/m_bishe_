"""
日志配置工具
"""
import sys
from loguru import logger


def setup_logger(log_file: str = None, level: str = "INFO"):
    """
    配置日志

    Args:
        log_file: 日志文件路径，None则只输出到终端
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
    """
    # 移除默认处理器
    logger.remove()

    # 终端输出格式
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # 添加终端处理器
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True
    )

    # 添加文件处理器
    if log_file:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )
        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation="10 MB",      # 日志文件大小限制
            retention="7 days",    # 保留7天
            compression="zip",     # 压缩旧日志
            encoding="utf-8"
        )

    return logger


# 默认配置
logger = setup_logger()