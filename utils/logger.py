"""
日志记录模块

功能:
    - 配置和初始化全局日志记录器 (Logger)。
    - 使用 Loguru 库提供功能强大且易于使用的日志功能。
    - 支持控制台和文件两种日志输出，并应用不同的格式和级别。
    - 统一项目所有模块的日志记录行为。

输入:
    - 日志消息 (字符串)。
    - 日志级别 (如 INFO, WARNING, ERROR)。
    - (可选) 模块名，用于获取特定的记录器实例。

输出:
    - 格式化的日志消息，输出到控制台和/或日志文件。
"""

import sys
from loguru import logger
import logging


class InterceptHandler(logging.Handler):
    """
    一个将标准logging日志重定向到loguru的处理器。
    """
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(console_level="INFO"):
    """
    配置全局日志记录器。
    
    - 使用 Loguru 作为核心日志库。
    - 拦截标准库 logging，统一管理第三方库的日志输出。
    - 为控制台和文件设置不同的日志级别和格式。
    """
    # 移除所有现有的处理器，确保从干净的状态开始
    logger.remove()

    # --- 配置 Loguru ---
    # 1. 配置控制台输出
    #    - INFO 级别及以上
    #    - 使用丰富的颜色和格式
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(sys.stderr, level=console_level, format=console_format, colorize=True)

    # 2. 配置文件输出
    #    - DEBUG 级别及以上，用于详细排查问题
    log_file_path = "outputs/logs/app_{time}.log"
    logger.add(log_file_path, level="DEBUG", format="{time} {level} {name}:{function} {message}", rotation="10 MB", retention="7 days", encoding="utf-8")

    # --- 拦截标准 logging ---
    # 1. 设置要拦截的第三方库和它们的日志级别
    logging_config = {
        "uvicorn": "INFO",
        "uvicorn.error": "WARNING",
        "uvicorn.access": "WARNING",
        "sqlalchemy": "WARNING",
        "aiosqlite": "WARNING",
        "watchfiles": "WARNING"
    }
    
    # 2. 用我们的处理器替换标准 logging 的处理器
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # 3. 为指定的记录器设置级别
    for name, level in logging_config.items():
        logging.getLogger(name).setLevel(level)
    
    logger.info("日志系统初始化完成。")

# 导出配置好的 logger 实例，供项目其他模块使用
__all__ = ["logger", "setup_logging"]