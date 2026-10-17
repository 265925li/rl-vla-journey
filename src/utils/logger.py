"""日志工具：一次配置，控制台和文件同时输出。

用法：
    from src.utils.logger import get_logger
    logger = get_logger(__name__, log_file=REPORT)
    logger.info("hello")

────────────────────────────────────────────────────────────────
对外接口：
    get_logger(name, log_file=None, level=logging.INFO) -> logging.Logger
        拿到一个配好的 logger：控制台 + 文件（给了 log_file 才写文件）
        同一个 name 重复调用返回同一个对象，不会重复挂 handler
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from pathlib import Path

# ───────────────────────────────────────────────────────────────────
# 两种格式，因为两种场景的读者不同
# ───────────────────────────────────────────────────────────────────
# 控制台：给人实时看的，越短越好看
#   %(asctime)s   时间
#   %(levelname)-7s  级别，左对齐占 7 格（这样各行能对齐）
#   %(message)s   正文
CONSOLE_FORMAT = "[合并版] %(asctime)s | %(levelname)-7s | %(message)s"

CONSOLE_DATEFMT = "%H:%M:%S"

# 文件：事后排查问题用的，所以多带两样定位信息
#   %(name)s     哪个 logger 打的（一般就是模块名）
#   %(lineno)d   哪一行打的  ← 出问题时能直接跳过去
FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s"
FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_file: Path | None = None,
    level: int = logging.INFO,#日志级别
) -> logging.Logger:
    """拿到一个配置好的 logger。

    参数：
        name     : logger 名字。一般直接传 __name__，
                   这样日志里能看出是哪个模块打的
        log_file : 日志文件的路径。传 None 表示只输出到控制台
        level    : 日志级别，默认 INFO

    为什么开头要判 `if logger.handlers: return logger`：
        logging.getLogger("x") 对于同一个 name，返回的是**同一个对象**。
        如果这里每次都 addHandler，第二次调用时 logger 上就挂了两个
        handler —— 于是每条日志打印两遍、写文件两遍。
        这个坑非常隐蔽：代码不报错，只是输出翻倍。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 已经配过就直接返回，避免 handler 越挂越多
    if logger.handlers:
        return logger

    # 控制台 handler。不带参数：StreamHandler() 默认写到 sys.stderr，
    # 也就是你眼前这个终端窗口。
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, CONSOLE_DATEFMT))
    logger.addHandler(console)

    # 只有给了文件路径，才挂文件 handler
    if log_file is not None:
        log_file = Path(log_file)
        # 父目录可能还不存在（比如 outputs/2026-10-16_120000/），先建出来。
        # 这里直接用 Path 自带的方法，不 import paths 模块 —— 保持工具模块之间
        # 互不依赖，谁都不会因为对方坏掉而坏掉。
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # 文件 handler。两个参数都必须显式写：
        #   mode="w"        -> 每次运行**覆盖**。默认值是 "a"（追加），
        #                      不改的话跑第二次文件里就会堆两份报告
        #   encoding="utf-8"-> 不指定会用系统默认编码，中文会乱码
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter(FILE_FORMAT, FILE_DATEFMT))
        logger.addHandler(fh)

    # 不要往上层（root logger）冒泡。
    # 否则将来某处代码在 root 上又配了一次 handler，我们的日志会被打两遍。
    logger.propagate = False
    return logger
