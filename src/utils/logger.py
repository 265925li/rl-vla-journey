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

    为什么开头要判 `if logger.handlers` 并复用：
        logging.getLogger("x") 对于同一个 name，返回的是**同一个对象**。
        如果这里每次都 addHandler，第二次调用时 logger 上就挂了两个
        handler —— 于是每条日志打印两遍、写文件两遍。
        这个坑非常隐蔽：代码不报错，只是输出翻倍。

    ═══════════════════════════════════════════════════════════════
    ⚠️ Day 23 修的一个真 bug：只判 handlers 非空是不够的
    ═══════════════════════════════════════════════════════════════
    第一版就是「handlers 非空 → 直接 return」。这在「一个进程只跑一次」
    的用法下完全正确，但 Day 23 的 sweep 在**同一个进程里连续 new 了 5 个
    Trainer**，每个都调 get_logger("train", log_file=自己的目录)。

    第二次调用时 handlers 非空，于是直接返回了**第一次**的 logger ——
    它的 FileHandler 还指着第一组的目录！

    后果：5 组实验的日志全部写进第 1 组的 train.log，后面 4 组的目录里
    根本没有 train.log（只有 history.json 和 curve.png，因为它们走
    self.out 而不是 logger）。而且**完全不报错**，只是你打开第 2 组的
    目录想看日志时发现是空的。

    检测到它的方式很土：数日志里出现了几次「输出目录」——
    应该 1 次，实际 4 次。

    正确行为是：**同一个文件 → 复用；换文件 → 换 handler。**
    判据不是「handler 列表空不空」，而是「现有 FileHandler 指向哪个文件」。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 这次想要的日志文件（绝对路径，方便和现有 handler 比较）
    want = Path(log_file).resolve() if log_file is not None else None

    # ── 1. 处理已有的 FileHandler ──────────────────────────────────
    # 注意 logging.FileHandler 是 StreamHandler 的子类，
    # 所以下面判断"控制台 handler"时要用 not isinstance(h, FileHandler)。
    existing = next(
        (h for h in logger.handlers if isinstance(h, logging.FileHandler)), None
    )
    if existing is not None:
        if want is not None and Path(existing.baseFilename).resolve() == want:
            # 指向同一个文件 → 复用。这才是原版想表达的"配过就别再配"
            return logger
        # 指向别的文件（或者这次根本不想要文件）→ 摘掉旧的
        logger.removeHandler(existing)
        existing.close()

    # ── 2. 控制台 handler：没有才挂，避免重复打印 ──────────────────
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console:
        # 不带参数：StreamHandler() 默认写到 sys.stderr，也就是你眼前这个终端窗口
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(CONSOLE_FORMAT, CONSOLE_DATEFMT))
        logger.addHandler(console)

    # ── 3. 文件 handler ────────────────────────────────────────────
    if want is not None:
        # 父目录可能还不存在（比如 outputs/2026-10-16_120000/），先建出来。
        # 这里直接用 Path 自带的方法，不 import paths 模块 —— 保持工具模块之间
        # 互不依赖，谁都不会因为对方坏掉而坏掉。
        want.parent.mkdir(parents=True, exist_ok=True)

        # 文件 handler。两个参数都必须显式写：
        #   mode="w"        -> 每次运行**覆盖**。默认值是 "a"（追加），
        #                      不改的话跑第二次文件里就会堆两份报告
        #   encoding="utf-8"-> 不指定会用系统默认编码，中文会乱码
        fh = logging.FileHandler(want, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter(FILE_FORMAT, FILE_DATEFMT))
        logger.addHandler(fh)

    # 不要往上层（root logger）冒泡。
    # 否则将来某处代码在 root 上又配了一次 handler，我们的日志会被打两遍。
    logger.propagate = False
    return logger
