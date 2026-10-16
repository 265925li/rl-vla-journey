"""env_info.py —— 环境探针

把当前 Python / PyTorch / CUDA 环境写成一份报告文件，同时打印到控制台。

用法（PowerShell，必须用解释器完整路径，不能用裸 python）：
    cd "E:\\work Buddy\\强化学习\\rl-vla-journey"
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\env_info.py

────────────────────────────────────────────────────────────────
本文件里有 6 处 TODO，都长这样：

    xxx = None
    assert xxx is not None, "TODO n 未完成：..."

你要做的：把 `None` 换成正确的代码，然后把紧跟的 assert 删掉。

**每跑一次，它会告诉你下一个没写完的地方** —— 所以不用一次想全，
改一处、跑一次、再改下一处。6 处都改完，脚本就通了。

在编辑器里搜索 "TODO" 可以看到全部位置。
────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import logging
import sys
import torch

# ═══════════════════════════════════════════════════════════════════
# 1. 路径（这部分是你已经写好的，正确）
# ═══════════════════════════════════════════════════════════════════
# 为什么是两层 parent：
#   __file__        = .../rl-vla-journey/scripts/env_info.py
#   .parent         = .../rl-vla-journey/scripts
#   .parent.parent  = .../rl-vla-journey      <- 项目根目录
#
# resolve() 把路径里的相对成分清掉、变成绝对路径。
# 不做这一步，脚本"从哪个目录被调用"会影响结果 —— 那是很难查的 bug。
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
REPORT = OUTPUTS / "env_report.txt"


# ═══════════════════════════════════════════════════════════════════
# 2. 日志
# ═══════════════════════════════════════════════════════════════════
def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """返回一个同时往控制台和文件写日志的 logger。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 重复调用时不要重复挂 handler，否则每条日志会打印多份
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── TODO 1 ───────────────────────────────────────────────────
    # 建一个「输出到控制台」的 handler。
    # 提示：logging.StreamHandler()  不带参数时默认就打到你眼前这个终端
    ch = logging.StreamHandler()#创建一个控制台日志处理器
    # ─────────────────────────────────────────────────────────────
    ch.setFormatter(fmt)#设置日志格式
    logger.addHandler(ch)#把控制台 handler 挂到 logger 上

    if log_file is not None:
        # ── TODO 2 ───────────────────────────────────────────────
        # 建一个「输出到文件」的 handler。
        # 提示：logging.FileHandler(路径, mode=?, encoding=?)
        #
        # 两个关键点在参数里：
        #   ① mode 默认是 "a"（追加）—— 跑第二次时新报告会接在旧报告后面，
        #      文件里就堆了两份。你要的是每次覆盖 → 该传什么？
        #   ② 不指定 encoding 会用系统默认编码，中文可能变乱码 → 传 "utf-8"
        fh = logging.FileHandler( log_file,mode = 'w',encoding = "utf-8")
        # ─────────────────────────────────────────────────────────
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ═══════════════════════════════════════════════════════════════════
# 3. 收集环境信息
# ═══════════════════════════════════════════════════════════════════
def collect_env() -> list[tuple[str, str]]:
    """返回 [(名称, 值), ...]。打印和写文件共用这一份数据。"""
    rows: list[tuple[str, str]] = []

    # sys.version 长这样："3.10.20 (main, ...) [MSC v.1933 64 bit (AMD64)]"
    # .split()[0] 取第一段，只要 "3.10.20"
    rows.append(("Python", sys.version.split()[0]))

    # ── TODO 3 ───────────────────────────────────────────────────
    # 取 torch 的版本号。
    # 提示：它是一个【属性】，不是函数调用 —— 别写成 torch.__version__()
    torch_ver = torch.__version__
    # ─────────────────────────────────────────────────────────────
    rows.append(("PyTorch", str(torch_ver)))

    # str() 是必要的：这个列表要求每一项都是字符串
    rows.append(("CUDA 可用", str(torch.cuda.is_available())))

    if torch.cuda.is_available():
        # ── TODO 4 ───────────────────────────────────────────────
        # 显卡名字。提示：torch.cuda.get_device_name(0)
        # 那个 0 表示「第 0 块卡」—— 多卡机器上会有区别，养成带上的习惯
        gpu_name = torch.cuda.get_device_name(0)
        # ─────────────────────────────────────────────────────────
        rows.append(("GPU", gpu_name))

        # ── TODO 5 ───────────────────────────────────────────────
        # 显存大小，单位 GB，保留 1 位小数，结果是字符串。
        # 提示：torch.cuda.get_device_properties(0).total_memory 拿到的是【字节】
        #       1 GB = 1024 ** 3 字节
        #       保留 1 位小数：f-string 写成 f"{数字:.1f}"
        vram = f"{(torch.cuda.get_device_properties(0).total_memory)/(1024**3):.1f}GB"
        # ─────────────────────────────────────────────────────────
        rows.append(("显存", vram))

        cap = torch.cuda.get_device_capability(0)
        rows.append(("算力等级", f"sm_{cap[0]}{cap[1]}"))
    else:
        # ── TODO 6 ───────────────────────────────────────────────
        # CUDA 不可用时的占位值。
        #
        # 这是整个脚本最重要的设计点：它存在的意义就是
        # 「在环境坏掉的时候还能跑起来，告诉你哪里坏了」。
        # 如果 CUDA 一不可用它就崩，那它恰好在你最需要它的时候失效。
        # 所以这里不要抛异常 —— 给一个人类可读的占位值（一个字符串就行）。
        gpu_name = "不可用"
        # ─────────────────────────────────────────────────────────
        rows.append(("GPU", gpu_name))

    return rows


# ═══════════════════════════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════════════════════════
def main() -> None:
    # 不加 exist_ok=True 的话，第二次运行会直接 FileExistsError 崩掉
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    logger = get_logger("env_info", REPORT)

    logger.info("-" * 52)
    logger.info("环境探针")
    logger.info("-" * 52)
    for key, value in collect_env():
        logger.info(f"{key:<10}: {value}")
    logger.info("-" * 52)

    # 思考题：上面全用 logger，为什么这一句偏偏用 print？
    print(f"报告已写入: {REPORT}")


if __name__ == "__main__":
    main()