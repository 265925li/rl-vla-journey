"""env_info.py —— 环境探针（Day 2：模块化改造版）

与 Day 1 版本的区别：
  1. 路径推导和日志配置不再写在这里，改成从 src/utils/ 里 import
     —— 因为这两样东西以后每个脚本都要用，不该复制粘贴
  2. torch 的导入改成「优雅降级」：装不上时不甩 traceback，而是给可读提示
  3. 文件末尾包了一层异常兜底

用法（PowerShell）：
    cd "E:\\work Buddy\\强化学习\\rl-vla-journey"
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\env_info.py

或者在 PyCharm 里：打开这个文件 → Ctrl+Shift+F10
（前提：右下角解释器显示 Python 3.10 (pytorch)）

────────────────────────────────────────────────────────────────
三种运行结果（都是设计好的）：

  正常（pytorch 环境）            -> 打出完整环境报告，退出码 0
  没装 torch（换别的解释器跑）     -> 三条中文提示 + 修复建议，退出码 1
  输出目录不可写                   -> 中文提示而非 traceback，退出码 1

验证第 2、3 条的方法见 学习计划/Day2-填空指南.md。
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# 0. 让 `python scripts/env_info.py` 能 import 到 src 包
# ═══════════════════════════════════════════════════════════════════
# 为什么需要这段：
#   Python 只在 sys.path 列出的目录里找模块。默认包含的是
#   「当前工作目录」，**不包含项目根**。所以从 scripts/ 里跑脚本时，
#   `from src.utils...` 会直接 ModuleNotFoundError。
#
#   （你在 PyCharm 里点 Run 时可能不会遇到 —— PyCharm 默认会把
#     项目根加进 PYTHONPATH。但换到终端就必炸。显式写上最保险。）
#
# 这两行只影响这一次进程，不写进任何配置文件，对系统零影响。
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger  # noqa: E402
from src.utils.paths import OUTPUTS, ensure_dir  # noqa: E402

REPORT = OUTPUTS / "env_report.txt"


# ═══════════════════════════════════════════════════════════════════
# 1. torch 的条件导入
# ═══════════════════════════════════════════════════════════════════
# 这是 Day 1 那个 TODO 6 的"升级版"，思路一样：
#   **诊断工具不能在诊断对象出问题时失效。**
#
# 区别是这里处理得更彻底 —— 连 import 本身都保护起来了。
# 因为 import torch 在很多情况下真的会失败：
#   · 用错了解释器（比如 PATH 里那个没装 torch 的 Python 3.12）
#   · torch 装坏了 / 装的是别的平台的包
#   · CUDA 驱动版本和 torch 对不上，载入动态库时直接崩
#
# 裸 import 的情况下，以上每一种都会甩一段英文 traceback 结束，
# 而你恰恰是在"环境出问题"的时候才来跑这个脚本的。
try:
    import torch

    HAS_TORCH = True
    TORCH_ERROR: Exception | None = None
except ImportError as _exc:  # noqa: N816
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False
    TORCH_ERROR = _exc


# ═══════════════════════════════════════════════════════════════════
# 2. 收集环境信息
# ═══════════════════════════════════════════════════════════════════
def collect_env() -> list[tuple[str, str]]:
    """收集环境信息，返回 [(名称, 值), ...]。

    控制台打印和写文件共用这一份数据，所以只收集一次。
    函数签名里的 list[tuple[str, str]] 是类型注解：
    告诉读代码的人（和 IDE）这个函数会返回什么。
    """
    rows: list[tuple[str, str]] = []
    rows.append(("Python", sys.version.split()[0]))

    # 当前解释器的完整路径。这可能是整份报告里最有价值的一行：
    # 以后会遇到无数种「装了什么却不生效」的怪现象，99% 的原因是
    # 你敲的 python 和你以为的不是同一个，打出来一眼就能确认。
    rows.append(("解释器", sys.executable))

    rows.append(("PyTorch", torch.__version__))
    rows.append(("CUDA 可用", str(torch.cuda.is_available())))

    if torch.cuda.is_available():
        # 0 表示「第 0 块卡」。多卡机器上有区别，养成带上的习惯。
        rows.append(("GPU", torch.cuda.get_device_name(0)))

        # total_memory 拿到的是【字节】，除 1024**3 换成 GB，
        # :.1f 保留一位小数（顺便把数字变成字符串，符合列表的类型要求）
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        rows.append(("显存", f"{vram_bytes / 1024**3:.1f}GB"))

        cap = torch.cuda.get_device_capability(0)
        rows.append(("算力等级", f"sm_{cap[0]}{cap[1]}"))
    else:
        # CUDA 不可用时的占位值。
        # 这里绝不能去调 get_device_name() —— 那会直接抛异常，
        # 于是脚本恰好在你最需要它的时候崩掉。
        rows.append(("GPU", "不可用"))

    return rows


# ═══════════════════════════════════════════════════════════════════
# 3. 主流程
# ═══════════════════════════════════════════════════════════════════
def main() -> int:
    """跑一遍探针。返回退出码：0 = 正常，1 = 出错。"""
    ensure_dir(OUTPUTS)
    logger = get_logger("env_info", REPORT)

    if not HAS_TORCH:
        logger.error("导入 torch 失败 —— 当前解释器里没有 PyTorch。")
        logger.error(f"当前解释器: {sys.executable}")
        logger.error(r"正确的解释器应该指向:E:\anaconda\envs\pytorch\python.exe")
        logger.error(f"原始错误: {TORCH_ERROR}")
        return 1

    logger.info("-" * 52)
    logger.info("环境探针")
    logger.info("-" * 52)
    for key, value in collect_env():
        logger.info(f"{key:<10}: {value}")
    logger.info("-" * 52)
    logger.info(f"报告已写入: {REPORT}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:

        print(f"环境探针执行失败: {exc}", file=sys.stderr)
        print("提示：检查 outputs 所在磁盘是否可写、路径是否存在",
              file=sys.stderr)
        code = 1
    sys.exit(code)
