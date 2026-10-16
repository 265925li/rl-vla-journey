"""项目路径常量 + 目录工具。

设计原则：所有路径都从【本文件的位置】反推，**不依赖当前工作目录**。
这样无论你在哪个目录下运行脚本，拿到的路径都一样。

────────────────────────────────────────────────────────────────
对外接口：
    ROOT / SRC / CONFIGS / SCRIPTS / NOTEBOOKS / OUTPUTS / WEEKLY
              各目录的 Path 常量，全部由 ROOT 拼出来
    ensure_dir(p)      目录不存在就创建，返回同一个 Path
    run_dir(prefix)    按时间戳建唯一输出子目录，避免覆盖上次结果
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# 1. 路径常量
# ═══════════════════════════════════════════════════════════════════
# 本文件在磁盘上的位置：  <项目根>/src/utils/paths.py
#
#   Path(__file__)              = .../src/utils/paths.py
#   .resolve()                  -> 转成绝对路径（清掉 ../ 之类的相对成分）
#   .parent                     = .../src/utils
#   .parent.parent              = .../src
#   .parent.parent.parent       = <项目根>          ← 我们要的就是它
#
# 注意这里比 Day 1 的 env_info.py 多了一层 —— 因为文件位置更深了。
# 这就是为什么不能把路径写死，必须从 __file__ 反推：
# 文件一挪位置，写死的路径就错了，反推出来的永远是对的。
ROOT: Path = Path(__file__).resolve().parent.parent.parent

SRC: Path = ROOT / "src"
CONFIGS: Path = ROOT / "configs"
SCRIPTS: Path = ROOT / "scripts"
NOTEBOOKS: Path = ROOT / "notebooks"
OUTPUTS: Path = ROOT / "outputs"
WEEKLY: Path = ROOT / "weekly"


# ═══════════════════════════════════════════════════════════════════
# 2. 目录工具
# ═══════════════════════════════════════════════════════════════════
def ensure_dir(p: Path) -> Path:
    """目录不存在就创建（含所有父目录），返回同一个 Path。

    已存在时**不报错** —— 这个函数注定要被反复调用。
    参数用 Path 对象（不是字符串），返回 Path 便于继续拼接。
    """
    # 两个参数都不能少，各自解决一个问题：
    #   parents=True   ->  outputs/2026-10-16_120000/ 这种多级目录要一次建出来，
    #                      否则父目录不存在时会报 FileNotFoundError
    #   exist_ok=True  ->  目录已存在时不要抛 FileExistsError
    #                      （这个函数会被反复调用，第二次就会撞上）
    p.mkdir(parents=True, exist_ok=True)  # make directory
    return p


def run_dir(prefix: str, base: Path | None = None) -> Path:
    """按时间戳生成一个唯一的输出子目录，并创建好它。

    形如：outputs/2026-10-16_165911_env_info/

    为什么需要它：以后每次跑实验都会产出日志、图、模型权重。
    如果都写进同一个目录，第二次实验会把第一次的结果覆盖掉，
    你就再也回不去看"上次那个好结果是怎么来的"。
    带上时间戳，每次实验一个独立目录，历史永远留着。

    参数：
        prefix : 目录名后缀，用来标记这次是干什么的（如 "dqn" / "ppo"）
        base   : 放在哪个目录下，默认 OUTPUTS
    """
    if base is None:
        base = OUTPUTS

    # 时间戳格式： 2026-10-16_165911
    #   %Y 年  %m 月  %d 日  %H 时  %M 分  %S 秒
    # 用下划线分隔日期和时间，避免路径里出现空格
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # 先把目标目录拼出来（/ 运算符就是 Path 的"拼接"）
    d: Path = base / f"{stamp}_{prefix}"

    # 目录拼出来了，但还没真的创建 —— 交给 ensure_dir()。
    # 这就是"复用"第一次真正发生的地方：同一个逻辑写一次，
    # 在多少个地方用都只维护一处。
    d = ensure_dir(d)

    return d


if __name__ == "__main__":
    # 自测：直接跑这个文件，应该能看到项目根路径和一个新建的输出目录。
    #     & "E:\anaconda\envs\pytorch\python.exe" -m src.utils.paths
    print(f"项目根   : {ROOT}")
    print(f"输出目录 : {run_dir('selftest')}")
