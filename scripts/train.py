"""训练入口 —— Day 3 只搭壳，训练逻辑待 W2 实现。

它现在只做四件事：
    1. 解析命令行 → 读 yaml → **命令行覆盖 yaml**
    2. 用 logger 打印最终生效的完整配置
    3. 在 outputs/<时间戳>_train/ 下建目录，把生效配置存一份 config.yaml
    4. 打一行 [dry-run] 提示

用法（PowerShell，在项目根目录跑）：
    cd "E:\\work Buddy\\强化学习\\rl-vla-journey"
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\train.py --dry-run
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\train.py --dry-run --seed 42 --device cpu

三样东西在这里第一次串起来：
    paths.run_dir()       建带时间戳的输出目录（Day 2）
    logger.get_logger()   打日志，不再用 print（Day 2）
    config.load_config()  读配置 + 命令行覆盖（Day 3）
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# 让 `python scripts/train.py` 能 import 到 src 包（和 env_info.py 同一套路）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, save_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.paths import run_dir  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# 1. 命令行
# ═══════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="训练入口（当前只有配置系统）")
    p.add_argument("--config", type=Path,
                   default=ROOT / "configs" / "default.yaml",
                   help="yaml 配置文件（默认用项目里的 configs/default.yaml）")

    # 下面这几个的 default 全是 None —— 这不是偷懒，是必须。
    # 写成具体值（比如 0）的话，用户不传时 args.seed=0，0 不是 None，
    # 就会被当成"用户传了"，把 yaml 里的 seed 反向覆盖掉。
    p.add_argument("--seed", type=int, default=None, help="随机种子")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                   help="训练设备")
    p.add_argument("--output-dir", type=str, default=None, help="输出根目录")
    p.add_argument("--log-level", type=str, default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="日志级别")

    # 开关：出现即 True，不出现即 None。
    # default 同样必须是 None —— 写成 False 的话，用户不传时
    # args.dry_run=False，而 False 不是 None，会被当成"用户传了"。
    p.add_argument("--dry-run", action="store_true", default=None,
                   help="只打印配置，不训练")

    return p


# ═══════════════════════════════════════════════════════════════════
# 2. 从命令行参数里挑出「覆盖项」
# ═══════════════════════════════════════════════════════════════════
def cli_overrides(args: argparse.Namespace) -> dict:
    """只把「用户真的传了」的键挑出来，供覆盖 yaml 用。

    这个函数是整套配置系统的关键：
        用户没传 -> 值仍是 None -> 不进 overrides -> yaml 的值保留
        用户传了 -> 值是真实值   -> 进 overrides  -> 覆盖 yaml

    排除的两个键各有原因：
        config    它是文件路径，不是配置项
        dry_run   它是流程开关，Config 里没这个字段
                  （不排除的话 Config(**raw) 会 TypeError）
    """
    return {
        k: v
        for k, v in vars(args).items()
        if k not in ("config", "dry_run") and v is not None
    }


# ═══════════════════════════════════════════════════════════════════
# 3. 主流程
# ═══════════════════════════════════════════════════════════════════
def main() -> int:
    args = build_parser().parse_args()

    # 第一步：读 yaml + 命令行覆盖 → Config 对象
    # overrides 只含用户真传的键，所以这一步天然实现了
    # 「命令行 > yaml > 默认值」的优先级。
    overrides = cli_overrides(args)
    cfg = load_config(args.config, overrides=overrides)

    # 第二步：建输出目录 + 建 logger
    #   · prefix 用 "train"，目录名就能看出这次跑的是什么
    #   · 日志写在 out 里而不是 outputs 根目录 —— 每次实验一个独立日志
    #   · level 用 getattr 把校验过的字符串转成 logging 常量
    #     （"DEBUG" -> logging.DEBUG -> 10）。不转的话 yaml 里的 log_level 白配。
    out = run_dir("train")
    logger = get_logger("train", out / "train.log",
                        level=getattr(logging, cfg.log_level))

    # 第三步：把生效配置打全，并存一份
    logger.info("=" * 52)
    logger.info(f"生效配置    （来源: {args.config.name}）")
    logger.info(f"命令行覆盖  （{len(overrides)} 项: {overrides or '无'}）")
    logger.info("-" * 52)
    for key, value in asdict(cfg).items():
        logger.info(f"  {key:<12} = {value}")
    logger.info("=" * 52)

    # 为什么要存：半年后想复现「当时那个好结果」，
    # 手上只有日志和权重文件，没有这一份 config，
    # 你根本不知道那一跑用了什么参数。
    save_config(cfg, out / "config.yaml")

    # 第四步：收尾提示
    if args.dry_run:
        logger.info("[dry-run] 配置就绪，训练逻辑待 W2 实现")
    else:
        logger.info("[未实现] 训练逻辑待 W2 实现")

    logger.info(f"输出目录: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
