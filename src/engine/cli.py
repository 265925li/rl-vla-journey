"""命令行入口：把「一堆参数」变成一个 Config，再交给 Trainer。

════════════════════════════════════════════════════════════════════
这个文件为什么能被叫「通用模板」
════════════════════════════════════════════════════════════════════

判断标准只有一条：**换实验的时候，这个文件要不要改？**

    ❌ 不通用：参数表里写死 --dropout / --linear / --momentum ...
              → 每加一个配置字段，就得回来加一行 argparse

    ✅ 通用：除了少量高频快键，还有一个 `--set key=value` 兜底
              → 新字段**零改动**就能从命令行覆盖

加 --set 这一条，是这个模板和"把参数搬个家"之间唯一实质的区别。

════════════════════════════════════════════════════════════════════
三级优先级（后到者胜）
════════════════════════════════════════════════════════════════════

    Config 里的默认值  <  yaml 文件  <  命令行

这条规则本身由 load_config() 实现，cli 只负责把"用户真的说了什么"传下去。
所以下面的每个 argparse 参数都设 default=None ——
None 表示"用户没说话"，就不去覆盖 yaml。
如果写成 default=10，你就永远分不清"用户想要 10"和"用户没说话"。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# 触发注册。没有这一行，注册表是空的，build_model("mlp") 会报「未知模型」。
import src.engine.builtin  # noqa: F401
from src.engine.registry import DATASETS, MODELS
from src.engine.trainer import Trainer
from src.utils.config import load_config
from src.utils.paths import CONFIGS


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="通用训练入口：配置驱动，模型/数据集按名字注册",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/train.py --config configs/mnist_mlp.yaml\n"
            "  python scripts/train.py --epochs 3 --tag quick\n"
            "  python scripts/train.py --set dropout=0.2 --set hidden=512\n"
            "  python scripts/train.py --list          # 看有哪些模型/数据集\n"
            "  python scripts/train.py --dry-run       # 只打印最终配置，不训练\n"
        ),
    )

    ap.add_argument("--config", type=Path, default=CONFIGS / "mnist_mlp.yaml",
                    help="yaml 配置文件路径")

    # ── 高频快键。设成 None = "用户没传"，就不覆盖 yaml ──────────────
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--optimizer", type=str, default=None, choices=["sgd", "adam", "mysgd"])
    ap.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=None)

    # ── 正则化三个旋钮 ────────────────────────────────────────────
    ap.add_argument("--dropout", type=float, default=None, help="丢弃概率，0 表示不加")
    ap.add_argument("--weight-decay", type=float, default=None, help="L2 惩罚系数")
    ap.add_argument("--patience", type=int, default=None,
                    help="早停耐心值：验证指标连续这么多轮不改善就停（0 = 不早停）")
    ap.add_argument("--es-metric", type=str, default=None, choices=["loss", "acc"],
                    help="早停看哪个指标。默认 loss（连续、灵敏）；选 acc 会看到相反的结论")

    # ── 手写算子开关（Day 16 / 17）────────────────────────────────
    ap.add_argument("--loss", type=str, default=None, choices=["ce", "myce"],
                    help="ce = nn.CrossEntropyLoss，myce = Day 16 手写版")
    ap.add_argument("--linear", type=str, default=None, choices=["torch", "my"],
                    help="torch = nn.Linear，my = Day 17 手写版")

    # ── 通用兜底 ──────────────────────────────────────────────────
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="覆盖任意配置字段，可重复。例：--set hidden=512 --set act=tanh。"
                         "值用 yaml 语法解析，所以 0.2 是 float、true 是 bool、mlp 是 str")

    ap.add_argument("--tag", type=str, default="run", help="输出目录后缀，方便区分多次实验")

    # ── 只做检查、不真训练的两个开关 ──────────────────────────────
    ap.add_argument("--dry-run", action="store_true",
                    help="只读配置 + 打印最终生效值，不建目录、不训练")
    ap.add_argument("--list", action="store_true",
                    help="列出所有已注册的模型和数据集")
    return ap


def parse_sets(items: list[str]) -> dict:
    """把 ['hidden=512', 'dropout=0.2'] 变成 {'hidden': 512, 'dropout': 0.2}。

    值用 yaml.safe_load 解析，而不是手动猜类型：
        "512"   → int 512        （不是字符串 "512"）
        "0.2"   → float 0.2
        "true"  → bool True
        "adam"  → str "adam"
    手动猜类型的写法（先试 int()，失败再试 float()...）有一堆边界情况，
    yaml 已经把这套规则定义好了，直接复用最不容易出错。

    ⚠️ 一个必须挡掉的坑：值里带等号时不能简单 split("=")。
    `--set "output_dir=a=b"` 会 split 成 3 段。用 split("=", 1) 只切第一个。
    """
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式，收到 {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--set 的键不能为空，收到 {item!r}")
        out[key] = yaml.safe_load(raw)
    return out


def main(argv: list[str] | None = None) -> None:
    """所有训练入口的公共实现。

    scripts/train.py 和 scripts/train_mnist.py 都只是它的薄壳 ——
    两个脚本共用同一份实现，就不存在"两个入口行为不一致"这种问题。
    """
    args = build_parser().parse_args(argv)

    # ── --list：看一眼有什么可选，不用去翻代码 ───────────────────
    if args.list:
        print("已注册的数据集：")
        for name, fn in sorted(DATASETS.items()):
            print(f"  {name:<10} {fn.__module__}.{fn.__name__}")
        print("\n已注册的模型：")
        for name, fn in sorted(MODELS.items()):
            print(f"  {name:<10} {fn.__module__}.{fn.__name__}")
        print("\n新增一个模型：在 src/engine/builtin.py 里加 @register_model('名字')")
        return

    # ── 合并命令行覆盖。三级优先级在这里体现 ─────────────────────
    overrides = {
        k: v
        for k, v in {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "device": args.device,
            "seed": args.seed,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "early_stop_patience": args.patience,
            "early_stop_metric": args.es_metric,
            "loss": args.loss,
            "linear": args.linear,
        }.items()
        if v is not None
    }
    # --set 放在最后合并，所以它能盖掉上面那些快键
    # （用户写了 --epochs 5 --set epochs=3，就是想要 3 —— 越明确越靠后）
    overrides.update(parse_sets(args.set))

    cfg = load_config(args.config, overrides)

    # ── --dry-run：只体检，不训练 ────────────────────────────────
    # 为什么值得有：训练要 95 秒，跑完才发现 device 写错、路径不存在、
    # 模型名拼错，那 95 秒就白花了。dry-run 把这些前置检查压到 1 秒内。
    if args.dry_run:
        print(f"配置文件 {args.config}")
        print("最终生效配置：")
        for k in cfg.__dataclass_fields__:
            print(f"  {k:<24} {getattr(cfg, k)!r}")
        print(f"\n模型 {cfg.model!r} → {'已注册' if cfg.model in MODELS else '❌ 未注册'}")
        print(f"数据集 {cfg.dataset!r} → {'已注册' if cfg.dataset in DATASETS else '❌ 未注册'}")
        return

    Trainer(cfg, tag=args.tag).fit()


if __name__ == "__main__":
    main()
