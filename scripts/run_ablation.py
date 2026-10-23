"""Day 15 · 正则化对照实验（ablation）：一次跑完四组，汇总成一张表

跑法（PowerShell，项目根目录）：
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\run_ablation.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\run_ablation.py --epochs 10

════════════════════════════════════════════════════════════════════
为什么要有这个脚本

  「加 dropout 有用吗」这种问题，**不能靠感觉回答**。
  你手动跑一次 baseline，再手动跑一次 dropout，中间隔了 3 分钟，
  你已经开始记不清第一次的数字了 —— 更别说还有随机种子在捣鬼。

  对照实验（ablation）的三个铁律：
    1. 一次只改一个变量      —— 否则不知道是哪个变量起的作用
    2. 种子必须相同          —— 否则差异可能只是"这次运气好"
    3. 结果必须并排放在一起  —— 分散在四个目录里的数字人眼比不出来

  这个脚本就是把这三点固化下来。以后做 RL 实验（PPO 的各种变体）
  也是同一套流程，只是把「训练 MNIST」换成「训练 PPO」。

════════════════════════════════════════════════════════════════════
四组怎么设计的

    A  baseline   dropout=0     wd=0        ← Day 13 的配置，当参照物
    B  dropout    dropout=0.2   wd=0        ← 只加 dropout
    C  weight_decay  dropout=0  wd=5e-4     ← 只加 L2 惩罚
    D  both       dropout=0.2   wd=5e-4     ← 两个一起加

  A 是参照物，B/C 各测一个变量，D 看两个变量会不会叠加。
  如果只有 A 一组，你永远不知道 dropout 有没有用；
  如果一次加两个（只跑 A 和 D），就算变好了也不知道是谁的功劳。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.paths import OUTPUTS, ensure_dir

# 四组实验。字典里**只写和 baseline 不一样的东西**，这样一眼能看出每组改了啥
GROUPS: list[dict] = [
    {"name": "A_baseline",     "dropout": 0.0, "wd": 0.0},
    {"name": "B_dropout",      "dropout": 0.2, "wd": 0.0},
    {"name": "C_weight_decay", "dropout": 0.0, "wd": 5e-4},
    {"name": "D_both",         "dropout": 0.2, "wd": 5e-4},
]


def run_one(group: dict, epochs: int, seed: int) -> Path | None:
    """用 subprocess 跑一组训练，返回那一跑的 outputs 目录。

    为什么要开子进程，而不是在同一个进程里循环调训练函数？
    因为 PyTorch 的随机状态、CUDA 上下文在同一个进程里会残留，
    第二组实验的"随机"其实已经被第一组影响过了。
    开一个全新的进程 = 全新的随机状态，四组才是真正独立的。
    （代价是每组都要重新 import torch、读一遍数据，多花几秒钟，值得。）
    """
    cmd = [
        sys.executable, str(ROOT / "scripts" / "train_mnist.py"),
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--tag", group["name"],
        "--dropout", str(group["dropout"]),
        "--weight-decay", str(group["wd"]),
    ]
    # flush=True 必须加：不加的话这行会留在 Python 的缓冲区里，
    # 而子进程的输出是直接写终端的，结果就是「训练日志先刷出来、
    # 『▶ 开始跑哪一组』后出来」，看起来像是顺序乱了。
    print(f"\n{'=' * 66}\n▶ {group['name']}  dropout={group['dropout']}  wd={group['wd']}\n{'=' * 66}",
          flush=True)

    # capture_output=False → 训练日志直接流到屏幕上，你能看到进度
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        print(f"✗ {group['name']} 失败，returncode={proc.returncode}")
        return None

    # train_mnist.py 用时间戳建目录，我们不知道名字 —— 从日志反查太脆，
    # 改用「找最新的那个目录」.
    # 更稳的做法是让 train_mnist.py 把目录名打到 stdout 再解析，
    # 但那要改训练脚本的接口，这里先用简单办法。
    cands = sorted(OUTPUTS.glob(f"*_mnist_{group['name']}"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def load_summary(run_dir: Path) -> dict:
    """从一跑的 history.json 里抽出汇总要用的几个数。"""
    rec = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    hist = rec["history"]
    last = hist[-1]
    # 前 3 轮的平均 gap —— 用平均而不是单轮，避免被某一轮的抖动带偏
    early = hist[:3]
    return {
        "dir": run_dir.name,
        "epochs_run": len(hist),
        "test_acc": rec["test"]["acc"],
        "test_loss": rec["test"]["loss"],
        "best_val_acc": rec["best_val_acc"],
        "last_train_acc": last["train_acc"],
        "last_val_acc": last["val_acc"],
        "last_gap": last["gap"],
        "best_val_loss": min(h["val_loss"] for h in hist),
        "final_wnorm": rec.get("final_wnorm", float("nan")),
        "stopped_at": rec.get("early_stop", {}).get("stopped_at"),
        "history": hist,
    }


def write_table(rows: list[tuple[dict, dict]], out_md: Path) -> None:
    """把四组结果排成 markdown 表。"""
    lines = [
        "# Day 15 · 正则化对照实验结果",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "| 组 | dropout | wd | 跑了几轮 | 末轮 train | 末轮 val | gap | best val loss | 权重范数 | test acc |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g, s in rows:
        lines.append(
            f"| {g['name']} | {g['dropout']} | {g['wd']} | {s['epochs_run']} | "
            f"{s['last_train_acc']:.4f} | {s['last_val_acc']:.4f} | "
            f"{s['last_gap']:+.4f} | {s['best_val_loss']:.4f} | "
            f"{s['final_wnorm']:.3f} | **{s['test_acc']:.4f}** |"
        )
    lines += [
        "",
        "## 怎么读这张表",
        "",
        "- **gap** = 末轮 train_acc − val_acc。**正数**才是过拟合（train 比 val 好）。",
        "  用了 dropout 之后 gap 可能变成**负数**，那不是欠拟合，",
        "  而是训练时被随机丢弃拖累了准确率 —— train 那条线本来就该比 val 差一点。",
        "- **权重范数**是 Σp² 开方。开了 weight_decay 之后它应该变小，",
        "  没变小就说明 wd 没生效。",
        "- **test acc** 是唯一能对外报的数字，其它列都是给自己看的诊断指标。",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def plot_compare(rows: list[tuple[dict, dict]], out_png: Path) -> None:
    """两组曲线并排：左边 val_loss，右边 gap。四组各一条线。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for g, s in rows:
        ep = [h["epoch"] for h in s["history"]]
        ax1.plot(ep, [h["val_loss"] for h in s["history"]], marker="o", ms=3, label=g["name"])
        ax2.plot(ep, [h["gap"] for h in s["history"]], marker="o", ms=3, label=g["name"])

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("val loss")
    ax1.set_title("val loss: lower is better")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.axhline(0, color="gray", lw=1, ls="--")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("train acc - val acc")
    ax2.set_title("overfitting gap")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Day 15 正则化对照实验")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="四组必须用同一个种子")
    ap.add_argument("--only", type=str, default=None,
                    help="只跑某一组（写名字，如 B_dropout），调试时省时间")
    args = ap.parse_args()

    groups = [g for g in GROUPS if args.only is None or g["name"] == args.only]
    if not groups:
        print(f"没有叫 {args.only!r} 的组，可选：{[g['name'] for g in GROUPS]}")
        sys.exit(1)

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = ensure_dir(OUTPUTS / f"{stamp}_ablation")

    rows: list[tuple[dict, dict]] = []
    for g in groups:
        d = run_one(g, args.epochs, args.seed)
        if d is None:
            continue
        rows.append((g, load_summary(d)))
        # 每跑完一组就立刻落盘一次 summary，防止后面某组崩掉导致前面白跑
        write_table(rows, out / "summary.md")

    if not rows:
        print("一组都没跑成")
        sys.exit(1)

    write_table(rows, out / "summary.md")
    plot_compare(rows, out / "compare.png")

    print("\n" + "=" * 66)
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"汇总目录：{out}")


if __name__ == "__main__":
    main()
