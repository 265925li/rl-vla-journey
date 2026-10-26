"""画训练曲线。

放在 utils 里而不是 engine 里，因为**它不参与训练** ——
删掉这个文件，训练照样跑，只是没有曲线图。
engine 里的东西是"训练必需"，utils 里的是"训练之外的辅助"。
分清楚这条线，以后做后台训练（不需要画图）时就知道能省掉哪一块。
"""
from __future__ import annotations

from pathlib import Path

# Agg 是"只画到文件、不弹窗"的后端。
# 必须在 import pyplot **之前**设置 —— 顺序反了不生效（pyplot 一被 import
# 就把后端定下来了）。不加这一行，某些环境（无显示设备的服务器、
# 无 GUI 的容器）里 plt.show() 会卡住或直接报错。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_curves(history: list[dict], out_path: Path) -> None:
    """把每个 epoch 的 loss / 准确率 / 两者差距画成三张图。

    train 和 val 一定画在同一张里 —— 单看 train 曲线你以为自己很厉害，
    两条放一起才能看出"是学会了，还是背下来了"（过拟合）。

    第三张 gap = train_acc - val_acc 是过拟合的"体温计"，
    单调往上爬就是过拟合在加重。看 loss 那两张也能判断，
    但 gap 这种"把差距单独拎出来"的画法更不容易被
    "两边都在涨"的表象骗过去。

    ⚠️ 一个真实的反例：Day 15 实测 gap 从 +0.54 爬到 +1.86，
    看着像在恶化，但同期 val_loss 从 0.0789 降到 0.0773（还在变好）。
    所以 **gap 只是警报，val_loss 的绝对值才是判决书**。
    这三张图要一起看，不能只看第三张。
    """
    epochs = [h["epoch"] for h in history]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.6))

    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], marker="s", label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["train_acc"] for h in history], marker="o", label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], marker="s", label="val")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 老格式的 history 里没有 gap 字段，用算式兜一下（不然画图会 KeyError）
    gaps = [h.get("gap", h["train_acc"] - h["val_acc"]) for h in history]
    ax3.plot(epochs, gaps, marker="^", color="crimson", label="train - val")
    ax3.axhline(0, color="gray", lw=1, ls="--")   # 0 线：两条曲线在这里交叉
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("acc gap")
    ax3.set_title("overfitting gap")
    ax3.legend()
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)   # 关掉，否则连着跑多次会积累一堆图对象（内存缓慢上涨）
