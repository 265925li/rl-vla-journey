"""Day 13 · 看 10 张图：数据没读错才谈得上训练

跑法（PowerShell，项目根目录）：
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\vis_mnist.py

产出：outputs/<时间戳>_mnist_vis/samples.png（10 张图 + 标签）

════════════════════════════════════════════════════════════════════
为什么这 5 分钟永远不能省

  数据读错是深度学习里最贵的一种 bug：**它不报错。**
  常见的长相：
    · 标签和图像错位（归一化参数、transpose 顺序写反）
    · 图像被转成 0~255 的 uint8，模型照样能训，只是准确率莫名其妙上不去
    · 训练集和测试集用了不同的 transform，导致"训练时好好的，测试崩了"
    · 切验证集时顺序切，验证集全是 0 和 1

  这些错误里没有一个是会抛异常的。唯一的发现方式是**看一眼图**。
  所以先画图，再训练 —— 顺序不能反。
════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.mnist import build_datasets
from src.utils.paths import run_dir
from src.utils.seed import make_generator


def main() -> None:
    # 只为了画图，参数随便造一个"够用就行"的配置对象。
    # 真实项目里这里应该也是 load_config，但画图脚本不该依赖配置文件存在。
    class _Cfg:
        data_dir = "data"
        seed = 0
        val_size = 6000

    train_set, val_set, test_set = build_datasets(_Cfg)

    out = run_dir("mnist_vis")
    print(f"输出目录 {out}")

    # 从训练集里随机挑 10 个下标。用固定种子，保证同一张图每次挑的是同一批样本
    g = make_generator(0)
    idx = torch.randperm(len(train_set), generator=g)[:10].tolist()

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.2))
    for ax, i in zip(axes.flat, idx):
        x, y = train_set[i]          # x: (1,28,28) float32 0~1，y: int
        img = x.squeeze(0).numpy()   # (1,28,28) → (28,28)，去掉通道维才能画

        # cmap="gray" 且 vmin/vmax 固定：不固定的话 matplotlib 会自动拉伸对比度，
        # 一张很淡的图会被它"调"得很黑，你就看不出数据原本的亮度分布了
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"y = {y}", fontsize=11)
        ax.axis("off")               # 关掉坐标轴，看图像时它只是干扰

    fig.suptitle("MNIST training samples", fontsize=12)
    fig.tight_layout()
    png = out / "samples.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)

    # ── 顺手把几个"必须知道"的数字打出来 ────────────────────────
    x0, y0 = train_set[0]
    print(f"\n单条样本：x.shape={tuple(x0.shape)}  dtype={x0.dtype}  "
          f"取值 [{x0.min():.2f}, {x0.max():.2f}]  y={y0}")
    print(f"数据集大小：train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    # 标签分布：如果不均衡，准确率这个指标会误导你
    labels = torch.tensor([train_set[i][1] for i in range(2000)])
    counts = torch.bincount(labels, minlength=10).tolist()
    print(f"前 2000 条的标签分布：{counts}")

    print(f"\n图像已保存 {png}")


if __name__ == "__main__":
    main()
