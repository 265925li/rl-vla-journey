"""Day 24 · 批级数据增强：把 3.46 倍的开销压回去

════════════════════════════════════════════════════════════════════
问题

  Day 23 的增强有一个硬代价：每轮从 9.48s 涨到 32.84s，慢 3.46 倍。
  多出来的 23.4 秒全花在 `grid_sample` 上，而且是**一串极小的调用**：

      每一批 128 张图 → __getitem__ 被调用 128 次
                      → 每次 warp 形状是 (1,1,28,28)，一张图
                      → 每个 epoch 54000 次（batch_size=128 → 422 批）

  28×28 的图太小了。一次 grid_sample 的**固定开销**（算子调度、
  内存分配、Python 侧的参数检查）远大于它真正做的计算量。
  调 54000 次，绝大部分时间花在"调用的仪式"上，不是花在插值上。

  把批内 128 张图合成一次 (128,1,28,28) 的调用，
  固定开销从 128 次变成 1 次。这是今天要拿回来的东西。

════════════════════════════════════════════════════════════════════
为什么这件事"看起来简单，其实很容易做错"

  直觉做法：把 `AugmentedDataset.__getitem__` 里的增强挪到 `collate_fn`。

      数据集 → 取出 128 个 (x,y) → 拼成批 → 整批 warp 一次

  代码只改了十几行，但**随机序列被悄悄换掉了**：

      Day 23（逐样本）：样本 i 的随机数 = 第 i 次调用的三个数
      今天（整批）  ：样本 i 的随机数 = 第 i 行的三个数

  如果抽样还是"分三次各抽 n 个"，那么整批时
  样本 i 拿到 (r[i], r[n+i], r[2n+i])，而逐样本时拿到 (r[3i], r[3i+1], r[3i+2])
  —— **同一张图拿到的平移量完全不同**。

  于是你会看到一个非常难受的局面：优化前后 loss 曲线对不上，
  但你分不清是"优化写错了"还是"随机序列变了"。
  **可对拍性一旦丢失，正确性就没有判据了。**

  Day 24 的做法是先解决随机数：把 sample_params 改成一次抽 3n 再
  reshape 成 (n, 3)。这样"第 i 个样本的参数"被钉死成 r[3i..3i+2]，
  与批大小、与分几次调用都无关。于是整批增强和逐样本增强
  **逐位相同**，对拍回到 0.00e+00。

  ⚠️ 这块推导写在 src/data/augment.py 的 sample_params 里，
  那边有旧/新两种抽法的对照，改这里之前先读那边。

════════════════════════════════════════════════════════════════════
三种"换实现"，对拍能力各不相同（Day 22 → 24 的完整阶梯）

  | 换的是什么 | 例子 | 能做到 | 判据 |
  |---|---|---|---|
  | 换算法   | im2col vs 直接卷积 | 逐位 0.00e+00 | 同样输入必须同样输出 |
  | 换数值路径 | 手写 CE vs F.cross_entropy | 单步 1e-7、多步分叉 | 终值差 << 统计噪声 |
  | 换执行时机 | 逐样本 vs 整批 | **先改随机消耗，就回到逐位** | 同 seed 同顺序，逐位相同 |

  第三行是今天的收获：执行方式本身不改变数值，真正会破坏对拍的
  是"随机数什么时候被抽"。把这一点控制住，优化就能被硬验证。

════════════════════════════════════════════════════════════════════
num_workers 的坑（从 Day 23 继承下来，但今天变轻了）

  Day 23 的警告：Dataset 里的 generator 在 num_workers>0 时会被
  fork 成多份，各 worker 状态相同 → 所有 worker 生成同一串随机数。

  今天换成 collate_fn 之后，collate 仍然在 worker 里执行
  （每个 worker 拿到一个 batch 的样本列表后自己 collate），
  所以**这个坑没有消失**，只是位置从 Dataset 挪到了 collate。
  本项目 num_workers=0，暂时安全。真要用多进程，必须用
  `torch.initial_seed()` 或 `worker_init_fn` 给每个 worker 换 seed。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.augment import AugmentedDataset, random_affine, sample_params  # noqa: E402
from src.utils.seed import make_generator  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# 1. collate_fn：把增强从"每张图一次"改成"每批一次"
# ═══════════════════════════════════════════════════════════════════

def make_augment_collate(
    max_shift: float = 0.0,
    max_angle: float = 0.0,
    gen: torch.Generator | None = None,
    seed: int = 0,
):
    """返回一个带增强的 collate_fn。

    ⚠️ Day 27 改过：原来返回的是**闭包**，现在返回一个类实例。
    原因：闭包**不能被 pickle**，而 Windows 上 DataLoader 用 spawn，
    子进程要靠 pickle 把 collate_fn 传过去 ——
    于是 `num_workers>0` 在挂增强时**根本起不来**，报

        AttributeError: Can't pickle local object
        'make_augment_collate.<locals>.collate'

    这个错误在 Day 23/24 从没暴露过，因为那两天一直是 num_workers=0。
    **不是新引入的 bug，是原来那条路根本没被走过。**

    DataLoader 的工作流程是：

        Dataset.__getitem__  × batch_size     ← 逐张取，纯读盘/去标准化
        collate_fn           × 1              ← 拼成一个批张量

    collate_fn 拿到的是 `[(x0,y0), (x1,y1), ..., (x127,y127)]`，
    它要返回一个批。默认实现（default_collate）就是把 list 里的张量
    沿新维度 stack 起来。

    我们在这里插一脚：先默认拼好，再对**整个批**做一次 warp。

    ── 为什么必须先拼再 warp ────────────────────────────────────
    如果对 list 里的每张图单独 warp 再 stack，那就是 Day 23 的做法，
    一点没省。省下来的全部来自"一次算子调用处理 128 张"。

    ── 关闭增强时不留痕迹 ────────────────────────────────────────
    max_shift 和 max_angle 都是 0 时，random_affine 直接 return x，
    **连一个随机数都不抽**。所以 augment=0 的路径和 Day 22 的原始管线
    在数值上完全等价（这一点在关 5 里断言）。

    ── per_worker_gen ────────────────────────────────────────────
    见 AugmentCollate 里那段：这是 num_workers>0 时唯一能保住
    "增强多样性"的开关，代价是失去与 num_workers=0 的逐位可比性。
    """
    return AugmentCollate(
        max_shift=max_shift, max_angle=max_angle, gen=gen, seed=seed
    )


class AugmentCollate:
    """可 pickle 的批级增强 collate。

    为什么是一个类而不是闭包：spawn 只能传**可 pickle** 的东西，
    而闭包捕获了局部变量，pickle 无法重建它。类可以（只要字段可 pickle，
    `torch.Generator` 恰好是可 pickle 的）。

    ── 一个必须讲清楚的坑：generator 会被"复制"给每个 worker ──────
    `num_workers=W` 时，这个对象被 pickle 到 W 个子进程，**每份都是
    一模一样的副本**，包括 `self.gen` 的当前状态。

    于是 worker 0 抽的第 1 组随机数 == worker 1 抽的第 1 组 == ……
    而 batch i 由 worker (i mod W) 处理，所以 **batch 0..W-1 拿到的是
    同一组增强参数**。W=4 时效果就是"每 4 个批里有 1 组参数被重复用 4 次"，
    增强剂量实际上被打了折 —— 而且 loss 曲线完全正常，看不出来。

    两种应对：
      · per_worker_gen=False（默认）：保持"与 num_workers=0 逐位一致"，
        但 W>1 时增强会重复。适合"我要可复现"的对照实验。
      · per_worker_gen=True：每个 worker 用 `seed + worker_id` 派生
        自己的流，多样性恢复；但这时 num_workers=0/2/4 的结果**不再相同**。
        适合"我只关心效果、不要求逐位对拍"的长训练。

    **这两个目标不能同时满足。** 因为单条顺序流被切成 W 条之后，
    就再也拼不回原来那条的顺序 —— 除非把随机流改成"按批号索引"，
    而那需要 sampler、dataset、collate 三处一起改（见讲义「还能怎么做」）。
    """

    def __init__(self, max_shift=0.0, max_angle=0.0, gen=None, seed=0,
                 per_worker_gen: bool = False) -> None:
        self.max_shift = float(max_shift)
        self.max_angle = float(max_angle)
        self.seed = int(seed)
        self.per_worker_gen = bool(per_worker_gen)
        self.gen = gen
        # 打个标记，让外面能判断"这个 DataLoader 到底有没有挂增强"。
        # 为什么不靠 `collate_fn is not default_collate` 来判：那只能说明
        # "不是默认的"，说不出它是不是增强。
        # （Day 23 的关 7 就是靠 isinstance 判挂载位置的，换成 collate 之后
        #  必须跟着换成断言"行为特征"—— 这条教训见 Day 23。）
        self.__augment__ = {"max_shift": self.max_shift,
                            "max_angle": self.max_angle}

    def _gen(self):
        """拿到本进程该用的 generator。

        per_worker_gen=False：直接用构造时传进来的那条（与单进程一致）。
        per_worker_gen=True ：按 worker_id 派生。**注意要在子进程里懒创建**，
        不能在 __init__ 里建 —— __init__ 跑在父进程，那时还不知道 worker_id。
        """
        if not self.per_worker_gen:
            if self.gen is None:
                self.gen = torch.Generator()
                self.gen.manual_seed(self.seed)
            return self.gen

        if self.gen is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            # 用一个大质数错开，避免 seed=0,1 与 wid=0,1 撞出同一条流
            g = torch.Generator()
            g.manual_seed(self.seed * 1_000_003 + wid)
            self.gen = g
        return self.gen

    def __call__(self, batch):
        xs, ys = default_collate(batch)
        xs = random_affine(xs, self.max_shift, self.max_angle, self._gen())
        return xs, ys

    def __repr__(self) -> str:
        return (f"AugmentCollate(shift={self.max_shift}, angle={self.max_angle}, "
                f"per_worker_gen={self.per_worker_gen})")


def has_augment(loader: DataLoader) -> bool:
    """这个 DataLoader 有没有挂增强。给自测用。"""
    return getattr(loader.collate_fn, "__augment__", None) is not None


def build_train_loader(
    train_set: Dataset,
    batch_size: int,
    seed: int,
    max_shift: float = 0.0,
    max_angle: float = 0.0,
    num_workers: int = 0,
    per_worker_gen: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """把「打乱 + 增强」装进一个 DataLoader。

    抽出来单独一个函数，是为了让 src/data/mnist.py 里那一段
    "到底怎么装增强"只有一处，改的时候不会漏。

    参数 per_worker_gen：只在 num_workers>0 且挂了增强时起作用。
        False（默认）→ 与 num_workers=0 逐位一致，但 W>1 时增强会重复
        True         → 增强多样性正确，但失去与 num_workers=0 的逐位可比性
        两者不可兼得，详见 AugmentCollate 的 docstring。
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    kwargs = {}
    if max_shift > 0 or max_angle > 0:
        kwargs["collate_fn"] = make_augment_collate(
            max_shift, max_angle, gen, seed=seed
        )
        # 只在需要时把 per_worker_gen 打开 —— 默认关闭，保住可对拍性
        if per_worker_gen and num_workers > 0:
            kwargs["collate_fn"].per_worker_gen = True
            kwargs["collate_fn"].gen = None      # 让它在子进程里懒创建

    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        # 与 Day 22/23 完全同一个 seed、同一个构造方式，所以样本顺序一位没变。
        # 这一点是 baseline 回归检查能逐位通过的前提。
        generator=make_generator(seed),
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        # ⚠️ 不要再加 worker_init_fn 去改 generator：generator 是 collate
        # 对象自己的字段，worker_init_fn 拿不到它。
        # 按 worker 分流只有一条路：在子进程里懒创建（per_worker_gen）。
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════
# 2. 六关
# ═══════════════════════════════════════════════════════════════════

def _toy(n: int = 512, channels: int = 1, size: int = 28, seed: int = 0):
    """造一个小数据集，形状和 MNIST 一致，免得每关都去下数据。

    ⚠️ 注意这里**不用** RandomDataset 之类的随机图 —— 用固定的
    非零图案，否则"两张图都是全 0"时，对拍 0.00e+00 毫无意义
    （两个零本来就相等）。Day 22 的关 5 就吃过这个亏。
    """
    g = torch.Generator().manual_seed(seed)
    # 每张图放一个亮块，位置随样本变化 —— 这样平移/旋转一定看得见
    x = torch.zeros(n, channels, size, size)
    for i in range(n):
        r = (i * 7) % (size - 8) + 4
        c = (i * 13) % (size - 8) + 4
        x[i, :, r : r + 4, c : c + 4] = 1.0
    x += 0.01 * torch.randn(x.shape, generator=g)     # 加一点噪声，避免退化
    y = torch.arange(n) % 10
    return torch.utils.data.TensorDataset(x, y)


def gate1_sampling_equivalence() -> bool:
    """关 1：批级抽样 == 逐样本抽样（逐位）。

    这是今天所有优化的地基。如果这里不逐位，后面就没法对拍。
    """
    print("\n[关 1] 批级抽样 vs 逐样本抽样")
    n, shift, angle = 8, 2.0, 10.0

    g1 = torch.Generator(); g1.manual_seed(3)
    tx_a, ty_a, ang_a = sample_params(n, shift, angle, g1)

    # 模拟逐样本：调 8 次，每次 n=1
    g2 = torch.Generator(); g2.manual_seed(3)
    rows = [sample_params(1, shift, angle, g2) for _ in range(n)]
    tx_b = torch.cat([r[0] for r in rows])
    ty_b = torch.cat([r[1] for r in rows])
    ang_b = torch.cat([r[2] for r in rows])

    d = max(
        (tx_a - tx_b).abs().max().item(),
        (ty_a - ty_b).abs().max().item(),
        (ang_a - ang_b).abs().max().item(),
    )
    print(f"    一次抽 n=8  vs  8 次抽 n=1   最大差 {d:.2e}")
    print(f"    样本 0 参数 (tx,ty,ang) = ({tx_a[0]:+.4f}, {ty_a[0]:+.4f}, {ang_a[0]:+.4f})")
    print(f"    样本 7 参数 (tx,ty,ang) = ({tx_a[7]:+.4f}, {ty_a[7]:+.4f}, {ang_a[7]:+.4f})")
    assert d == 0.0, (
        f"批级与逐样本抽样不等价（差 {d:.3e}）。\n"
        "      原因通常是 sample_params 里没有用「一次抽 3n 再 reshape」，\n"
        "      或者 reshape 顺序不是行优先（应保证样本 i 拿 r[3i..3i+2]）。"
    )
    print("    ✓ 通过（逐位相同，所以整批增强可以被硬对拍）")
    return True


def gate2_batch_size_invariance() -> bool:
    """关 2：每样本的参数与 batch_size 无关。

    这一关测的是**性质**，不是实现：同一张图在 epoch 里的位置一样，
    不管它落在 128 的批里还是 32 的批里，都应该拿到同样的变换。

    为什么重要：batch_size 应该是"只影响速度"的量。如果它偷偷影响
    增强强度，那扫剂量实验里"只改了批大小"的对照就全是假的。
    """
    print("\n[关 2] 每样本参数与 batch_size 无关")
    total, shift, angle = 16, 2.0, 10.0

    def collect(batch_size: int) -> torch.Tensor:
        g = torch.Generator(); g.manual_seed(11)
        out = []
        for start in range(0, total, batch_size):
            k = min(batch_size, total - start)
            tx, ty, ang = sample_params(k, shift, angle, g)
            out.append(torch.stack([tx, ty, ang], dim=1))
        return torch.cat(out)

    ref = collect(total)              # 一次抽 16 组
    for bs in (1, 2, 4, 8):
        got = collect(bs)
        d = (got - ref).abs().max().item()
        print(f"    batch={bs:<3} 拼 16 组参数   与一次抽 vs 最大差 {d:.2e}")
        assert d == 0.0, (
            f"batch={bs} 时每样本参数变了（差 {d:.3e}）。\n"
            "      说明抽样方式依赖了 n —— 那么改批大小就会悄悄改增强剂量。"
        )
    print("    ✓ 通过（batch_size 只影响速度，不影响每一张图怎么变）")
    return True


def gate3_warp_batch_equivalence() -> bool:
    """关 3：给定同一组 theta，整批 warp == 逐样本 warp。

    关 1 保证"抽到同样的参数"，这一关保证"同样的参数得到同样的图"。
    两关合起来才有资格做端到端对拍（关 4）。
    """
    print("\n[关 3] 整批 warp vs 逐样本 warp")
    ds = _toy(16)
    x = torch.stack([ds[i][0] for i in range(16)])

    from src.data.augment import warp, make_theta

    g = torch.Generator(); g.manual_seed(5)
    tx, ty, ang = sample_params(16, 2.0, 10.0, g)
    theta = make_theta(tx, ty, ang, image_size=x.shape[-1])

    whole = warp(x, theta)
    pieces = torch.cat([warp(x[i : i + 1], theta[i : i + 1]) for i in range(16)])
    d = (whole - pieces).abs().max().item()

    print(f"    warp(16 张, 一次)  vs  warp(1 张) × 16   最大差 {d:.2e}")
    assert d == 0.0, (
        f"整批与逐样本 warp 不等价（差 {d:.3e}）。\n"
        "      若这是 1e-7 量级，说明 grid_sample 在批维上有不同的内核/累加顺序，\n"
        "      那端到端就只能比统计量，必须把这件事写进讲义，不能假装逐位。"
    )
    print("    ✓ 通过（grid_sample 在批维上逐样本独立计算，可逐位对拍）")
    return True


def gate4_end_to_end() -> bool:
    """关 4（最硬的一关）：两种挂法在真实 DataLoader 上逐位相同。

    左边：Day 23 的做法 —— 增强在 Dataset.__getitem__ 里，逐样本
    右边：今天的做法   —— 增强在 collate_fn 里，整批

    同 seed、同 shuffle 顺序、同 batch_size，要求每一个 batch 的
    x 和 y 都逐位相同。这一关过了，"优化没改变任何数值"就是实测结论，
    不是我的推理。
    """
    print("\n[关 4] 端到端：Dataset 逐样本 vs collate 整批")
    ds = _toy(500)
    shift, angle, seed, bs = 2.0, 10.0, 0, 128

    def run(kind: str):
        if kind == "sample":
            source = AugmentedDataset(ds, shift, angle, seed=seed)
            loader = DataLoader(
                source, batch_size=bs, shuffle=True,
                # ⚠️ 这条 generator 必须与 build_train_loader 里那条同 seed。
                # 它控制的是**样本顺序**，是独立于增强的另一条随机流 ——
                # 对拍时如果只对齐了增强的 seed，忘了 shuffle 的 seed，
                # 两边看到的样本顺序都不一样，比出来的差是没有意义的。
                # （Day 24 第一版就踩了这个，是关 4 自己抓出来的。）
                generator=torch.Generator().manual_seed(seed),
                num_workers=0, drop_last=False,
            )
        else:
            loader = build_train_loader(
                ds, batch_size=bs, seed=seed,
                max_shift=shift, max_angle=angle, num_workers=0,
            )
        out = []
        for xb, yb in loader:
            out.append((xb.clone(), yb.clone()))
        return out

    a = run("sample")
    b = run("batch")

    assert len(a) == len(b), f"批个数不同：{len(a)} vs {len(b)}"
    print(f"    {'批':>3}{'样本数':>8}{'x 最大差':>12}{'y 相同':>9}")
    worst = 0.0
    for i, ((xa, ya), (xb, yb)) in enumerate(zip(a, b)):
        d = (xa - xb).abs().max().item()
        worst = max(worst, d)
        same_y = torch.equal(ya, yb)
        print(f"    {i:>3}{xa.shape[0]:>8}{d:>12.2e}{str(same_y):>9}")
        assert same_y, f"第 {i} 批标签顺序不同 —— shuffle 的随机流没对齐"
    print(f"    总体最大差 {worst:.2e}")

    # 批大小不整除时最后一批会不满，这里必须覆盖到（500/128 → 128,128,128,116）
    assert a[-1][0].shape[0] == 500 % bs or 500 % bs == 0, "构造问题"

    assert worst == 0.0, (
        f"端到端对不上（最大差 {worst:.3e}）。按这个顺序查：\n"
        "      1. collate 里的 gen 与 AugmentedDataset 的 seed 是否相同\n"
        "      2. 两个 DataLoader 的 shuffle generator 是否同 seed（否则顺序就不同）\n"
        "      3. sample_params 是否真做到了『与 n 无关』（回到关 1、关 2）"
    )
    print("    ✓ 通过（整批增强与逐样本增强逐位相同，优化没有改动任何数值）")
    return True


def gate5_augment_off() -> bool:
    """关 5：关掉增强时，必须和原始管线逐位相同。

    Day 23 的 AugmentedDataset 在 active=False 时短路；今天的 collate
    在幅度全 0 时根本不挂上去。两条路径都要落到"完全不碰数据"。
    """
    print("\n[关 5] 关闭增强 == 原始管线")
    ds = _toy(300)
    bs = 64

    def run(use_collate: bool):
        if use_collate:
            loader = build_train_loader(ds, bs, seed=0, max_shift=0.0, max_angle=0.0)
        else:
            loader = DataLoader(
                ds, batch_size=bs, shuffle=True,
                generator=torch.Generator().manual_seed(0), drop_last=False,
            )
        return [(x.clone(), y.clone()) for x, y in loader]

    a, b = run(False), run(True)
    worst = max((xa - xb).abs().max().item() for (xa, _), (xb, _) in zip(a, b))
    print(f"    DataLoader 默认 vs build_train_loader(幅度 0)   最大差 {worst:.2e}")

    # 再确认一次：幅度为 0 时 sample_params 里的 rand 压根没被调用
    g = torch.Generator(); g.manual_seed(0)
    before = g.get_state().clone()
    random_affine(torch.zeros(4, 1, 28, 28), 0.0, 0.0, g)
    after = g.get_state()
    untouched = torch.equal(before, after)
    print(f"    幅度为 0 时 random_affine 是否消耗随机数：{not untouched}")
    assert untouched, "幅度为 0 却消耗了随机数 —— 会让 baseline 与 Day 22 的序列错位"

    assert worst == 0.0, f"关闭增强时管线不一致（差 {worst:.3e}）"
    print("    ✓ 通过（关掉增强后，baseline 仍是 Day 22 那条一模一样的管线）")
    return True


def gate6_speed() -> bool:
    """关 6：批级到底快多少。"""
    print("\n[关 6] 速度对比（128 张 × 10 次）")
    x = torch.rand(128, 1, 28, 28)
    shift, angle, repeat = 2.0, 10.0, 10

    g1 = torch.Generator(); g1.manual_seed(0)
    t0 = time.perf_counter()
    for _ in range(repeat):
        random_affine(x, shift, angle, g1)
    t_batch = (time.perf_counter() - t0) / repeat

    g2 = torch.Generator(); g2.manual_seed(0)
    t0 = time.perf_counter()
    for _ in range(repeat):
        for i in range(128):
            random_affine(x[i : i + 1], shift, angle, g2)
    t_sample = (time.perf_counter() - t0) / repeat

    print(f"    整批 (128,1,28,28) 一次      {t_batch * 1000:>8.2f} ms")
    print(f"    逐样本 128 次 (1,1,28,28)    {t_sample * 1000:>8.2f} ms")
    print(f"    加速 {t_sample / t_batch:.1f} 倍")
    # 54000 是 MNIST 训练集切掉 6000 验证后的实际张数（batch_size=128 → 422 批）
    print(f"    折算到 54000 张/轮：{t_sample * 54000 / 128:.1f}s → {t_batch * 54000 / 128:.1f}s")
    assert t_batch < t_sample, "批级反而更慢？检查是不是在 Python 里又循环了一遍"
    print("    ✓ 通过")
    return True


# ═══════════════════════════════════════════════════════════════════
# 3. 入口
# ═══════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="Day 24 批级数据增强自测")
    ap.add_argument("--train", action="store_true", help="跑全部六关（含速度）")
    args = ap.parse_args()

    print("=" * 68)
    print("Day 24 · 批级数据增强（collate_fn）自测")
    print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    print("=" * 68)

    gates = [
        ("关 1 批级抽样 == 逐样本抽样", gate1_sampling_equivalence),
        ("关 2 与 batch_size 无关", gate2_batch_size_invariance),
        ("关 3 整批 warp == 逐样本 warp", gate3_warp_batch_equivalence),
        ("关 4 端到端逐位对拍", gate4_end_to_end),
        ("关 5 关闭增强 == 原始管线", gate5_augment_off),
    ]
    if args.train:
        gates.append(("关 6 速度", gate6_speed))

    ok, passed = True, 0
    for name, fn in gates:
        try:
            if fn():
                passed += 1
        except AssertionError as e:
            print(f"    ✗ 未通过：{e}")
            ok = False
        except Exception as e:                       # noqa: BLE001
            print(f"    ✗ 意外错误：{type(e).__name__}: {e}")
            ok = False

    print("\n" + "=" * 68)
    print(f"{passed}/{len(gates)} 关通过")
    if ok:
        print("下一步：跑 25 轮，看 aug3 到底在哪见顶")
        print('  & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\sweep_augment.py --epochs 25 --only 0,3')
    else:
        print("有检查没过，别急着跑长训练 —— 优化错了跑出来的曲线全是垃圾")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
