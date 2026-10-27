"""Day 27 · 数据加载提速：先量，再改，最后验证没改坏

════════════════════════════════════════════════════════════════════
一、这一天是从一个"错的假设"开始的

  Day 24 的结论说：每轮 13.55s，CNN 前反向 9.48s，剩下 4.07s（30%）
  是"数据加载与拷贝"。于是 Day 27 的既定任务是优化那 30%。

  结果第一步 profiling 就把这个前提打翻了：

      只调 Dataset.__getitem__ × 54000        7.97s   ← 98% 的时间在这
      default_collate × 422                   0.07s
      collate + 增强 × 422                    0.40s   （增强净 0.8ms/批）
      422 次 H2D 拷贝                         0.09s
      422 步前反向 + 更新                     1.5s 左右

  **数据增强只花 0.40s，而"把一张图变成张量"花了 8 秒。**
  前者是 Day 24 花了一整天优化的对象，后者从来没被看过一眼。

  为什么会这样：torchvision 的 MNIST 在 __getitem__ 里做的是

      Image.fromarray(self.data[i].numpy())   ← 造一个 PIL 对象
      ToTensor()                              ← 再转回张量

  也就是说，数据本来就在内存里（uint8 张量），却要**绕一圈 PIL**
  再转回来。单张 150µs，其中真正的数值计算不到 5µs。

  这就是"先量再改"的价值：不量的话，你会去优化一个占 3% 的东西。

════════════════════════════════════════════════════════════════════
二、改法是"换后端"，不是"加并发"

  三个后端，每一级都更快，代价是内存：

      pil    torchvision 原样（PIL 往返）      6.10s / 轮     0 额外内存
      u8     直接索引 uint8 张量，逐样本转       1.96s / 轮     0 额外内存
      f32    启动时一次性转成 float32 常驻       0.62s / 轮    +179MB

  **三者在数值上都必须逐位相同**（关 1/关 2 证明），
  因为它们做的数学就是同一个：`float(uint8) / 255`。
  ToTensor() 对 uint8 图做的事也就是这个。

  所以这是一个"纯提速"的改动 —— 没有精度权衡，只有内存权衡。
  这种优化是最值得做的：判据明确（0.00e+00），收益明确（最多 9.8 倍）。

════════════════════════════════════════════════════════════════════
三、num_workers 的结论：不是"设个数字"，而且这里不该用

  两个独立的问题，都不能靠调参解决：

  ① **挂增强的 collate 原来不可 pickle。** 它是闭包，
     spawn 传不过去 → `num_workers>0` 直接报
     "Can't pickle local object"。这个 bug 一直存在，
     只是因为前三天一直用 num_workers=0 所以从没被触发。
     Day 27 把它改成了 `AugmentCollate` 类。

  ② **generator 会被原样复制给每个 worker。** 改了①之后能跑了，
     但 W 个 worker 的 generator 起始状态**一模一样**，于是
     batch 0..W-1 拿到同一组增强参数，增强剂量被打了折 ——
     而 loss 曲线完全正常，看不出来。

  ⇒ 所以"num_workers=0 与 4 逐位一致"这个验收标准**做不到**，
    除非把随机流从"顺序消耗"改成"按批号索引"（要同时改
    sampler / dataset / collate 三处，见讲义最后一节）。

  而做完第二节那个 9.8 倍提速之后，**num_workers 已经没有意义了**：
  数据侧从 8 秒压到 0.6 秒，模型侧是 1.5 秒 —— 瓶颈换了位置。
  在 Windows 上 spawn 每个 epoch 还要重启进程，净收益是负的。

  这条是 Amdahl 定律的反面用法：
  **优化瓶颈之后，原来的第二瓶颈会变成新瓶颈，而它的优化方案未必还成立。**
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALLOWED_BACKENDS = ("pil", "u8", "f32")


class TensorBackedDataset(Dataset):
    """把 torchvision 的 MNIST 换成"直接索引张量"。

    ── 为什么绕开 PIL 是安全的 ───────────────────────────────────
    ToTensor() 对 uint8 图做的事是：

        torch.from_numpy(np.array(pic)).permute(2,0,1).to(float32).div(255)

    对灰度图（C=1）就是 `float(x) / 255`。这是**逐元素的同一个除法**，
    没有插值、没有近似，所以逐位相同是可证的，不只是"应该差不多"。
    关 1 用 0.00e+00 把它钉死。

    ── u8 和 f32 的取舍 ─────────────────────────────────────────
    u8 ：每次取都做一遍 float()/255。省内存（MNIST 本来就以 uint8
         在内存里，等于零额外开销），但 54000 次除法的总开销是 1.3s。
    f32：启动时一次性转，之后每次只是切片 + unsqueeze。
         MNIST 全集 60000×28×28×4 = 179MB。换来 3.2 倍。

    为什么 f32 值得：这个数据集小到可以整个装进内存，
    而训练要跑几十轮。**一次性的 179MB 换每轮省 1.3 秒，25 轮就是 33 秒。**

    ⚠️ f32 的 179MB 是**整个 60000 张**，不是切出来的 54000 张。
    因为 Subset 只是下标视图，底层张量是共享的 —— 验证集也要用。
    """

    def __init__(self, mnist, indices, backend: str = "u8") -> None:
        if backend not in ("u8", "f32"):
            raise ValueError(
                f"TensorBackedDataset 只接受 u8 / f32，收到 {backend!r}。"
                "pil 表示「不换后端」，应该由 wrap_backend 处理。"
            )
        self.backend = backend
        # 保留 .dataset / .indices 两个属性：
        # 上层的 len(loader.dataset)、以及 Day 23/24 那些自测都靠它们。
        # 破坏这两个属性会让"换后端"变成一次全仓库搜索。
        self.dataset = mnist
        self.indices = indices
        self.targets = mnist.targets

        if backend == "f32":
            # 原地除：省掉一份 179MB 的中间张量
            self._f32 = mnist.data.to(torch.float32).div_(255.0)
        else:
            self._f32 = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        if self._f32 is not None:
            x = self._f32[j].unsqueeze(0)
        else:
            x = self.dataset.data[j].to(torch.float32).div(255.0).unsqueeze(0)
        # targets 是张量，必须转成 Python int ——
        # 否则 default_collate 会把它 stack 成 (B,) 张量，
        # 而下游 yb.to(device) 也能接受，看起来没问题，
        # 但 `int(self.targets[j])` 这条路径是原实现的行为，保持一致更安全。
        return x, int(self.targets[j])

    def __repr__(self) -> str:
        return (f"TensorBackedDataset(backend={self.backend}, n={len(self.indices)}, "
                f"mem={self.mem_mb():.0f}MB)")

    def mem_mb(self) -> float:
        if self._f32 is None:
            return 0.0
        return self._f32.numel() * 4 / 2**20


def wrap_backend(ds, backend: str):
    """把「MNIST」或「Subset(MNIST)」换成指定后端。pil 时原样返回。

    只做一件事，但要处理两种输入形状：
        datasets.MNIST            → indices = range(len)
        Subset(MNIST, indices)    → 用它的 indices
    这两种正是 src/data/mnist.py 里 pass 来 pass 去的两种。
    """
    if backend not in ALLOWED_BACKENDS:
        raise ValueError(f"data_backend 只能是 {ALLOWED_BACKENDS}，收到 {backend!r}")
    if backend == "pil":
        return ds

    from torch.utils.data import Subset

    if isinstance(ds, Subset):
        base, indices = ds.dataset, list(ds.indices)
    else:
        # torchvision 的 MNIST 有 .data / .targets 两个张量
        if not hasattr(ds, "data") or not hasattr(ds, "targets"):
            raise TypeError(
                f"wrap_backend 只支持 torchvision MNIST 或 Subset(MNIST)，"
                f"收到 {type(ds).__name__}。换数据集时要给它写一个对应的后端实现，"
                "不能靠猜它的内部结构。"
            )
        base, indices = ds, list(range(len(ds)))

    return TensorBackedDataset(base, indices, backend=backend)


# ═══════════════════════════════════════════════════════════════════
# 六关
# ═══════════════════════════════════════════════════════════════════

def _get_train_subset():
    """拿一个真实的 train Subset（和 build_datasets 切出来的完全一致）。"""
    from src.data.mnist import build_datasets

    class _Cfg:
        data_dir = "data"
        val_size = 6000
        seed = 0

    train_set, _, _ = build_datasets(_Cfg())
    return train_set


def gate1_u8_bitwise() -> bool:
    """关 1：u8 后端与 PIL 路径**逐位**相同。

    这一关是全部提速的地基。它不成立的话，
    "换后端没改数值"就是我的猜测，不是事实。
    """
    print("\n[关 1] u8 后端 vs PIL 路径（逐位）")
    sub = _get_train_subset()
    fast = wrap_backend(sub, "u8")

    probes = [0, 1, 2, 127, 128, 1000, 27777, len(sub) - 1]
    worst = 0.0
    print(f"    {'下标':>8}{'形状':>14}{'x 最大差':>14}{'标签':>7}")
    print("    " + "-" * 44)
    for i in probes:
        a, ya = sub[i]
        b, yb = fast[i]
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        print(f"    {i:>8}{str(tuple(a.shape)):>14}{d:>14.2e}{'相同' if ya == yb else '不同':>7}")
        assert ya == yb, f"下标 {i} 的标签不一致：{ya} vs {yb}"

    print(f"\n    总体最大差 {worst:.2e}")
    assert worst == 0.0, (
        f"u8 后端与 PIL 路径不是逐位相同（最大差 {worst:.3e}）。\n"
        "  最可能的原因：ToTensor 在 uint8 上做的不是 float(x)/255 而是别的\n"
        "  （比如先除以 255 再取整，或者带额外的归一化）。\n"
        "  这时不能硬说'等价'，要么改公式对齐，要么承认差异并改判据。"
    )
    print("    ✓ 通过（float(uint8)/255 与 ToTensor 逐位相同）")
    return True


def gate2_f32_bitwise() -> bool:
    """关 2：f32 后端也逐位相同，并给出它的内存代价。"""
    print("\n[关 2] f32 后端 vs PIL 路径（逐位 + 内存代价）")
    sub = _get_train_subset()
    fast = wrap_backend(sub, "f32")
    print(f"    {fast!r}")

    worst = 0.0
    for i in (0, 500, 20000, len(sub) - 1):
        a, _ = sub[i]
        b, _ = fast[i]
        worst = max(worst, (a - b).abs().max().item())
    print(f"    最大差 {worst:.2e}")
    assert worst == 0.0, f"f32 后端不是逐位相同（最大差 {worst:.3e}）"
    print(f"    内存代价 {fast.mem_mb():.0f}MB（整个 60000 张常驻，不是切出来的 54000）")
    print("    ✓ 通过（预转 float32 只是把除法提前做，数值一位不变）")
    return True


def gate3_end_to_end() -> bool:
    """关 3（最硬的一关）：换后端后，整条训练管线的 422 个批逐位相同。

    含增强、含 shuffle。单条对拍过了不代表端到端对拍过 ——
    中间还有 shuffle 的随机流、collate 的 stack 顺序。
    """
    print("\n[关 3] 端到端：422 批 × 增强 × shuffle 逐位对拍")
    from src.data.batch_aug import build_train_loader

    sub = _get_train_subset()
    fast = wrap_backend(sub, "u8")

    def run(ds):
        loader = build_train_loader(ds, 128, seed=0,
                                    max_shift=2.0, max_angle=10.0, num_workers=0)
        return [(x.clone(), y.clone()) for x, y in loader]

    a = run(sub)
    b = run(fast)
    assert len(a) == len(b), f"批数不同：{len(a)} vs {len(b)}"

    worst, same_y = 0.0, 0
    for (xa, ya), (xb, yb) in zip(a, b):
        worst = max(worst, (xa - xb).abs().max().item())
        same_y += int(torch.equal(ya, yb))
    print(f"    批数 {len(a)}   x 最大差 {worst:.2e}   标签全同的批 {same_y}/{len(a)}")
    assert same_y == len(a), "标签顺序变了 —— shuffle 的随机流没对齐"
    assert worst == 0.0, (
        f"端到端不是逐位相同（最大差 {worst:.3e}）。查这个顺序：\n"
        "  1. 两个 loader 的 seed 是否相同（shuffle 与增强是两条独立随机流）\n"
        "  2. batch_size 是否相同\n"
        "  3. 后端里除的是不是 /255.0（写 /255 在某些情况下会走整数除法）"
    )
    print("    ✓ 通过（换后端没有改动任何一个数值）")
    return True


def gate4_speed_ladder() -> bool:
    """关 4：三级后端的实测速度 —— 每一级的收益都要能对上。"""
    print("\n[关 4] 速度阶梯（各跑一整轮 54000 张）")
    from src.data.batch_aug import build_train_loader

    sub = _get_train_subset()
    rows = []
    for backend in ALLOWED_BACKENDS:
        ds = wrap_backend(sub, backend)
        # ① 纯读取（不打乱不增强）
        t0 = time.perf_counter()
        for i in range(0, len(ds), 97):
            ds[i]
        per = (time.perf_counter() - t0) / len(range(0, len(ds), 97))
        # ② 端到端（打乱 + 增强）
        loader = build_train_loader(ds, 128, seed=0,
                                    max_shift=2.0, max_angle=10.0, num_workers=0)
        t0 = time.perf_counter()
        for _ in loader:
            pass
        ep = time.perf_counter() - t0
        mem = ds.mem_mb() if hasattr(ds, "mem_mb") else 0.0
        rows.append((backend, per, ep, mem))

    base = rows[0][2]
    print(f"    {'后端':>6}{'单条读取':>12}{'一轮(增强)':>13}{'相对 pil':>11}{'额外内存':>11}")
    print("    " + "-" * 54)
    for backend, per, ep, mem in rows:
        print(f"    {backend:>6}{per*1e6:>10.1f}µs{ep:>11.2f}s"
              f"{base/ep:>10.2f}x{mem:>9.0f}MB")

    # 单调性：三级必须一级比一级快
    eps = [r[2] for r in rows]
    assert eps[0] > eps[1] > eps[2], (
        f"速度没有单调改善：{eps}。\n"
        "  要么后端实现有问题，要么机器正在跑别的负载（计时不稳定）。"
    )
    print(f"\n    最快一级把数据侧从 {eps[0]:.2f}s 压到 {eps[2]:.2f}s（{eps[0]/eps[2]:.1f}x）")
    print("    ✓ 通过")
    return True


def gate5_collate_picklable() -> bool:
    """关 5：挂增强的 collate 必须能被 pickle —— 否则 num_workers>0 起不来。

    这一关断言的是"可 pickle"这个**行为特征**，不是"是不是类"。
    （Day 23 的教训：断言实现细节会过期，断言行为活得久。）
    """
    print("\n[关 5] 挂增强的 collate 能不能传给子进程")
    import pickle
    from src.data.batch_aug import make_augment_collate

    gen = torch.Generator()
    gen.manual_seed(0)
    coll = make_augment_collate(2.0, 10.0, gen, seed=0)

    try:
        blob = pickle.dumps(coll)
        back = pickle.loads(blob)
        ok_old = True
        kind = type(coll).__name__
        print(f"    {kind}: pickle 成功，{len(blob)} 字节，往返后类型 "
              f"{type(back).__name__}")
    except Exception as e:                            # noqa: BLE001
        ok_old = False
        print(f"    pickle 失败：{type(e).__name__}: {e}")

    # 往返之后行为要一样（不只是"能存能读"）
    if ok_old:
        xs = torch.rand(16, 1, 28, 28)
        ys = torch.arange(16) % 10
        batch = [(xs[i], ys[i]) for i in range(16)]
        g1 = torch.Generator(); g1.manual_seed(7)
        coll_a = make_augment_collate(2.0, 10.0, g1, seed=0)
        xa, _ = coll_a(batch)
        g2 = torch.Generator(); g2.manual_seed(7)
        coll_b = make_augment_collate(2.0, 10.0, g2, seed=0)
        coll_b = pickle.loads(pickle.dumps(coll_b))
        xb, _ = coll_b(batch)
        d = (xa - xb).abs().max().item()
        print(f"    pickle 往返后输出最大差 {d:.2e}")
        assert d == 0.0, "pickle 往返改变了行为 —— 那子进程里的增强就和主进程不一样了"

    print(f"""
    为什么这一关重要：Windows 上 DataLoader 用 **spawn**，
    collate_fn 要靠 pickle 送进子进程。原来它是闭包
    （`def collate(batch): ...` 定义在 make_augment_collate 内部），
    pickle 无法重建捕获的局部变量，于是

        AttributeError: Can't pickle local object
        'make_augment_collate.<locals>.collate'

    **这个 bug 从 Day 24 就存在，但三天里一直 num_workers=0，从没触发过。**
    改成类之后就过了。
    """)
    assert ok_old, "collate 仍然不可 pickle，num_workers>0 会在第一个批就报错"
    print("    ✓ 通过（collate 可以被 pickle 到子进程）")
    return True


def gate6_worker_generator_duplication() -> bool:
    """关 6：W 个 worker 的 generator 起始状态相同 → 增强被重复。

    用"所有图都长一样"的数据集，这样"两个批的输入相同"时，
    输出是不是相同就只取决于增强参数。若 batch 0..W-1 的输入全同，
    输出也全同，就说明它们抽到了同一组参数。
    """
    print("\n[关 6] num_workers>0 时增强会不会重复")
    from src.data.batch_aug import build_train_loader

    n, size = 64, 28
    x = torch.ones(n, 1, size, size) * 0.5
    # 全同图：故意让输入可辨认。只要增强参数一样，输出就一样。
    x[:, :, 10:18, 10:18] = 1.0
    y = torch.arange(n) % 10
    ds = torch.utils.data.TensorDataset(x, y)

    def collect(workers, per_worker):
        loader = build_train_loader(ds, 8, seed=0, max_shift=3.0, max_angle=15.0,
                                    num_workers=workers, per_worker_gen=per_worker)
        return [xb.clone() for xb, _ in loader]

    ref = collect(0, False)
    print(f"    {'配置':<34}{'批数':>6}{'和 num_workers=0 逐位相同':>26}")
    print("    " + "-" * 68)
    print(f"    {'num_workers=0':<34}{len(ref):>6}{'（基准）':>24}")

    results = {}
    for workers, pw in ((2, False), (2, True)):
        got = collect(workers, pw)
        same = sum(int(torch.equal(a, b)) for a, b in zip(ref, got))
        tag = "per_worker_gen=False" if not pw else "per_worker_gen=True"
        results[(workers, pw)] = (len(got), same)
        print(f"    {f'num_workers={workers} {tag}':<34}{len(got):>6}"
              f"{f'{same}/{len(got)}':>24}")

    # 关键观察：per_worker_gen=False 时，前 W 个批应该彼此相同（参数被重复）
    dup = collect(2, False)
    first_two_same = torch.equal(dup[0], dup[1])
    print(f"\n    per_worker_gen=False 时 batch0 与 batch1 相同：{first_two_same}")
    print(f"""    这就是那个坑：输入全同时，batch0 和 batch1 逐位相同 ——
    说明两个 worker 抽到了**同一组**增强参数（generator 被原样复制）。

    而 per_worker_gen=True 之后：{results[(2, True)][1]}/{results[(2, True)][0]} 批与单进程相同。
    """)
    assert first_two_same, (
        "预期 per_worker_gen=False 时不同 worker 会抽到同一组参数，但实测不同。\n"
        "  可能原因：PyTorch 版本改了 generator 在 worker 间的处理方式。\n"
        "  那就把这条结论改成实测到的样子 —— 别照抄讲义。"
    )
    print("""    ✓ 通过（复制行为被实测确认）

    ⚠️ 由此得出今天最重要的否定结论：
       **「num_workers=0 与 4 逐位一致」做不到。**
       因为单条顺序的随机流被切成 W 条之后拼不回去。
       要么保住逐位（per_worker_gen=False，但增强重复），
       要么保住多样性（per_worker_gen=True，但失去可比性）。
       这不是调参能解决的，是随机流的组织方式决定的。
    """)
    return True


# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════

def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Day 27 数据加载提速自测")
    ap.add_argument("--train", action="store_true",
                    help="跑全部六关（含速度与多进程；默认只跑前两关）")
    args = ap.parse_args()

    print("=" * 72)
    print("Day 27 · 数据加载提速")
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    print("=" * 72)
    print("""
Day 24 说剩下的 4.07s 是"数据加载与拷贝"。今天先量了一遍：

    只调 __getitem__ × 54000     7.97s   ← 真正的大头
    collate + 增强 × 422         0.40s
    H2D 拷贝 × 422               0.09s

**瓶颈不在增强，在"把一张图变成张量"这一步绕了 PIL。**
这一天的价值一半来自这个发现本身。
""")

    gates = [
        ("关 1 u8 后端逐位", gate1_u8_bitwise),
        ("关 2 f32 后端逐位", gate2_f32_bitwise),
        ("关 3 端到端 422 批逐位", gate3_end_to_end),
        ("关 4 速度阶梯", gate4_speed_ladder),
        ("关 5 collate 可 pickle", gate5_collate_picklable),
        ("关 6 worker 间参数重复", gate6_worker_generator_duplication),
    ]
    if not args.train:
        print("（预览模式只跑前两关。加 --train 跑全六关）\n")
        gates = gates[:2]

    results, ok = [], True
    for name, fn in gates:
        try:
            good = bool(fn())
        except AssertionError as e:
            print(f"    ✗ 未通过：{e}")
            good = False
        except Exception as e:                        # noqa: BLE001
            print(f"    ✗ 意外错误：{type(e).__name__}: {e}")
            good = False
        results.append((name, good))
        ok = ok and good

    passed = sum(1 for _, g in results if g)
    print("\n" + "=" * 72)
    print(f"总计：{passed}/{len(results)} 关通过")
    for name, good in results:
        print(f"  {'✓' if good else '✗'}  {name}")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
