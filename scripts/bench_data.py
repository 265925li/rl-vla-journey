"""Day 27 的实测基准：数据侧到底花在哪、换后端能赚多少、num_workers 值不值。

三个问题，三张表：

    ① 数据侧拆解     __getitem__ / collate / 增强 / H2D 各占多少
    ② 后端速度阶梯   pil → u8 → f32
    ③ num_workers    spawn 成本 vs 并行收益（含 persistent_workers）

结论（实测）：
    ① 瓶颈是 Dataset.__getitem__ 里的 PIL 往返（7.97s），不是增强（0.40s）
    ② pil 12.1s → u8 5.6s → f32 3.4s（整轮含模型，端到端 3.5x）
    ③ num_workers>0 在 Windows 上慢 8 倍；persistent_workers 也只能缓到 3.5 倍慢
       而且它会改变数值（增强参数在 worker 间重复），不能用于对照实验

用法：
    python scripts/bench_data.py                # 跑全部三张表
    python scripts/bench_data.py --only 2       # 只跑后端阶梯
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, default_collate

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.batch_aug import build_train_loader, make_augment_collate  # noqa: E402
from src.data.fast_mnist import ALLOWED_BACKENDS, wrap_backend  # noqa: E402
from src.data.mnist import build_datasets  # noqa: E402
from src.utils.seed import make_generator  # noqa: E402


class Cfg:
    """最小配置：只给 build_datasets 需要的四个字段。"""

    data_dir = "data"
    val_size = 6000
    seed = 0


def _subset():
    train_set, val_set, test_set = build_datasets(Cfg())
    return train_set, val_set, test_set


def table1_breakdown() -> None:
    train_set, _, _ = _subset()
    N = len(train_set)
    print("\n① 数据侧拆解（54000 张，batch=128 → 422 批）")
    print("-" * 64)

    # 抽样再线性放大，省时间（逐条测 54000 次要 10 秒，没必要）
    step = 97
    t0 = time.perf_counter()
    for i in range(0, N, step):
        train_set[i]
    t_item = (time.perf_counter() - t0) / len(range(0, N, step)) * N
    print(f"  __getitem__ × {N}（PIL 解码 + ToTensor）   {t_item:>7.2f}s")

    cached = [train_set[i] for i in range(128)]
    t0 = time.perf_counter()
    for _ in range(422):
        default_collate(cached)
    t_collate = time.perf_counter() - t0
    print(f"  default_collate × 422                     {t_collate:>7.2f}s")

    coll = make_augment_collate(2.0, 10.0, torch.Generator().manual_seed(0), seed=0)
    t0 = time.perf_counter()
    for _ in range(422):
        coll(cached)
    t_aug = time.perf_counter() - t0
    print(f"  collate + 增强 × 422                      {t_aug:>7.2f}s"
          f"   （增强净 {(t_aug - t_collate) * 1000 / 422:.2f}ms/批）")

    xb = torch.randn(128, 1, 28, 28)
    t0 = time.perf_counter()
    for _ in range(422):
        xb.to("cuda")
    torch.cuda.synchronize()
    print(f"  H2D 拷贝 × 422                            "
          f"{time.perf_counter() - t0:>7.2f}s")

    print(f"""
  读法：__getitem__ 占了数据侧的绝大部分。而"增强"——
        那是 Day 24 花一整天优化的对象 —— 只有 0.4 秒。
        **量过之后再动手，和凭直觉动手，优化的对象完全不同。**
""")


def table2_backend_ladder() -> None:
    train_set, _, _ = _subset()
    print("\n② 后端速度阶梯（各跑一整轮，含打乱 + 增强）")
    print("-" * 64)
    rows = []
    for backend in ALLOWED_BACKENDS:
        ds = wrap_backend(train_set, backend)
        step = 97
        t0 = time.perf_counter()
        for i in range(0, len(ds), step):
            ds[i]
        per = (time.perf_counter() - t0) / len(range(0, len(ds), step))

        loader = build_train_loader(ds, 128, seed=0,
                                    max_shift=2.0, max_angle=10.0, num_workers=0)
        t0 = time.perf_counter()
        for _ in loader:
            pass
        ep = time.perf_counter() - t0
        mem = ds.mem_mb() if hasattr(ds, "mem_mb") else 0.0
        rows.append((backend, per, ep, mem))

    base = rows[0][2]
    print(f"  {'后端':>6}{'单条读取':>12}{'一轮(含增强)':>15}{'相对 pil':>11}{'额外内存':>11}")
    print("  " + "-" * 56)
    for backend, per, ep, mem in rows:
        print(f"  {backend:>6}{per * 1e6:>10.1f}µs{ep:>13.2f}s"
              f"{base / ep:>10.2f}x{mem:>9.0f}MB")

    print(f"""
  读法：三级都快，而且**数值逐位相同**（见 src/data/fast_mnist.py 关 1/2/3）。
        u8 零额外内存拿 {base / rows[1][2]:.2f}x，f32 用 {rows[2][3]:.0f}MB 再拿 {rows[1][2] / rows[2][2]:.2f}x。
        这是"纯提速"—— 没有精度权衡，只有内存权衡。
""")


def table3_workers() -> None:
    train_set, _, _ = _subset()
    print("\n③ num_workers 值不值（3 个 epoch 的总账）")
    print("-" * 64)

    def run(backend, workers, persistent, epochs=3):
        ds = wrap_backend(train_set, backend)
        gen = torch.Generator()
        gen.manual_seed(0)
        loader = DataLoader(
            ds, batch_size=128, shuffle=True,
            generator=make_generator(0),
            num_workers=workers,
            persistent_workers=(persistent and workers > 0),
            collate_fn=make_augment_collate(2.0, 10.0, gen, seed=0),
        )
        t0 = time.perf_counter()
        for _ in range(epochs):
            for _ in loader:
                pass
        return time.perf_counter() - t0

    print(f"  {'后端':>5}{'workers':>9}{'persistent':>12}{'3 epoch':>10}{'每 epoch':>10}")
    print("  " + "-" * 48)
    for backend in ("u8", "f32"):
        for w in (0, 2, 4):
            for p in ((False, True) if w else (False,)):
                t = run(backend, w, p)
                print(f"  {backend:>5}{w:>9}{str(p):>12}{t:>9.2f}s{t / 3:>9.2f}s")

    print("""
  读法：num_workers=0 永远最快。原因不是"并行没意义"，是
        **Windows 用 spawn，每个 worker 都要重新 import torch
        并把整个 dataset 序列化过去**（实测 ~5.1 秒/worker）。

        这个 epoch 只有 422 批，数据侧本来只要 1~3 秒，
        所以"启动一次的成本"远大于"干完这次活的收益"。
        persistent_workers=True 能摊薄启动成本，但也只从 8 倍慢降到 3.5 倍慢。

  判据（可以直接拿去用）：
        num_workers 值不值得
            = 每 epoch 能省下的时间 > spawn 成本 / 计划跑多少轮
        这里：能省 1.15s/epoch，spawn 要 20.6s（4 worker），
        所以要么轮数 > 18，要么单个 epoch 的数据量大 18 倍以上。

  ⚠️ 还有一个不是性能的问题：num_workers>0 会**改变数值**。
      collate 里那条 generator 被原样复制给每个 worker，
      于是 batch 0..W-1 抽到同一组增强参数（Day 27 关 6 实测）。
      所以它连"用于对照实验"都不够格 —— 除非开 per_worker_gen，
      而那又会失去与单进程的逐位可比性。
""")


def main() -> int:
    ap = argparse.ArgumentParser(description="数据侧基准（Day 27）")
    ap.add_argument("--only", type=int, nargs="*", default=[1, 2, 3],
                    help="1=拆解 2=后端阶梯 3=num_workers")
    args = ap.parse_args()

    print("=" * 64)
    print("Day 27 · 数据加载基准")
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    print("=" * 64)

    if 1 in args.only:
        table1_breakdown()
    if 2 in args.only:
        table2_backend_ladder()
    if 3 in args.only:
        table3_workers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
