"""Day 26 的实测基准：amp 在这台机器上值不值。

六关（src/utils/amp_scaler.py）只证明**实现是对的**，
这个脚本才回答**有没有用**。

结论（1660 Ti / sm_75 / TU116，实测）：
    所有 batch size 下 amp 都更慢，1.56x ~ 4.15x，**没有交叉点**。
    省显存是真的（25%~52%），提速是假的。

用法：
    python scripts/bench_amp.py                  # 扫 5 个 batch size
    python scripts/bench_amp.py --bs 128 --repeat 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cnn import CNN  # noqa: E402

DEV = "cuda"


def bench(bs: int, mode: str, width: int = 32, warmup: int = 3, repeat: int = 12):
    """测一步训练（前向+反向+更新）的时间与峰值显存。

    mode:
        fp32  纯 fp32
        amp   autocast(fp16) + GradScaler

    ⚠️ 一定要 warmup：第一次调用包含 cuDNN 选算法、显存池扩容，
    不 warmup 测出来的是启动成本，不是稳态速度。
    ⚠️ 一定要 reset_peak_memory_stats：否则拿到的是累积峰值。
    """
    model = CNN(in_channels=1, channels=(16, width), hidden=128, out_dim=10,
                dropout=0.0, act="relu", bn_cls=nn.BatchNorm2d).to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if mode == "amp" else None
    xb = torch.randn(bs, 1, 28, 28, device=DEV)
    yb = torch.randint(0, 10, (bs,), device=DEV)

    def one():
        opt.zero_grad(set_to_none=True)
        if mode == "fp32":
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        else:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(repeat):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat, torch.cuda.max_memory_allocated() / 2**20


def main() -> int:
    ap = argparse.ArgumentParser(description="amp 值不值（Day 26）")
    ap.add_argument("--bs", type=int, nargs="*", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--repeat", type=int, default=12)
    ap.add_argument("--width", type=int, default=32, help="第二层卷积的通道数")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("需要 CUDA")
        return 1

    p = torch.cuda.get_device_properties(0)
    print("=" * 74)
    print(f"amp 值不值  |  {torch.cuda.get_device_name(0)}  sm_{p.major}{p.minor}")
    print(f"torch {torch.__version__}  repeat={args.repeat}")
    print("=" * 74)

    print(f"{'batch':>7}{'fp32':>10}{'amp':>10}{'amp/fp32':>10}"
          f"{'fp32 显存':>11}{'amp 显存':>11}{'省显存':>9}")
    print("-" * 70)
    faster, slower = 0, 0
    for bs in args.bs:
        try:
            t32, m32 = bench(bs, "fp32", args.width, repeat=args.repeat)
            tamp, mamp = bench(bs, "amp", args.width, repeat=args.repeat)
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>7}  显存不够，跳过")
            continue
        ratio = tamp / t32
        if ratio < 1.0:
            faster += 1
        else:
            slower += 1
        print(f"{bs:>7}{t32 * 1000:>9.1f}ms{tamp * 1000:>9.1f}ms"
              f"{ratio:>10.2f}x{m32:>9.0f}MB{mamp:>9.0f}MB{1 - mamp / m32:>9.0%}")

    print()
    if faster == 0 and slower > 0:
        print("结论：**没有一个 batch size 上 amp 更快**。")
        print("      不是'某个配置下不划算'，是这个 GPU + 这个模型根本不划算。")
        print("      原因：sm_75 的消费卡（TU116）没有 Tensor Core，")
        print("            fp16 的 GEMM 不走加速路径，只剩类型转换的额外开销。")
        print("      省显存是真的（这一列有意义），提速是假的。")
    elif faster > 0:
        print(f"结论：有 {faster} 个配置上 amp 更快。")
        print("      注意交叉点在哪 —— 它说明'值不值'依赖 batch size，")
        print("      所以不能只测一个点就下结论（Day 15 的教训）。")

    print("""
反过来说，什么情况下该开 amp（即使这里更慢）：
  · 模型大到 fp32 装不下显存 → 省下的 25%~52% 能让你跑得起来
  · 换到有 Tensor Core 的卡（RTX 20 系以上 / A100 / T4）
    在这台机器上，这两个前提都不成立，所以默认 amp=none。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
