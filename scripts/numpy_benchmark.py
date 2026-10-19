# -*- coding: utf-8 -*-
"""Day 5 作业 A：验证「for 循环是性能杀手」。

在同一台机器上，用同一组数据，分别用三层 for 循环和 A @ B 做矩阵乘法，
把耗时和倍数打成一张表。

用法（在**项目根目录** `E:\\work Buddy\\强化学习\\rl-vla-journey` 下）：

    # 完整版，约 3 分 50 秒
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts/numpy_benchmark.py

    # 想先验证逻辑对不对，秒出结果
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts/numpy_benchmark.py --sizes 100 --runs 1

输出：
    控制台打表格；同时写进 outputs/numpy_benchmark.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ───────────────────────────────────────────────────────────────────
# 把项目根塞进 sys.path，这样无论从哪里运行，都能 import 到 src/
# 逻辑和 Day 2 的 env_info.py 一样：路径必须从 __file__ 反推
# ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger  # noqa: E402
from src.utils.paths import OUTPUTS, ensure_dir  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# 1. 两个待对比的实现
# ═══════════════════════════════════════════════════════════════════
def matmul_loop(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """三层 for 循环实现的矩阵乘法 (M,K) x (K,N) -> (M,N)。

    数学定义：C[i,j] = Σ_k A[i,k] * B[k,j]

    为什么把 a = A[i, k] 提到最内层循环外面：
        内层要循环 N 次，如果每次都写 A[i, k]，就是 N 次索引开销。
        提出来只取一次，是"手工做的最朴素优化"——即便如此，它照样慢一万倍。
    """
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N))

    for i in range(M):
        for k in range(K):
            a = A[i, k]
            for j in range(N):

                C[i,j] += a*B[k,j]


    return C


def matmul_blas(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """同一个数学运算，交给 NumPy（底层是 BLAS）。"""
    return A @ B


def time_it(fn, *args, runs: int = 3, **kwargs) -> float:
    """跑 runs 次，返回**平均**耗时（秒）。

    为什么取平均而不是取一次：
        第一次运行包含 cache 预热、BLAS 线程池初始化等一次性开销，
        单次计时波动可能有好几倍。取平均能压掉噪声。

    用 time.perf_counter() 而不是 time.time()：
        perf_counter 是单调递增的高精度计时器，不受系统时间调整影响。
    """
    times: list[float] = []
    for _ in range(runs):


        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)


    return sum(times) / len(times)


def format_row(n: int, t_loop: float, t_mm: float) -> str:
    """把一行结果格式化成固定宽度的表格行。"""
    return f"{n:>8} | {t_loop:>12.3f} | {t_mm:>12.6f} | {t_loop / t_mm:>9.0f}x"


# ═══════════════════════════════════════════════════════════════════
# 2. 主流程
# ═══════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="for 循环 vs 矩阵乘：性能对比")
    ap.add_argument("--sizes", default="200,500", help="逗号分隔的矩阵规模，默认 200,500")
    ap.add_argument("--runs", type=int, default=3, help="每个规模测几次取平均，默认 3")
    ap.add_argument("--seed", type=int, default=0, help="随机种子，保证每次数据一样")
    args = ap.parse_args()

    out = ensure_dir(OUTPUTS)
    logger = get_logger(__name__, log_file=out / "numpy_benchmark.txt")

    sizes = [int(s) for s in args.sizes.split(",")]
    logger.info(f"numpy {np.__version__} | sizes={sizes} | runs={args.runs}")

    header = f"{'M=K=N':>8} | {'for 循环(s)':>12} | {'A @ B(s)':>12} | {'倍数':>10}"
    logger.info(header)
    logger.info("-" * len(header))

    rng = np.random.default_rng(args.seed)

    # 先用 8x8 的小矩阵确认两版结果一致。
    # 这一步很重要：如果 TODO 1 写错了，在这里立刻就能发现，
    # 不用先在 500 那边白等 54 秒 × 3 次。
    _A = rng.standard_normal((8, 8))
    _B = rng.standard_normal((8, 8))
    assert np.allclose(matmul_loop(_A, _B), _A @ _B), "两版结果不一致，先检查 TODO 1"
    logger.info("小规模正确性检查通过（8x8 两版结果一致）")

    for n in sizes:
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))

        t_loop = time_it(matmul_loop, A, B, runs=args.runs)
        t_mm   = time_it(matmul_blas,  A, B, runs=args.runs)
        logger.info(format_row(n, t_loop, t_mm))

    logger.info("-" * len(header))
    logger.info("完成。结果同时写在 outputs/numpy_benchmark.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
