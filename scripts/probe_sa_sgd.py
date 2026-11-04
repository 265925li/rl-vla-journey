"""第 6 课 · 随机近似与随机梯度下降 —— 五个实验

配套讲义：docs/Ch6-随机近似与SGD.md
只做算术，不带训练循环。**讲义里出现的每个数字都应该出自本文件的输出。**

    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\probe_sa_sgd.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\probe_sa_sgd.py --only B

这一课是 54 讲里唯一的「纯数学工具课」：它不讲任何一个 RL 算法，
只讲三件事 —— 不存数据怎么求均值、不知道函数怎么求根、不知道期望怎么求梯度。
后面的 MC / TD / Q-Learning / Actor-Critic **全部**是它的应用。

五个实验：
  A 迭代均值：不存数据也能求平均；浮点上和「全量平均」差多少
  B Robbins-Monro：三种步长，一个收得准、一个一直抖、一个根本走不到
  C 常数步长的抖动幅度：∝ α，手算公式与实测对拍
  D BGD / MBGD / SGD：同样的样本预算，谁先到
  E 不变性：目标函数加常数 / 乘常数，轨迹怎么变
"""

from __future__ import annotations

import argparse

import numpy as np

# ── 全局常量（噪声、根的位置，全程固定）─────────────────────
ROOT = 3.0          # 实验 B/C 要解的根：g(w) = w - ROOT = 0
NOISE = 1.0         # 观测噪声标准差（σ）
N_BIG = 200_000     # A / C 的样本量


# ────────────────────────────────────────────────────────────
# A · 迭代均值：w_{k+1} = w_k + (1/k)(x_k − w_k)
# ────────────────────────────────────────────────────────────
def run_a(n: int = N_BIG, seed: int = 0) -> dict:
    """把 N 个数的平均值，用「只有 1 个变量」的方式算出来。

    这一步看起来平淡，但它是整门课的起点：
    **它是「全量平均」的唯一 O(1) 内存替代品**，而且它是后面所有
    增量式更新（MC 的 V ← V + α(G−V)、TD 的 δ、Q-Learning）的同一个式子。

    关键：a_k = 1/k 不是「随便选的衰减」，它是让 w_k **恰好等于**
    前 k 个数的算术平均的那个值。别的衰减都会得到一个偏的估计。
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(5.0, 2.0, n)

    w_batch = float(x.mean())          # 全量平均（要存 n 个数）

    w = 0.0                            # 迭代均值（只存 1 个数）
    for k in range(1, n + 1):
        w += (x[k - 1] - w) / k

    # 前几步：验证 w_1 就等于 x_1，且 w_k 真等于「前 k 个的均值」
    w2 = 0.0
    head = []
    for k in range(1, 6):
        w2 += (x[k - 1] - w2) / k
        head.append(w2)
    head_batch = [float(x[:k].mean()) for k in range(1, 6)]

    return {
        "n": n,
        "w_batch": w_batch,
        "w_iter": w,
        "diff": abs(w - w_batch),
        "head": head,
        "head_batch": head_batch,
        "head_diff": max(abs(a - b) for a, b in zip(head, head_batch)),
        "mem_batch_kb": n * 8 / 1024,
        "mem_iter_b": 8,
    }


# ────────────────────────────────────────────────────────────
# B · Robbins-Monro：只有带噪声的观测，怎么解 g(w)=0
# ────────────────────────────────────────────────────────────
def rm_run(a_fn, n_steps: int = 100_000, seed: int = 0,
           noise: float = NOISE, root: float = ROOT,
           tail_len: int = 10_000) -> tuple[float, np.ndarray]:
    """w_{k+1} = w_k − a_k · g̃(w_k, η_k)，g̃ 是 g 的一次带噪观测。

    这里 g(w) = w − ROOT，g̃ = g + ξ，ξ ~ N(0, noise²)。
    本来一步就到（w = ROOT），但**只能看到带噪声的 g̃**，
    所以必须靠 a_k → 0 把噪声平均掉。
    """
    rng = np.random.default_rng(seed)
    w = 0.0
    tail = []
    for k in range(1, n_steps + 1):
        g_tilde = (w - root) + rng.normal(0.0, noise)
        w -= a_fn(k) * g_tilde
        if k > n_steps - tail_len:
            tail.append(w)
    return w, np.asarray(tail)


def run_b() -> dict:
    """三种步长，三种命运。这是本课最该记住的一张表。

    理论（RM 收敛的两个条件，缺一不可）：
        Σ a_k = ∞        → 走得够远，能到根
        Σ a_k² < ∞       → 噪声被平均掉，最后不抖

    a_k = 1/k     ：两条都满足 → 收敛到根，抖动 → 0
    a_k = 常数 α   ：Σa_k=∞ 满足，Σa_k²=∞ 不满足 → **到得了根，但永远在抖**
    a_k = 1/k²    ：Σa_k < ∞（= π²/6 ≈ 1.645）不满足 → **走不到根就停了**
    """
    schedules = {
        "1/k": lambda k: 1.0 / k,
        "alpha=0.05 常数": lambda k: 0.05,
        "1/k^2": lambda k: 1.0 / (k * k),
    }
    out = {}
    for name, fn in schedules.items():
        w_final, tail = rm_run(fn)
        out[name] = {
            "final": w_final,
            "tail_mean": float(tail.mean()),
            "tail_std": float(tail.std()),
            "err_final": abs(w_final - ROOT),
            "err_mean": abs(float(tail.mean()) - ROOT),
            "sum_a": float(sum(fn(k) for k in range(1, 100_001))),
        }
    return out


# ────────────────────────────────────────────────────────────
# C · 常数步长：抖动幅度是 α 的什么函数
# ────────────────────────────────────────────────────────────
def steady_std(alpha: float, n_steps: int = 200_000, seed: int = 0,
               noise: float = NOISE, root: float = ROOT,
               tail_len: int = 50_000) -> float:
    _, tail = rm_run(lambda k: alpha, n_steps=n_steps, seed=seed,
                     noise=noise, root=root, tail_len=tail_len)
    return float(tail.std())


def run_c() -> dict:
    """手算 vs 实测：常数步长下的稳态标准差。

    在根附近令 e_k = w_k − ROOT，则
        e_{k+1} = e_k − α(e_k + ξ_k) = (1−α)e_k − αξ_k
    这是一个 AR(1)。稳态方差满足
        s² = (1−α)² s² + α²σ²
    ⟹ s² = α²σ² / (1 − (1−α)²) = α²σ² / (2α − α²) = ασ²/(2−α)

    所以 s = sqrt(ασ²/(2−α))。α 小的时候 ≈ σ·sqrt(α/2)。
    **这就是「学习率不能是常数」的全部理由** —— 不是玄学，是这个式子。
    """
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2]
    rows = []
    for a in alphas:
        s_emp = steady_std(a)
        s_th = float(np.sqrt(a * NOISE**2 / (2 - a)))
        rows.append({
            "alpha": a,
            "std_emp": s_emp,
            "std_th": s_th,
            "rel_err": abs(s_emp - s_th) / s_th,
        })
    # 顺带量一下「抖动 ∝ sqrt(alpha)」：std / sqrt(alpha) 应该大致恒定
    ratio = [r["std_emp"] / np.sqrt(r["alpha"]) for r in rows]
    return {"rows": rows, "ratio": ratio}


# ────────────────────────────────────────────────────────────
# D · BGD / MBGD / SGD：同样的样本预算，谁先到
# ────────────────────────────────────────────────────────────
def run_d(n: int = 10_000, n_epoch: int = 20, lr: float = 0.1,
          batch: int = 100, seed: int = 0) -> dict:
    """目标 J(w) = ½·E[(w − x)²]，真解 w* = E[x]。
    梯度：BGD 用 (w − x̄)，MBGD 用 (w − x̄_b)，SGD 用 (w − x_i)。

    **公平性声明**：三者用同一个 lr=0.1，这是「同一超参下的对比」，
    不是「各自调优后的对比」。真要各调各的，比较对象就变了。

    一个 epoch = 把 n 个样本各过一遍。此时：
        BGD  更新 1 次（每次成本 n）
        MBGD 更新 n/batch 次（每次成本 batch）
        SGD  更新 n 次（每次成本 1）
    总成本相同，**更新次数的差距是 batch 倍**。

    ⚠️ **只报「epoch 末那一个点的误差」是错的**（本项目已经栽过两次）：
    SGD 每一步都在抖，末尾那个点抖到哪纯属运气。所以这里报的是
    **每个 epoch 内所有时刻 |w−w*| 的均值**，以及最后的抖动幅度。
    均值曲线才代表「这段时间它一般有多准」。
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(5.0, 2.0, n)
    m = float(x.mean())

    out = {}
    for mode in ("bgd", "mbgd", "sgd"):
        w = 0.0
        mean_curve, std_curve = [], []
        for _ in range(n_epoch):
            errs = []
            if mode == "bgd":
                w -= lr * (w - m)
                errs.append(abs(w - m))
            else:
                order = rng.permutation(n)
                if mode == "sgd":
                    for idx in order:
                        w -= lr * (w - x[idx])
                        errs.append(abs(w - m))
                else:  # mbgd
                    for s in range(0, n, batch):
                        w -= lr * (w - x[order[s:s + batch]].mean())
                        errs.append(abs(w - m))
            e = np.asarray(errs)
            mean_curve.append(float(e.mean()))
            std_curve.append(float(e.std()))
        out[mode] = {
            "mean_curve": mean_curve,
            "std_curve": std_curve,
            "n_updates": n_epoch * (1 if mode == "bgd"
                                    else n // batch if mode == "mbgd" else n),
            "samples": n_epoch * n,
        }
    out["w_star"] = m
    out["n_epoch"] = n_epoch
    return out


# ────────────────────────────────────────────────────────────
# E · 不变性：常数项不影响轨迹，缩放等价于调学习率
# ────────────────────────────────────────────────────────────
def sgd_traj(x_seq: np.ndarray, lr: float, scale: float, shift: float,
             w0: float = 0.0) -> np.ndarray:
    """对 J(w) = scale · ½E[(w−x)²] + shift 做 SGD。

    注意 shift 根本没进梯度 —— 这就是「加常数没用」的全部证明。
    """
    w = w0
    out = np.empty_like(x_seq, dtype=np.float64)
    for i, xi in enumerate(x_seq):
        g = scale * (w - xi)          # shift 在这一行里消失了
        w -= lr * g
        out[i] = w
    return out


def run_e(n: int = 20_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.normal(5.0, 2.0, n)
    lr = 0.05

    base = sgd_traj(x, lr, scale=1.0, shift=0.0)
    shifted = sgd_traj(x, lr, scale=1.0, shift=100.0)
    scaled = sgd_traj(x, lr, scale=3.0, shift=0.0)
    lr3 = sgd_traj(x, lr * 3, scale=1.0, shift=0.0)

    return {
        "shift_max_diff": float(np.max(np.abs(base - shifted))),
        "shift_bitexact": bool(np.array_equal(base, shifted)),
        "scale_vs_lr3_max_diff": float(np.max(np.abs(scaled - lr3))),
        "base_tail_std": float(base[-2000:].std()),
    }


# ────────────────────────────────────────────────────────────
# 汇总
# ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="ABCDE",
                    help="只跑某几个实验，例如 --only B 或 --only CD")
    args = ap.parse_args()
    only = args.only.upper()

    if "A" in only:
        r = run_a()
        print("=" * 62)
        print("A · 迭代均值 vs 全量平均")
        print("=" * 62)
        print(f"  全量平均      : {r['w_batch']:.15f}")
        print(f"  迭代均值      : {r['w_iter']:.15f}")
        print(f"  差            : {r['diff']:.3e}   （浮点顺序不同，不是算法不同）")
        print(f"  前 5 步误差   : {r['head_diff']:.3e}")
        for k, (a, b) in enumerate(zip(r["head"], r["head_batch"]), 1):
            print(f"    k={k}  iter={a:.15f}  batch={b:.15f}")
        print(f"  内存：全量 {r['mem_batch_kb']:.0f} KB  vs  迭代 {r['mem_iter_b']} B"
              f"  （{r['mem_batch_kb'] * 1024 / r['mem_iter_b']:.0f} 倍）")

    if "B" in only:
        r = run_b()
        print("=" * 62)
        print("B · Robbins-Monro：三种步长，三种命运（根 = 3.0）")
        print("=" * 62)
        print(f"  {'步长':<16}{'Σa_k':>10}{'末尾均值':>12}{'末尾std':>10}{'|均值−根|':>12}")
        for name, v in r.items():
            print(f"  {name:<16}{v['sum_a']:>10.3f}{v['tail_mean']:>12.6f}"
                  f"{v['tail_std']:>10.6f}{v['err_mean']:>12.3e}")
        print("  读法：1/k 既是「能走到」又是「能停住」；")
        print("        α 常数能走到根，但 std 永远不降；")
        print("        1/k² 的 Σa_k 有限 → 走到半路就没步长了。")

    if "C" in only:
        r = run_c()
        print("=" * 62)
        print("C · 常数步长：抖动幅度 s = sqrt(ασ²/(2−α))")
        print("=" * 62)
        print(f"  {'α':>6}{'实测std':>12}{'理论std':>12}{'相对误差':>12}{'std/√α':>12}")
        for row, ratio in zip(r["rows"], r["ratio"]):
            print(f"  {row['alpha']:>6.3f}{row['std_emp']:>12.6f}"
                  f"{row['std_th']:>12.6f}{row['rel_err']:>12.2e}{ratio:>12.6f}")
        print("  最后一列近似常数 → 抖动 ∝ √α，不是 ∝ α。")

    if "D" in only:
        r = run_d()
        print("=" * 62)
        print(f"D · BGD / MBGD / SGD（同一 lr=0.1，{r['n_epoch']} 个 epoch）")
        print("=" * 62)
        print(f"  w* = {r['w_star']:.6f}   三者样本预算相同 = "
              f"{r['bgd']['samples']} 个（每 epoch {r['bgd']['samples'] // r['n_epoch']}）")
        print(f"  {'方法':<8}{'更新次数':>10}{'第1轮均值':>12}{'第5轮均值':>12}"
              f"{'末轮均值':>12}{'末轮抖动':>12}")
        for mode in ("bgd", "mbgd", "sgd"):
            v = r[mode]
            mc, sc = v["mean_curve"], v["std_curve"]
            print(f"  {mode.upper():<8}{v['n_updates']:>10}{mc[0]:>12.4f}"
                  f"{mc[4]:>12.4f}{mc[-1]:>12.4f}{sc[-1]:>12.4f}")
        print("  读法：同样的样本预算，更新次数差 batch 倍；")
        print("        SGD 一轮就把误差压下去，但末轮抖动卡在 sqrt(ασ²/2) 不再降；")
        print("        BGD 每次更新都很准，可惜一轮只有 1 次。")

    if "E" in only:
        r = run_e()
        print("=" * 62)
        print("E · 不变性")
        print("=" * 62)
        print(f"  J → J+100   轨迹最大差 : {r['shift_max_diff']:.3e}"
              f"   逐位相同: {r['shift_bitexact']}")
        print(f"  J → 3J  vs lr×3 最大差 : {r['scale_vs_lr3_max_diff']:.3e}")
        print(f"  基线末尾 std            : {r['base_tail_std']:.6f}")


if __name__ == "__main__":
    main()
