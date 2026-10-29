"""Day 30 的原理验证：贝尔曼期望方程到底是不是一个线性方程组。

和 Day 29 一样，这个脚本不练编程。它只回答五个问题：

    实验 A  把动作边缘化：手算的 P^π / R^π 和代码算的一样吗
    实验 B  精确解代回贝尔曼方程右边，残差是 0 吗
    实验 C  两条独立的路（直接解 / 不动点迭代）求出同一个 V^π 吗，误差怎么衰减
    实验 D  **把 P 的方向写反**会得到什么（一个不报错的经典坑）
    实验 E  采样要多少条轨迹，才追得上一次直接解

实验 E 是本周和 W6 的接缝：它用数字说明「为什么有模型时要解方程，而不是去试」。

用法：
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts/probe_bellman.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts/probe_bellman.py --only D
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════
#  今天的 MDP：2 个状态、2 个动作。故意做得小，好手算。
#
#  状态 0 = 工位   状态 1 = 休息室
#  动作 0 = 保守   动作 1 = 冒险
#
#  P[s, a, s'] = 在 s 做 a 之后到 s' 的概率
#  R[s, a]     = 在 s 做 a 之后立刻拿到的奖励
# ═══════════════════════════════════════════════════════════════════

P = np.array([
    [[0.9, 0.1],      # 工位 · 保守
     [0.5, 0.5]],     # 工位 · 冒险
    [[0.2, 0.8],      # 休息室 · 保守
     [0.0, 1.0]],     # 休息室 · 冒险
])

R = np.array([
    [1.0, 3.0],
    [2.0, 0.0],
])

# 策略 π(a|s)：两处都各 50% 概率。选均匀是为了让手算的数字整齐。
PI = np.array([
    [0.5, 0.5],
    [0.5, 0.5],
])

GAMMA = 0.5


# ═══════════════════════════════════════════════════════════════════
#  核心两个函数：边缘化、精确解
# ═══════════════════════════════════════════════════════════════════

def marginalize(pi: np.ndarray, P: np.ndarray, R: np.ndarray):
    """把动作维消掉，得到「策略下的」转移矩阵和奖励向量。

        P^π(s, s') = Σ_a π(a|s) P(s, a, s')
        R^π(s)     = Σ_a π(a|s) R(s, a)

    注意这两个式子的**求和指标是 a**，而且 π 的下标是 (s, a)。
    策略是「在状态 s 下选动作 a 的概率」，所以它对每个状态单独给一组分布。
    """
    P_pi = np.einsum("sa,sat->st", pi, P)
    R_pi = np.einsum("sa,sa->s", pi, R)
    return P_pi, R_pi


def exact_values(gamma: float, P_pi: np.ndarray, R_pi: np.ndarray) -> np.ndarray:
    """V^π = (I − γP^π)⁻¹ R^π。这是本周唯一能拿到精确答案的地方。"""
    n = P_pi.shape[0]
    return np.linalg.solve(np.eye(n) - gamma * P_pi, R_pi)


def bellman_rhs(V: np.ndarray, gamma: float, P_pi: np.ndarray, R_pi: np.ndarray) -> np.ndarray:
    """贝尔曼方程的右边：R^π + γP^π V。"""
    return R_pi + gamma * P_pi @ V


# ═══════════════════════════════════════════════════════════════════
#  实验 A：边缘化，手算 vs 代码
# ═══════════════════════════════════════════════════════════════════

def exp_a():
    print("\n[实验 A] 把动作边缘化掉（手算 vs 代码）")

    P_pi, R_pi = marginalize(PI, P, R)

    # 下面四行就是纸上手算的过程，逐项写出来，方便你对着核
    hand = np.array([
        [0.5 * P[0, 0, 0] + 0.5 * P[0, 1, 0], 0.5 * P[0, 0, 1] + 0.5 * P[0, 1, 1]],
        [0.5 * P[1, 0, 0] + 0.5 * P[1, 1, 0], 0.5 * P[1, 0, 1] + 0.5 * P[1, 1, 1]],
    ])
    hand_R = np.array([
        0.5 * R[0, 0] + 0.5 * R[0, 1],
        0.5 * R[1, 0] + 0.5 * R[1, 1],
    ])

    print("    P^π 手算：")
    print(f"      行 0: 0.5×{P[0, 0, 0]} + 0.5×{P[0, 1, 0]} = {hand[0, 0]:.2f}"
          f"   |  0.5×{P[0, 0, 1]} + 0.5×{P[0, 1, 1]} = {hand[0, 1]:.2f}")
    print(f"      行 1: 0.5×{P[1, 0, 0]} + 0.5×{P[1, 1, 0]} = {hand[1, 0]:.2f}"
          f"   |  0.5×{P[1, 0, 1]} + 0.5×{P[1, 1, 1]} = {hand[1, 1]:.2f}")
    print(f"    R^π 手算：{hand_R}")

    print(f"\n    代码 einsum 的结果：\n{P_pi}")
    print(f"    R^π = {R_pi}")

    d_p = np.abs(hand - P_pi).max()
    d_r = np.abs(hand_R - R_pi).max()
    print(f"\n    P^π 最大差 {d_p:.3e}，R^π 最大差 {d_r:.3e}")

    print(f"""
    两行的和都是 1（{P_pi[0].sum():.15f} / {P_pi[1].sum():.15f}）——
    这是必须自己检查的一条：**P^π 也必须是随机矩阵**。
    如果某一行和不是 1，说明 π 没归一，或者边缘化时下标搞错了。
    """)
    return d_p < 1e-15 and d_r < 1e-15


# ═══════════════════════════════════════════════════════════════════
#  实验 B：精确解必须满足贝尔曼方程
# ═══════════════════════════════════════════════════════════════════

def exp_b():
    print(f"\n[实验 B] 精确解代回贝尔曼方程，残差是多少（γ={GAMMA}）")

    P_pi, R_pi = marginalize(PI, P, R)
    V = exact_values(GAMMA, P_pi, R_pi)
    rhs = bellman_rhs(V, GAMMA, P_pi, R_pi)
    resid = np.abs(V - rhs).max()

    print(f"    V^π        = [{V[0]:.9f}, {V[1]:.9f}]")
    print(f"    R^π+γP^πV  = [{rhs[0]:.9f}, {rhs[1]:.9f}]")
    print(f"    最大残差   = {resid:.3e}")

    # 顺便把 V 展开成分数，讲课要用
    from fractions import Fraction
    fr = [Fraction(v).limit_denominator(1000) for v in V]
    print(f"    分数形式   = [{fr[0]}, {fr[1]}]   ← 手算应该得到这两个分数")

    print(f"""
    **残差为 0 不是巧合，是定义。** V^π 就是那个让 V = R^π + γP^πV 成立的向量。
    反过来说，**你要验自己推的贝尔曼方程对不对，最快的办法就是这个残差**：
    把解代回右边，看是不是原样回来。Day 31 的最优方程同样这么验。
    """)
    return resid < 1e-12


# ═══════════════════════════════════════════════════════════════════
#  实验 C：两条独立的路
# ═══════════════════════════════════════════════════════════════════

def exp_c():
    print(f"\n[实验 C] 直接解 vs 不动点迭代（γ={GAMMA}）")

    P_pi, R_pi = marginalize(PI, P, R)
    V_exact = exact_values(GAMMA, P_pi, R_pi)

    # 不动点迭代 = 策略评估：拿贝尔曼方程右边当作"下一版 V"
    V = np.zeros(2)
    traj = [V.copy()]
    for _ in range(200):
        V = bellman_rhs(V, GAMMA, P_pi, R_pi)
        traj.append(V.copy())
        if np.abs(V - V_exact).max() < 1e-12:
            break

    errs = [np.abs(v - V_exact).max() for v in traj]
    need = next((k for k, e in enumerate(errs) if e < 1e-12), None)

    print(f"    {'轮':>4}{'V[0]':>14}{'V[1]':>14}{'与精确解的误差':>18}{'误差比':>10}")
    print("    " + "-" * 60)
    for k in (0, 1, 2, 3, 5, 10, 20, 30, 40):
        if k >= len(traj):
            continue
        ratio = errs[k] / errs[k - 1] if k >= 1 and errs[k - 1] > 0 else float("nan")
        print(f"    {k:>4}{traj[k][0]:>14.9f}{traj[k][1]:>14.9f}{errs[k]:>18.3e}{ratio:>10.4f}")

    theory = int(np.ceil(np.log(1e-12 / errs[0]) / np.log(GAMMA)))
    print(f"""
    收敛到 1e-12 实际用了 {need} 轮；理论上界 k ≥ log(1e-12/‖e₀‖)/log(γ) = {theory} 轮。
    实测 ≤ 理论，符合 `‖e_k‖ ≤ γ^k‖e_0‖`。

    **看「误差比」那一列：每一轮误差都乘上不超过 γ 的系数，越接近 γ={GAMMA}。**
    这就是 Day 32 要证的压缩映射，今天先用数字看见它 ——
    注意它是「不超过 γ」而不是「等于 γ」：前几轮只有 0.44~0.47，
    因为误差在两个分量上的比例在变，∞-范数取的是最大那个。

    两条路的结果：
      直接解   V = {V_exact}
      迭代 {need} 轮 V = {traj[need]}
      差        {np.abs(traj[need] - V_exact).max():.3e}
    """)
    return need is not None and np.abs(traj[need] - V_exact).max() < 1e-9


# ═══════════════════════════════════════════════════════════════════
#  实验 D：把 P 的方向写反
# ═══════════════════════════════════════════════════════════════════

def exp_d():
    print(f"\n[实验 D] P 的方向写反会怎样（γ={GAMMA}）")

    P_pi, R_pi = marginalize(PI, P, R)
    V_ok = exact_values(GAMMA, P_pi, R_pi)

    # 错误写法：把 P^π 当成「列是 s、行是 s'」，等价于解 (I − γP^T)V = R
    n = P_pi.shape[0]
    V_bad = np.linalg.solve(np.eye(n) - GAMMA * P_pi.T, R_pi)

    # 另一种常见错法：矩阵乘写成 V @ P
    V_bad2 = R_pi + GAMMA * (V_ok @ P_pi)
    V_bad2 = np.linalg.solve(np.eye(n) - GAMMA * P_pi.T, R_pi)

    print(f"    正确的 (I − γP^π)V = R^π        V = [{V_ok[0]:.9f}, {V_ok[1]:.9f}]")
    print(f"    写反成 (I − γP^T)V = R^π        V = [{V_bad[0]:.9f}, {V_bad[1]:.9f}]")
    print(f"    两个答案差                      {np.abs(V_ok - V_bad).max():.9f}")

    # 把错的那个解代回正确的方程，看残差有多大
    resid_bad = np.abs(exp_b_resid(V_bad, GAMMA, P_pi, R_pi)).max()
    resid_ok = np.abs(exp_b_resid(V_ok, GAMMA, P_pi, R_pi)).max()
    print(f"\n    把「写反的解」代回正确的贝尔曼方程，残差 = {resid_bad:.6f}")
    print(f"    把「正确的解」代回正确的贝尔曼方程，残差 = {resid_ok:.3e}")

    print(f"""
    **这个错误不会报错，也不会 nan，只会给你一个"看起来很像"的答案。**
    两个解相差 {np.abs(V_ok - V_bad).max():.6f}（相对 {np.abs(V_ok - V_bad).max() / np.abs(V_ok).max() * 100:.1f}%），
    曲线照样画得出来，结论照样写。

    为什么容易写错：数学书写 V(s) = R(s) + γ Σ_s' P(s'|s) V(s')，
    `s'` 在**里面**被求和掉；而 numpy 里 `P @ V` 是「行乘列」，
    `P[s, s']` 恰好是「行 = s、列 = s'」。两者对得上 ——
    但只要你想成「V 是行向量」，就会写成 `V @ P`，等价于用了 `P^T`。

    **唯一的防线是实验 B 那个残差**：写完就代回去看残差是不是 0。
    这是本周每个方程都要做的一步，别省。
    """)
    return np.abs(V_ok - V_bad).max() > 1e-6 and resid_bad > 1e-6


def exp_b_resid(V, gamma, P_pi, R_pi):
    return V - bellman_rhs(V, gamma, P_pi, R_pi)


# ═══════════════════════════════════════════════════════════════════
#  实验 E：采样要多少条轨迹
# ═══════════════════════════════════════════════════════════════════

def _sample_returns(n_traj: int, horizon: int, gamma: float, seed: int = 0) -> np.ndarray:
    """从状态 0 出发，按 π 走 horizon 步，返回每条轨迹的折扣回报。

    向量化写法：N 条轨迹同时推进，每一步只做几次数组运算。
    （逐条 for 循环在这个 N 上要跑几十秒，而向量化只要毫秒 —— 这不是"优化"，
      是因为采样本来就是独立的，天然可并行。）
    """
    rng = np.random.default_rng(seed)
    s = np.zeros(n_traj, dtype=int)          # 所有轨迹都从状态 0 出发
    g = np.zeros(n_traj)
    disc = 1.0
    for _ in range(horizon):
        # 按 π(·|s) 抽动作：π[s,0] 是"选动作 0"的概率
        a = (rng.random(n_traj) < PI[s, 0]).astype(int)
        g += disc * R[s, a]
        disc *= gamma
        # 按 P(s'=1 | s, a) 抽下一个状态
        s = (rng.random(n_traj) < P[s, a, 1]).astype(int)
    return g


def exp_e():
    print(f"\n[实验 E] 采样要多少条轨迹才追上一次直接解（γ={GAMMA}）")

    P_pi, R_pi = marginalize(PI, P, R)
    V0 = exact_values(GAMMA, P_pi, R_pi)[0]
    horizon = 60      # γ^60 ≈ 8.7e-19，截断偏差可忽略

    # 先量"单条轨迹的回报波动有多大" —— 采样精度的分母就是它
    big = _sample_returns(200_000, horizon, GAMMA, seed=12345)
    sigma = float(big.std(ddof=1))

    print(f"    精确值 V^π(0) = {V0:.9f}（解一次 2×2 方程，0 条轨迹）")
    print(f"    单条轨迹回报的 std σ = {sigma:.6f}（20 万条估出来）")
    print()
    print(f"    {'N':>8}{'重复':>6}{'估计均值':>13}{'偏差':>12}{'估计的 std':>14}"
          f"{'实测 std×√N':>14}{'理论 σ':>10}")
    print("    " + "-" * 78)

    repeats = 40
    rows = []
    for N in (10, 100, 1000, 10000):
        ests = np.array([_sample_returns(N, horizon, GAMMA, seed=s).mean()
                         for s in range(repeats)])
        sd = float(ests.std(ddof=1))
        bias = float(ests.mean() - V0)
        rows.append((N, sd, bias))
        print(f"    {N:>8}{repeats:>6}{ests.mean():>13.6f}{bias:>12.6f}"
              f"{sd:>14.6f}{sd * np.sqrt(N):>14.4f}{sigma:>10.4f}")

    # 用最大那个 N 检验 "std×√N ≈ σ"
    N_last, sd_last, _ = rows[-1]
    ratio = sd_last * np.sqrt(N_last) / sigma

    for T in (10, 20, 30):
        print(f"    长度 {T} 的轨迹穷举：2^{T} = {2 ** T:,} 条（还没算状态随机性）")

    print(f"""
    三条读数，一条比一条重要：

    ① **估计值的 std 按 1/√N 缩小。** 看「实测 std×√N」那一列，
       基本稳定在 σ ≈ {sigma:.4f} 附近（最大那个 N 上是 σ 的 {ratio:.3f} 倍）。
       所以：**采样 100 倍轨迹只换来 10 倍精度。** 这是采样的天花板，不是实现问题。

    ② **偏差可以忽略，波动不能。** 「偏差」那一列最大也只有 {max(abs(r[2]) for r in rows) * 100:.2f}%，
       而且符号在变，没有系统性方向 —— 采样估计是**无偏但高方差**的。
       无偏不代表单个估计能用：N=10 时 std 有 {rows[0][1]:.3f}，而真值才 {V0:.3f}。

    ③ **解方程一次就是精确的。** 长度 30 的轨迹要穷举 {2 ** 30:,} 条；
       解 2×2 方程一次 O(n³)，误差 0。

    这就是「有模型」和「没模型」的全部差别：
      **有 P 和 R → 解方程（本周 / DP）；没有 → 采样（W6 / MC 与 TD）。**
    采样不是更"现代"的方法，它是没模型时**唯一**能做的事。
    今天先把有模型这条路走通，W6 才有对拍的对象 ——
    那时你会拿 MC/TD 的估计去和今天的精确解比，看它差多少、随 N 怎么缩。
    """)
    return 0.7 < ratio < 1.4


# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Day 30 原理验证")
    ap.add_argument("--only", default=None, help="只跑某个实验，如 A / D")
    args = ap.parse_args()

    print("=" * 70)
    print("Day 30 · 贝尔曼期望方程：它到底是不是一个线性方程组")
    print("=" * 70)
    print(f"""
环境（2 状态 2 动作，γ={GAMMA}，策略 π 两处都各 50%）：

    P[0,0,:] = {P[0, 0]}      R[0,0] = {R[0, 0]}
    P[0,1,:] = {P[0, 1]}      R[0,1] = {R[0, 1]}
    P[1,0,:] = {P[1, 0]}      R[1,0] = {R[1, 0]}
    P[1,1,:] = {P[1, 1]}      R[1,1] = {R[1, 1]}
""")

    plan = [
        ("A", "边缘化", exp_a),
        ("B", "残差为 0", exp_b),
        ("C", "两条路同一个解", exp_c),
        ("D", "P 的方向写反", exp_d),
        ("E", "采样 vs 解方程", exp_e),
    ]

    results = []
    for key, name, fn in plan:
        if args.only and args.only.upper() != key:
            continue
        try:
            ok = fn()
        except Exception as exc:                       # noqa: BLE001
            print(f"    ✗ 实验 {key} 抛异常：{exc}")
            ok = False
        results.append((key, name, ok))

    if not results:
        print(f"没有匹配的实验：--only {args.only}")
        return 1

    print("\n" + "=" * 70)
    print(f"总计：{sum(1 for _, _, ok in results if ok)}/{len(results)}")
    for key, name, ok in results:
        print(f"  {'✓' if ok else '✗'}  实验 {key}  {name}")
    print("=" * 70)
    return 0 if all(ok for _, _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
