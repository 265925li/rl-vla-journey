"""第 5 章 · 蒙特卡洛方法 —— 五个实验

配套讲义：docs/Ch05-蒙特卡洛方法.md
只做算术，不带训练循环。**讲义里出现的每个数字都应该出自本文件的输出。**

    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\probe_mc.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\probe_mc.py --only A

环境直接复用 Day 33 冻结的 GridWorld（5×5、无内墙、有风 slip=0.1）。
**接口不许改**：`P` 行和 1、`R` 是期望、`TERMINALS` 自环。

这一章要回答的唯一一个问题：**没有模型，怎么估 `V^π`？**
答案只有一个动作：**用采样平均代替期望**。剩下全是这个动作的后果。

五个实验：
  A 误差 ∝ 1/√N：MC 慢在哪，慢多少（实测斜率，不是「比较慢」）
  B 首次访问 vs 每次访问：都说无偏，差别在方差
  C MC Basic：**理论保证策略单调改进，但样本不够时会回退**
  D 样本效率：DP 0 次交互，MC 要多少次
  E ε-greedy 的代价：ε 扫一遍，差多少是有数的
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_gridworld import (  # noqa: E402
    build_gridworld,
    _reward,
    N_S,
    N_A,
    TERMINALS,
    GAMMA,
)

SLIP = 0.1
LIVE = [s for s in range(N_S) if s not in TERMINALS]
S0 = 0


# ────────────────────────────────────────────────────────────
# 环境与采样（全部基于 P，不额外假设任何东西）
# ────────────────────────────────────────────────────────────
def make_env():
    P, R = build_gridworld(slip=SLIP)
    cumP = np.cumsum(P, axis=2)          # 预计算，省掉每步重算
    return P, R, cumP


def sample_next(cum_row: np.ndarray, rng) -> int:
    """按累积分布抽一个后继状态。"""
    return int(np.searchsorted(cum_row, rng.random()))


def greedy_policy(P, R, V) -> np.ndarray:
    """用精确的 `V` 和模型算贪心策略（只在「需要真值」的地方用）。"""
    Q = R + GAMMA * P @ V
    pi = np.zeros((N_S, N_A))
    for s in range(N_S):
        pi[s, int(np.argmax(Q[s]))] = 1.0
    return pi


def exact_V(P, R, policy) -> np.ndarray:
    """解析解 `V^π = (I − γP^π)^{-1} R^π`。

    ⚠️ 这是**独立的真值来源**。Day 31 踩过「拿迭代输出当真值」的坑，
    所以这里一律用线性方程组，不用任何迭代。
    """
    Ppi = np.einsum("sa,sat->st", policy, P)
    Rpi = (policy * R).sum(axis=1)
    V = np.zeros(N_S)
    A = np.eye(len(LIVE)) - GAMMA * Ppi[np.ix_(LIVE, LIVE)]
    V[LIVE] = np.linalg.solve(A, Rpi[LIVE])
    return V


def run_episode(cumP, policy, start_s, rng, first_action=None,
                max_steps: int = 200):
    """从 `start_s` 出发按 `policy` 走，返回 [(s, a, r), ...]。"""
    traj = []
    s = start_s
    if first_action is None:
        a = int(np.searchsorted(np.cumsum(policy[s]), rng.random()))
    else:
        a = first_action
    for _ in range(max_steps):
        ns = sample_next(cumP[s, a], rng)
        r = _reward(ns)                  # 采样用**真实转移**的奖励，不是期望
        traj.append((s, a, r))
        if ns in TERMINALS:
            break
        s = ns
        a = int(np.searchsorted(np.cumsum(policy[s]), rng.random()))
    return traj


def discount_returns(traj) -> list[float]:
    """G_t 从后往前算，一次遍历。"""
    G = 0.0
    out = [0.0] * len(traj)
    for i in range(len(traj) - 1, -1, -1):
        G = traj[i][2] + GAMMA * G
        out[i] = G
    return out


def uniform_policy() -> np.ndarray:
    return np.full((N_S, N_A), 1.0 / N_A)


# ────────────────────────────────────────────────────────────
# A · 误差 ∝ 1/√N
# ────────────────────────────────────────────────────────────
def run_a(ns=(100, 400, 1600, 6400), n_seed: int = 8) -> dict:
    """从固定起点 `s0` 跑 N 条 episode，用两种方式估计，看误差怎么降。

    **方式 1（iid，理论上严格 1/√N）**：只用每条的 `G_0`，估 `V^π(s0)`。
    每条 episode 独立同分布 → 样本均值标准差 `= σ/√N` → 误差斜率 **−0.5**。

    **方式 2（全表，实际会偏平）**：顺手把所有途经状态都统计上。
    两个原因让它比 −0.5 平：
      ① 各状态被访问的次数**不同**（边缘状态样本少，是瓶颈）
      ② 同一 episode 内多次访问的 `G` **相关**（实验 B 测出冗余 2.81 倍）
    所以「误差 ∝ 1/√episode 数」这句话，**只对单状态 iid 估计成立**。
    """
    P, R, cumP = make_env()
    pi = uniform_policy()
    V_true = exact_V(P, R, pi)

    rows = []
    for N in ns:
        e_iid, e_tab, e_max = [], [], []
        for seed in range(n_seed):
            rng = np.random.default_rng(seed)
            g0_sum = 0.0
            sums = np.zeros(N_S)
            cnt = np.zeros(N_S)
            for _ in range(N):
                traj = run_episode(cumP, pi, S0, rng)
                rets = discount_returns(traj)
                g0_sum += rets[0]
                for (s, _a, _r), G in zip(traj, rets):
                    sums[s] += G
                    cnt[s] += 1
            e_iid.append(abs(g0_sum / N - V_true[S0]))
            with np.errstate(invalid="ignore", divide="ignore"):
                est = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)
            diff = np.abs(est[LIVE] - V_true[LIVE])
            e_tab.append(float(diff.mean()))
            e_max.append(float(diff.max()))
        # ⚠️ 用 RMSE 而不是 mean(|err|)：单个 |err| 的相对波动有 50%，
        #    8 个 seed 平均也压不住，会让斜率乱跳（3 个 seed 时测出过 −0.69）。
        rows.append({"N": N,
                     "err_iid": float(np.sqrt(np.mean(np.square(e_iid)))),
                     "err_tab": float(np.sqrt(np.mean(np.square(e_tab)))),
                     "err_max": float(np.sqrt(np.mean(np.square(e_max))))})

    logN = np.log([r["N"] for r in rows])
    fit = lambda k: float(np.polyfit(logN, np.log([r[k] for r in rows]), 1)[0])
    return {"rows": rows, "V_s0": float(V_true[S0]),
            "slope_iid": fit("err_iid"), "slope_tab": fit("err_tab"),
            "slope_max": fit("err_max"),
            "V_range": float(V_true[LIVE].max() - V_true[LIVE].min())}


# ────────────────────────────────────────────────────────────
# B · 首次访问 vs 每次访问
# ────────────────────────────────────────────────────────────
def mc_estimate(episodes, first_visit: bool) -> np.ndarray:
    sums = np.zeros(N_S)
    cnt = np.zeros(N_S)
    for traj in episodes:
        seen = set()
        rets = discount_returns(traj)
        for (s, _a, _r), G in zip(traj, rets):
            if first_visit and s in seen:
                continue
            seen.add(s)
            sums[s] += G
            cnt[s] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)


def run_b(n_ep: int = 200, n_rep: int = 80) -> dict:
    """两者都无偏，差别只在方差。

    首次访问：每个 episode 内每个状态只取第一次的 `G` → 样本**相互独立**（跨 episode）
    每次访问：同一 episode 内多次访问的 `G` 高度相关 → 有效样本更少 → 方差更大

    ⚠️ **30 次重复测不出差别**（实测方差比 1.02，全在噪声里）。
    这里加到 80 次重复，并同时报「访问冗余率」—— 总访问次数 / 首次访问次数。
    冗余率是每次访问法方差更大的**直接来源**：多出来的那些样本不是新的信息。
    """
    P, R, cumP = make_env()
    pi = uniform_policy()
    V_true = exact_V(P, R, pi)

    est_first, est_every = [], []
    redundancy = []
    for rep in range(n_rep):
        rng = np.random.default_rng(1000 + rep)
        episodes = []
        for _ in range(n_ep):
            start = LIVE[rng.integers(len(LIVE))]
            episodes.append(run_episode(cumP, pi, start, rng))
        est_first.append(mc_estimate(episodes, True))
        est_every.append(mc_estimate(episodes, False))

        tv = fv = 0
        for traj in episodes:
            seen = set()
            for (s, _a, _r) in traj:
                tv += 1
                if s not in seen:
                    seen.add(s)
                    fv += 1
        redundancy.append(tv / fv)

    a = np.asarray(est_first)[:, LIVE]
    b = np.asarray(est_every)[:, LIVE]
    return {
        "std_first": float(a.std(axis=0).mean()),
        "std_every": float(b.std(axis=0).mean()),
        "bias_first": float(np.abs(a.mean(axis=0) - V_true[LIVE]).mean()),
        "bias_every": float(np.abs(b.mean(axis=0) - V_true[LIVE]).mean()),
        "redundancy": float(np.mean(redundancy)),
        "n_rep": n_rep, "n_ep": n_ep,
    }


# ────────────────────────────────────────────────────────────
# C · MC Basic：样本不够，策略改进会回退
# ────────────────────────────────────────────────────────────
def mc_basic(P, R, cumP, pi, n_per_sa: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """一轮 MC Basic：对每个 (s,a) 用 N 次 exploring starts 估 q，然后贪心改进。

    **只估 q，不估 v。** 原因很简单：改进要 `argmax_a`，v 是一个数，选不出动作。
    没有模型时，从 v 推不出 q —— 这就是 MC 必须估 q 的全部理由。
    """
    Q = np.zeros((N_S, N_A))
    for s in LIVE:
        for a in range(N_A):
            tot = 0.0
            for _ in range(n_per_sa):
                traj = run_episode(cumP, pi, s, rng, first_action=a)
                # q(s,a) 就是从 (s,a) 出发那一刻的回报
                tot += discount_returns(traj)[0]
            Q[s, a] = tot / n_per_sa
    new_pi = np.zeros((N_S, N_A))
    for s in range(N_S):
        new_pi[s, int(np.argmax(Q[s]))] = 1.0
    return new_pi, Q


def run_c(n_per_sa=(5, 20, 50), rounds: int = 5, seed: int = 0) -> dict:
    """理论说「评估 → 贪心改进」一定不变差。**但那是 `q` 精确的时候。**

    `q` 是估的，估错就可能选错动作，选错就真的变差。
    这个实验要找「N 小到什么程度，回退开始变明显」。

    ⚠️ **数「回退几次」是不够的**：最优策略常常不唯一，
    采样噪声会让策略在并列的最优动作之间来回跳，`V` 只动 `1e-4` 量级。
    那不是改进失败，是同一个最优值的不同写法。所以要报**回退幅度**。
    """
    P, R, cumP = make_env()
    V_star = exact_V(P, R,
                     greedy_policy(P, R, _value_iteration(P, R)))

    out = {}
    for N in n_per_sa:
        rng = np.random.default_rng(seed)
        pi = uniform_policy()
        curve = [float(exact_V(P, R, pi)[S0])]
        for _ in range(rounds):
            pi, _Q = mc_basic(P, R, cumP, pi, N, rng)
            curve.append(float(exact_V(P, R, pi)[S0]))
        drops = [curve[i - 1] - curve[i]
                 for i in range(1, len(curve)) if curve[i] < curve[i - 1]]
        out[N] = {"curve": curve,
                  "n_drops": sum(1 for d in drops if d > 1e-6),
                  "max_drop": float(max(drops)) if drops else 0.0,
                  "final_gap": float(V_star[S0] - curve[-1])}
    out["V_star"] = float(V_star[S0])
    out["rounds"] = rounds
    return out


def _value_iteration(P, R, iters: int = 400):
    V = np.zeros(N_S)
    for _ in range(iters):
        V = np.max(R + GAMMA * P @ V, axis=1)
        V[list(TERMINALS)] = 0.0
    return V


# ────────────────────────────────────────────────────────────
# D · 样本效率：DP 0 次交互，MC 要多少
# ────────────────────────────────────────────────────────────
def run_d(n_per_sa: int = 100, seed: int = 0) -> dict:
    """同样的 5×5 GridWorld，两条路各要花多少。

    DP：每轮 `O(nS²·nA)` 次乘加，**环境交互 0 次**（模型在手）。
    MC：每轮 `|S|·|A|·N` 个 episode，每个 episode 要真走一遍。
    """
    P, R, cumP = make_env()
    rng = np.random.default_rng(seed)
    pi = uniform_policy()

    # 实测一个 episode 的平均长度
    lens = []
    for s in LIVE:
        for a in range(N_A):
            traj = run_episode(cumP, pi, s, rng, first_action=a)
            lens.append(len(traj))
    mean_len = float(np.mean(lens))

    n_sa = len(LIVE) * N_A
    dp_ops_per_iter = N_S * N_S * N_A
    mc_steps_per_round = n_sa * n_per_sa * mean_len

    return {
        "n_sa": n_sa, "mean_len": mean_len, "n_per_sa": n_per_sa,
        "dp_ops_per_iter": dp_ops_per_iter,
        "dp_iters": 9,                      # Day 33 实测：有吸收终态 → 9 轮精确收敛
        "mc_eps_per_round": n_sa * n_per_sa,
        "mc_steps_per_round": mc_steps_per_round,
        "ratio": mc_steps_per_round / (dp_ops_per_iter * 9),
    }


# ────────────────────────────────────────────────────────────
# E · ε-greedy 的代价
# ────────────────────────────────────────────────────────────
def epsilon_greedy_from_Q(Q: np.ndarray, eps: float) -> np.ndarray:
    pi = np.full((N_S, N_A), eps / N_A)
    for s in range(N_S):
        pi[s, int(np.argmax(Q[s]))] += 1.0 - eps
    return pi


def mc_epsilon_greedy(P, R, cumP, pi, n_per_sa: int, eps: float, rng):
    """用 ε-greedy 生成数据，也改进成 ε-greedy。
    改进后仍保证每个动作有正概率 —— 这就是「不用 exploring starts 也能覆盖」的正解。
    """
    Q = np.zeros((N_S, N_A))
    for s in LIVE:
        for a in range(N_A):
            tot = 0.0
            for _ in range(n_per_sa):
                traj = run_episode(cumP, pi, s, rng, first_action=a)
                tot += discount_returns(traj)[0]
            Q[s, a] = tot / n_per_sa
    return epsilon_greedy_from_Q(Q, eps), Q


def run_e(epss=(0.01, 0.05, 0.1, 0.2, 0.5), n_per_sa: int = 60,
          rounds: int = 3, seed: int = 0) -> dict:
    """ε 越大，探索越多，但最终策略离最优越远。**这是有价的，价格可以量出来。**"""
    P, R, cumP = make_env()
    V_star = exact_V(P, R, greedy_policy(P, R, _value_iteration(P, R)))

    rows = []
    for eps in epss:
        rng = np.random.default_rng(seed)
        pi = np.full((N_S, N_A), 1.0 / N_A)
        for _ in range(rounds):
            pi, _Q = mc_epsilon_greedy(P, R, cumP, pi, n_per_sa, eps, rng)
        gap = float(V_star[S0] - exact_V(P, R, pi)[S0])
        rows.append({"eps": eps, "gap": gap,
                     "eps_ratio": eps / (1.0 - eps),
                     "gap_per_ratio": gap / (eps / (1.0 - eps))})
    return {"rows": rows, "V_star": float(V_star[S0])}


# ────────────────────────────────────────────────────────────
# 汇总
# ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="ABCDE")
    args = ap.parse_args()
    only = args.only.upper()

    if "A" in only:
        r = run_a()
        print("=" * 66)
        print("A · MC 估 V^π：误差随 episode 数怎么降")
        print("=" * 66)
        print(f"  V^π 的取值范围（真值跨度）: {r['V_range']:.4f}")
        print(f"  {'N':>8}{'单状态|误差|':>16}{'全表平均|误差|':>18}{'全表max':>14}")
        for row in r["rows"]:
            print(f"  {row['N']:>8}{row['err_iid']:>16.6f}"
                  f"{row['err_tab']:>18.6f}{row['err_max']:>14.6f}")
        print(f"  斜率 单状态 iid   = {r['slope_iid']:>8.4f}   理论 −0.5000")
        print(f"  斜率 全表平均     = {r['slope_tab']:>8.4f}   偏平：访问不均 + 同 episode 内相关")
        print(f"  斜率 全表 max     = {r['slope_max']:>8.4f}   由最差状态决定，单次波动大，不作判据")
        print("  单状态那条 ∝ 1/√N ⟹ **误差减半要 4 倍 episode**")

    if "B" in only:
        r = run_b()
        print("=" * 66)
        print(f"B · 首次访问 vs 每次访问（{r['n_rep']} 次重复 × {r['n_ep']} episode）")
        print("=" * 66)
        print(f"  {'':<14}{'平均std':>12}{'平均|偏差|':>14}")
        print(f"  {'首次访问':<14}{r['std_first']:>12.6f}{r['bias_first']:>14.6f}")
        print(f"  {'每次访问':<14}{r['std_every']:>12.6f}{r['bias_every']:>14.6f}")
        print(f"  方差比（每次/首次）= {r['std_every'] / r['std_first']:.4f}")
        print(f"  访问冗余率（总访问/首次访问）= {r['redundancy']:.4f}")
        print(f"  ⟹ 每次访问多用了 {r['redundancy'] - 1:.1%} 的样本，")
        print(f"     但有效样本没那么多 —— 同一条 episode 内的 G 是相关的")

    if "C" in only:
        r = run_c()
        print("=" * 66)
        print(f"C · MC Basic：N 小到什么程度，策略改进开始回退（{r['rounds']} 轮）")
        print("=" * 66)
        print(f"  V*(s0) = {r['V_star']:.6f}")
        for N in (5, 20, 50):
            v = r[N]
            curve = " → ".join(f"{c:.4f}" for c in v["curve"])
            print(f"  N={N:<3}  回退 {v['n_drops']} 次   最大幅度 {v['max_drop']:.6f}")
            print(f"        {curve}")
            print(f"        末轮距最优 {v['final_gap']:.6f}")
        print("  读法：N 太小时「评估 → 改进」照样会变差，理论保证的前提是 q 精确")

    if "D" in only:
        r = run_d()
        print("=" * 66)
        print("D · 样本效率：DP vs MC（同一个 5×5 GridWorld）")
        print("=" * 66)
        print(f"  可达 (s,a) 对数        : {r['n_sa']}")
        print(f"  实测平均 episode 长度  : {r['mean_len']:.3f} 步")
        print(f"  DP  每轮乘加           : {r['dp_ops_per_iter']} 次"
              f"（{r['dp_iters']} 轮收敛 → {r['dp_ops_per_iter'] * r['dp_iters']}）")
        print(f"  DP  环境交互           : 0 步        ← 模型在手")
        print(f"  MC  每轮 episode       : {r['mc_eps_per_round']} 个"
              f"（N={r['n_per_sa']}）")
        print(f"  MC  每轮环境交互       : {r['mc_steps_per_round']:.0f} 步"
              f"   ← 没模型，只能真走")
        print("  这就是 model-free 的账单：同一个 5×5，DP 一步不用走，")
        print("  MC 一轮评估就要走十几万步。")

    if "E" in only:
        r = run_e()
        print("=" * 66)
        print("E · ε-greedy 的代价：ε 越大离最优越远")
        print("=" * 66)
        print(f"  V*(s0) = {r['V_star']:.6f}")
        print(f"  {'ε':>6}{'V*(s0)−V_ε(s0)':>18}{'ε/(1−ε)':>12}{'gap/比值':>12}")
        for row in r["rows"]:
            print(f"  {row['eps']:>6.2f}{row['gap']:>18.6f}"
                  f"{row['eps_ratio']:>12.4f}{row['gap_per_ratio']:>12.4f}")


if __name__ == "__main__":
    main()
