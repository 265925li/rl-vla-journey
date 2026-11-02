"""Day 31 的原理验证：把一个 Σ 换成 max，到底变了什么。

和前两天一样，这个脚本不练编程。它只回答五个问题：

    实验 A  `max` 能不能从 `Σ` 外面提出来（一个「看着像分配律」的错）
    实验 B  `V*` 是不是某个线性方程的解，以及它和「所有 V^π 的最大值」什么关系
    实验 C  `V*` 唯一吗（三个完全不同的初值，值迭代收敛到同一点吗）
    实验 D  `V*` 唯一，但最优策略唯一吗（一个反推构造的 tie 环境）
    实验 E  从 `V*` 出发做一步贪心，能不能直接拿到最优策略

实验 E 是明天的种子：它就是策略迭代里「策略改进」那一步。

用法：
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts/probe_optimal.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts/probe_optimal.py --only D
"""
from __future__ import annotations

import argparse
import itertools
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

np.set_printoptions(precision=9, suppress=True, linewidth=110)


# ═══════════════════════════════════════════════════════════════════
#  三个环境。都小到能手算，但各自负责不同的事。
#
#  P[s, a, s'] = 在 s 做 a 之后到 s' 的概率
#  R[s, a]     = 在 s 做 a 之后立刻拿到的奖励
# ═══════════════════════════════════════════════════════════════════

# ── 环境一：CHAIN ── 昨天那个 2 状态 2 动作，用来和 Day 30 接上 ──
#  状态 0 = 工位   状态 1 = 休息室
#  动作 0 = 保守   动作 1 = 冒险
CHAIN_P = np.array([
    [[0.9, 0.1],
     [0.5, 0.5]],
    [[0.2, 0.8],
     [0.0, 1.0]],
])
CHAIN_R = np.array([
    [1.0, 3.0],
    [2.0, 0.0],
])
CHAIN_GAMMA = 0.5

# ── 环境二：TIE ── 奖励是反推出来的，目的只有一个：制造一个精确的平局 ──
#  先定 V* = [5, 2]、γ = 0.5，然后倒着解出 R，让 s1 的两个动作 Q 值恰好相等。
#
#  s1 · a0：转移到 s0，Q = R10 + 0.5·V*(0) = R10 + 2.5 = 2  →  R10 = -0.5
#  s1 · a1：停在 s1，Q = R11 + 0.5·V*(1) = R11 + 1.0 = 2    →  R11 =  1.0
#  **两个动作的 Q 都是 2.0，但一个往 s0 走、一个留在 s1。**
#
#  s0 · a0：Q = R00 + 0.5·(0.5·5 + 0.5·2) = R00 + 1.75 = 5  →  R00 = 3.25（最优）
#  s0 · a1：Q = R01 + 0.5·(0.2·5 + 0.8·2) = R01 + 1.30        →  R01 = 1.0 → 2.3（次优）
TIE_P = np.array([
    [[0.5, 0.5],
     [0.2, 0.8]],
    [[1.0, 0.0],
     [0.0, 1.0]],
])
TIE_R = np.array([
    [3.25, 1.0],
    [-0.5, 1.0],
])
TIE_GAMMA = 0.5

# ── 环境三：HUB ── 3 状态 3 动作，用来穷举 27 个确定性策略 ──
#  s0 = 岔路口，s1 = 金矿（自循环，每步 8），s2 = 陷阱（自循环，0）
#  在 s0 上：a0 去金矿但立刻没奖励；a1 去陷阱但立刻拿 10。这正好是
#  「奖励大的动作」和「未来好的动作」分家的地方 —— 实验 A 需要它。
HUB_P = np.array([
    [[0.0, 1.0, 0.0],     # s0 · a0 → 金矿
     [0.0, 0.0, 1.0],     # s0 · a1 → 陷阱，但立刻 +10
     [0.5, 0.5, 0.0]],    # s0 · a2 → 一半一半
    [[0.0, 1.0, 0.0],     # s1 · a0 → 留在金矿
     [0.0, 0.0, 1.0],     # s1 · a1 → 跳进陷阱
     [0.0, 1.0, 0.0]],    # s1 · a2 → 和 a0 完全一样（冗余动作）
    [[0.0, 0.0, 1.0],     # s2 · a0 → 困在陷阱
     [0.0, 0.0, 1.0],     # s2 · a1 → 困在陷阱，还倒扣
     [0.0, 0.0, 1.0]],    # s2 · a2 → 和 a0 完全一样
])
HUB_R = np.array([
    [0.0, 10.0, 5.0],
    [8.0, -5.0, 8.0],
    [0.0, -1.0, 0.0],
])
HUB_GAMMA = 0.9


# ═══════════════════════════════════════════════════════════════════
#  核心函数。只有四个，都很短。
# ═══════════════════════════════════════════════════════════════════

def q_values(V: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """给定一个 V（长度 nS），算每个 (s, a) 的「一步前瞻」值。

        Q(s, a) = R(s, a) + γ Σ_{s'} P(s, a, s') V(s')

    形状是 (nS, nA)。这个函数是今天所有实验的地基 ——
    贝尔曼最优方程就是 `V*(s) = max_a Q(s, a)`，一字不差。
    """
    return R + gamma * (P @ V)          # P 是 (nS, nA, nS)，@ V 得 (nS, nA)


def bellman_optimal(V: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """贝尔曼最优算子 T*：把 V 推进一轮。

        (T* V)(s) = max_a [ R(s,a) + γ Σ_{s'} P(s,a,s') V(s') ]

    和 Day 30 的 (T^π V)(s) = Σ_a π(a|s) [...] 只差一个东西：
    **Σ_a π(a|s) 换成了 max_a**。就这一个符号。
    """
    return q_values(V, P, R, gamma).max(axis=1)


def value_iteration(P, R, gamma, V0=None, tol=1e-12, max_iter=20000):
    """反复套 T*，直到不动。返回 (V, 轮数, 每轮误差列表)。"""
    nS = P.shape[0]
    V = np.zeros(nS) if V0 is None else np.array(V0, dtype=float)
    errs = []
    for k in range(max_iter):
        V_new = bellman_optimal(V, P, R, gamma)
        err = np.abs(V_new - V).max()
        errs.append(err)
        V = V_new
        if err < tol:
            return V, k + 1, errs
    return V, max_iter, errs


def policy_matrices(policy: tuple, P, R):
    """把确定性策略压成 (P^π, R^π)，形状 (nS, nS) 和 (nS,)。"""
    idx = np.arange(P.shape[0])
    return P[idx, list(policy), :], R[idx, list(policy)]


def policy_values(policy: tuple, P, R, gamma) -> np.ndarray:
    """给定确定性策略（每状态一个动作编号），解出它的 V^π。

    这一步用的是**昨天**的东西：策略定了，问题就退化成线性方程组
        (I − γ P^π) V = R^π
    所以今天「V* 不是线性方程的解」这句话要小心说 ——
    对**已知**策略它仍然是线性方程；难的是不知道哪个策略最优。
    """
    P_pi, R_pi = policy_matrices(policy, P, R)
    nS = P.shape[0]
    return np.linalg.solve(np.eye(nS) - gamma * P_pi, R_pi)


def dedupe_policies(policies, P, R):
    """按 (P^π, R^π) 去重，只留**实质不同**的策略。

    为什么需要这一步：如果一个状态上有两个动作的转移和奖励完全一样
    （冗余动作），那「动作编号不同的策略」其实是同一个策略的两种写法。
    不区分的话，会把「冗余」误当成「不唯一」。
    """
    groups = {}
    for pol in policies:
        P_pi, R_pi = policy_matrices(pol, P, R)
        key = (tuple(np.round(P_pi.ravel(), 12)), tuple(np.round(R_pi, 12)))
        groups.setdefault(key, []).append(pol)
    return list(groups.values())


def greedy_policy(V: np.ndarray, P, R, gamma) -> tuple:
    """拿着 V 做一步贪心：每个状态挑 Q 最大的动作。"""
    return tuple(int(a) for a in q_values(V, P, R, gamma).argmax(axis=1))


def all_deterministic_policies(P):
    """穷举全部确定性策略。nS 个状态、nA 个动作 → nA ** nS 个。"""
    nS, nA = P.shape[0], P.shape[1]
    return list(itertools.product(range(nA), repeat=nS))


# ═══════════════════════════════════════════════════════════════════
#  实验 A：max 能不能从 Σ 外面提出来
# ═══════════════════════════════════════════════════════════════════

def exp_a():
    print("\n[实验 A] 把 max 从 Σ 外面提出来 —— 一个「看着像分配律」的错")
    print(f"    环境 HUB（γ={HUB_GAMMA}），只看 s0 这个岔路口。\n")

    V_star, _, _ = value_iteration(HUB_P, HUB_R, HUB_GAMMA)
    print(f"    先解出 V* = {V_star}")

    R0 = HUB_R[0]                                  # s0 三个动作的即时奖励
    nxt = HUB_GAMMA * (HUB_P[0] @ V_star)          # s0 三个动作的「γΣ P V*」
    Q0 = R0 + nxt

    print(f"\n    {'动作':>6}{'R':>10}{'γΣPV*':>12}{'Q = 两者之和':>16}")
    print("    " + "-" * 46)
    for a in range(len(R0)):
        print(f"    {a:>6}{R0[a]:>10.4f}{nxt[a]:>12.4f}{Q0[a]:>16.4f}")

    right = Q0.max()
    wrong = R0.max() + nxt.max()        # ← 两个 max 各挑各的
    print(f"""
    正确：V*(s0) = max_a (R + γΣPV*) = {right:.4f}          ← 挑 a={int(Q0.argmax())}
    天真：max_a R + max_a γΣPV*      = {wrong:.4f}          ← 挑 a={int(R0.argmax())} 和 a={int(nxt.argmax())}
    差 {wrong - right:.4f}，相对 {(wrong - right) / right * 100:.1f}%

    ⚠️ 先看一眼**最优的是哪个动作**：a{int(Q0.argmax())}。
    它既不是奖励最大的（那是 a{int(R0.argmax())}，+10），
    也不是下一步最好的（那是 a{int(nxt.argmax())}，γΣPV* = {nxt.max():.4f}）。
    它是两者的**折中** —— 一半概率回 s0 继续选，一半概率去金矿。

    这个 74.5455 不是整数，因为它自引用（a2 有 0.5 概率回到 s0）。
    手算就是解一个一元方程：

        V(0) = 5 + 0.9 · (0.5·V(0) + 0.5·80)
             = 5 + 0.45·V(0) + 36
        0.55·V(0) = 41      →   V(0) = 820/11 = {820 / 11:.6f}

    注意上面 `nxt` 那一列用的是**已经收敛的** V*，也就是说
    「V(0) 里面含 V(0)」这个循环已经被迭代解掉了。
    值迭代做的就是这件事：拿上一轮的 V(0) 当这一轮的输入，直到自洽。

    天真的算法把 a={int(R0.argmax())} 的奖励和 a={int(nxt.argmax())} 的未来拼在了一起 ——
    **那不是任何一个动作**。它算出来的东西对应一个在这个世界里
    根本不存在的策略，所以它必然偏大：取到两个 max，只会高不会低。

    这就是 (max, +) 和 (Σ, +) 的根本区别：
      Σ 可以逐项拆开再拼回去（线性），max 不行（只满足 ≤ 的三角不等式）。
    能用 max 拆的只有**同一个 a** 上的拆分：

        max_a (R_a + γG_a)   ≠   max_a R_a + γ max_a G_a

    左边要求一个 a 从头用到尾，右边允许中途换人。
    """)
    return wrong > right + 1e-9


# ═══════════════════════════════════════════════════════════════════
#  实验 B：V* 是哪个方程的解
# ═══════════════════════════════════════════════════════════════════

def exp_b():
    print("\n[实验 B] V* 不是某个线性方程的解 —— 但它是「所有 V^π 逐分量最大的那个」")
    print(f"    环境 HUB（γ={HUB_GAMMA}），3 状态 3 动作 → 穷举 3^3 = 27 个确定性策略\n")

    V_star, n_iter, _ = value_iteration(HUB_P, HUB_R, HUB_GAMMA)
    policies = all_deterministic_policies(HUB_P)

    print(f"    值迭代给出的 V* = {V_star}（{n_iter} 轮）")
    print(f"\n    {'策略':>10}{'V^π(0)':>12}{'V^π(1)':>12}{'V^π(2)':>12}{'与 V* 的差':>14}")
    print("    " + "-" * 60)

    best = np.full(3, -np.inf)
    best_pols = []
    for pol in policies:
        V_pi = policy_values(pol, HUB_P, HUB_R, HUB_GAMMA)
        best = np.maximum(best, V_pi)
        gap = np.abs(V_pi - V_star).max()
        if gap < 1e-9:
            best_pols.append(pol)
        if pol in [(0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 0, 2), (0, 2, 0)]:
            print(f"    {str(pol):>10}{V_pi[0]:>12.6f}{V_pi[1]:>12.6f}{V_pi[2]:>12.6f}{gap:>14.3e}")

    print(f"    {'...':>10}（27 个里只列了 5 个）")
    print(f"""
    逐分量取最大：{best}
    值迭代的 V* ：{V_star}
    差 {np.abs(best - V_star).max():.3e}

    **`V* = max_π V^π`（逐分量）。** 这个等式是今天最重要的一句，
    它把「最优」从形容词变成了一个可以穷举算出来的东西。

    达到 V* 的策略：{len(best_pols)} 个 —— {best_pols}

    ⚠️ 但这 {len(best_pols)} 个**不算「策略不唯一」**，得先做一步甄别：""")

    groups = dedupe_policies(best_pols, HUB_P, HUB_R)
    print(f"""
    按 (P^π, R^π) 去重后只剩 {len(groups)} 组：""")
    for i, g in enumerate(groups, 1):
        print(f"      第 {i} 组：{g}")
    print(f"""
    原因是 HUB 里 s1 的 a0 和 a2 转移、奖励**逐位相同**（s2 的 a0/a2 也是），
    这种叫**冗余动作** —— 换个编号，策略没换。

    不区分「冗余」和「不唯一」，就会把一个平庸的环境误判成
    「我的算法不稳定」。真平局长什么样，看实验 D。
    """)

    return np.abs(best - V_star).max() < 1e-9 and len(groups) == 1


# ═══════════════════════════════════════════════════════════════════
#  实验 C：V* 唯一吗
# ═══════════════════════════════════════════════════════════════════

def exp_c():
    print("\n[实验 C] V* 唯一吗 —— 从三个完全不同的初值出发")
    print(f"    环境 HUB（γ={HUB_GAMMA}）\n")

    starts = [
        ("全 0", [0.0, 0.0, 0.0]),
        ("全 1000", [1000.0, 1000.0, 1000.0]),
        ("全 -1000", [-1000.0, -1000.0, -1000.0]),
    ]
    print(f"    {'初值':>10}{'轮数':>8}{'V(0)':>16}{'V(1)':>14}{'V(2)':>14}")
    print("    " + "-" * 64)

    results, iters = [], []
    for name, V0 in starts:
        V, n, _ = value_iteration(HUB_P, HUB_R, HUB_GAMMA, V0=V0)
        results.append(V)
        iters.append(n)
        print(f"    {name:>10}{n:>8}{V[0]:>16.9f}{V[1]:>14.9f}{V[2]:>14.9f}")

    d = max(np.abs(results[0] - r).max() for r in results[1:])

    # HUB 的解析解：V(1)=80（8/(1-0.9)），V(2)=0，V(0) 由 §2.4 那个一元方程得 820/11
    exact = np.array([820 / 11, 80.0, 0.0])
    gap_at_stop = np.abs(results[0] - exact).max()

    print(f"""
    三个终点之间的最大差：{d:.3e}   → **同一个点**

    所以 V* 与初值无关。这条不只是好消息，它是「值迭代可以随便
    初始化」这个做法的全部依据 —— 你不知道 V* 在哪，从哪出发都行。

    ⚠️ 但轮数值得看一眼：{iters[0]} / {iters[1]} / {iters[2]}。
    从 1000 起步和从 -1000 起步，初值离真值分别是 925 和 1080（差不多），
    轮数只差 1 轮。全 0 起步快 {iters[1] - iters[0]} 轮 —— 因为它一开始就在真值附近。
    所以精确的说法是：**初值只影响速度，而且影响得很小**（对数级）。

    更大的问题在这里：**判据是「相邻两轮之差」，不是「与真值的距离」。**

    全 0 那条在 k={iters[0]} 停下来（相邻差刚低于 1e-12）。
    但那一刻它离解析解 [820/11, 80, 0] 还有 **{gap_at_stop:.3e}** ——
    比 1e-12 大了 {gap_at_stop / 1e-12:.1f} 倍。

    这个倍数不是随机的，它约等于 1/(1−γ) = {1 / (1 - HUB_GAMMA):.0f}：

        |V_k − V*|        ≈  80 × 0.9^k     到真值的距离
        |V_{{k+1}} − V_k|   ≈   8 × 0.9^k     相邻两轮之差
                              ↑ 系数正好差一个 (1−γ)

    也就是说：**「值迭代收敛了」这句话，比你以为的要乐观 1/(1−γ) 倍。**
    γ=0.9 时是 10 倍（实测 {gap_at_stop / 1e-12:.1f} 倍），
    γ=0.99 时是 100 倍，γ=0.999 时是 1000 倍 —— 而你不会收到任何提示。

    Day 32 讲压缩映射时会给出这个倍数的来源，
    它也是 PPO 里 GAE 的 λ、DQN 里 target network 的同一个根。

    最后一行对照：从 -1000 起步时相邻差的系数是 108（不是 8），
    大了 13.5 倍，轮数只多 {np.log(108 / 8) / np.log(1 / HUB_GAMMA):.0f} 轮 —— 因为轮数随系数是**对数**增长。
    """)
    return d < 1e-9


# ═══════════════════════════════════════════════════════════════════
#  实验 D：V* 唯一，最优策略唯一吗
# ═══════════════════════════════════════════════════════════════════

def exp_d():
    print("\n[实验 D] V* 唯一，但最优策略可以不止一个")
    print(f"    环境 TIE（γ={TIE_GAMMA}），奖励是反推构造的，就为了造一个精确的平局\n")

    V_star, n_iter, _ = value_iteration(TIE_P, TIE_R, TIE_GAMMA)
    print(f"    值迭代 V* = {V_star}（{n_iter} 轮），目标就是 [5, 2]")

    print(f"\n    在 s1 上的两个动作：")
    for a in range(2):
        q = TIE_R[1, a] + TIE_GAMMA * (TIE_P[1, a] @ V_star)
        tgt = "留在 s1" if a == 1 else "回到 s0"
        print(f"      a{a}（{tgt}）：R={TIE_R[1, a]:>6.2f} + {TIE_GAMMA}·ΣPV* = {q:.4f}")

    print(f"""
    **两个动作的 Q 都是 2.0000。** 一个往 s0 走，一个留在 s1 ——
    转移完全不同的两个动作，价值一模一样。

    所以下面两个确定性策略都是最优的：""")

    policies = all_deterministic_policies(TIE_P)
    print(f"\n    {'策略':>10}{'V^π(0)':>14}{'V^π(1)':>14}{'是 V* 吗':>12}")
    print("    " + "-" * 50)
    optimal = []
    for pol in policies:
        V_pi = policy_values(pol, TIE_P, TIE_R, TIE_GAMMA)
        ok = np.abs(V_pi - V_star).max() < 1e-12
        if ok:
            optimal.append(pol)
            print(f"    {str(pol):>10}{V_pi[0]:>14.9f}{V_pi[1]:>14.9f}{'✅':>12}")
        else:
            print(f"    {str(pol):>10}{V_pi[0]:>14.9f}{V_pi[1]:>14.9f}{'—':>12}")

    print(f"""
    共 {len(optimal)} 个最优策略：{optimal}

    它们的 V^π **完全一样**（都等于 V*），但走法不同。

    这就是为什么明天的断言要断言值、不断言策略：
      V* 唯一（可以断言），π* 可能不唯一（不能断言）。

    工程含义比这条数学事实更值钱：**精确的平局几乎不会发生，
    但「近似平局」到处都有。** 我的构造是把 R 反推出来的，
    小数点后两位就凑上了；真实环境里不会有这么巧的事，
    但两个动作差 0.001 到处都是 —— 于是训练时策略会在它们之间
    来回跳，看上去像「不稳定」，其实是「两个都对」。
    """)
    return len(optimal) == 2


# ═══════════════════════════════════════════════════════════════════
#  实验 E：从 V* 出发做一步贪心
# ═══════════════════════════════════════════════════════════════════

def exp_e():
    print("\n[实验 E] 拿着 V* 做一步贪心，能不能直接拿到最优策略")
    print("    （这就是策略迭代里「策略改进」那一步，明天的主角）\n")

    envs = [
        ("CHAIN", CHAIN_P, CHAIN_R, CHAIN_GAMMA),
        ("TIE", TIE_P, TIE_R, TIE_GAMMA),
        ("HUB", HUB_P, HUB_R, HUB_GAMMA),
    ]

    print(f"    {'环境':>8}{'贪心得到的策略':>18}{'它的 V^π 与 V* 之差':>22}{'是不是最优':>12}")
    print("    " + "-" * 62)

    ok = True
    for name, P, R, g in envs:
        V_star, _, _ = value_iteration(P, R, g)
        pol = greedy_policy(V_star, P, R, g)
        V_pi = policy_values(pol, P, R, g)
        gap = np.abs(V_pi - V_star).max()
        is_opt = gap < 1e-9
        ok = ok and is_opt
        print(f"    {name:>8}{str(pol):>18}{gap:>22.3e}{'✅' if is_opt else '❌':>12}")

    print(f"""
    三个环境都是：**从 V* 贪心一次，直接得到最优策略**，V^π 和 V* 差 0。

    这件事叫「策略改进定理」，内容是：对任何策略 π，
    用 V^π 做贪心得到的 π' 一定满足 V^{{π'}} ≥ V^π。
    取 π 就是最优策略时，等号成立 —— 所以你贪心一次还是它。

    反过来的方向才是值迭代在做的：固定策略、把 V 算到不动。
    一个在策略空间里跳，一个在值空间里跳，明天会看到两者其实是
    同一个压缩映射的两种写法（Day 32 的不动点）。

    ⚠️ 一个必须记住的边界：**这条只在「能精确算出 V^π」时成立。**
    W6 你拿不到 P，V^π 只能估 —— 估偏了贪心就选错，
    这就是 Q-Learning 里 max 带来的高估偏差（overestimation）的来源。
    今天先记住有这个坑，W6 再拆。
    """)
    return ok


# ═══════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════

EXPERIMENTS = {"A": exp_a, "B": exp_b, "C": exp_c, "D": exp_d, "E": exp_e}


def main():
    ap = argparse.ArgumentParser(description="Day 31 贝尔曼最优方程的原理验证")
    ap.add_argument("--only", choices=list(EXPERIMENTS), help="只跑一个实验")
    args = ap.parse_args()

    print("=" * 74)
    print("  Day 31 · 贝尔曼最优方程：把一个 Σ 换成 max")
    print("=" * 74)

    names = [args.only] if args.only else list(EXPERIMENTS)
    results = []
    for k in names:
        results.append((k, EXPERIMENTS[k]()))

    if not args.only:
        print("\n" + "=" * 74)
        print("  汇总")
        print("=" * 74)
        for k, ok in results:
            print(f"    实验 {k}  {'✅ 通过' if ok else '❌ 失败'}")
        print(f"\n    {sum(ok for _, ok in results)}/{len(results)} 通过")


if __name__ == "__main__":
    main()
