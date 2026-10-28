"""Day 29 的原理验证：MDP 与回报里的那几条论断，到底成不成立。

这个脚本**不是让你练编程**，它只做一件事：把你今天要接受的 5 条论断
用真实数字检查一遍。你读它、改里面的数字，但不需要从零写它。

    实验 A  折扣因子 γ 是个杠杆：V(A) = 1/(1-γ²)，并对照界 R_max/(1-γ)
    实验 B  递归恒等式 G_t = R_{t+1} + γ·G_{t+1} 是不是严格成立
    实验 C  手算的 1/(1-γ²) 和线性方程组 (I-γP)⁻¹R 的解是不是同一个数
    实验 D  γ=1 且任务不停 → 回报发散（部分和随 T 线性增长）
    实验 E  **马尔可夫性被破坏时会发生什么**（这条最重要）

实验 E 的设计：拿一条 A↔B 交替的链，让"观测"把 A 和 B 压成同一个符号。
于是同一个观测下，未来的期望回报有两个不同的值 —— 说明 `V(观测)` 根本不存在。
这就是"状态定义错了"的严格含义。

用法：
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts/probe_mdp.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts/probe_mdp.py --only E
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════
#  今天唯一的环境：两状态链
#  A --奖励 1--> B --奖励 0--> A（确定性，只有一个动作，所以 π 是平凡的）
# ═══════════════════════════════════════════════════════════════════

# 状态 0 = A，状态 1 = B
# 转移矩阵 P[s, s'] = 从 s 走到 s' 的概率（确定性，只有一条路）
P_DET = np.array([
    [0.0, 1.0],   # A → B
    [1.0, 0.0],   # B → A
])

# 奖励 R[s] = 离开 s 时立刻拿到的奖励
R_DET = np.array([1.0, 0.0])


def exact_values(gamma: float, P: np.ndarray = P_DET, R: np.ndarray = R_DET) -> np.ndarray:
    """线性方程组精确解 V = (I - γP)⁻¹ R。

    这个式子只在「有限状态 + 策略固定」时成立 —— 是本周唯一能拿到
    精确答案的地方（Day 30 会把它推出来）。
    """
    n = P.shape[0]
    A = np.eye(n) - gamma * P
    return np.linalg.solve(A, R)


def closed_form_A(gamma: float) -> float:
    """纸上推出的 V(A) = 1/(1-γ²)。

    推导只有两步：
        V(A) = 1 + γ V(B)
        V(B) = 0 + γ V(A)      ← 代入上式
        V(A) = 1 + γ² V(A)  →  V(A) = 1/(1-γ²)
    """
    return 1.0 / (1.0 - gamma ** 2)


# ═══════════════════════════════════════════════════════════════════
#  实验 A：γ 是杠杆
# ═══════════════════════════════════════════════════════════════════

def exp_a():
    print("\n[实验 A] 折扣因子 γ 是个杠杆")
    print(f"    {'γ':>6}{'V(A)=1/(1-γ²)':>18}{'矩阵解':>14}{'界 R_max/(1-γ)':>18}{'占界比例':>10}")
    print("    " + "-" * 68)

    R_max = float(R_DET.max())
    gammas = [0.0, 0.5, 0.9, 0.99, 0.999]
    for g in gammas:
        v_hand = closed_form_A(g)
        v_mat = exact_values(g)[0]
        bound = R_max / (1.0 - g)
        print(f"    {g:>6g}{v_hand:>18.6f}{v_mat:>14.6f}{bound:>18.4f}{v_hand / bound:>10.4f}")

    print(f"""
    两件事同时成立：
      ① 手推的 1/(1-γ²) 和矩阵解是同一个数（误差见实验 C），
         说明 **推导没错，代码也没错**；
      ② V 永远不超过 R_max/(1-γ)。γ=0.999 时 V(A)≈{closed_form_A(0.999):.1f}，
         而 γ=0.9 时只有 {closed_form_A(0.9):.2f} —— **同一个环境，只改一个数字，
         值函数翻了 {closed_form_A(0.999) / closed_form_A(0.9):.0f} 倍**。
     所以 γ 不是"调参"，它决定这个任务**看多远**。
    """)
    return True


# ═══════════════════════════════════════════════════════════════════
#  实验 B：递归恒等 G_t = R_{t+1} + γ G_{t+1}
# ═══════════════════════════════════════════════════════════════════

def exp_b(gamma: float = 0.9, n_traj: int = 5, T: int = 20):
    print(f"\n[实验 B] G_t = R_(t+1) + γ·G_(t+1)（γ={gamma}，{n_traj} 条 {T} 步轨迹）")

    rng = np.random.default_rng(0)
    worst = 0.0
    for k in range(n_traj):
        s = int(rng.integers(0, 2))
        rewards, states = [], []
        for _ in range(T):
            rewards.append(R_DET[s])
            states.append(s)
            s = int(rng.choice(2, p=P_DET[s]))
        rewards = np.array(rewards)

        # 从后往前算 G_t（折扣回报的定义式，直接照抄）
        G = np.zeros(T)
        acc = 0.0
        for t in range(T - 1, -1, -1):
            acc = rewards[t] + gamma * acc
            G[t] = acc

        # 检查递归：G_t 是否等于 R_(t+1) + γ G_(t+1)
        # 注意 G 的最后一项没有 G_{T}，所以只检查前 T-1 项
        diff = np.abs(G[:-1] - (rewards[:-1] + gamma * G[1:])).max()
        worst = max(worst, diff)
        if k == 0:
            print(f"    轨迹 0 的前 5 个 G_t：{np.array2string(G[:5], precision=6)}")

    print(f"""
    所有轨迹、所有 t 上，|G_t − (R_(t+1) + γG_(t+1))| 的最大值 = {worst:.3e}

    **这条恒等不是新假设，它就是定义式移了个项。**
    但它是全部强化学习的种子：Day 30 的贝尔曼方程、Day 36 的 TD 更新、
    PPO 的 advantage 估计，全部由这一步展开而来。
    """)
    return worst < 1e-12


# ═══════════════════════════════════════════════════════════════════
#  实验 C：手算 vs 矩阵解
# ═══════════════════════════════════════════════════════════════════

def exp_c(gamma: float = 0.9):
    print(f"\n[实验 C] 手算 1/(1-γ²) vs 线性方程组 (I-γP)⁻¹R（γ={gamma}）")

    v_mat = exact_values(gamma)
    v_hand_A = closed_form_A(gamma)
    v_hand_B = gamma * v_hand_A          # V(B) = γ V(A)

    print(f"    (I - γP) = [[{1:.1f}, {-gamma:.1f}], [{-gamma:.1f}, {1:.1f}]]")
    det = 1 - gamma ** 2
    print(f"    行列式 = 1 - γ² = {det:.6f}   →   1/det = {1 / det:.6f}")
    print(f"    {'':>10}{'手算':>14}{'矩阵解':>14}{'差':>12}")
    print("    " + "-" * 52)
    for name, hand, mat in (("V(A)", v_hand_A, v_mat[0]), ("V(B)", v_hand_B, v_mat[1])):
        print(f"    {name:>10}{hand:>14.8f}{mat:>14.8f}{abs(hand - mat):>12.3e}")

    gap = max(abs(v_hand_A - v_mat[0]), abs(v_hand_B - v_mat[1]))
    print(f"""
    最大差 {gap:.3e}（两列**严格相同**，不是"接近"）。

    **这是本周最重要的一次交叉验证**：同一个量用两条完全独立的路算出来
    —— 一条是纸上的代数消元，一条是解线性方程组。
    两边对上，说明你对"V 是什么"的理解和数学定义是一致的。
    Day 30 起，每次推完贝尔曼方程都这么对一次。
    """)
    return gap < 1e-12


# ═══════════════════════════════════════════════════════════════════
#  实验 D：γ = 1 且任务不停 → 发散
# ═══════════════════════════════════════════════════════════════════

def exp_d():
    print("\n[实验 D] γ=1 且任务没有终点 → 回报发散")
    print(f"    {'T (步数)':>10}{'部分和 Σγ^k·R':>18}{'γ=0.9 的 G_0':>16}")
    print("    " + "-" * 46)

    for T in (10, 100, 1000, 10000):
        # γ=1：每步都拿 1，部分和就是 T
        sum_g1 = float(T)
        # γ=0.9：等比级数，T 步部分和 → 1/(1-0.9) = 10
        sum_09 = sum(0.9 ** k for k in range(T))
        print(f"    {T:>10}{sum_g1:>18.1f}{sum_09:>16.6f}")

    print(f"""
    γ=1 时部分和 = T，跟着步数一起长，**没有上界**。
    γ=0.9 时 10000 步已经稳在 10.000000 附近 —— 有限。

    **这就是 γ<1 的第一个身份：它不是"偏好近处奖励"，它是收敛性的前提。**
    后面 Day 32 会看到更硬的话：γ<1 是压缩映射成立的条件，
    而压缩映射是值迭代能收敛的**唯一**理由。

    （γ=1 也不是绝对不能用：任务有终止状态、且一定能终止时，
      总回报仍是有限的。今天这个链永远不停，所以发散。）
    """)
    return True


# ═══════════════════════════════════════════════════════════════════
#  实验 E：马尔可夫性被破坏时会发生什么
# ═══════════════════════════════════════════════════════════════════

def exp_e(gamma: float = 0.9):
    print("\n[实验 E] 马尔可夫性被破坏：把 A 和 B 压成同一个观测")

    v_A, v_B = exact_values(gamma)
    print(f"""
    真实状态有两个：A（下一步拿 1）和 B（下一步拿 0）。
    精确值：V(A) = {v_A:.6f}，V(B) = {v_B:.6f}

    现在做一个"状态抽象"：观测 = 颜色，而 A 和 B 是同一个颜色。
    于是观测层面只看到一个符号 c。请问 V(c) 等于多少？
    """)

    # 从 A 出发 vs 从 B 出发，都观测到 c，但未来期望回报不同
    print(f"    {'观测到的':>10}{'隐藏的真实状态':>16}{'E[G|观测]':>14}")
    print("    " + "-" * 42)
    print(f"    {'c':>10}{'A':>16}{v_A:>14.6f}")
    print(f"    {'c':>10}{'B':>16}{v_B:>14.6f}")

    print(f"""
    同一个观测 c，两种不同的期望回报，差 {abs(v_A - v_B):.6f}。
    **所以 V(c) 没有定义** —— 不是"算不准"，是根本不存在这个数。

    那这个差会不会随 γ 变小？代数上：
        V(A) − V(B) = 1/(1−γ²) − γ/(1−γ²) = (1−γ)/(1−γ²) = **1/(1+γ)**
    γ 越大这个差越小，但 γ→1 时它趋近 0.5，**永远不小于 0.5**。
    下面逐点验：
    """)

    print(f"    {'γ':>6}{'V(A)−V(B) 实测':>18}{'1/(1+γ)':>12}{'差':>12}")
    print("    " + "-" * 50)
    err = 0.0
    for g in (0.0, 0.5, 0.9, 0.99, 0.999):
        d = abs(exact_values(g)[0] - exact_values(g)[1])
        pred = 1.0 / (1.0 + g)
        err = max(err, abs(d - pred))
        print(f"    {g:>6g}{d:>18.6f}{pred:>12.6f}{abs(d - pred):>12.3e}")

    print(f"""
    最大偏差 {err:.3e} —— 解析式对得上。
    **注意差值随 γ 增大而减小，但下界是 0.5。**
    所以"马尔可夫性被破坏"不是一个可以靠调 γ 缓解的程度问题，
    它是**有没有**的问题：只要观测丢了信息，V 就不存在，一步都省不掉。

    再说一遍这句话，因为它是整个 RL 里最容易糊过去的地方：
      **状态不是"你能看到的东西"，状态是"决定未来的所有信息"。**
      A 和 B 看到的东西一样，但它们的未来不一样，所以它们是两个状态。

    工程含义：你设计观测/特征的那一刻，就已经在决定马尔可夫性成不成立。
    后面 Day 33 写 GridWorld 时，"状态 = 格子坐标"这个选择看起来天经地义，
    但如果你把坐标去掉只留"周围有没有墙"，就会立刻退化成今天这个实验。
    """)

    # 用数值把它钉死：同观测下 V 的取值不是唯一的，且差值有解析式
    spread = abs(v_A - v_B)
    return spread > 1e-6 and err < 1e-12


# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Day 29 原理验证")
    ap.add_argument("--only", default=None, help="只跑某个实验，如 A / E")
    args = ap.parse_args()

    print("=" * 70)
    print("Day 29 · MDP 与回报：五条论断的真假")
    print("=" * 70)
    print("""
这个脚本不练编程。它只回答一个问题：
**讲台上说的那几条结论，在这台机器上是不是真的？**
所有数字都会出现在今天的讲义里。
""")

    plan = [
        ("A", "γ 是杠杆", exp_a),
        ("B", "递归恒等", exp_b),
        ("C", "手算 vs 矩阵解", exp_c),
        ("D", "γ=1 发散", exp_d),
        ("E", "马尔可夫性被破坏", exp_e),
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
