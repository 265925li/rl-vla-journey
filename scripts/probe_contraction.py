"""Day 32 · 压缩映射与不动点 —— 六个实验

配套讲义：docs/Day32-压缩映射与不动点.md
只做算术，不带训练循环。**讲义里出现的每个数字都应该出自本文件的输出。**

    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts/probe_contraction.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts/probe_contraction.py --only D

设计原则（Day 31 的教训）：**误差的基准必须独立于被测物。**
所以「真值」不走 `value_iteration()`，而是
   ① 先迭代到很紧，拿到贪心策略 π
   ② 用 `(I − γP^π) V = R^π` **精确解**出 V
   ③ 再把 V 代回 T*，验证 `T*V = V` 的残差
第 ③ 步是独立检查：如果 π 不是最优的，残差不会消失。
"""

from __future__ import annotations

import argparse

import numpy as np

GAMMA = 0.9


# ────────────────────────────────────────────────────────────
# 一、环境与算子
# ────────────────────────────────────────────────────────────

def make_mdp(n_s: int = 4, n_a: int = 3, seed: int = 0):
    """随机 MDP。

    P 形状 (nS, nA, nS)，每行（固定 s,a 后对 s' 求和）为 1。
    R 形状也是 (nS, nA, nS)：**奖励依赖 s'**，这是 Day 29 练习里那个
    「逼你发现 R 写成 (nS,nA) 不够用」的设定。
    """
    rng = np.random.default_rng(seed)
    w = rng.random((n_s, n_a, n_s)) ** 2 + 0.05
    prob = w / w.sum(axis=2, keepdims=True)
    reward = rng.integers(-2, 6, size=(n_s, n_a, n_s)).astype(float)
    return prob, reward


def bellman_optimal(v, prob, reward, gamma):
    """T* 作用一次，同时返回 Q。

        (T*V)(s) = max_a Σ_s' P(s'|s,a) [ R(s,a,s') + γ V(s') ]
    """
    q = ((reward + gamma * v[None, None, :]) * prob).sum(axis=2)
    return q.max(axis=1), q


def bellman_policy(v, prob, reward, gamma, policy):
    """T^π 作用一次（π 是确定性策略，(nS,) 的整数数组）。"""
    idx = np.arange(prob.shape[0])
    return ((reward[idx, policy, :] + gamma * v[None, :]) * prob[idx, policy, :]).sum(axis=1)


def exact_from_policy(policy, prob, reward, gamma):
    """给定策略，用线性方程组精确解出 V^π（不迭代）。

        (I − γ P^π) V = R^π
    """
    idx = np.arange(prob.shape[0])
    p_pi = prob[idx, policy, :]
    r_pi = (reward[idx, policy, :] * p_pi).sum(axis=1)
    return np.linalg.solve(np.eye(len(policy)) - gamma * p_pi, r_pi)


def value_iteration(prob, reward, gamma, v0=None, tol=1e-13, max_iter=200000, track=False):
    """值迭代，返回 (V, 轮数, 轨迹)。轨迹元素为 (k, V_k, 相邻差)。"""
    n_s = prob.shape[0]
    v = np.zeros(n_s) if v0 is None else np.array(v0, dtype=float)
    hist = []
    for k in range(1, max_iter + 1):
        v_next, _ = bellman_optimal(v, prob, reward, gamma)
        delta = np.abs(v_next - v).max()
        v = v_next
        if track:
            hist.append((k, v.copy(), delta))
        if delta < tol:
            return v, k, hist
    return v, max_iter, hist


def solve_star(prob, reward, gamma):
    """拿到**独立的**真值，并验证它真的是不动点。

    返回 (V*, 残差, 是否贪心一致)。
    """
    v, _, _ = value_iteration(prob, reward, gamma, tol=1e-15, max_iter=400000)
    policy = np.argmax(bellman_optimal(v, prob, reward, gamma)[1], axis=1)
    v_exact = exact_from_policy(policy, prob, reward, gamma)
    residual = np.abs(bellman_optimal(v_exact, prob, reward, gamma)[0] - v_exact).max()
    greedy = np.argmax(bellman_optimal(v_exact, prob, reward, gamma)[1], axis=1)
    return v_exact, residual, bool(np.array_equal(greedy, policy))


# ────────────────────────────────────────────────────────────
# 实验 A：压缩比会不会超过 γ
# ────────────────────────────────────────────────────────────

def exp_a():
    print("\n[实验 A] 压缩比 ‖TV − TU‖∞ / ‖V − U‖∞ 会不会超过 γ")
    prob, reward = make_mdp(seed=0)
    rng = np.random.default_rng(1)

    print(f"    {'γ':>6}{'随机对最大比':>16}{'超过 γ 的组数':>14}{'上界可达':>12}")
    print("    " + "-" * 52)
    ok = True
    for gamma in (0.1, 0.5, 0.9, 0.99):
        ratios = []
        for _ in range(3000):
            v = rng.normal(0.0, 10.0, prob.shape[0])
            u = rng.normal(0.0, 10.0, prob.shape[0])
            gap = np.abs(v - u).max()
            if gap < 1e-9:
                continue
            tv = bellman_optimal(v, prob, reward, gamma)[0]
            tu = bellman_optimal(u, prob, reward, gamma)[0]
            ratios.append(np.abs(tv - tu).max() / gap)
        worst = max(ratios)
        exceed = sum(1 for r in ratios if r > gamma + 1e-12)
        # 上界可达：U = V + c·1 → T U = T V + γc·1，比值恰为 γ
        v = rng.normal(0.0, 10.0, prob.shape[0])
        c = 3.7
        hit = np.abs(bellman_optimal(v + c, prob, reward, gamma)[0]
                     - bellman_optimal(v, prob, reward, gamma)[0]).max() / c
        print(f"    {gamma:>6}{worst:>16.9f}{exceed:>14}{hit:>12.6f}")
        ok = ok and exceed == 0 and worst <= gamma + 1e-12 and abs(hit - gamma) < 1e-12

    print(f"""
    三列读数：

    ① **随机 3000 组，比值一次都没有超过 γ。** 这是「压缩」两个字的全部内容。
    ② **上界是可达的**（最后一列恰好等于 γ）：取 U = V + c·1 时
       T(V + c·1) = TV + γc·1，两边距离正好缩到 γ 倍。
       所以 γ 不是「一个宽松的界」，它是**最坏情况的精确值**。
    ③ 比值随 γ 单调靠近 1 —— γ → 1 时压缩性还在，但**越来越没用**。
       实验 E 会把这个「没用」量化。

    ⚠️ 顺便看第一列：γ=0.9 时随机 3000 组最大只到 0.863，比 γ 小了一截。
    **随机抽样永远碰不到最坏情况**，最坏情况得构造（就是第二列那个 U = V + c·1）。
    这是「跑一个例子就下结论」的又一种失效方式。

    这里没有证明，只是没找到反例。证明在讲义 §3，只有四行。
    """)
    return ok


# ────────────────────────────────────────────────────────────
# 实验 B：两个引理，以及把 P 的随机性破坏掉会发生什么
# ────────────────────────────────────────────────────────────

def exp_b():
    print("\n[实验 B] 证明靠的两个引理 —— 以及各自依赖哪一条前提")
    prob, reward = make_mdp(seed=0)
    rng = np.random.default_rng(2)
    gamma = GAMMA

    # 引理 1：单调性。V ≤ U ⟹ T V ≤ T U。只依赖 P ≥ 0。
    bad = 0
    for _ in range(2000):
        v = rng.normal(0.0, 5.0, prob.shape[0])
        u = v + rng.random(prob.shape[0]) * 3.0      # 保证 V ≤ U
        tv = bellman_optimal(v, prob, reward, gamma)[0]
        tu = bellman_optimal(u, prob, reward, gamma)[0]
        bad += int((tv > tu + 1e-12).sum())
    print(f"    引理 1 单调性：2000 组 V ≤ U，违反的分量数 {bad}")

    # 引理 2：常数平移。T(V + c·1) = T V + γc·1。只依赖 P·1 = 1。
    worst_shift = 0.0
    for _ in range(200):
        v = rng.normal(0.0, 5.0, prob.shape[0])
        c = rng.normal(0.0, 2.0)
        lhs = bellman_optimal(v + c, prob, reward, gamma)[0]
        rhs = bellman_optimal(v, prob, reward, gamma)[0] + gamma * c
        worst_shift = max(worst_shift, np.abs(lhs - rhs).max())
    print(f"    引理 2 常数平移：200 组最大偏差 {worst_shift:.3e}")

    # 破坏行和为 1：P 乘 (1+δ)
    print(f"\n    把 P 的行和改成 1+δ（不再是随机矩阵）：")
    print(f"    {'δ':>6}{'行和':>10}{'平移偏差':>14}{'压缩比':>12}{'应等于 γ(1+δ)':>16}")
    print("    " + "-" * 60)
    ratios_ok = True
    for delta in (0.0, 0.05, 0.2):
        p_bad = prob * (1.0 + delta)
        v = rng.normal(0.0, 5.0, prob.shape[0])
        c = 2.0
        lhs = bellman_optimal(v + c, p_bad, reward, gamma)[0]
        rhs = bellman_optimal(v, p_bad, reward, gamma)[0] + gamma * c
        shift_err = np.abs(lhs - rhs).max()
        u = v + c
        ratio = np.abs(bellman_optimal(u, p_bad, reward, gamma)[0]
                       - bellman_optimal(v, p_bad, reward, gamma)[0]).max() / c
        pred = gamma * (1.0 + delta)
        print(f"    {delta:>6}{1.0 + delta:>10.2f}{shift_err:>14.3e}{ratio:>12.6f}{pred:>16.6f}")
        ratios_ok = ratios_ok and abs(ratio - pred) < 1e-12

    print(f"""
    这是今天最该带走的一条**结构性**认识：

      「单调性」只要 P ≥ 0；
      「常数平移」要求 P·1 = 1，也就是 **P 是随机矩阵**。

    而压缩定理 = 单调性 + 常数平移 + 对称性。
    所以 **P 的行和一旦不是 1，压缩因子就从 γ 变成 γ(1+δ)**，
    δ = 0.2 时变成 1.08 —— **大于 1，不再是压缩**，Banach 定理整条失效。

    ⚠️ 而 transpose / 归一化写错一个除法，正好就会造出这种 P。
    不会有任何报错，只是值迭代从此不保证收敛。
    """)
    return bad == 0 and worst_shift < 1e-12 and ratios_ok


# ────────────────────────────────────────────────────────────
# 实验 C：Banach 三件事
# ────────────────────────────────────────────────────────────

def exp_c():
    print("\n[实验 C] 存在、唯一、收敛 —— 一次全拿到")
    prob, reward = make_mdp(seed=0)
    gamma = GAMMA
    v_star, residual, greedy = solve_star(prob, reward, gamma)
    print(f"    真值（线性方程组精确解）V* = {np.array2string(v_star, precision=9)}")
    print(f"    代回 T* 的残差 {residual:.3e}   ·  该策略在 V* 处贪心一致：{greedy}")

    rng = np.random.default_rng(3)
    starts = [np.zeros(prob.shape[0])] + [rng.normal(0.0, 50.0, prob.shape[0]) for _ in range(4)]
    print(f"\n    {'初值':>14}{'轮数':>8}{'与 V* 的差':>16}{'初值离 V* 的距离':>18}")
    print("    " + "-" * 58)
    ends = []
    for i, v0 in enumerate(starts):
        v, n, _ = value_iteration(prob, reward, gamma, v0=v0, tol=1e-13)
        ends.append(v)
        name = "全 0" if i == 0 else f"随机 {i}"
        print(f"    {name:>14}{n:>8}{np.abs(v - v_star).max():>16.3e}"
              f"{np.abs(np.asarray(v0, float) - v_star).max():>18.3e}")

    spread = max(np.abs(ends[0] - e).max() for e in ends[1:])

    # 收敛速率：误差比是否恒 ≤ γ
    v = np.zeros(prob.shape[0])
    ratios, errs = [], []
    for _ in range(40):
        errs.append(np.abs(v - v_star).max())
        v = bellman_optimal(v, prob, reward, gamma)[0]
    for k in range(1, len(errs)):
        if errs[k - 1] > 0:
            ratios.append(errs[k] / errs[k - 1])
    print(f"""
    ① **存在**：五条路全部停下，且停下的地方代回 T* 残差 {residual:.1e} —— 确实是不动点。
    ② **唯一**：五个完全不同的初值（有的离 V* 超过 50），终点两两最大差 {spread:.3e}。
    ③ **收敛速率**：误差逐轮乘上的系数，最大 {max(ratios):.6f}，最小 {min(ratios):.6f}（γ = {gamma}）。

    第 ③ 列就是 γ^k 的形状：{np.array2string(np.array(errs[:6]), precision=4)} 是前 6 轮的误差。
    每轮大约乘 {gamma}，20 轮就是 {gamma ** 20:.3f} 倍，40 轮 {gamma ** 40:.4f} 倍。
    **指数收敛，不是「跑够久总会停」那种收敛。**

    ⚠️ 看第 ② 行那个数：{spread:.2e}，比一个 float64 的舍入大得多。
    这不是精度不够，而是**五个终点各自的误差都被放大过**：停止判据是「相邻差 < 1e-13」，
    所以每个终点离真值本身就有 1e-13 量级的偏差，五个点一比较自然会看到 1e-12。
    **容差必须匹配被测精度** —— 拿 1e-13 当断言线去要求 1e-15 的一致性，只会在噪声里找信号。
    """)
    # 断言用 1e-11：终点的差本身被 1/(1-γ) 放大，不能拿 tol 本身当线
    return spread < 1e-11 and residual < 1e-12 and greedy and max(ratios) <= gamma + 1e-9


# ────────────────────────────────────────────────────────────
# 实验 D：1/(1−γ) 这个放大倍数的来源
# ────────────────────────────────────────────────────────────

def exp_d():
    print("\n[实验 D] 停止判据宽松了多少 —— 1/(1−γ) 的确切出处")
    prob, reward = make_mdp(seed=0)
    gamma = GAMMA
    v_star, _, _ = solve_star(prob, reward, gamma)
    sharp = gamma / (1.0 - gamma)      # 用同一轮的相邻差
    loose = 1.0 / (1.0 - gamma)        # 用下一轮的相邻差（常见写法）
    print(f"    γ = {gamma}")
    print(f"    紧的界 γ/(1−γ) = {sharp:.4f}   ·   常见写法 1/(1−γ) = {loose:.4f}\n")

    v = np.zeros(prob.shape[0])
    prev = v.copy()
    print(f"    {'轮 k':>6}{'相邻差 d_k':>15}{'真误差 e_k':>15}{'e_k / d_k':>12}")
    print("    " + "-" * 50)
    rows = []
    for k in range(1, 401):
        v = bellman_optimal(v, prob, reward, gamma)[0]
        delta = np.abs(v - prev).max()
        err = np.abs(v - v_star).max()
        prev = v.copy()
        if k in (1, 2, 3, 5, 10, 20, 40, 80, 160, 240, 280, 320):
            ratio = err / delta if delta > 0 else float("nan")
            rows.append((k, delta, err, ratio))
            print(f"    {k:>6}{delta:>15.3e}{err:>15.3e}{ratio:>12.4f}")

    worst = max(r[3] for r in rows)
    tol = 1e-12
    stop = next((r for r in rows if r[1] < tol), None)

    print(f"""
    **读数一：「误差 / 相邻差」这一列不超过 {sharp:.0f}，并且在第 10 轮之后几乎钉死在它上面。**

    为什么恰好是 γ/(1−γ) 而不是 1/(1−γ)？三行（讲义 §5.2）：

        e_k = ‖V* − V_k‖ = ‖T*V* − T*V_{{k−1}}‖ ≤ γ‖V* − V_{{k−1}}‖ = γ·e_{{k−1}}
        e_{{k−1}} ≤ d_k + e_k                       （把 V*−V_{{k−1}} 拆成两段）
        代入：  e_k ≤ γ(d_k + e_k)  ⟹  e_k ≤ γ·d_k/(1−γ)

    常见写法 e_k ≤ d_{{k+1}}/(1−γ) 也对，但用的是**下一轮**的相邻差，
    而 d_{{k+1}} ≤ γ d_k，所以那个版本更松。**用同一轮的 d_k 就该是 γ/(1−γ)。**

    读数二：**判据「相邻差 < ε」保证的是「真误差 < γε/(1−γ) ≈ {sharp:.0f}ε」，不是 ε。**
    {"停在第 " + str(stop[0]) + " 轮时 d_k = " + f"{stop[1]:.3e}" + " 已经小于 " + f"{tol:.0e}"
     + "，但 e_k = " + f"{stop[2]:.3e}" + f"，是容差的 {stop[2] / tol:.1f} 倍。" if stop else ""}

    Day 31 在另一个环境上报的是 8.1 倍，正好落在 γ/(1−γ) = {sharp:.0f} 之内 —— **两天的数不冲突，
    今天只是把「1/(1−γ)」这个习惯说法收紧成了 γ/(1−γ)。**
    """)
    return worst <= sharp + 1e-6


# ────────────────────────────────────────────────────────────
# 实验 E：γ 扫描，两种判据各要跑多少轮
# ────────────────────────────────────────────────────────────

def exp_e():
    print("\n[实验 E] 收敛轮数、两种判据，以及 float64 能给出的最好精度")
    prob, reward = make_mdp(seed=0)
    tol = 1e-12
    n_s = prob.shape[0]
    print(f"    目标：真误差 ‖V_k − V*‖∞ < {tol:.0e}，起点 V₀ = 0\n")
    print(f"    {'γ':>7}{'公式上界':>10}{'按真误差':>10}{'按相邻差':>10}"
          f"{'float64 地板':>15}")
    print("    " + "-" * 58)
    ok = True
    floors = []
    for gamma in (0.5, 0.9, 0.99, 0.999):
        v_star, _, _ = solve_star(prob, reward, gamma)
        span = np.abs(bellman_optimal(np.zeros(n_s), prob, reward, gamma)[0]).max()
        formula = np.log(tol * (1.0 - gamma) / span) / np.log(gamma)

        # 一次跑到底，同时记录三件事：按真误差何时达标、按相邻差何时达标、误差最低能到多少
        v = np.zeros(n_s)
        prev = v.copy()
        k_err = k_delta = None
        floor = np.inf
        for k in range(1, int(np.ceil(formula)) + 5000 + 1):
            v = bellman_optimal(v, prob, reward, gamma)[0]
            err = np.abs(v - v_star).max()
            floor = min(floor, err)
            if k_err is None and err < tol:
                k_err = k
            d = np.abs(v - prev).max()
            prev = v.copy()
            if k_delta is None and d < tol:
                k_delta = k

        s_err = str(k_err) if k_err else "不可达"
        print(f"    {gamma:>7}{formula:>10.1f}{s_err:>10}{str(k_delta):>10}{floor:>15.3e}")
        floors.append((gamma, floor))
        if k_err is not None:
            ok = ok and k_err <= np.ceil(formula) + 1
        if k_delta is not None and k_err is not None:
            ok = ok and k_delta <= k_err

    print(f"""
    三列读数：

    ① **公式是上界，实测贴着它**（差 0~1 轮）。§5.3 的推导没错。
       γ 每靠近 1 一个数量级，轮数也涨一个数量级：几十轮 → 三万多轮。

    ② **「按相邻差」永远不晚于「按真误差」。** 省下的那些轮，代价就是实验 D 那个
       γ/(1−γ) —— **你省掉的不只是轮数，还有精度。**

    ③ **最后一列是我原来没算到的：γ 大了之后，float64 根本到不了 1e-12。**

        γ=0.99 的地板是 {floors[2][1]:.2e}，γ=0.999 是 {floors[3][1]:.2e} —— 而目标写的是 1e-12。
        所以那两行的「按真误差」只能写「不可达」。这不是慢，是**不可能**。

    原因不是代码写错，是**舍入误差本身也被 1/(1−γ) 放大了**：
    每一步浮点运算注入约 eps·|V| 的噪声，被压缩映射累积成 eps·|V|/(1−γ)，
    而 |V| 本身又是 R_max/(1−γ) 量级 → 地板约 R_max·eps/(1−γ)²。
    γ=0.999 时 |V| ≈ 2826，eps·|V| ≈ 6e-13，再除以 0.001 → 6e-10，与实测 {floors[3][1]:.1e} 同量级。

    ⚠️ **所以「γ=0.999 时值迭代收到 1e-12」这句话做不到，不是慢。**
    看到有人这么报，要么他用了更高精度，要么他报的是相邻差。
    """)
    return ok


# ────────────────────────────────────────────────────────────
# 实验 F：前提不在，结论就不在
# ────────────────────────────────────────────────────────────

def exp_f():
    print("\n[实验 F] 压缩因子究竟是多少 —— γ ≥ 1 时前提就没了")
    prob, reward = make_mdp(seed=0)
    rng = np.random.default_rng(1)      # 与实验 A 同一个流，两表可以直接对读
    n_s = prob.shape[0]
    print(f"    {'γ':>7}{'随机对最大比':>14}{'最坏情况':>12}{'小于 1？':>10}"
          f"{'400 轮后 |V|':>16}")
    print("    " + "-" * 62)
    rows = []
    for gamma in (0.9, 1.0, 1.5):
        ratios = []
        for _ in range(3000):
            v = rng.normal(0.0, 10.0, n_s)
            u = rng.normal(0.0, 10.0, n_s)
            gap = np.abs(v - u).max()
            if gap < 1e-9:
                continue
            tv = bellman_optimal(v, prob, reward, gamma)[0]
            tu = bellman_optimal(u, prob, reward, gamma)[0]
            ratios.append(np.abs(tv - tu).max() / gap)
        # 最坏情况不是随机的，是结构性的：U = V + c·1
        v = rng.normal(0.0, 5.0, n_s)
        c = 2.0
        worst = np.abs(bellman_optimal(v + c, prob, reward, gamma)[0]
                       - bellman_optimal(v, prob, reward, gamma)[0]).max() / c
        v = np.zeros(n_s)
        for _ in range(400):
            v = bellman_optimal(v, prob, reward, gamma)[0]
        rows.append((gamma, max(ratios), worst, np.abs(v).max()))
        print(f"    {gamma:>7}{max(ratios):>14.6f}{worst:>12.6f}"
              f"{'是' if worst < 1 else '否':>10}{np.abs(v).max():>16.3e}")

    # γ=1 时值是否线性增长（不收敛但也不爆）
    v = np.zeros(n_s)
    marks = {}
    for k in range(1, 401):
        v = bellman_optimal(v, prob, reward, 1.0)[0]
        if k in (100, 200, 400):
            marks[k] = float(np.abs(v).max())

    print(f"""
    读数：

    · **γ = 0.9**：随机对最大比 {rows[0][1]:.4f}，**最坏情况恰为 {rows[0][2]:.6f} = γ**。
    · **γ = 1.0**：最坏情况 {rows[1][2]:.6f} —— **正好等于 1，不是压缩。**
      Banach 要求严格小于 1，差这一点就整条不成立。
    · **γ = 1.5**：最坏情况 {rows[2][2]:.6f} > 1，**扩张**，400 轮后值到 {rows[2][3]:.1e}。

    ⚠️ 注意「随机对」和「最坏情况」的差距（{rows[1][1]:.4f} vs {rows[1][2]:.4f}）：
    **随机抽样取不到最坏情况。** 最坏情况是结构性的（U = V + c·1），
    靠随机试是试不出来的 —— 这也是「只跑一个例子就下结论」为什么危险。

    γ = 1 时值的大小：100 轮 {marks[100]:.1f} → 200 轮 {marks[200]:.1f}
    → 400 轮 {marks[400]:.1f}，**每 100 轮涨大约同样的量，是线性增长，不是收敛**。
    它不报错、不 nan，只是永远停不下来。
    """)
    ok = (abs(rows[0][2] - 0.9) < 1e-12 and abs(rows[1][2] - 1.0) < 1e-12
          and abs(rows[2][2] - 1.5) < 1e-12 and marks[400] > marks[200] > marks[100])
    return ok


# ────────────────────────────────────────────────────────────
# 汇总
# ────────────────────────────────────────────────────────────

EXPERIMENTS = [
    ("A", "压缩比不超 γ、上界可达", exp_a),
    ("B", "两个引理 + 破坏行和", exp_b),
    ("C", "存在 / 唯一 / 收敛", exp_c),
    ("D", "1/(1−γ) 的来源", exp_d),
    ("E", "γ 扫描对账", exp_e),
    ("F", "γ ≥ 1 前提失效", exp_f),
]


def main():
    parser = argparse.ArgumentParser(description="Day 32 压缩映射六个实验")
    parser.add_argument("--only", default=None, help="只跑某个实验，如 A")
    args = parser.parse_args()

    picked = [e for e in EXPERIMENTS if args.only is None or e[0] == args.only.upper()]
    if not picked:
        print(f"没有实验 {args.only}，可选 {[e[0] for e in EXPERIMENTS]}")
        return

    results = []
    for key, title, fn in picked:
        print("=" * 66)
        print(f"实验 {key} · {title}")
        print("=" * 66)
        try:
            results.append((key, title, fn()))
        except AssertionError as err:
            results.append((key, title, False))
            print(f"    ✗ 断言失败：{err}")

    print("\n" + "=" * 66)
    print("汇总")
    print("=" * 66)
    passed = 0
    for key, title, ok in results:
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] 实验 {key} · {title}")
        passed += int(bool(ok))
    print(f"\n  {passed}/{len(results)} 通过")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
