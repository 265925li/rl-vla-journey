"""Day 33 · GridWorld：状态怎么定义，决定 MDP 成不成立 —— 六个实验

配套讲义：docs/Day33-GridWorld与状态定义.md
只做算术，不带训练循环。**讲义里出现的每个数字都应该出自本文件的输出。**

    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\probe_gridworld.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\probe_gridworld.py --only D

今天的问题只有一个：**`P` 和 `R` 是从哪来的？**
答案是「你写的」。而写错有两种方式，两种都不报错：

  ① 行和不为 1  → Day 32 的压缩映射整条失效（值迭代发散，但不抛异常）
  ② 状态压得太狠 → 马尔可夫性破掉，`V` 根本不存在（Day 29 实验 E 的工程版）

六个实验：
  A 手算对拍：角落格与非角落格的一行
  B 撞墙处理：三种写法，行和与 V 各差多少
  C 奖励：`R(s,a)` 不够，必须写成 `r(s,a,s')` 的期望
  D 局部观测：一个**连墙都没有**的空房间就够破掉马尔可夫性
  E 加动量：坐标不够，状态要变成 (坐标, 上一步动作)
  F 终态：自环 / 全零行 / 忘记特判
"""

from __future__ import annotations

import argparse

import numpy as np

# ── 环境常量（全程固定，改这里就能做参数实验）──────────────
H = W = 5
N_S = H * W
N_A = 4
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ("上", "下", "左", "右")
DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

GOAL = (0, 4)
TRAP = (3, 1)
GOAL_S = GOAL[0] * W + GOAL[1]
TRAP_S = TRAP[0] * W + TRAP[1]
TERMINALS = {GOAL_S, TRAP_S}

GAMMA = 0.9
STEP_REWARD = -0.04
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0

# Day 32 的结论，今天要拿来用
def contraction_factor(gamma: float, row_sum: float) -> float:
    """行和 = row_sum 时，T 的压缩因子是 γ·row_sum。

    Day 32 实验 B 已经量过：δ=0.2 时因子变成 γ(1+δ) = 1.08 > 1。
    今天它换了个身份：**写环境时不小心把行和搞成 2.0，因子就是 1.8。**
    """
    return gamma * row_sum


# ────────────────────────────────────────────────────────────
# 一、环境：一行一行手写 P 和 R
# ────────────────────────────────────────────────────────────

def _step(r: int, c: int, a: int, walls: frozenset) -> tuple[int, int]:
    """执行一个动作。撞墙 / 出界 **留在原地**。

    「留在原地」不是一个细节，是一个设计决策。另一种常见选择是
    「撞墙时随机滑向一个可行方向」，两者的 P 完全不同。
    选哪个都行，但**必须写下来**，因为它决定了 `P` 长什么样。
    """
    dr, dc = DELTA[a]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < H and 0 <= nc < W) or (nr, nc) in walls:
        return r, c
    return nr, nc


def _successors(r, c, a, walls, slip):
    """动作 a 的后继列表 [(下一格坐标, 概率), ...]，概率之和为 1。

    `slip` 是「风」：以 1−slip 走向意图方向，以 slip/2 各滑向两个垂直方向。
    **三个分支的概率必须加起来等于 1** —— 这是行和的来源。
    """
    if slip <= 0.0:
        return [(_step(r, c, a, walls), 1.0)]
    pa, pb = (LEFT, RIGHT) if a in (UP, DOWN) else (UP, DOWN)
    return [
        (_step(r, c, a, walls), 1.0 - slip),
        (_step(r, c, pa, walls), slip / 2.0),
        (_step(r, c, pb, walls), slip / 2.0),
    ]


def _reward(ns: int) -> float:
    """**奖励属于「转移」，不属于「动作」。**（实验 C 会把这句话量化）"""
    if ns == GOAL_S:
        return GOAL_REWARD
    if ns == TRAP_S:
        return TRAP_REWARD
    return STEP_REWARD


def build_gridworld(walls=frozenset(), slip: float = 0.0, mode: str = "correct"):
    """构造 (P, R)。P 形状 (nS, nA, nS)，R 形状 (nS, nA)。

    mode 用来注入四种「会真实发生的」错误写法：
      correct              —— 撞墙留原地、终态自环、奖励取期望
      drop_blocked         —— 撞墙时把那份概率直接丢掉（行和 < 1）
      double_count         —— 撞墙处理完忘了 return，又走了一遍通用路径（行和 > 1）
      no_self_loop         —— 终态不写自环，留一整行 0（行和 = 0）
      terminal_not_special —— 忘记特判终态，智能体能从终点走出去继续拿奖励
    """
    P = np.zeros((N_S, N_A, N_S))
    R = np.zeros((N_S, N_A))

    for r in range(H):
        for c in range(W):
            s = r * W + c

            # ── 终态：动作无意义，两种写法 ──
            if s in TERMINALS and mode != "terminal_not_special":
                if mode == "no_self_loop":
                    continue                      # 故意留一行 0
                P[s, :, s] = 1.0                  # 自环
                continue                          # R 保持 0

            for a in range(N_A):
                succ = _successors(r, c, a, walls, slip)

                if mode == "drop_blocked":
                    # 只保留「真的走动了」的后继，撞墙那份概率凭空消失
                    succ = [(rc, p) for rc, p in succ if rc != (r, c)]

                for (nr, nc), p in succ:
                    if p <= 0.0:
                        continue
                    ns = nr * W + nc               # ← 坐标 → 状态下标，这一步不能省
                    # double_count：撞墙时先 += 1.0 却没 return，又落到这里再加一次
                    mult = 2.0 if (mode == "double_count" and ns == s) else 1.0
                    P[s, a, ns] += p * mult
                    R[s, a] += p * _reward(ns)     # 奖励取期望 → 见实验 C
    return P, R


def value_iteration(P, R, gamma, tol=1e-13, max_iter=200000):
    """值迭代。返回 (V, 轮数, 是否正常收敛)。

    注意这里**没有**任何关于行和的检查 —— 这正是今天的重点：
    行和错了，它不会抛异常，只会给你一个假的数或者直接 nan。
    """
    n_s = P.shape[0]
    v = np.zeros(n_s)
    for k in range(max_iter):
        with np.errstate(over="ignore", invalid="ignore"):
            q = ((R[:, :, None] + gamma * v[None, None, :]) * P).sum(axis=2)
            v_new = q.max(axis=1)
        if not np.all(np.isfinite(v_new)):
            return v_new, k + 1, False
        if np.abs(v_new - v).max() < tol:
            return v_new, k + 1, True
        v = v_new
    return v, max_iter, False


def greedy_policy(P, R, gamma, v):
    q = ((R[:, :, None] + gamma * v[None, None, :]) * P).sum(axis=2)
    return q.argmax(axis=1)


def row_sums(P):
    """每条 (s,a) 的行和。**这一列永远是 1 才对。**"""
    return P.sum(axis=2)


def _coord(s: int) -> str:
    return f"({s // W},{s % W})"


# ────────────────────────────────────────────────────────────
# 实验 A：手算对拍
# ────────────────────────────────────────────────────────────

def exp_a():
    print(f"\n[实验 A] 手算 vs 代码：P 的一行长什么样（γ={GAMMA}）")

    P, R = build_gridworld()

    for s, label in ((0, "角落 (0,0)"), (1 * W + 1, "非角落 (1,1)")):
        print(f"\n    {label}：")
        print(f"      {'动作':>4}{'后继':>12}{'概率':>12}")
        print("      " + "-" * 28)
        for a in range(N_A):
            nz = np.nonzero(P[s, a])[0]
            for ns in nz:
                print(f"      {ACTION_NAMES[a]:>4}{_coord(int(ns)):>12}{P[s, a, ns]:>12.6f}")

    rs = row_sums(P)
    print(f"""
    两处的读数：

    ① **角落 (0,0) 的「上」和「左」都指向自己。** 因为撞墙留在原地 ——
       这不是 bug，是设计。如果换成「撞墙随机滑开」，这两行会完全不同。
    ② **非角落 (1,1) 的四个动作严格都指向不同的格子。** 没有自环，
       因为没有墙可以撞。所以「P 里有自环」这件事本身就是一个诊断信号：
       自环只该出现在撞墙和终态。

    全表 {P.shape[0] * P.shape[1]} 个 (s,a) 的行和：
      最小 {rs.min():.15f}   最大 {rs.max():.15f}
      **与 1 的最大偏差 {np.abs(rs - 1.0).max():.3e}**

    「行和 = 1」就是 Day 32 引理 2 的全部前提（`P·1 = 1`）。
    它在这张表上只是一个断言；在你自己写的环境里，它是**一行代码的自检**：
        assert np.allclose(P.sum(axis=2), 1.0), "行和不为 1"
    """)
    return np.abs(rs - 1.0).max() < 1e-15


# ────────────────────────────────────────────────────────────
# 实验 B：撞墙处理的三种写法
# ────────────────────────────────────────────────────────────

def exp_b():
    print(f"\n[实验 B] 撞墙怎么写 —— 四种实现，行和与 V 各偏多少（γ={GAMMA}）")

    cases = [
        ("正确：撞墙留在原地", "correct", 0.0),
        ("丢概率（无风）", "drop_blocked", 0.0),
        ("丢概率（有风 slip=0.3）", "drop_blocked", 0.3),
        ("双计（先处理后忘了 return）", "double_count", 0.0),
    ]

    print(f"    {'写法':<28}{'slip':>6}{'行和最小':>10}{'行和最大':>10}{'max|ΔV|':>12}{'收敛':>7}")
    print("    " + "-" * 74)
    rows = []
    for label, mode, slip in cases:
        P, R = build_gridworld(slip=slip, mode=mode)
        P_ref, R_ref = build_gridworld(slip=slip, mode="correct")
        v_ref, _, _ = value_iteration(P_ref, R_ref, GAMMA)
        rs = row_sums(P)
        v, n, ok = value_iteration(P, R, GAMMA)
        d = np.abs(v - v_ref).max() if np.all(np.isfinite(v)) else np.inf
        rows.append((label, slip, rs.min(), rs.max(), d, ok))
        d_s = f"{d:.4f}" if np.isfinite(d) else "inf"
        print(f"    {label:<28}{slip:>6.1f}{rs.min():>10.4f}{rs.max():>10.4f}{d_s:>12}"
              f"{'是' if ok else '否':>7}")

    worst = rows[3][3]
    print(f"""
    四行的读数，从最不明显的一行开始：

    ① **「丢概率」在无风时，`max|ΔV| = {rows[1][4]:.6f}` —— 完全看不出来。**
       因为撞墙那份概率消失了，那个动作变成一个「几乎什么都不做」的动作，
       值很低，于是**最优策略根本不会选它**。错的那个格子被策略绕开了。

       ⚠️ 这是最危险的一种错：**端到端的测试（跑一遍训练、看 loss 降没降）
       永远抓不到它**，因为效果根本没进入最优路径。它只在某些初始状态、
       某些探索策略下才会露头，然后表现为「偶尔卡在墙角」。

    ② **同一个 bug，加上风就显形了**：`max|ΔV| = {rows[2][4]:.4f}`。
       因为风让**每个动作**都有概率撞墙，那份丢掉的概率从「很少用到」
       变成了「每一步都在漏」。错的东西被放大了，不是被修好了。

    ③ **「双计」是灾难**：行和最大 {worst:.4f}，压缩因子从 γ={GAMMA}
       变成 **γ·{worst:.1f} = {contraction_factor(GAMMA, worst):.2f} > 1**，
       值迭代在几十轮内溢出成 inf —— 这是唯一会被**下游**发现的一种。

       它就是 Day 32 实验 B 那张 δ 表的工程版本：那里 δ=0.2 时因子 1.08，
       只偏 12% 就越界；这里直接偏了 100%，所以爆得很响。
       **偏一点点的时候不会爆，只会慢得像不收敛** —— 那才是常态。

    ⚠️ 三种错法里，只有第三种会自己暴露。前两种都得靠一行断言：

        assert np.allclose(P.sum(axis=2), 1.0), "P 的行和不为 1"

    这行必须写在**环境里**，不能写在训练脚本里 ——
    训练脚本拿到 P 的时候，它已经错了。
    """)
    return (abs(rows[0][4]) < 1e-15            # 正确写法基线
            and rows[1][4] < 1e-6              # 无风时错误被遮住
            and rows[2][4] > 1e-3              # 有风时暴露
            and not rows[3][5])                # 双计不收敛


# ────────────────────────────────────────────────────────────
# 实验 C：奖励依赖 s'，不是依赖 (s,a)
# ────────────────────────────────────────────────────────────

def exp_c():
    print(f"\n[实验 C] R(s,a) 不够 —— 一个 (s,a) 有两个后继，奖励不一样（slip=0.3）")

    slip = 0.3
    P, _ = build_gridworld(slip=slip)

    # 找偏离最大的一处：同一个 (s,a) 的后继里，r 的取法差最多的
    best = None
    for s in range(N_S):
        if s in TERMINALS:
            continue
        for a in range(N_A):
            nz = np.nonzero(P[s, a])[0]
            if len(nz) < 2:
                continue
            vals = np.array([_reward(int(ns)) for ns in nz])
            probs = P[s, a, nz]
            r_correct = float((probs * vals).sum())
            r_big = float(vals[np.argmax(probs)])              # 最可能后继
            r_max = float(vals.max())                          # 乐观
            gap = max(abs(r_big - r_correct), abs(r_max - r_correct))
            if best is None or gap > best[0]:
                best = (gap, s, a, list(zip(nz, probs, vals)),
                        r_correct, r_big, r_max)

    gap, s, a, succ, r_correct, r_big, r_max = best
    print(f"    最典型的一处：状态 {_coord(s)}，动作「{ACTION_NAMES[a]}」")
    print(f"      {'后继':>12}{'概率':>10}{'进这一格的奖励':>16}")
    print("      " + "-" * 38)
    for ns, p, v in succ:
        print(f"      {_coord(int(ns)):>12}{p:>10.4f}{v:>16.4f}")

    print(f"""
    三种写法的后果（下表全部指 ({_coord(s)}, 「{ACTION_NAMES[a]}」) 这一个动作）。
    为了只看 `R` 的影响，Q 统一用**正确的** `V*` 算，不动下游：
    """)

    # 用三种 R 各解一次 V*
    def _R_from(rule: str):
        out = np.zeros((N_S, N_A))
        for ss in range(N_S):
            if ss in TERMINALS:
                continue
            for aa in range(N_A):
                nz = np.nonzero(P[ss, aa])[0]
                vals = np.array([_reward(int(x)) for x in nz])
                if rule == "expect":
                    out[ss, aa] = (P[ss, aa, nz] * vals).sum()
                elif rule == "likely":
                    out[ss, aa] = vals[np.argmax(P[ss, aa, nz])]
                else:
                    out[ss, aa] = vals.max()
        return out

    R_ok = _R_from("expect")
    R_big = _R_from("likely")
    R_max = _R_from("max")

    # 正确的 V*，作为固定参照
    v_ok, _, _ = value_iteration(P, R_ok, GAMMA)
    downstream = GAMMA * (v_ok[None, None, :] * P).sum(axis=2)   # (nS, nA)

    print(f"      {'写法':<28}{'R(s,a)':>10}{'Q(s,a)':>12}{'ΔQ = ΔR':>12}")
    print("      " + "-" * 62)
    rows = []
    for lab, Rm in (("期望 Σ P(s'|s,a)·r(s,a,s')", R_ok),
                    ("用最可能后继的 r", R_big),
                    ("用 max r", R_max)):
        q = Rm[s, a] + downstream[s, a]
        rows.append((lab, Rm[s, a], q, q - (R_ok[s, a] + downstream[s, a])))
        print(f"      {lab:<28}{rows[-1][1]:>10.4f}{rows[-1][2]:>12.6f}{rows[-1][3]:>+12.6f}")

    # 把错的 R 一路解到底，看全局偏多少
    v_big, _, _ = value_iteration(P, R_big, GAMMA)
    v_max, _, _ = value_iteration(P, R_max, GAMMA)
    d_big = np.abs(v_big - v_ok).max()
    d_max = np.abs(v_max - v_ok).max()

    print(f"""
    读数：

    同一个 (s,a)，三个后继一个进终点两个留在附近，奖励差 {GOAL_REWARD - STEP_REWARD:.2f}。
    **所以「这个动作的奖励是多少」这句话本身没有答案** —— 只能问
    「这个动作的**期望**奖励是多少」。

    正确答案是把 r 定义在**转移**上，再对后继取期望：

        R(s,a) = Σ_s' P(s'|s,a) · r(s,a,s')

    两种错写法都错在同一个地方 —— **它们丢掉了「进终点的那一份概率」**：

      · 「用最可能后继的 r」：`R` 变成 **{rows[1][1]:.4f}**，因为「上」有 70% 撞墙不动，
        于是那 15% 进终点的机会被完全抹掉。Q 偏 **{rows[1][3]:+.4f}**。
        这不是精度问题，是**信息丢失**。

      · 「用 max r」：`R` 变成 **{rows[2][1]:.4f}**，Q 偏 **{rows[2][3]:+.4f}**。
        它必然偏大，理由和 Day 31 实验 A 一模一样：**挑最大只会高不会低。**

    把错的 `R` 一路解到底（连 `V` 一起重算），全局最大偏：
      「最可能后继」 {d_big:.4f}   「max r」 {d_max:.4f}

    工程含义：`R` 存成 `(nS, nA)` 是**可以的**，但那个数必须是期望值，
    而且你得先想清楚 `r(s,a,s')` 是什么。**先有 r，再压成 R，别反过来。**
    ⚠️ 反过来的症状是：策略明明「看起来对」，但总在终点前一步犹豫 ——
    因为那一步的期望奖励被算成了负数。
    """)
    return abs(rows[1][3]) > 1e-3 and rows[2][3] > 0


# ────────────────────────────────────────────────────────────
# 实验 D：局部观测 —— 空房间就够
# ────────────────────────────────────────────────────────────

def local_pattern(r: int, c: int, walls=frozenset()) -> tuple:
    """3×3 邻域的墙形（含出界）。这就是「能看到的东西」。"""
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr, nc = r + dr, c + dc
            blocked = (not (0 <= nr < H and 0 <= nc < W)) or (nr, nc) in walls
            out.append(1 if blocked else 0)
    return tuple(out)


def exp_d():
    print(f"\n[实验 D] 把坐标压成「3×3 局部墙形」—— 一个空房间就够破（γ={GAMMA}）")

    P, R = build_gridworld()          # 注意：**一面内墙都没有**
    v, _, _ = value_iteration(P, R, GAMMA)

    groups: dict[tuple, list[int]] = {}
    for r in range(H):
        for c in range(W):
            s = r * W + c
            if s in TERMINALS:
                continue
            groups.setdefault(local_pattern(r, c), []).append(s)

    print(f"    {N_S} 个格子，去掉 2 个终态 → {N_S - 2} 个有效状态")
    print(f"    压成「3×3 局部墙形」后只剩 **{len(groups)} 种观测**\n")

    print(f"    {'观测':>6}{'格子数':>8}{'成员':>32}{'V 最小':>12}{'V 最大':>12}{'差':>12}")
    print("    " + "-" * 84)
    widest = None
    for gi, (pat, members) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        vals = v[members]
        spread = vals.max() - vals.min()
        names = " ".join(_coord(s) for s in members[:4])
        if len(members) > 4:
            names += " …"
        print(f"    {gi:>6}{len(members):>8}{names:>32}{vals.min():>12.6f}"
              f"{vals.max():>12.6f}{spread:>12.6f}")
        if widest is None or spread > widest[0]:
            widest = (spread, members, vals)

    spread, members, vals = widest
    i_lo = int(np.argmin(vals))
    i_hi = int(np.argmax(vals))
    s_lo, s_hi = int(members[i_lo]), int(members[i_hi])
    print(f"""
    最宽的一组是 {len(members)} 个格子，`V` 从 {vals.min():.6f} 到 {vals.max():.6f}，
    **差 {spread:.6f}**。

    其中 {_coord(s_lo)} 的 V 是 {v[s_lo]:.6f}，{_coord(s_hi)} 的 V 是 {v[s_hi]:.6f} ——
    但它们看到的 3×3 墙形**逐位相同**（都是 9 个 0：周围一圈全是空地）。

    所以 `V(观测)` 有两个值。**不是「算不准」，是这个数不存在。**
    这和 Day 29 实验 E 是同一件事，只是那次是我手搭的两个状态，
    这次是**一个连墙都没有的 5×5 空房间**：
    光靠局部观测，你连「自己离终点多远」都不知道。

    ⚠️ 最要命的是：**这整段代码没有一个地方会报错。**
    你会得到一个 25 维的 V，画出来还挺好看，然后拿它去训练，
    结果一切正常 —— 直到你发现策略在几个位置之间来回震荡。

    压缩映射（Day 32）证得再漂亮也救不了这里：
    它保证的是「`T` 有唯一不动点」，**没保证那个不动点是不是你要的**。
    定理的前提是「你给的是一个 MDP」。前提不成立时，定理什么都不说。
    """)
    return spread > 0.1


# ────────────────────────────────────────────────────────────
# 实验 E：加动量 —— 坐标不够了
# ────────────────────────────────────────────────────────────

def build_momentum(p_keep: float = 0.7, walls=frozenset()):
    """有惯性：上一步走的方向有 p_keep 的概率延续。

    状态必须写成 (格子, 上一步动作)，nS = 25 × 4 = 100。
    只写坐标的 25 状态版本，**根本不是良定义的 MDP**。
    """
    n_s = N_S * N_A
    P = np.zeros((n_s, N_A, n_s))
    R = np.zeros((n_s, N_A))

    for r in range(H):
        for c in range(W):
            pos = r * W + c
            for la in range(N_A):
                s = pos * N_A + la
                if pos in TERMINALS:
                    P[s, :, s] = 1.0
                    continue
                for a in range(N_A):
                    for d, pr in ((la, p_keep), (a, 1.0 - p_keep)):
                        nr, nc = _step(r, c, d, walls)
                        ns = (nr * W + nc) * N_A + a
                        P[s, a, ns] += pr
                        R[s, a] += pr * _reward(nr * W + nc)
    return P, R, n_s


def exp_e():
    print(f"\n[实验 E] 加惯性之后，坐标不够 —— 状态 = (格子, 上一步动作)（γ={GAMMA}）")

    P, R, n_s = build_momentum()
    rs = row_sums(P)
    v, iters, ok = value_iteration(P, R, GAMMA)

    P0, R0 = build_gridworld()
    v0, iters0, _ = value_iteration(P0, R0, GAMMA)

    print(f"    状态数 {N_S} → {n_s}（×{N_A}），行和偏差 {np.abs(rs - 1.0).max():.3e}")
    print(f"    值迭代：无惯性 {iters0} 轮 / 有惯性 {iters} 轮\n")

    # ── 收敛轨迹：γ^k 到底是不是实际速率？──
    def _trace(Pm, Rm, k_max=9):
        vv = np.zeros(Pm.shape[0])
        seq = []
        for _ in range(k_max):
            q = ((Rm[:, :, None] + GAMMA * vv[None, None, :]) * Pm).sum(axis=2)
            vn = q.max(axis=1)
            seq.append(np.abs(vn - vv).max())
            vv = vn
        return seq

    tr0 = _trace(P0, R0)
    tr1 = _trace(P, R)
    print(f"    相邻差 ‖V_k−V_(k-1)‖ 的前 9 轮（γ = {GAMMA}）：")
    print(f"      {'k':>4}{'无惯性':>16}{'有惯性':>16}")
    print("      " + "-" * 36)
    for k in range(9):
        print(f"      {k + 1:>4}{tr0[k]:>16.3e}{tr1[k]:>16.3e}")

    print(f"""
    这一段比下面 V 的差值更重要，因为它**修正了 Day 32 的一个读法**：

    无惯性那列 1.0 → 0.9 → 0.81 → 0.729 …（比值**精确等于 γ**），
    第 9 轮直接变成 **0.000e+00**，然后永远不动。

    不是 bug，是「最优路径最长 9 步」的直接后果 ——
    值迭代从 `V₀ = 0` 出发，第 k 轮的 `V_k` 就是「做 k 步规划能拿到多少」。
    k 一旦超过最优路径长度，`T*` 再作用不改变任何东西：**有限步精确收敛**。

    ⚠️ 所以 Day 32 的 `γ^k` 是**最坏情况的界，不是实际速率**：
      · 有吸收终态、最优策略必达终态 → 有限步精确停，轮数 = 最长路径
      · 没有终态（Day 30 那条 2 状态链）→ 真正的几何收敛，42 轮
    界不会骗你（有限步当然也满足 `γ^k` 的界），但**照着界去估轮数会估多**。

    有惯性那列慢两轮，第 5 轮起开始分岔（{tr0[4]:.3e} 对 {tr1[4]:.3e}）——
    那就是「上一步动作」这个信息第一次进入值里。
    """)


    rows = []
    for r in range(H):
        for c in range(W):
            pos = r * W + c
            if pos in TERMINALS:
                continue
            vals = np.array([v[pos * N_A + la] for la in range(N_A)])
            rows.append((vals.max() - vals.min(), pos, vals))

    rows.sort(reverse=True)
    print(f"    {'格子':>8}{'上一步=上':>12}{'上一步=下':>12}{'上一步=左':>12}{'上一步=右':>12}{'差':>10}")
    print("    " + "-" * 68)
    for sp, pos, vals in rows[:5]:
        print(f"    {_coord(pos):>8}" + "".join(f"{x:>12.6f}" for x in vals) + f"{sp:>10.6f}")

    worst_sp, worst_pos, worst_vals = rows[0]
    print(f"""
    读数：

    **同一个格子，只因为「上一步往哪走」，`V` 差最多 {worst_sp:.4f}**
    （{_coord(worst_pos)}：{worst_vals.min():.6f} 到 {worst_vals.max():.6f}）。

    所以「状态 = 坐标」这个假设**在有惯性的环境里直接不成立**。
    你可以说「那我不用坐标，用坐标 + 速度」—— 对的，但这个 MDP 的状态数从
    {N_S} 变成 {n_s}，DP 每轮的代价从 O(nS²nA) 变成 O(({n_s})²·{N_A})，
    **是 {N_A}² = {N_A ** 2} 倍**。

    这是一条贯穿整个 RL 的张力：
      **状态加得越多 → 马尔可夫性越好 → 状态空间越大 → 越算不动。**
    W6 起函数近似（用神经网络代替表格）就是为了打破这一头。
    但在 P1 里，你要**先看见它有多贵**，才知道后面那些方法在省什么。

    顺带一句：这个环境的正确状态数其实是 {n_s}，但很多人会写成 {N_S}，
    然后发现策略在「刚撞完墙」的位置特别奇怪 —— 因为那一格的价值取决于
    上一步是被哪个动作撞的。**这种 bug 不报错，只表现成「训练不稳定」。**
    """)
    return worst_sp > 0.05 and np.abs(rs - 1.0).max() < 1e-12


# ────────────────────────────────────────────────────────────
# 实验 F：终态那一行
# ────────────────────────────────────────────────────────────

def exp_f():
    print(f"\n[实验 F] 终态怎么写 —— 自环 / 全零行 / 忘记特判（γ={GAMMA}）")

    P_ok, R_ok = build_gridworld()
    v_ok, _, _ = value_iteration(P_ok, R_ok, GAMMA)

    cases = [
        ("自环  P[s,:,s]=1", "correct"),
        ("全零行  什么都不写", "no_self_loop"),
        ("忘记特判", "terminal_not_special"),
    ]
    print(f"    {'写法':<20}{'终态行和':>10}{'V(终点)':>12}{'V(陷阱)':>12}{'max|ΔV|':>12}")
    print("    " + "-" * 68)
    rows = []
    for label, mode in cases:
        P, R = build_gridworld(mode=mode)
        v, _, _ = value_iteration(P, R, GAMMA)
        rs_term = row_sums(P)[GOAL_S].mean()
        d = np.abs(v - v_ok).max()
        rows.append((label, rs_term, v[GOAL_S], v[TRAP_S], d))
        print(f"    {label:<20}{rs_term:>10.4f}{v[GOAL_S]:>12.6f}{v[TRAP_S]:>12.6f}{d:>12.6f}")

    print(f"""
    三行的读数，一条一条说：

    ① **自环**：行和 1，`V(终点) = V(陷阱) = 0`。值是 0 是对的 ——
       到了终点之后不再有奖励，但也不是「没有未来」，而是「未来恒为 0」。

    ② **全零行**：行和 0，`V` 和 ① **完全相同**（差 {rows[1][4]:.1e}）。
       所以这是个**纯记账**问题，不是数学问题：
         行和 0 时 `P·1 ≠ 1`，Day 32 的引理 2（常数平移）在这一行失效，
         虽然它不影响结果（0 ≤ 1 仍是压缩的），但你以后每次引用昨天的定理
         都要额外加一句「除了终态那几行」。
       **自环的唯一好处是让这类例外消失。** 值 1 分，整洁 10 分。

    ③ **忘记特判是唯一真会错的**：行和还是 1，但智能体从终点**能走出去**、
       绕一圈再回来继续领奖，于是 `V(终点) = {rows[2][2]:.6f} > 0`，
       全局最大偏 **{rows[2][4]:.4f}**。

       ⚠️ 这个数不是随便来的：`{rows[2][2]:.6f} = 1/(1−γ) = {1 / (1 - GAMMA):.0f}`。
       它等于「在终点无限次领 +1 的折现和」——
       也就是 Day 29 那个 `|V| ≤ R_max/(1−γ)` 的界**被顶到了等号**。

       还记得 Day 29 怎么说的吗：**`V` 超过这个界一定是代码错。**
       现在它正好等于界 —— 这就是「智能体学会了刷分」的签名。
       遇到 `V` 卡在 `R_max/(1−γ)` 上，第一个要查的就是终态有没有特判。

       ⚠️ 而且它偏得很「合理」：值都变大了，loss 曲线照常下降。
       唯一能抓住它的是断言 `V(终态) == 0`。
    """)
    return (abs(rows[1][4]) < 1e-12          # 全零行数值上等价
            and rows[2][2] > 1e-6            # 忘记特判确实错了
            and rows[2][4] > 1e-3)


# ────────────────────────────────────────────────────────────
EXPERIMENTS = [
    ("A", "手算对拍：P 的一行", exp_a),
    ("B", "撞墙的四种写法", exp_b),
    ("C", "奖励依赖 s'", exp_c),
    ("D", "局部观测破马尔可夫性", exp_d),
    ("E", "加惯性：坐标不够", exp_e),
    ("F", "终态那一行", exp_f),
]


def main():
    parser = argparse.ArgumentParser(description="Day 33 GridWorld 六个实验")
    parser.add_argument("--only", default=None, help="只跑某个实验，如 D")
    args = parser.parse_args()

    picked = [e for e in EXPERIMENTS if args.only is None or e[0] == args.only.upper()]
    if not picked:
        print(f"没有实验 {args.only}，可选 {[e[0] for e in EXPERIMENTS]}")
        return

    results = []
    for key, title, fn in picked:
        print("=" * 70)
        print(f"实验 {key} · {title}")
        print("=" * 70)
        try:
            results.append((key, title, fn()))
        except AssertionError as err:
            results.append((key, title, False))
            print(f"    ✗ 断言失败：{err}")

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
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
