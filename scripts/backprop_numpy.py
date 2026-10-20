# -*- coding: utf-8 -*-
"""Day 9 作业：纯 numpy 手写两层 MLP 的前向 + 反向，与 torch 对拍。

网络结构（x → W1 → ReLU → W2 → softmax+CE）：

    z1 = X @ W1 + b1        (N, H)
    a1 = relu(z1)           (N, H)
    z2 = a1 @ W2 + b2       (N, C)      ← logits
    p  = softmax(z2)        (N, C)
    L  = -mean_i log p[i, y[i]]         ← 标量

你要写的只有 4 处 TODO，落地总共 7 行代码——就是推导笔记里那七行。
其余全部现成：初始化、softmax、前向、交叉熵、有限差分、torch 对拍、表格输出。

── 今天的三道保险（我埋的）──────────────────────────────────
    1. 每个 TODO 后面紧跟 assert 检查形状 —— 转置转错了当场死，不会拖到最后
    2. dlogits 的行和断言为 0 —— 漏 onehot 会立刻被抓
    3. main() 先单独用有限差分验 TODO 1（最小单元），再验整体
       —— 这样 TODO 1 的 bug 不会和 W2/W1 的问题混在一起
────────────────────────────────────────────────────────────

用法（在**项目根目录**下）：

    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts/backprop_numpy.py
"""

from __future__ import annotations

import numpy as np
import torch

SEED = 0

# 两组规模：小规模用来精确定位错误，大规模用来抓"N 相关"的 bug
# （漏 /N 在 N=1 时完全不显形，忘 ReLU 在 N 大时相对误差反而变小 —— 详见文档第五节）
SMALL = {"N": 8, "D": 5, "H": 4, "C": 3}
LARGE = {"N": 100, "D": 20, "H": 16, "C": 10}


# ═══════════════════════════════════════════════════════════════════
# 1. 参数初始化（已给）
# ═══════════════════════════════════════════════════════════════════
def init_params(rng: np.random.Generator, n_in: int, n_hidden: int, n_out: int) -> dict:
    """He 初始化。乘 sqrt(2/n) 是为了让 ReLU 层的激活值方差保持不变。

    Day 10 会专门讲为什么这么除，现在只要知道它比 "randn 直接乘 0.01" 强得多。
    """
    W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)
    b1 = np.zeros(n_hidden)
    W2 = rng.standard_normal((n_hidden, n_out)) * np.sqrt(2.0 / n_hidden)
    b2 = np.zeros(n_out)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# ═══════════════════════════════════════════════════════════════════
# 2. 前向（已给，不用改）
# ═══════════════════════════════════════════════════════════════════
def softmax(z: np.ndarray) -> np.ndarray:
    """沿最后一维归一化。先减 max 是数值安全，不影响结果。"""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def forward(params: dict, X: np.ndarray) -> tuple[np.ndarray, dict]:
    """返回 (p, cache)。cache 里存反向时需要的一切，反向绝不重算前向。"""
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ params["W2"] + params["b2"]
    p = softmax(z2)
    return p, {"X": X, "z1": z1, "a1": a1, "p": p}


def cross_entropy(p: np.ndarray, y: np.ndarray) -> float:
    """mean 版本的交叉熵。注意是 mean —— 这里决定了反向里那个 1/N。"""
    n = p.shape[0]
    return float(-np.log(p[np.arange(n), y] + 1e-12).mean())


# ═══════════════════════════════════════════════════════════════════
# 3. TODO 1：logits 的梯度（今天最核心的一行）
# ═══════════════════════════════════════════════════════════════════
def grad_logits(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """softmax + 交叉熵 对 logits z2 的梯度。形状 (N, C)。

    你推导出来的那个"漂亮形式"就填在这里。三个提醒：
      ① 它只用到 p 和 onehot(y)，完全不需要 z2
      ② 代价函数是 mean，所以别忘了 1/N
      ③ onehot 已经给你构造好了，你要做的是"整体"那一步
    """
    n, n_class = p.shape
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0
    # ── TODO 1 ────────────────────────────────────────────────────
    # 你的推导结果：∂L/∂z2 = ?
    # 形状必须是 (n, n_class)
    # 删掉下面这行，写成你的实现
    dlogits = (p - onehot) / n
    # 填完后下面两道自检自动生效 —— 它们会先于任何后续代码拦住错误
    assert dlogits.shape == p.shape, (
        f"dlogits 形状应为 {p.shape}（和 p 一致），实际是 {dlogits.shape}。"
        "检查是不是把 p 和 onehot 做成了外积，或者对 p 转了置。"
    )
    assert np.allclose(dlogits.sum(axis=1), 0.0, atol=1e-10), (
        "dlogits 每一行加起来必须是 0（softmax 输出和恒为 1 的必然结果）。\n"
        f"实测行和 = {dlogits.sum(axis=1)}\n"
        "常见原因：漏了 onehot（只用了 p），或者 1/N 加在了错的位置。"
    )
    return dlogits


# ═══════════════════════════════════════════════════════════════════
# 4. TODO 2/3/4：反向传播
# ═══════════════════════════════════════════════════════════════════
def backward(params: dict, cache: dict, y: np.ndarray) -> dict:
    """返回 {"W1","b1","W2","b2"} 四个梯度，形状与对应参数一致。"""
    X, z1, a1, p = cache["X"], cache["z1"], cache["a1"], cache["p"]

    dlogits = grad_logits(p, y)     # (N, C)
    dW2 = a1.T @ dlogits
    db2 = dlogits.sum(axis=0)
    # ── TODO 2 ────────────────────────────────────────────────────
    # ∂L/∂W2 和 ∂L/∂b2
    # 形状约束：dW2 必须和 W2 同形 (H,C)；db2 必须和 b2 同形 (C,)
    # 提示：b2 被广播加到每一行，所以 db2 要沿 axis=0 求和
    # 删掉下面这行，写两行

    assert dW2.shape == params["W2"].shape, (
        f"dW2 形状应为 {params['W2'].shape}，实际 {dW2.shape}。"
        "转置方向反了？看一眼文档 4.3：要让 N 在中间相遇。"
    )
    assert db2.shape == params["b2"].shape, (
        f"db2 形状应为 {params['b2'].shape}，实际 {db2.shape}。"
        "是不是用了 axis=1？b2 是沿 N 维广播的，要消掉的是 N。"
    )

    da1 = dlogits @ params["W2"].T
    dz1 = da1 * (z1 > 0)

    assert da1.shape == a1.shape, (
        f"da1 形状应为 {a1.shape}，实际 {da1.shape}。检查 W2 有没有转置。"
    )
    assert dz1.shape == z1.shape, (
        f"dz1 形状应为 {z1.shape}，实际 {dz1.shape}。"
        "逐元素运算不应该改变形状 —— 检查是不是误用了 @。"
    )

    # ── TODO 4 ────────────────────────────────────────────────────
    # ∂L/∂W1 和 ∂L/∂b1
    # 和第 2 步完全同构，只是把 a1 换成 X
    # 删掉下面这行，写两行
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    assert dW1.shape == params["W1"].shape, (
        f"dW1 形状应为 {params['W1'].shape}，实际 {dW1.shape}。"
    )
    assert db1.shape == params["b1"].shape, (
        f"db1 形状应为 {params['b1'].shape}，实际 {db1.shape}。"
    )

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ═══════════════════════════════════════════════════════════════════
# 5. 有限差分：逐元素扰动的独立裁判（已给）
# ═══════════════════════════════════════════════════════════════════
def loss_of(params: dict, X: np.ndarray, y: np.ndarray) -> float:
    p, _ = forward(params, X)
    return cross_entropy(p, y)


def finite_diff(params: dict, X: np.ndarray, y: np.ndarray, h: float = 1e-5) -> dict:
    """对每个参数的每个元素做中心差分。O(参数个数) 次前向，只配当裁判。

    h 用 1e-5：Day 8 测过 U 型曲线，太小会被舍入误差吃掉，太大是截断误差。
    """
    grads = {}
    for key, val in params.items():
        g = np.zeros_like(val)
        for idx in np.ndindex(val.shape):
            old = val[idx]
            val[idx] = old + h
            lp = loss_of(params, X, y)
            val[idx] = old - h
            lm = loss_of(params, X, y)
            val[idx] = old          # ← 必须还原，否则后面全错
            g[idx] = (lp - lm) / (2 * h)
        grads[key] = g
    return grads


def fd_grad_logits(params: dict, X: np.ndarray, y: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """只对 z2 逐元素求有限差分。专门用来单独验证 TODO 1，不受后面几步影响。"""
    a1 = np.maximum(X @ params["W1"] + params["b1"], 0.0)
    z2 = a1 @ params["W2"] + params["b2"]
    g = np.zeros_like(z2)
    for idx in np.ndindex(z2.shape):
        old = z2[idx]
        z2[idx] = old + h
        lp = cross_entropy(softmax(z2), y)
        z2[idx] = old - h
        lm = cross_entropy(softmax(z2), y)
        z2[idx] = old
        g[idx] = (lp - lm) / (2 * h)
    return g


# ═══════════════════════════════════════════════════════════════════
# 6. torch 对拍（已给）
# ═══════════════════════════════════════════════════════════════════
def torch_grads(params: dict, X: np.ndarray, y: np.ndarray) -> tuple[dict, float]:
    """把同一份权重喂给 torch，用 autograd 求梯度。

    注意 dtype 统一 float64 —— 对拍的目的不是"够用"，是"证明公式同一份"。
    float32 会混进 ~3e-8 的舍入误差，虽然也过 1e-5，但没必要给自己留模糊地带。
    """
    Xt = torch.tensor(X, dtype=torch.float64)
    W1 = torch.tensor(params["W1"], dtype=torch.float64, requires_grad=True)
    b1 = torch.tensor(params["b1"], dtype=torch.float64, requires_grad=True)
    W2 = torch.tensor(params["W2"], dtype=torch.float64, requires_grad=True)
    b2 = torch.tensor(params["b2"], dtype=torch.float64, requires_grad=True)

    z1 = Xt @ W1 + b1
    a1 = torch.relu(z1)
    z2 = a1 @ W2 + b2
    loss = torch.nn.functional.cross_entropy(z2, torch.tensor(y))
    loss.backward()

    return {
        "W1": W1.grad.numpy(), "b1": b1.grad.numpy(),
        "W2": W2.grad.numpy(), "b2": b2.grad.numpy(),
    }, float(loss.item())


# ═══════════════════════════════════════════════════════════════════
# 7. 主流程（已给）
# ═══════════════════════════════════════════════════════════════
def make_data(spec: dict, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((spec["N"], spec["D"]))
    y = rng.integers(0, spec["C"], size=spec["N"])
    params = init_params(rng, spec["D"], spec["H"], spec["C"])
    return X, y, params, rng


def report_row(name: str, v1: float, v2: float) -> str:
    return f"  {name:<4} {v1:>18.3e} {v2:>22.3e}"


def main() -> int:
    print("=" * 72)
    print("Day 9 · 两层 MLP 前向 + 手写反向")
    print(f"numpy {np.__version__} | torch {torch.__version__}")
    print("=" * 72)

    # ---------- 第一组：小规模，逐项精确定位 ----------
    s = SMALL
    print(f"\n[1] 小规模  N={s['N']} D={s['D']} H={s['H']} C={s['C']}")
    X, y, params, _ = make_data(s, SEED)
    p, cache = forward(params, X)
    loss = cross_entropy(p, y)
    print(f"    loss = {loss:.10f}   p.shape = {p.shape}")
    print(f"    每行和 = {p.sum(axis=1)}")

    # 1a. 单独验 TODO 1（用有限差分直接对 z2 求梯度）
    dl = grad_logits(p, y)
    dl_fd = fd_grad_logits(params, X, y)
    err_dl = float(np.abs(dl - dl_fd).max())
    print(f"\n    [1a] TODO 1 单独验证：dlogits vs 有限差分 = {err_dl:.3e}")
    if err_dl > 1e-9:
        print("         ✗ TODO 1 有问题。先只查这一行，别管后面。")
        print("           误差约 N-1 倍 → 漏了 1/N")
        print("           每行和不全为 0 → 漏了 onehot")
        return 1
    print("         ✓ 通过（这一行对了，后面才有意义）")

    # 1b. 完整 backward vs torch
    g_np = backward(params, cache, y)
    g_th, loss_th = torch_grads(params, X, y)
    print(f"\n    [1b] numpy vs torch        （torch loss = {loss_th:.10f}，差 {abs(loss - loss_th):.1e}）")

    # 1c. 完整 backward vs 有限差分
    g_fd = finite_diff(params, X, y)
    print(f"\n    [1c] 三方对拍表")
    print(f"    {'参数':<4} {'numpy vs torch':>16} {'numpy vs 有限差分':>22}")
    print("    " + "-" * 44)
    worst_th = worst_fd = 0.0
    bad = []
    for k in ("W1", "b1", "W2", "b2"):
        e1 = float(np.abs(g_np[k] - g_th[k]).max())
        e2 = float(np.abs(g_np[k] - g_fd[k]).max())
        worst_th, worst_fd = max(worst_th, e1), max(worst_fd, e2)
        if e1 > 1e-9:
            bad.append(k)
        print("  " + report_row(k, e1, e2))

    if worst_th > 1e-9 or worst_fd > 1e-6:
        print("\n  ✗ 对拍未通过。按下面顺序判读：")
        if worst_th > 1e-9 and worst_fd < 1e-6:
            print("     numpy vs torch 大、vs 有限差分小 → torch 对拍那边写错了（不该改的地方动了？）")
        elif worst_th < 1e-9 and worst_fd > 1e-6:
            print("     numpy vs torch 小、vs 有限差分大 → 有限差分那边的问题（h 或还原）")
        elif bad == ["W1", "b1"]:
            # 这个模式最能定位：只有第一层的梯度错，说明断点在第一层往回的路上
            print("     只有 W1/b1 错、W2/b2 完全对 —— 这个模式很有用，说明：")
            print("        dlogits 是对的（否则 W2 也错），W2 那一行也是对的。")
            print("        错在 da1 或 dz1：最常见是 dz1 忘了乘 ReLU 导数 (z1 > 0)。")
        elif bad == ["W2", "b2"]:
            print("     只有 W2/b2 错 —— 但 dlogits 在 [1a] 已经单独验过了，")
            print("        所以看 a1.T @ dlogits 那一行：是不是转置方向反了？")
        else:
            print(f"     出错参数：{bad}（两列都大）→ 你的公式错了。看误差倍数：")
            print("        约 N-1 倍 → 漏 /N；0.x 倍 → 忘 ReLU 导数；1.2 倍 → 多套了 softmax")
        return 1

    # ---------- 第二组：大规模，抓"N 相关"的 bug ----------
    L = LARGE
    print(f"\n[2] 大规模  N={L['N']} D={L['D']} H={L['H']} C={L['C']}")
    X2, y2, params2, _ = make_data(L, SEED + 1)
    _, cache2 = forward(params2, X2)
    g_np2 = backward(params2, cache2, y2)
    g_th2, _ = torch_grads(params2, X2, y2)
    worst2 = max(float(np.abs(g_np2[k] - g_th2[k]).max()) for k in g_np2)
    print(f"    numpy vs torch 最大误差 = {worst2:.3e}")
    if worst2 > 1e-9:
        print("    ✗ 小规模过了、大规模没过 → 一定是和 N 有关的 bug（最可能是 /N 的位置）")
        return 1

    print("\n" + "=" * 72)
    print("全部通过。三方一致 = 公式对、实现对、前向也对。")
    print("=" * 72)
    print(f"\n记下你的数字：")
    print(f"  TODO 1 单独对拍      : {err_dl:.3e}")
    print(f"  小规模 vs torch      : {worst_th:.3e}")
    print(f"  小规模 vs 有限差分   : {worst_fd:.3e}")
    print(f"  大规模 vs torch      : {worst2:.3e}")
    print("\n然后去 notebooks/w2_backprop_derivation.md 填第八节那张表。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
