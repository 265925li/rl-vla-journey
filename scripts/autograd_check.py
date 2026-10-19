# -*- coding: utf-8 -*-
"""Day 8 作业：手算 vs autograd vs 有限差分 三方对拍。

为什么非要三种方法？一种方法只能证明"我算出了一个数"：

    手算（解析式） —— 证明你的数学没问题
    autograd       —— 证明框架的反向传播没问题
    有限差分       —— 证明前向函数 f 本身没写错
                      （它完全不看解析式，只用 f 的值，是唯一的独立裁判）

三者一致，才叫算对了。

用法（在**项目根目录** `E:\\work Buddy\\强化学习\\rl-vla-journey` 下）：

    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts/autograd_check.py

为什么全程 float64：
    float32 的相对精度约 1e-7，而对拍门槛是 1e-6 —— 分不清"真 bug"和"精度噪声"。
    规矩：对拍用 float64，训练用 float32。两件事目的不同。
"""

from __future__ import annotations

import math

import torch

TOL = 1e-6          # 过关标准：误差 < 1e-6
H = 1e-6            # 中心差分步长（实测最优区间 1e-5 ~ 1e-6）


# ═══════════════════════════════════════════════════════════════════
# 0. 两个前向函数（纯 Python 版，只给有限差分用）
# ═══════════════════════════════════════════════════════════════════
def f_scalar(xv: float, yv: float) -> float:
    """f(x, y) = x²y + y³"""
    return xv ** 2 * yv + yv ** 3


def f_composite(xv: float, yv: float) -> float:
    """z(x, y) = exp(sin(x·y))"""
    return math.exp(math.sin(xv * yv))


# ═══════════════════════════════════════════════════════════════════
# 1. 方法一：手算的解析式
# ═══════════════════════════════════════════════════════════════════
def hand_grad_f(xv: float, yv: float) -> tuple[float, float]:
    """f = x²y + y³ 的解析式梯度。

    纸上推：
        y 当常数对 x 求导：  ∂f/∂x = 2xy
        x 当常数对 y 求导：  ∂f/∂y = x² + 3y²

    注意返回顺序：(对 x 的, 对 y 的)
    """
    # ── TODO 1 ──────────────────────────────────────────────────────
    # 把上面两个解析式写成代码（用 xv / yv），然后 return
    df_dx = 2*xv*yv
    df_dy = xv**2 + 3*(yv**2)
    return df_dx, df_dy
    # 删掉下面这行，写出你的实现


# ═══════════════════════════════════════════════════════════════════
# 2. 方法二：autograd
# ═══════════════════════════════════════════════════════════════════
def auto_grad_f(xv: float, yv: float) -> tuple[float, float]:
    """交给 autograd 求同一个东西。

    三个必须写对的点：
        ① dtype=torch.float64 —— 别用默认的 float32（见文件头）
        ② requires_grad=True  —— 不给它，就不进计算图，梯度是 None
        ③ 用 torch.autograd.grad(f, (x, y)) 而不是 f.backward()
           前者直接把梯度返回给你，不写 .grad，适合对拍
    """
    x = torch.tensor(xv, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(yv, dtype=torch.float64, requires_grad=True)
    f = x ** 2 * y + y ** 3
    gx,gy = torch.autograd.grad(f,(x,y))
    return gx.item(),gy.item()



# ═══════════════════════════════════════════════════════════════════
# 3. 方法三：中心差分
# ═══════════════════════════════════════════════════════════════════
def numeric_grad(fn, xv: float, yv: float, h: float = H) -> tuple[float, float]:
    """用 f 的值估算梯度，完全不看解析式。

    导数定义：   f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
                 ↑ 中心差分，误差 O(h²)；前向差分 [f(x+h)-f(x)]/h 是 O(h)，更差

    要对 x 和 y 各做一次（另一次保持不动）。
    """
    # ── TODO 3 ──────────────────────────────────────────────────────
    # 两行：分别对 x 和对 y 做中心差分，然后 return
    dx = (fn(xv + h, yv) - fn(xv - h, yv)) / (2 * h)
    dy = (fn(xv, yv + h) - fn(xv, yv - h)) / (2 * h)
    return dx, dy

# ═══════════════════════════════════════════════════════════════════
# 4. 矩阵版：L = Σ(Av)²，梯度是 2Aᵀ(Av)
# ═══════════════════════════════════════════════════════════════════
def hand_grad_matrix(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """L = sum((A @ v)²) 的手推梯度。

    纸上推（设 r = A@v）：
        L = Σ rᵢ²
        ∂L/∂r = 2r                    ← 逐元素平方求导
        ∂r/∂v = A                     ← 线性变换的雅可比就是 A 本身
        链式法则：∂L/∂v = Aᵀ · 2r = 2Aᵀ(Av)
                          ↑
        注意 Aᵀ 不是我随便转的：线性映射反传要用转置。Day 9 手推反传会遇到同一个东西。

    形状自查：v 是 (3,)，梯度也必须是 (3,)。Aᵀ 是 (3,4)、Av 是 (4,) → (3,) ✓
    """
    return 2 * A.T @ (A @ v)

def auto_grad_matrix(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """autograd 版（已写好，供对照）。"""
    v2 = v.clone().detach().requires_grad_(True)
    L = ((A @ v2) ** 2).sum()
    (gv,) = torch.autograd.grad(L, v2)
    return gv


# ═══════════════════════════════════════════════════════════════════
# 5. 对比与报告
# ═══════════════════════════════════════════════════════════════════
def report(name: str, values: list[tuple[str, float]]) -> None:
    """打印一组对拍结果，并断言所有值互相一致。

    values 形如 [("手算", 12.0), ("autograd", 12.0), ("有限差分", 12.0000)]
    """
    labels = " / ".join(f"{k}={v:.10f}" for k, v in values)
    ref = values[0][1]
    worst = max(abs(v - ref) for _, v in values)
    ok = worst < TOL
    mark = "✓" if ok else "✗"
    print(f"[{mark}] {name}")
    print(f"      {labels}")
    print(f"      最大偏差 = {worst:.3e}  (门槛 {TOL:.0e})")
    assert ok, f"{name} 对不上！最大偏差 {worst:.3e}。先查解析式，再用有限差分定位是哪一段错"


def main() -> None:
    print("=" * 66)
    print("第一题：f(x,y) = x²y + y³   在 (x=2, y=3)")
    print("=" * 66)

    x0, y0 = 2.0, 3.0
    hx, hy = hand_grad_f(x0, y0)
    ax, ay = auto_grad_f(x0, y0)
    nx, ny = numeric_grad(f_scalar, x0, y0)

    report("对 x 求偏导", [("手算", hx), ("autograd", ax), ("有限差分", nx)])
    report("对 y 求偏导", [("手算", hy), ("autograd", ay), ("有限差分", ny)])

    print()
    print("=" * 66)
    print("第二题（已给答案，看清链式法则怎么落成代码）：z = exp(sin(x·y)) 在 (0.5, 1.2)")
    print("=" * 66)

    x1, y1 = 0.5, 1.2
    e = math.exp(math.sin(x1 * y1))
    c = math.cos(x1 * y1)
    hx1, hy1 = c * y1 * e, c * x1 * e        # 手算：cos(xy)·y·exp(sin(xy)) / cos(xy)·x·exp(sin(xy))

    # autograd 版：注意必须把 x1t、y1t 存成变量，
    # 因为 autograd.grad 要拿它们的引用去要梯度（别去爬 grad_fn，那条路走不通）
    x1t = torch.tensor(x1, dtype=torch.float64, requires_grad=True)
    y1t = torch.tensor(y1, dtype=torch.float64, requires_grad=True)
    zt = torch.exp(torch.sin(x1t * y1t))
    az1, az2 = torch.autograd.grad(zt, (x1t, y1t))
    nz1, nz2 = numeric_grad(f_composite, x1, y1)

    report("对 x 求偏导", [("手算", hx1), ("autograd", az1.item()), ("有限差分", nz1)])
    report("对 y 求偏导", [("手算", hy1), ("autograd", az2.item()), ("有限差分", nz2)])

    print()
    print("=" * 66)
    print("第三题：L = Σ(Av)²     A:(4,3)  v:(3,)")
    print("=" * 66)

    torch.manual_seed(0)
    A = torch.randn(4, 3, dtype=torch.float64)
    v = torch.randn(3, dtype=torch.float64)

    g_hand = hand_grad_matrix(A, v)
    g_auto = auto_grad_matrix(A, v)
    print(f"      手算   2Aᵀ(Av) = {g_hand.numpy()}")
    print(f"      autograd        = {g_auto.numpy()}")
    same = torch.allclose(g_hand, g_auto, atol=1e-10)
    print(f"[{'✓' if same else '✗'}] 最大偏差 = {(g_hand - g_auto).abs().max().item():.3e}")
    assert g_hand.shape == v.shape, f"梯度形状应该是 {tuple(v.shape)}，你给的是 {tuple(g_hand.shape)}"
    assert same, "矩阵梯度和 autograd 不一致，检查是不是漏了 Aᵀ 或漏了 2"
    assert torch.allclose(g_hand, g_auto, atol=1e-10)

    print()
    print("=" * 66)
    print("全部通过。三张表都一致 = 手算对、框架对、f 也对。")
    print("=" * 66)


if __name__ == "__main__":
    main()
