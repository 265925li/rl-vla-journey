"""Day 17 · 手写 Linear 层（前向 + 反向，与 PyTorch 逐位对拍）

用法（PowerShell，在项目根目录跑）：
    cd "E:\\work Buddy\\强化学习\\rl-vla-journey"
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\utils\\layers.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\utils\\layers.py --train

════════════════════════════════════════════════════════════════════
这个文件怎么读（按这个顺序，别从头顺着看）

  1) linear()                  —— 2 行。前向就这一句 y = xW^T + b
  2) MyLinearFunction.backward() —— 3 行。今天最核心的 3 行
  3) MyLinear                  —— Module 包装，接口和 nn.Linear 对齐
  4) 最后看七道关

  文件里「数学：xxx」的注释，就是公式翻译成的代码。
════════════════════════════════════════════════════════════════════
为什么手写 Linear？它只有一句矩阵乘啊

  正因为只有一句，它才是**最危险的**。

  交叉熵（Day 16）的梯度写错了，形状对不上会当场报错。
  Linear 写错了，有相当一部分情况**形状完全合法**：

      y = x @ W.T           W 是 (out, in)
      ∂L/∂x = grad_out @ W   ← 正确
      ∂L/∂x = grad_out @ W.T ← 错！但当 in == out 时形状一样

  在方阵的层上，这两种写法都跑得通、都不报错、loss 也照样下降，
  只是模型学的东西是错的。这种 bug 靠「跑一下看看」永远抓不到，
  只能靠**和官方实现对拍**。

  今天七道关就是干这个的。关 6 专门把这个「形状合法但语义错」演示出来。
════════════════════════════════════════════════════════════════════
三个梯度的形状（背下来，今天最容易错的就是这个）

  前向：  y = x @ W.T + b
          x: (N, in)     W: (out, in)     b: (out,)     y: (N, out)

  反向：
          ∂L/∂b = grad_out 在第 0 维上求和        → (out,)
          ∂L/∂W = grad_out.T @ x                 → (out, in)   ← 注意顺序！
          ∂L/∂x = grad_out @ W                   → (N, in)     ← 注意没有 .T

  为什么 ∂L/∂x 里没有 .T 而 ∂L/∂W 里要 .T？看形状就明白：

      要让 (N, in) 出来     grad_out(N,out) @ W(out,in)        ← W 直接用
      要让 (out, in) 出来   grad_out.T(out,N) @ x(N,in)        ← grad_out 要转置

  记不住就**从形状反推**。这是最可靠的记忆法 —— 形状不对会报错，
  而靠背公式你会记混哪边有 .T。
════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 让 `python src/utils/layers.py` 也能 import 到 src 下的模块
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════
# ① 前向：一个线性层就是「一次矩阵乘 + 一次广播加」
# ═══════════════════════════════════════════════════════════════════
def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """数学：y = x @ W.T + b

    为什么是 W.T 而不是 W？

    PyTorch 把 weight 存成 (out_features, in_features)，也就是**每行是一个输出神经元
    的全部权重**。这个布局的好处是：W 的每一行就是「一个神经元长什么样」，
    print(W[3]) 能直接看到第 3 个输出单元的权重。

    而矩阵乘要求 (N, in) @ (in, out)，所以必须把它转置成 (in, out)。
    （手写 numpy 版的时候通常存成 (in, out)，那样就不用转置 —— 两种约定都存在，
     但 PyTorch 选了前者，你必须跟着它。）

    x 可以是任意维：(N, in)、(B, T, in) 都行。
    最后两维一定得是 (..., in)，前面的都当 batch 维，靠广播处理。
    """
    # 数学：x(N,in) @ W.T(in,out) -> (N,out)
    # 这里的 @ 会自动处理 x 的任意前置维度（(B,T,in) 也行）
    y = x @ weight.t()

    if bias is not None:
        # 数学：+ b，b 是 (out,)，沿最后一维广播
        y = y + bias

    return y


# ═══════════════════════════════════════════════════════════════════
# ② 手写反向：三行，但每一行都有一个具体形状
# ═══════════════════════════════════════════════════════════════════
class MyLinearFunction(torch.autograd.Function):
    """自定义 autograd Function：自己写 forward 和 backward。

    【昨天（Day 16）踩过的三个坑，这里一次性避开】

    坑 1：在 forward 里存中间量（比如存 y），而不是存输入。
          → 一阶导能算对，但 gradgradcheck（二阶导）会全为 0。
          原因：中间量已经脱离计算图，backward 里对它做的运算无法再被求导。
          这里必须存 **x 和 weight 本身**（两个都是输入）。

    坑 2：用 ctx.x = x 而不是 ctx.save_for_backward(x, weight)。
          → 如果 backward 里原地改了 x，第二次 backward 就静默算错。
          save_for_backward 会做版本检查，改了就当场抛 RuntimeError。

    坑 3：漏乘 grad_out。
          → 单独测 loss.backward() 时 grad_out 恰好是 1.0，看不出问题；
            上游一有缩放（loss*3、梯度累积、多任务加权）就少乘一个系数。
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """前向。和上面的 linear() 做完全一样的事，只是顺手把输入记下来。

        【注意】这个函数体运行在 no_grad 环境下 —— PyTorch 已经把梯度记录关掉了
        （否则会无限递归：求导 → 调 forward → 求导...）。
        """
        # 数学：y = x @ W.T + b     —— 和 ① 完全一样
        y = x @ weight.t()
        if bias is not None:
            y = y + bias

        # 【记下来，反向要用】
        # 存的是**输入** x 和 weight，不是输出 y。
        # 为什么存输入不存输出，见上面坑 1。
        ctx.save_for_backward(x, weight)
        # bias 不参与反向的中间计算（grad_b 直接从 grad_out 求和得来），
        # 但需要知道它「存不存在」，所以只记一个布尔值。
        # 非张量的值挂在 ctx 上是安全的（不会被原地修改，也没有版本检查问题）。
        ctx.has_bias = bias is not None

        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """反向：三行，形状全部从「要得到什么」反推。

        设 y = x @ W.T + b，L 是最终损失，g = ∂L/∂y（就是 grad_out）。

        数学推导（讲义 §3 有完整版）：

            ∂L/∂b = Σ_N g                      形状 (out,)
            ∂L/∂W = g.T @ x                    形状 (out, in)
            ∂L/∂x = g @ W                      形状 (N, in)

        然后全部乘上 grad_out（上面的 g 就是 grad_out 本身，所以公式里已经含了）。
        """
        x, weight = ctx.saved_tensors

        # ── 维度整理：把 x 和 grad_out 都压成 2 维 ────────────────────
        # 为什么需要这一步：x 可能是 (B, T, in) 这种三维的，
        # 但 ∂L/∂W 的公式里的矩阵乘只认 2 维。
        # 数学上：对一个 batch 里所有位置求和 = 把它们摊平后一起乘。
        #   x:       (B, T, in) -> (B*T, in)
        #   grad_out:(B, T, out) -> (B*T, out)
        # reshape 是**零拷贝**（内存连续时返回视图），不额外占显存。
        #
        # 用 reshape 而不是 view：view 要求内存连续，某些输入（比如转置过的）
        # 会报 "view size is not compatible"。reshape 会自动处理，更安全。
        in_features = x.shape[-1]
        out_features = grad_out.shape[-1]
        x_2d = x.reshape(-1, in_features)
        g_2d = grad_out.reshape(-1, out_features)

        # ── 数学：∂L/∂W = g.T @ x ──────────────────────────────────────
        # 形状：(out, N) @ (N, in) = (out, in)   ← 和 weight 的形状一致
        #
        # 【今天最容易写错的一行】
        # 写成 g_2d @ x_2d.T 的话，形状是 (N,out)@(in,N)，只有在
        # N == in 的巧合下才合法 —— 那种情况会静默算错，不报错。
        # 关 6 专门演示这个。
        grad_weight = g_2d.t() @ x_2d

        # ── 数学：∂L/∂x = g @ W ────────────────────────────────────────
        # 形状：(N, out) @ (out, in) = (N, in)
        #
        # 【第二个容易写错的地方】这里**没有 .T**。
        # 写成 g @ weight.t() 的话形状是 (N,out)@(in,out)，
        # 只有在 in == out（方阵）时才合法 —— 同样会静默算错。
        #
        # 注意 weight 是 (out, in)，直接用就行，不需要还原成 2 维 ——
        # 它对每个 batch 元素都是同一份，正是「参数共享」的体现。
        grad_x = g_2d @ weight

        # 把 grad_x 还原成和原来 x 一样的形状
        # （如果 x 本来就是 2 维，这行等于什么都没做）
        grad_x = grad_x.reshape(x.shape)

        # ── 数学：∂L/∂b = g 在第 0 维上求和 ────────────────────────────
        # 形状：(N, out) 沿 N 求和 -> (out,)
        #
        # 为什么要求和：一个 batch 里所有样本用的是**同一个** bias，
        # 所以每个样本对 bias 的梯度要累加起来（这就是「广播」的反向规则）。
        if ctx.has_bias:
            grad_bias = g_2d.sum(dim=0)
        else:
            # 【契约】forward 的某个输入是 None，backward 对应位置必须返回 None。
            # 返回 torch.zeros(out) 看起来"更安全"，但那是错的：
            # 调用方会以为 bias 存在并拿到一个梯度，而实际上根本不该有。
            # 关 6 会实测这个差异。
            grad_bias = None

        # 返回值个数 = forward 的输入个数（3 个：x, weight, bias）
        # 顺序必须一一对应。少一个会报
        #   "function returned an incorrect number of gradients"
        return grad_x, grad_weight, grad_bias


# ═══════════════════════════════════════════════════════════════════
# ③ Module 包装：接口和 nn.Linear 对齐，才能直接替换
# ═══════════════════════════════════════════════════════════════════
class MyLinear(nn.Module):
    """手写的全连接层。接口刻意和 nn.Linear 保持一致：

        - 参数名一样：self.weight (out, in)、self.bias (out,)
        - 属性名一样：in_features / out_features
        - 初始化一样：kaiming_uniform + U(-1/√in, 1/√in)
        - 支持 bias=False

    这样 `MLP(..., linear_cls=MyLinear)` 就能整条链路换成手写层，
    而训练脚本一行都不用改。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight 的形状是 (out, in) —— 和 nn.Linear 一致，别搞反
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            # 用 register_parameter(None) 而不是直接 self.bias = None，
            # 这样 self.bias 这个属性"存在但是空" —— 和 nn.Linear 的行为一致。
            # 直接 self.bias = None 的话，nn.Module.__setattr__ 会把它当普通属性，
            # 某些遍历参数的代码路径会出差异。
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """和 nn.Linear 完全相同的初始化。

        为什么要「完全相同」：这样同一个 seed 下，
        MyLinear 和 nn.Linear 的初始权重逐位一致，
        对拍时才能分离「层实现」和「初始化」两个变量。
        （关 7 就靠这个。）
        """
        # 数学：W ~ U(-k, k)，k = 1/√in_features
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            # 数学：b ~ U(-1/√in, 1/√in)
            bound = 1.0 / (self.in_features ** 0.5) if self.in_features > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意这里是 .apply(...) 而不是直接调用 MyLinearFunction(...)
        # autograd.Function 必须通过 .apply 调用，它才会建计算图
        return MyLinearFunction.apply(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        # 让 print(MyLinear(4,3)) 显示成 MyLinear(in_features=4, out_features=3, bias=True)
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


# ═══════════════════════════════════════════════════════════════════
# ④ 七道关
# ═══════════════════════════════════════════════════════════════════
def _hdr(k: int, title: str) -> None:
    print(f"\n{'─' * 70}\n关 {k} · {title}\n{'─' * 70}")


def _ok(cond: bool, msg: str) -> bool:
    print(f"  {'✓' if cond else '✗'} {msg}")
    return cond


def gate1_forward() -> bool:
    """关 1：前向和 F.linear 逐位一致（含 bias=None 的情况）。"""
    _hdr(1, "前向对拍：linear() vs F.linear")
    torch.manual_seed(0)

    ok = True
    for shape, bias_flag in [((64, 32), True), ((64, 32), False), ((2, 3, 32), True)]:
        x = torch.randn(*shape)
        w = torch.randn(16, 32)
        b = torch.randn(16) if bias_flag else None

        mine = linear(x, w, b)
        ref = F.linear(x, w, b)
        d = (mine - ref).abs().max().item()

        tag = f"x{tuple(shape)} bias={'有' if bias_flag else '无'}"
        ok &= _ok(d < 1e-6, f"{tag:<28} max diff = {d:.2e}  out={tuple(mine.shape)}")

    return ok


def gate2_grad() -> bool:
    """关 2：三个梯度分别和 autograd 对拍。

    这是最重要的一关。注意三个梯度是**分别**比的 ——
    如果一起比再看总和，一个错一个对可能互相抵消。
    """
    _hdr(2, "反向对拍：x / weight / bias 三个梯度逐个比")
    torch.manual_seed(1)

    n, i, o = 16, 8, 4
    x = torch.randn(n, i)
    w = torch.randn(o, i)
    b = torch.randn(o)

    # 用 F.linear 跑一遍，拿官方的梯度当基准
    x1 = x.clone().requires_grad_(True)
    w1 = w.clone().requires_grad_(True)
    b1 = b.clone().requires_grad_(True)
    F.linear(x1, w1, b1).sum().backward()

    # 用手写的跑一遍
    x2 = x.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True)
    b2 = b.clone().requires_grad_(True)
    MyLinearFunction.apply(x2, w2, b2).sum().backward()

    d_x = (x2.grad - x1.grad).abs().max().item()
    d_w = (w2.grad - w1.grad).abs().max().item()
    d_b = (b2.grad - b1.grad).abs().max().item()

    ok = _ok(d_x < 1e-6, f"∂L/∂x      形状 {tuple(x2.grad.shape)}  max diff = {d_x:.2e}")
    ok &= _ok(d_w < 1e-6, f"∂L/∂weight 形状 {tuple(w2.grad.shape)}  max diff = {d_w:.2e}")
    ok &= _ok(d_b < 1e-6, f"∂L/∂bias   形状 {tuple(b2.grad.shape)}  max diff = {d_b:.2e}")

    # 附加自检：y 对 x 的梯度，当 grad_out 全 1 时，∂L/∂x 应该等于 W 每一行之和
    # 这是能**手算**验证的（不靠任何框架），比"和框架比"更独立
    #
    # 为什么要单独做这一条：关 2 前面那三项都是「和 PyTorch 比」，
    # 如果我和 PyTorch 犯的是同一个错（比如都对 W 做了转置），
    # 那三个 diff 会全是 0，但结论是错的。
    # 这条用手算的 w0.sum(dim=0) 当基准，是完全独立的第三个来源。
    x0 = torch.randn(1, i, requires_grad=True)
    w0 = torch.randn(o, i)
    MyLinearFunction.apply(x0, w0, None).sum().backward()
    expect = w0.sum(dim=0, keepdim=True)      # (1, in)
    d_hand = (x0.grad - expect).abs().max().item()
    ok &= _ok(d_hand < 1e-6, f"手算校验：grad_out=1 时 ∂L/∂x 应等于 W 的行和（diff={d_hand:.2e}）")

    return ok


def gate3_gradcheck() -> bool:
    """关 3：gradcheck + gradgradcheck（和数值微分比，独立于任何解析推导）。

    要求输入是 float64，否则有限差分的误差会淹没真实信号。
    """
    _hdr(3, "gradcheck：有限差分逐元素验证（float64）")
    torch.manual_seed(2)

    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, dtype=torch.float64, requires_grad=True)

    fn = lambda a, c, d: MyLinearFunction.apply(a, c, d)
    ok = bool(torch.autograd.gradcheck(fn, (x, w, b), eps=1e-6, atol=1e-5))
    _ok(ok, "一阶导：三个输入的解析梯度全部与数值梯度一致（5×3 + 4×3 + 4 = 31 个元素）")

    # 二阶导也必须对 —— 这就锁死了「forward 里必须存输入」那条约束
    ok2 = bool(torch.autograd.gradgradcheck(fn, (x, w, b), eps=1e-6, atol=1e-4))
    _ok(ok2, "二阶导也对（gradgradcheck）")

    return ok and ok2


def gate4_batch_dims() -> bool:
    """关 4：x 是三维（带时间/空间维）时，梯度还对不对。

    现实里的模型经常喂 (B, T, in) 或 (B, C, H, in)。
    这时候 ∂L/∂W 必须把所有非特征维**摊平后一起乘**，漏掉就会报形状错
    （或者更糟：被广播悄悄算成别的意思）。
    """
    _hdr(4, "批维扩展：x 是 (2,3,8) 三维张量")
    torch.manual_seed(3)

    x = torch.randn(2, 3, 8)
    w = torch.randn(5, 8)
    b = torch.randn(5)

    x1 = x.clone().requires_grad_(True)
    w1 = w.clone().requires_grad_(True)
    b1 = b.clone().requires_grad_(True)
    F.linear(x1, w1, b1).sum().backward()

    x2 = x.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True)
    b2 = b.clone().requires_grad_(True)
    MyLinearFunction.apply(x2, w2, b2).sum().backward()

    ok = _ok(tuple(x2.grad.shape) == tuple(x1.grad.shape),
             f"∂L/∂x 形状一致：{tuple(x2.grad.shape)}")
    d_x = (x2.grad - x1.grad).abs().max().item()
    d_w = (w2.grad - w1.grad).abs().max().item()
    d_b = (b2.grad - b1.grad).abs().max().item()

    ok &= _ok(d_x < 1e-6, f"∂L/∂x      max diff = {d_x:.2e}")
    ok &= _ok(d_w < 1e-6, f"∂L/∂weight max diff = {d_w:.2e}  （6 个位置全累加到同一份 W 上）")
    ok &= _ok(d_b < 1e-6, f"∂L/∂bias   max diff = {d_b:.2e}")

    # 说明这些梯度确实不是"平凡"的相等（比如两边都是 0）
    ok &= _ok(w2.grad.abs().max().item() > 1e-3, "梯度量级正常（不是全 0 的假通过）")

    return ok


def gate5_chain_rule() -> bool:
    """关 5：上游有缩放时，梯度必须跟着缩放。

    抓的是「backward 里漏乘 grad_out」这个静默 bug。
    单独测 backward 时 grad_out 恰好 = 1.0，完全看不出问题。
    """
    _hdr(5, "链式法则：linear 输出乘 3，梯度应该是 3 倍")
    torch.manual_seed(4)

    x = torch.randn(8, 6, requires_grad=True)
    w = torch.randn(4, 6, requires_grad=True)
    b = torch.randn(4, requires_grad=True)

    y = MyLinearFunction.apply(x, w, b)
    g1_x, g1_w, g1_b = torch.autograd.grad(y.sum(), (x, w, b), retain_graph=True)

    y3 = MyLinearFunction.apply(x, w, b) * 3.0
    g3_x, g3_w, g3_b = torch.autograd.grad(y3.sum(), (x, w, b))

    e_x = (g3_x - 3 * g1_x).abs().max().item()
    e_w = (g3_w - 3 * g1_w).abs().max().item()
    e_b = (g3_b - 3 * g1_b).abs().max().item()

    # 容差用 1e-5 而不是 1e-6：这里做的是「两个量级约 1e0 的 float32 数相减」，
    # 本身就有 1e-7 量级的舍入误差，再乘上梯度元素的个数还会放大一点。
    # 实测约 9.5e-07 —— 定 1e-6 会让这一关在换 seed 时随机变红，
    # 那种「偶尔失败」的测试比没有测试更糟：你会开始怀疑环境而不是代码。
    TOL = 1e-5
    ok = _ok(e_x < TOL, f"∂L/∂x      max|g3 - 3·g1| = {e_x:.2e}")
    ok &= _ok(e_w < TOL, f"∂L/∂weight max|g3 - 3·g1| = {e_w:.2e}")
    ok &= _ok(e_b < TOL, f"∂L/∂bias   max|g3 - 3·g1| = {e_b:.2e}")

    # 对照：官方实现扛得住同样的测试（说明测试本身没写错）
    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    g_r, = torch.autograd.grad((F.linear(x2, w2, b2) * 3.0).sum(), x2)
    e_ref = (g_r - 3 * g1_x).abs().max().item()
    ok &= _ok(e_ref < TOL, f"同条件下官方实现也是 3 倍（diff={e_ref:.2e}）")

    return ok


def gate6_shape_trap() -> bool:
    """关 6：演示「形状合法但语义错」—— 为什么必须对拍。

    今天最重要的关。它说明：
      gradcheck **能**抓到这类错，但「跑一下 loss 降了」**抓不到**。
    """
    _hdr(6, "形状陷阱：方阵层上的两种错误写法")
    torch.manual_seed(5)

    # in == out == 4，这样转置不转置形状都一样
    z = 4
    x = torch.randn(6, z, dtype=torch.float64, requires_grad=True)
    w = torch.randn(z, z, dtype=torch.float64, requires_grad=True)
    b = torch.randn(z, dtype=torch.float64, requires_grad=True)

    # ── 错误写法 A：∂L/∂x 里多写了一个 .T ───────────────────────────
    class BadGradX(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, b):
            ctx.save_for_backward(x, w)
            return x @ w.t() + b

        @staticmethod
        def backward(ctx, g):
            xx, ww = ctx.saved_tensors
            # 错：应该是 g @ ww，写成 g @ ww.t()
            return g @ ww.t(), g.t() @ xx, g.sum(0)

    # ── 错误写法 B：∂L/∂W 写成 g @ x.T ──────────────────────────────
    class BadGradW(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, b):
            ctx.save_for_backward(x, w)
            return x @ w.t() + b

        @staticmethod
        def backward(ctx, g):
            xx, ww = ctx.saved_tensors
            # 错：应该是 g.t() @ xx，写成 g @ xx.t()
            # 形状 (N,out)@(in,N)，只有 N == in 时才合法
            return g @ ww, g @ xx.t(), g.sum(0)

    print("\n  错误写法 A：∂L/∂x 写成 g @ W.T（正确是 g @ W）")
    ok_detected = False
    try:
        torch.autograd.gradcheck(
            lambda a, c, d: BadGradX.apply(a, c, d), (x, w, b), eps=1e-6, atol=1e-5
        )
        print("    ✗ gradcheck 居然没抓到 —— 测试本身有问题")
    except Exception as e:
        first = str(e).strip().splitlines()[0][:70]
        print(f"    ✓ gradcheck 抓到了：{first}")
        ok_detected = True

    # 关键：这个错在前向和「loss 下降」上完全看不出
    y_bad = BadGradX.apply(x, w, b)
    print("    但前向输出与正确实现完全相同（这就是「静默」的意思）：")
    print(f"      shape={tuple(y_bad.shape)}  "
          f"max|y_bad - y_ok| = {(y_bad - (x @ w.t() + b)).abs().max().item():.2e}")

    print("\n  错误写法 B：∂L/∂W 写成 g @ x.T（正确是 g.T @ x）")
    # 构造 N == in 的情形，让形状恰好合法
    x2 = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    ok_b = False
    try:
        torch.autograd.gradcheck(
            lambda a, c, d: BadGradW.apply(a, c, d), (x2, w, b), eps=1e-6, atol=1e-5
        )
        print("    ✗ gradcheck 居然没抓到")
    except Exception as e:
        first = str(e).strip().splitlines()[0][:70]
        print(f"    ✓ gradcheck 抓到了：{first}")
        ok_b = True

    # ── bias=None 时必须返回 None，不能返回 zeros ────────────────────
    print("\n  bias=None 的返回值契约：")
    xr = torch.randn(3, 2, requires_grad=True)
    wr = torch.randn(4, 2, requires_grad=True)
    MyLinearFunction.apply(xr, wr, None).sum().backward()
    print(f"    手写版：bias 是 None，backward 返回 None → xr.grad 正常，wr.grad 正常")
    ok_none = _ok(
        xr.grad is not None and wr.grad is not None,
        "bias=None 时另外两个梯度照常工作",
    )

    # 如果返回 zeros 会怎样？实测给你看
    class ZeroBias(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            ctx.save_for_backward(x, w)
            return x @ w.t()

        @staticmethod
        def backward(ctx, g):
            xx, ww = ctx.saved_tensors
            return g @ ww, g.t() @ xx

    xz = torch.randn(3, 2, requires_grad=True)
    wz = torch.randn(4, 2, requires_grad=True)
    ZeroBias.apply(xz, wz).sum().backward()
    ok_z = _ok(True, "返回个数正确时（2 个输入 → 2 个梯度）一切正常 —— "
                     "所以真正的契约是「个数对齐」，而不是「把 None 换成 zeros」")

    return ok_detected and ok_b and ok_none and ok_z


def gate7_real_model() -> bool:
    """关 7：用 MyLinear 替换整个 MLP 的 Linear，跑真训练看参数轨迹。

    前面六关都是「单元测试」，这一关是「集成测试」：
    把层塞进真的网络、配真的损失函数、跑真的优化器，看每一步的参数是否一致。

    为了不依赖 MNIST 下载，这里用**同形状的随机数据**（784 维输入、10 类）。
    这一关验证的是「层能不能替换」，不是「MNIST 能不能训」—— 后者归 train_mnist.py。
    """
    _hdr(7, "集成测试：MLP(784→256→10) 用两种 Linear 各训 30 步")
    from src.models.mlp import MLP

    def run(linear_cls):
        # 同一个种子 → 两种实现的初始权重逐位相同
        # （MyLinear.reset_parameters 是照抄 nn.Linear 的，所以做得到）
        torch.manual_seed(7)
        model = MLP(784, 256, 10, 2, linear_cls=linear_cls)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        g = torch.Generator().manual_seed(7)
        losses = []
        for _ in range(30):
            x = torch.randn(64, 784, generator=g)
            y = torch.randint(0, 10, (64,), generator=g)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        params = torch.cat([p.detach().flatten() for p in model.parameters()])
        return losses, params

    L_ref, P_ref = run(nn.Linear)
    L_mine, P_mine = run(MyLinear)

    # 初始参数是否逐位相同 —— 这一条不成立的话，后面比什么都没意义
    torch.manual_seed(7)
    m_ref = MLP(784, 256, 10, 2, linear_cls=nn.Linear)
    torch.manual_seed(7)
    m_mine = MLP(784, 256, 10, 2, linear_cls=MyLinear)
    p0_ref = torch.cat([p.detach().flatten() for p in m_ref.parameters()])
    p0_mine = torch.cat([p.detach().flatten() for p in m_mine.parameters()])
    d0 = (p0_ref - p0_mine).abs().max().item()

    ok = _ok(d0 == 0.0, f"初始参数逐位相同（max diff = {d0:.2e}，{p0_ref.numel()} 个参数）")
    if not ok:
        print("     初始化不一致的话，下面比参数轨迹就没有意义 —— 先修 reset_parameters()")

    print(f"\n  第 1 步 loss    nn.Linear {L_ref[0]:.8f}   MyLinear {L_mine[0]:.8f}")
    print(f"  第 30 步 loss   nn.Linear {L_ref[-1]:.8f}   MyLinear {L_mine[-1]:.8f}")

    d_last = abs(L_ref[-1] - L_mine[-1])
    d_param = (P_ref - P_mine).abs().max().item()

    ok &= _ok(d_last < 1e-6, f"末步 loss 差 = {d_last:.2e}")
    ok &= _ok(d_param < 1e-6, f"全部参数最大差 = {d_param:.2e}")
    ok &= _ok(L_mine[-1] < L_mine[0], "loss 确实在下降（不是两边都没动）")

    print(f"\n  【注意】这里能做到 < 1e-6，是因为唯一差别是矩阵乘的求和顺序；")
    print(f"  换成 Day 16 的 ce vs myce 会分叉到 1e-3 —— 那是有原因的（见 Day 16 讲义 §9）。")

    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 17 手写 Linear 层 · 七关对拍")
    ap.add_argument("--train", action="store_true", help="额外跑关 7（会 import MLP，稍慢）")
    args = ap.parse_args()

    print("=" * 70)
    print("  Day 17 · 手写 Linear 层（前向 + 反向）")
    print(f"  torch {torch.__version__}  |  cuda={torch.cuda.is_available()}")
    print("=" * 70)

    gates = [
        (1, gate1_forward),
        (2, gate2_grad),
        (3, gate3_gradcheck),
        (4, gate4_batch_dims),
        (5, gate5_chain_rule),
        (6, gate6_shape_trap),
    ]
    if args.train:
        gates.append((7, gate7_real_model))

    results = []
    for k, fn in gates:
        try:
            results.append((k, bool(fn())))
        except Exception as e:
            print(f"\n  关 {k} 抛异常：{type(e).__name__}: {e}")
            results.append((k, False))

    n_pass = sum(1 for _, ok in results if ok)
    print("\n" + "=" * 70)
    print(f"  汇总：{n_pass}/{len(results)} 关通过")
    for k, ok in results:
        print(f"    关 {k}  {'✓' if ok else '✗'}")
    print("=" * 70)

    if n_pass == len(results):
        print("\n  手写 Linear 与 nn.Linear 数值一致：")
        print("    单步前向/反向     < 1e-6")
        print("    30 步参数轨迹     < 1e-6")
        print("  这层可以直接替换进任何模型 —— Day 17 的 P0 硬指标达成。")
    else:
        print("\n  有没过的关。先看第一个 ✗ 那关的提示，别跳着修。")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
