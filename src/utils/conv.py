"""Day 22 · 手写卷积 conv2d（三种实现互相对拍 + 手写反向 + 七关）

用法（PowerShell，在项目根目录跑）：
    cd "E:\\work Buddy\\强化学习\\rl-vla-journey"
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\utils\\conv.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\utils\\conv.py --train

════════════════════════════════════════════════════════════════════
这个文件怎么读（按这个顺序，别从头顺着看）

  1) conv2d_naive()             —— 4 重循环 + 一句乘加。卷积的**定义**
  2) conv2d_im2col()            —— im2col + 一次矩阵乘。卷积的**实现**
  3) MyConv2dFunction.backward() —— 6 行。今天最核心的 6 行
  4) MyConv2d                   —— Module 包装，接口和 nn.Conv2d 对齐
  5) 最后看七道关

  文件里「数学：xxx」的注释，就是公式翻译成的代码。
════════════════════════════════════════════════════════════════════
为什么 P1 第一件事是 CNN？

  P0 的 MLP 停在 98.0%。三条独立实验都撞到同一堵墙：

      hidden 256 → 0.9800      hidden 512 → 0.9818（+0.18pt，不显著）
      wd 最优 1e-4 → 0.9812（+0.12pt，小于一个标准误）
      四组正则化 → 全部下降（−0.18 ~ −0.46pt）

  「再调调看」已经没意义了。墙不是超参数，是**结构**：

      MLP 第一步就把 28×28 摊平成 784 个独立数字
           ↓
      相邻像素的"相邻"关系、笔画的方向和连续性，**全部永久丢失**
           ↓
      它只能背下"784 个位置各是什么"，学不到"这里有个拐角"

  卷积干的事就是**不摊平**：用一个小窗口在图上滑，同一个窗口扫全图。
  于是它天生知道"东西在哪不重要、长什么样才重要"（平移不变），
  而且参数极少 —— 这是它能在小数据上赢的原因。
════════════════════════════════════════════════════════════════════
为什么还要手写卷积？直接 nn.Conv2d 不行吗？

  和 Day 16 / Day 17 完全同一个理由：**nn.Conv2d 是一行你不认识的代码**。

  但卷积比前面两个都更容易写错，因为它的形状有四维
  （N 批 / C 通道 / H 高 / W 宽），错一个维度可能**完全不报错**：

      weight 是 (out, in, kh, kw)  不是 (out, kh, kw, in)  ← TF 才是反的
      张量的逻辑顺序是 NHWC 还是 NCHW？PyTorch 是 NCHW

  写成 (out, kh, kw, in) 在 in==kh==kw 的方阵上形状合法、能跑、loss 也降，
  和 Day 17 的转置错误一模一样。

  今天的打法是把「卷积」拆成三层，让它们互相指认：

      朴素 4 重循环  ← 定义（绝对正确，但慢）
           ↕ 对拍
      im2col + GEMM  ← 真实实现（快，PyTorch 内部也是这个思路）
           ↕ 对拍
      F.conv2d       ← 官方（cuDNN kernel，可能有 Winograd 等优化）

  三个独立的实现同时错得一样的概率，基本为零。
  只对拍两个（手写 vs 官方）时，你没法判断是自己的错还是"优化带来的
  数值差异"—— 有了朴素版当裁判，就能分清。
════════════════════════════════════════════════════════════════════
输出尺寸公式（背下来，今天最容易错的第二处）

      H_out = floor( (H_in + 2·padding − dilation·(kh − 1) − 1) / stride ) + 1

  stride=1, padding=1, kh=3 时：  H_out = H_in        ← "same" 卷积
  stride=2, padding=0, kh=3 时：  H_out = (H_in − 3)//2 + 1

  算错这个不会报错 —— 会在后面接全连接层时突然报形状不匹配，
  而且报的是**很远的位置**，让你以为是全连接层的错。
════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 让 `python src/utils/conv.py` 也能 import 到 src 下的模块
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════
# ① 卷积的定义：一个窗口在图上滑，每个位置做一次点积
# ═══════════════════════════════════════════════════════════════════
def conv2d_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """最朴素的卷积，用循环写。慢，但**定义直白到不可能写错**。

    形状约定（PyTorch 是 NCHW）：
        x      : (N, C_in,  H,  W )    一批图，C_in 个通道
        weight : (C_out, C_in, kh, kw) 每个输出通道有一套 C_in×kh×kw 的窗口
        bias   : (C_out,)              每个输出通道一个标量
        out    : (N, C_out, H_out, W_out)

    数学：out[n, o, i, j] = Σ_{c, u, v} x[n, c, i·s+u, j·s+v] · W[o, c, u, v] + b[o]

    注意 Σ 覆盖三个维度：**输入通道 c、窗口行 u、窗口列 v**。
    这是卷积和全连接唯一实质的区别 —— 这个 Σ 是"局部"的（窗口只有 kh×kw），
    而全连接的 Σ 覆盖整个输入。

    为什么 weight 的通道维在**第二**位（C_out, C_in, ...）而不是最后？
      历史原因（PyTorch 沿用 NCHW 的 C 在前）。
      TensorFlow 是 (kh, kw, C_in, C_out) —— C_out 在最后。
      这两个顺序互转时漏一次 permute，在 C_in == C_out 时**形状合法**、静默出错。
    """
    N, C_in, H, W = x.shape
    C_out, C_in_w, kh, kw = weight.shape

    # 通道数必须对上。这个检查很重要：它是唯一一个"一定会报错"的形状错误，
    # 所以要善用它 —— 剩下的形状错误都可能静默。
    assert C_in == C_in_w, f"输入通道 {C_in} != 权重通道 {C_in_w}"

    # padding：在四周补 0。为什么要补？因为不补的话每次卷积图都变小，
    # 而且边缘像素被窗口扫到的次数远少于中心像素（信息不公平）。
    xp = F.pad(x, (padding, padding, padding, padding)) if padding else x
    Hp, Wp = xp.shape[2], xp.shape[3]

    # 数学：H_out = floor((Hp − kh)/stride) + 1，dilation=1 时 Hp 就是 H+2p
    H_out = (Hp - kh) // stride + 1
    W_out = (Wp - kw) // stride + 1

    # 先建全 0 的输出。注意 dtype/device 必须跟输入一致，
    # 否则后面 += 会报 "expected scalar type Double but found Float"
    out = torch.zeros(N, C_out, H_out, W_out, dtype=x.dtype, device=x.device)

    # ── 四层循环：批 / 输出通道 / 输出行 / 输出列 ─────────────────
    # 这里**故意不给 c/u/v 写循环**，而是用切片 + .sum() —— 效果完全等价，
    # 但代码短一半，而且 PyTorch 会向量化，快得多。
    # 想真正看 7 重循环长什么样，把下面 (patch * weight[o]).sum() 展开成
    # 对 c、u、v 的三层循环即可。
    for n in range(N):
        for o in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    # 取窗口：从 (n, 全部通道) 里切出 kh×kw 那块
                    # 形状 (C_in, kh, kw)
                    patch = xp[
                        n, :,
                        i * stride : i * stride + kh,
                        j * stride : j * stride + kw,
                    ]
                    # 数学：Σ_{c,u,v} patch[c,u,v] · W[o,c,u,v]
                    # 逐元素乘再全加 → 标量。
                    # 注意 weight[o] 的形状也是 (C_in, kh, kw)，和 patch 对齐，
                    # 所以**不需要任何转置**。这是 NCHW 顺序带来的便利。
                    out[n, o, i, j] = (patch * weight[o]).sum()
            # bias 加在通道维上。放在 o 循环里 = 每个输出通道加一次。
            if bias is not None:
                out[n, o] += bias[o]

    return out


# ═══════════════════════════════════════════════════════════════════
# ② 卷积的真实实现：im2col + 一次矩阵乘
# ═══════════════════════════════════════════════════════════════════
def conv2d_im2col(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """和 conv2d_naive 数学上完全等价，但快几十倍。核心思想只有一句话：

        卷积 = 「把每个窗口摊平成一列」+「一次矩阵乘」

    im2col（image to column）干的事：
        输入 (N, C, H, W)
          ↓ 每个位置抠出 C×kh×kw 的窗口，摊平成一列
        输出 (N, C·kh·kw, H_out·W_out)       ← 注意第二维是"窗口内容"

    然后：
        weight.reshape(C_out, C·kh·kw)  @  col(N, C·kh·kw, L)
          → (N, C_out, L)  →  view 成 (N, C_out, H_out, W_out)

    为什么这样会快：矩阵乘是 BLAS/cuBLAS 里优化了四十年的算子，
    让 GPU 一次算一大堆，远胜过让 Python 循环一个个抠窗口。

    ⚠️ 代价是内存：col 的大小是 N·C·kh·kw·H_out·W_out。
       28×28 图、C=1、k=3、stride=1 时，col 是输入的 **9 倍**。
       这也是为什么大图 + 大 kernel 时显存会炸。
    """
    N, C_in, H, W = x.shape
    C_out, _, kh, kw = weight.shape

    # F.unfold 就是 im2col。参数顺序是 (kernel_size, dilation, padding, stride)。
    # 它输出 (N, C_in*kh*kw, L)，L = H_out*W_out。
    col = F.unfold(x, (kh, kw), dilation=1, padding=padding, stride=stride)
    L = col.shape[-1]

    # 权重摊平：(C_out, C_in, kh, kw) → (C_out, C_in*kh*kw)
    # 摊平的顺序必须和 unfold 一致 —— 都是 (C, kh, kw) 的行优先后展开。
    # 如果这里用 permute 换过顺序，两边的"列含义"就对不上，
    # 结果是**每个位置都是乱序点积**：数值完全不报错，但学不到东西。
    w_flat = weight.reshape(C_out, -1)

    # 数学：(C_out, K) @ (N, K, L) → (N, C_out, L)，matmul 自动广播批维
    out = torch.matmul(w_flat, col)

    # 算输出尺寸。用和朴素版**同一个公式** —— 两处不一致就是 bug 源头，
    # 所以关 3 专门验这个公式。
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1
    out = out.view(N, C_out, H_out, W_out)

    if bias is not None:
        # bias 的形状是 (C_out,)，要广播到 (N, C_out, H_out, W_out)
        # 所以中间补两个维度：view(1, C_out, 1, 1)
        out = out + bias.view(1, -1, 1, 1)

    return out


# ═══════════════════════════════════════════════════════════════════
# ③ 手写反向
# ═══════════════════════════════════════════════════════════════════
class MyConv2dFunction(torch.autograd.Function):
    """卷积的 forward + 手写 backward。

    ── 三个梯度的形状（先看形状，再看公式）─────────────────────────
        前向： out = conv(x, W, b)
               x:(N,C_in,H,W)  W:(C_out,C_in,kh,kw)  b:(C_out,)
               out:(N,C_out,H_out,W_out)

        反向： grad_out:(N,C_out,H_out,W_out)

               ∂L/∂b = grad_out 在 (N, H_out, W_out) 上求和   → (C_out,)
               ∂L/∂W = Σ_n  grad_out[n] @ col[n].T           → (C_out, C_in·kh·kw)
               ∂L/∂x = fold( W_flat.T @ grad_out )           → (N, C_in, H, W)

    ── 为什么 ∂L/∂W 要「Σ_n」而 ∂L/∂x 不要 ─────────────────────────
        这是卷积反向最容易糊涂的地方，根因是**参数共享**：

            W 是**同一套**权重，被 N 个样本、每个样本的 L 个位置**共用**
            → 每个样本都对它贡献一次梯度 → 必须把 N 个样本的贡献加起来

            x[n] 只属于第 n 个样本，别的样本影响不到它
            → 不需要跨样本求和

        对照 Day 17 的 Linear：那里 W 也是共享的，所以 ∂L/∂W 也要
        把 N 个样本加起来（写成 grad_out.T @ x，矩阵乘里已经隐含求和）。
        卷积这里因为 L 个位置也要加，所以是「N 和 L 两个维度都求和」。

    ── 为什么 ∂L/∂x 用 fold，而不是「反过来卷积一遍」────────────────
        数学上 ∂L/∂x 确实等于「grad_out 与翻转 180° 的 W 做全卷积」，
        公式更漂亮但实现更绕（要处理 padding 的裁剪）。
        fold 是 unfold 的逆运算：把 (N, C·kh·kw, L) 的列**加回**原图位置。
        重叠区域自动累加，正好就是链式法则要的"每个输入像素被几个窗口用到，
        它的梯度就是那几个窗口贡献之和"。

        一句话：**unfold 把重叠的窗口摊开，fold 把梯度按重叠次数加回去。**
    """

    @staticmethod
    def forward(ctx, x, weight, bias, stride: int, padding: int):
        N, C_in, H, W = x.shape
        C_out, _, kh, kw = weight.shape

        col = F.unfold(x, (kh, kw), dilation=1, padding=padding, stride=stride)
        w_flat = weight.reshape(C_out, -1)
        out = torch.matmul(w_flat, col)          # (N, C_out, L)

        H_out = (H + 2 * padding - kh) // stride + 1
        W_out = (W + 2 * padding - kw) // stride + 1
        out = out.view(N, C_out, H_out, W_out)

        if bias is not None:
            out = out + bias.view(1, -1, 1, 1)

        # ── 存什么：x 和 weight，**不存 col** ────────────────────────
        # Day 16 的坑①：存中间量（col）会让二阶导断链 ——
        # col 是 x 经过 unfold 得到的，它不是叶子张量，autograd 记录不到
        # 「col 是怎么从 x 来的」，于是 gradgradcheck 的 grad_x 会是 0。
        # 正确做法是存**输入**（x），backward 里重新算一次 col。
        # 重算的代价是一点算力，换来的是二阶导正确。
        #
        # 另外：不能用 ctx.p = p 存张量（Day 16 的坑②），必须 save_for_backward，
        # 否则原地修改时不会报错，第一次 backward 对、第二次才错。
        ctx.save_for_backward(x, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        stride, padding = ctx.stride, ctx.padding
        N, C_in, H, W = x.shape
        C_out, _, kh, kw = weight.shape

        # 检查这个被用到了没：漏乘 grad_out 是 Day 16 的坑③，
        # 单测时 grad_out 恰好是 1.0 看不出来，上游一有缩放就少乘系数。
        # 这里直接用，所以没问题 —— 但记住它是个**张量**，不是 1。
        grad_out = grad_out.contiguous()

        # 重新算 col（为了二阶导，见 forward 里的说明）
        col = F.unfold(x, (kh, kw), dilation=1, padding=padding, stride=stride)
        L = col.shape[-1]

        # (N, C_out, H_out, W_out) → (N, C_out, L)
        g2d = grad_out.view(N, C_out, L)

        # ── ∂L/∂W ────────────────────────────────────────────────
        # 数学：∂L/∂W_flat = Σ_n  g2d[n] (C_out, L)  @  col[n].T (L, K)
        #                   → (C_out, K)
        # col.transpose(1, 2) 是 (N, L, K)，matmul 批乘 (N, C_out, K)，
        # 最后 .sum(0) 把 N 个样本的贡献加起来 —— 这就是「参数共享」的代价。
        grad_w = torch.matmul(g2d, col.transpose(1, 2)).sum(0)
        grad_w = grad_w.view_as(weight)

        # ── ∂L/∂b ────────────────────────────────────────────────
        # 数学：Σ over N, H_out, W_out 三个维度，只保留通道维 (C_out,)
        grad_b = grad_out.sum(dim=(0, 2, 3)) if ctx.has_bias else None

        # ── ∂L/∂x ────────────────────────────────────────────────
        # 数学：grad_col[n] = W_flat.T (K, C_out) @ g2d[n] (C_out, L) → (K, L)
        # 然后 fold 回 (N, C_in, H, W)。重叠区域自动相加。
        grad_col = torch.matmul(weight.reshape(C_out, -1).t(), g2d)   # (N, K, L)
        grad_x = F.fold(
            grad_col,
            output_size=(H, W),
            kernel_size=(kh, kw),
            dilation=1,
            padding=padding,
            stride=stride,
        )

        # 返回值个数必须和 forward 的**参数个数**一致（5 个）。
        # stride / padding 是 int，不是张量，对它们求导无意义 → 返回 None。
        # 少返回一个不会报错，会静默把梯度错配给下一个参数 —— 极难查。
        return grad_x, grad_w, grad_b, None, None


class MyConv2d(nn.Module):
    """手写卷积的 Module 包装，构造签名与 nn.Conv2d **完全一致**。

    这是 Day 17 学到的「接口对齐」：只要签名和属性名一致，
    任何用 nn.Conv2d 的地方都能无痛换成 MyConv2d，调用方零改动。

    nn.Conv2d 必有的属性：in_channels / out_channels / kernel_size /
                          stride / padding / dilation / groups / weight / bias
    我们只做 groups=1、dilation=1 的子集，其余参数显式拒绝 ——
    **宁可报错，也不要静默忽略一个参数**。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        # 不支持的参数必须当场报错。如果默默接受 groups=4 却按 groups=1 算，
        # 结果是「能跑、Loss 降、精度低一点」—— 又一个静默 bug。
        if dilation != 1:
            raise NotImplementedError("MyConv2d 只支持 dilation=1")
        if groups != 1:
            raise NotImplementedError("MyConv2d 只支持 groups=1（深度可分离卷积留到以后）")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 权重形状 (C_out, C_in, kh, kw)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            # register_parameter(name, None) 是让 nn.Module 认识"这个属性存在但是空"的
            # 正确写法。直接 self.bias = None 也行，但 register 更明确。
            self.register_parameter("bias", None)

        # 初始化必须和 nn.Conv2d 用同一套（kaiming_uniform + U(-1/√fan_in, 1/√fan_in)），
        # 否则关 7「同 seed 训练一致」永远对不上 —— 起始点不同，后面怎么比都白搭。
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """照抄 nn.Conv2d 的初始化。

        fan_in = C_in · kh · kw （**不含** C_out，因为每个输出通道有自己的窗口）
        kaiming_uniform_(a=√5) 恰好等价于 U(-1/√fan_in, 1/√fan_in)，
        这是 PyTorch 内部的实际做法，照抄才能保证同 seed 下初始权重逐位相同。
        """
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MyConv2dFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding
        )

    def extra_repr(self) -> str:
        # 让 print(model) 的输出里带上关键参数，和 nn.Conv2d 一样
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}"
        )
        if self.padding:
            s += f", padding={self.padding}"
        if self.bias is None:
            s += ", bias=False"
        return s


# ═══════════════════════════════════════════════════════════════════
# ④ 七道关
# ═══════════════════════════════════════════════════════════════════
def _ok(cond: bool, msg: str) -> bool:
    print(f"  {'✓' if cond else '✗'} {msg}")
    return bool(cond)


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def gate1_naive_vs_official() -> bool:
    """关 1：朴素版 vs F.conv2d —— 先把「卷积到底是什么」钉死。

    这一关如果不过，后面全都不用看：说明我对卷积的定义理解就是错的。
    """
    print("\n关1  朴素 4 重循环  vs  F.conv2d")
    torch.manual_seed(0)
    ok = True

    cases = [
        # (N, C_in, H, W, C_out, k, stride, padding)
        (2, 1, 6, 6, 3, 3, 1, 0),
        (2, 1, 6, 6, 3, 3, 1, 1),   # 有 padding
        (2, 3, 7, 7, 4, 3, 2, 1),   # stride=2 + padding=1
        (1, 2, 5, 8, 1, 2, 1, 0),   # 非方图 + 非方 kernel
    ]

    for (N, Cin, H, W, Cout, k, s, p) in cases:
        x = torch.randn(N, Cin, H, W, dtype=torch.float64)
        w = torch.randn(Cout, Cin, k, k, dtype=torch.float64)
        b = torch.randn(Cout, dtype=torch.float64)
        mine = conv2d_naive(x, w, b, stride=s, padding=p)
        ref = F.conv2d(x, w, b, stride=s, padding=p)
        d = _maxdiff(mine, ref)
        ok &= _ok(
            d < 1e-12 and mine.shape == ref.shape,
            f"N={N} C={Cin} {H}x{W} k={k} s={s} p={p}  "
            f"形状 {tuple(mine.shape)}=={tuple(ref.shape)}  最大差 {d:.2e}",
        )
    return ok


def gate2_im2col_vs_both() -> bool:
    """关 2：im2col 版 vs 朴素版 vs 官方 —— 三方互拍。

    为什么要有朴素版当裁判：
      只对拍「im2col vs 官方」时，如果差 1e-7，你无法判断是
        (a) 我写错了，还是
        (b) 官方用了 Winograd / FFT 等算法，本来就有舍入差。
      有朴素版这个「定义级正确」的第三方在，就能定位到底是哪个错。
    """
    print("\n关2  im2col 版  vs  朴素版  vs  F.conv2d（三方互拍）")
    torch.manual_seed(1)
    ok = True

    cases = [
        (2, 1, 8, 8, 4, 3, 1, 0),
        (2, 3, 8, 8, 4, 3, 1, 1),
        (2, 3, 9, 9, 5, 3, 2, 1),
        (1, 4, 7, 7, 2, 5, 1, 2),   # 大 kernel + 大 padding
    ]

    for (N, Cin, H, W, Cout, k, s, p) in cases:
        x = torch.randn(N, Cin, H, W, dtype=torch.float64)
        w = torch.randn(Cout, Cin, k, k, dtype=torch.float64)
        b = torch.randn(Cout, dtype=torch.float64)

        a = conv2d_naive(x, w, b, stride=s, padding=p)
        m = conv2d_im2col(x, w, b, stride=s, padding=p)
        r = F.conv2d(x, w, b, stride=s, padding=p)

        d1, d2, d3 = _maxdiff(a, m), _maxdiff(m, r), _maxdiff(a, r)
        ok &= _ok(
            max(d1, d2, d3) < 1e-12,
            f"k={k} s={s} p={p}  |  朴素-官方 {d3:.2e}  "
            f"im2col-官方 {d2:.2e}  im2col-朴素 {d1:.2e}",
        )

    # 顺便量一下 im2col 到底省了多少时间（这是它的唯一存在理由）
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(8, 3, 32, 32, dtype=torch.float32, device=dev)
    w = torch.randn(16, 3, 3, 3, dtype=torch.float32, device=dev)
    import time

    reps = 5   # 朴素版每次 100ms 级，跑 5 次够了；20 次会让整个脚本慢一分多钟
    t0 = time.perf_counter()
    for _ in range(reps):
        conv2d_naive(x[:2], w)              # 只跑 2 个样本，朴素版太慢
    t_naive = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        conv2d_im2col(x, w)                 # 跑满 8 个样本
    t_im2col = (time.perf_counter() - t0) / reps

    print(f"    参考：8x3x32x32 → 16 通道 k=3")
    print(f"      朴素版(2 样本)  {t_naive * 1000:8.2f} ms")
    print(f"      im2col(8 样本)  {t_im2col * 1000:8.2f} ms"
          f"   ← 4 倍数据量，快了 {t_naive * 4 / t_im2col:.0f} 倍")
    return ok


def gate3_output_shape() -> bool:
    """关 3：输出尺寸公式 —— 会算错但不会报错的东西。

    公式：H_out = floor((H + 2p − d·(k−1) − 1) / s) + 1

    这个错**不会当场报错**，会在两三层之后接全连接层时炸，
    而且炸在很远的位置。所以必须单独验一遍。
    """
    print("\n关3  输出尺寸公式（手算 vs 实际）")
    ok = True

    def hand(H, k, s, p, d=1):
        return (H + 2 * p - d * (k - 1) - 1) // s + 1

    cases = [
        # (H, k, s, p, 期望输出, 说明)
        (28, 3, 1, 1, 28, "same 卷积，尺寸不变"),
        (28, 3, 2, 1, 14, "stride=2 减半"),
        (28, 3, 2, 0, 13, "无 padding 的 stride=2"),
        (28, 5, 1, 0, 24, "k=5 不补边，每边少 2"),
        (28, 2, 2, 0, 14, "偶数 kernel 刚好整除"),
        (7, 3, 1, 0, 5, "最小值，下一层就不能再卷了"),
    ]

    for (H, k, s, p, expect, why) in cases:
        x = torch.randn(1, 1, H, H)
        w = torch.randn(1, 1, k, k)
        y = F.conv2d(x, w, stride=s, padding=p)
        by_hand = hand(H, k, s, p)
        ok &= _ok(
            y.shape[-1] == by_hand == expect,
            f"H={H:<3} k={k} s={s} p={p} → {by_hand:<3} "
            f"(实测 {y.shape[-1]}, 期望 {expect})  {why}",
        )

    # 故意演示算错的后果：两层卷积后尺寸对不上，报错在很远的位置
    print("    反例：把 28x28 接到 32*7*7 的全连接层上")
    x = torch.randn(1, 1, 28, 28)
    conv = nn.Conv2d(1, 32, 3, stride=2, padding=0)   # 28 → 13，不是 14！
    y = conv(x)
    print(f"      conv(k=3,s=2,p=0) 之后是 {tuple(y.shape)}，不是 (1,32,14,14)")
    print(f"      如果按 14 算：32*14*14={32 * 14 * 14}，实际展平后是 {y.numel()}")
    print(f"      → 全连接层会报 mat1 and mat2 shapes cannot be multiplied")
    print(f"      → 而报错位置在**全连接层**，离真正的错因隔了两层")
    return ok


def gate4_backward() -> bool:
    """关 4：手写 backward vs autograd。

    三个梯度逐个比。**必须三个都比**，不能只比 ∂L/∂x ——
    实测过：只比 ∂L/∂x 会把 ∂L/∂W 求和维度写错的情况漏掉（Day 17 的教训）。
    """
    print("\n关4  手写反向  vs  autograd（三个梯度逐个比）")
    ok = True
    torch.manual_seed(2)

    cases = [
        (2, 1, 6, 6, 3, 3, 1, 0),
        (2, 3, 7, 7, 4, 3, 1, 1),
        (2, 3, 8, 8, 4, 3, 2, 1),
    ]

    for (N, Cin, H, W, Cout, k, s, p) in cases:
        # 用 double 跑，把浮点噪声压到 1e-15 级，这样 1e-9 的差就是真错
        x1 = torch.randn(N, Cin, H, W, dtype=torch.float64, requires_grad=True)
        w1 = torch.randn(Cout, Cin, k, k, dtype=torch.float64, requires_grad=True)
        b1 = torch.randn(Cout, dtype=torch.float64, requires_grad=True)

        # 路线 A：官方 conv2d + autograd
        y_ref = F.conv2d(x1, w1, b1, stride=s, padding=p)
        # 用一个固定权重的标量函数当 loss，保证两条路线的 grad_out 完全相同
        gout = torch.randn_like(y_ref)
        (y_ref * gout).sum().backward()

        # 路线 B：手写 Function
        x2 = x1.detach().clone().requires_grad_(True)
        w2 = w1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)
        y_my = MyConv2dFunction.apply(x2, w2, b2, s, p)
        (y_my * gout).sum().backward()

        dx = _maxdiff(x2.grad, x1.grad)
        dw = _maxdiff(w2.grad, w1.grad)
        db = _maxdiff(b2.grad, b1.grad)
        # 同时确认梯度不是全 0（全 0 也会让"差为 0"通过 —— 这是最阴的假通过）
        nx = x2.grad.abs().max().item()
        ok &= _ok(
            max(dx, dw, db) < 1e-12 and nx > 1e-6,
            f"k={k} s={s} p={p}  ∂x {dx:.2e}  ∂W {dw:.2e}  ∂b {db:.2e}"
            f"  (梯度幅度 {nx:.4f})",
        )

    # ── 顺便演示：∂W 漏掉「Σ_n」会怎样 ──────────────────────────
    # 这是卷积反向最常见的错：把 .sum(0) 忘了，只算了一个样本的贡献。
    print("    反例：∂L/∂W 漏掉跨样本求和（少写 .sum(0)）")
    N, Cin, H, W, Cout, k = 4, 2, 6, 6, 3, 3
    x = torch.randn(N, Cin, H, W, dtype=torch.float64)
    col = F.unfold(x, (k, k))
    gout = torch.randn(N, Cout, col.shape[-1], dtype=torch.float64)
    right = torch.matmul(gout, col.transpose(1, 2)).sum(0)
    wrong = torch.matmul(gout[:1], col[:1].transpose(1, 2))[0]   # 只算第 0 个样本
    print(f"      正确（N={N} 样本求和）{right.abs().mean():.4f}")
    print(f"      错误（只算 1 个样本）  {wrong.abs().mean():.4f}"
          f"   → 差了 {right.abs().mean() / wrong.abs().mean():.1f} 倍")
    print(f"      形状完全合法 (C_out, K) = {tuple(wrong.shape)}，不报错")
    return ok


def gate5_gradcheck() -> bool:
    """关 5：gradcheck + gradgradcheck —— 不依赖 PyTorch 卷积实现的独立验证。

    gradcheck 用**有限差分**（数值求导）来对拍解析梯度。
    它不关心你怎么实现的，只关心「输入扰动一点点，输出变化对不对」。

    这是四种验证手段里唯一**完全不依赖参照实现**的一种 ——
    即使 PyTorch 和我的代码错得一模一样，gradcheck 也能抓到。

    gradgradcheck 验二阶导。它专门抓 Day 16 的坑①（存中间量而非输入）。
    """
    print("\n关5  gradcheck + gradgradcheck（有限差分，不看参照实现）")
    ok = True
    torch.manual_seed(3)

    # gradcheck 必须用 double：float32 的有限差分噪声就有 1e-3 量级，
    # 根本分不清"误差"和"真错"。这是新手最常踩的坑。
    for (Cout, Cin, k, s, p) in [(2, 1, 3, 1, 0), (3, 2, 3, 1, 1), (2, 2, 2, 2, 0)]:
        x = torch.randn(2, Cin, 4, 4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(Cout, Cin, k, k, dtype=torch.float64, requires_grad=True)
        b = torch.randn(Cout, dtype=torch.float64, requires_grad=True)

        try:
            torch.autograd.gradcheck(
                lambda x_, w_, b_: MyConv2dFunction.apply(x_, w_, b_, s, p),
                (x, w, b), eps=1e-6, atol=1e-8, rtol=1e-6,
            )
            ok &= _ok(True, f"gradcheck   k={k} s={s} p={p}")
        except Exception as e:
            ok &= _ok(False, f"gradcheck   k={k} s={s} p={p}  → {type(e).__name__}")

        try:
            torch.autograd.gradgradcheck(
                lambda x_, w_, b_: MyConv2dFunction.apply(x_, w_, b_, s, p),
                (x, w, b), eps=1e-6, atol=1e-8, rtol=1e-6,
            )
            ok &= _ok(True, f"gradgradcheck（二阶导）k={k} s={s} p={p}")
        except Exception as e:
            ok &= _ok(False, f"gradgradcheck k={k} s={s} p={p}  → {type(e).__name__}")

    print("    说明：gradgradcheck 过不了只有一个典型原因 ——")
    print("          forward 里存了中间量（col）而不是输入（x）。")
    print("          存 col 的话一阶导完全正确，二阶导全是 0，而且不报错。")
    return ok


def gate6_parameter_accounting() -> bool:
    """关 6：参数账 —— 卷积为什么参数这么少，以及参数量手算对账。

    这一关不验数值，验的是**理解**。CNN 赢过 MLP 的原因不在"更深"，
    在"参数共享"：同一套 3×3 权重扫全图，它学的是"边缘"这种**与位置无关**的东西。
    """
    print("\n关6  参数账：卷积 vs 全连接")
    ok = True

    # 一个 1→16 的 3×3 卷积，和"等价的"全连接层比
    conv = nn.Conv2d(1, 16, 3, padding=1)
    conv_p = sum(p.numel() for p in conv.parameters())
    # 等价全连接：输入 784 → 输出 16*784（每个输出位置都要独立权重）
    fc_p = 784 * (16 * 784) + 16 * 784
    print(f"    Conv2d(1, 16, k=3)          {conv_p:>12,} 参数   ← 16*1*3*3 + 16")
    print(f"    等价的逐位置全连接           {fc_p:>12,} 参数   ← 784*16*784 + 16*784")
    print(f"    比值                       {fc_p / conv_p:>12,.0f} 倍")
    ok &= _ok(conv_p == 16 * 1 * 3 * 3 + 16, "卷积参数量 = C_out·(C_in·kh·kw + 1)")

    # 手算公式 vs 程序：和 P0 里 expected_params() 同一套做法
    print("    逐层手算对账（今天用的 CNN 结构）：")
    layers = [
        ("conv1  1→16  k3 p1", 16 * (1 * 3 * 3) + 16),
        ("conv2  16→32 k3 p1", 32 * (16 * 3 * 3) + 32),
        ("fc1    1568→128", 1568 * 128 + 128),
        ("fc2    128→10", 128 * 10 + 10),
    ]
    total_hand = 0
    for name, n in layers:
        total_hand += n
        print(f"      {name:<22} {n:>10,}")
    print(f"      {'合计':<22} {total_hand:>10,}")

    # 用真实模块算一遍，确认手算没错
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(1568, 128), nn.ReLU(), nn.Linear(128, 10),
    )
    total_prog = sum(p.numel() for p in model.parameters())
    print(f"      程序数出来的             {total_prog:>10,}")
    ok &= _ok(total_hand == total_prog, f"手算 == 程序（{total_hand}）")

    # 关键对比：CNN 参数几乎全在 FC 层
    fc_share = (1568 * 128 + 128 + 1290) / total_prog * 100
    print(f"    ⚠️ 其中 {fc_share:.1f}% 的参数在最后两个全连接层 ——")
    print(f"       真正的卷积部分只占 {100 - fc_share:.1f}%")
    print(f"       这是早期 CNN 的典型形态，也是后来「全局平均池化」被发明的原因")
    return ok


def gate7_replace_in_training() -> bool:
    """关 7：真模型上替换 —— 用 MyConv2d 顶掉 nn.Conv2d 训 30 步。

    前面六关都是单步数学。这一关验的是**多步累积后还对不对**。

    ⚠️ 预期结果和 Day 16/17 一样：单步逐位一致，多步会分叉。
       原因不是"写错了"，是 col 的重算顺序、cuDNN 的 kernel 不同，
       浮点加法顺序不同 → 1e-16 的差被 SGD 放大。
       判据要看**终值差远小于统计噪声**，而不是"差为 0"。
    """
    print("\n关7  真模型训练：MyConv2d 顶替 nn.Conv2d（30 步）")
    ok = True

    def tiny_cnn(conv_cls):
        # 注意：这里不能写 isinstance(m, nn.Conv2d) 之类的判断（Day 17 的坑）。
        # 直接在构造时传类，最干净。
        def C(a, b, k, p):
            return conv_cls(a, b, k, padding=p)
        return nn.Sequential(
            C(1, 4, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            C(4, 8, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(8 * 7 * 7, 16), nn.ReLU(), nn.Linear(16, 10),
        )

    def run(conv_cls):
        torch.manual_seed(42)          # 种子必须在建模型之前（Day 13 的教训）
        model = tiny_cnn(conv_cls)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        crit = nn.CrossEntropyLoss()
        torch.manual_seed(0)           # 数据也固定
        x = torch.randn(64, 1, 28, 28)
        y = torch.randint(0, 10, (64,))
        losses = []
        for _ in range(30):
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        params = torch.cat([p.detach().flatten() for p in model.parameters()])
        return losses, params

    L_ref, P_ref = run(nn.Conv2d)
    L_my, P_my = run(MyConv2d)

    d_first = abs(L_my[0] - L_ref[0])
    d_last = abs(L_my[-1] - L_ref[-1])
    d_param = (P_my - P_ref).abs().max().item()

    print(f"    第 1 步 loss    {L_my[0]:.8f}  vs  {L_ref[0]:.8f}   差 {d_first:.2e}")
    print(f"    第 30 步 loss   {L_my[-1]:.8f}  vs  {L_ref[-1]:.8f}   差 {d_last:.2e}")
    print(f"    全部参数最大差  {d_param:.2e}")

    ok &= _ok(d_first < 1e-9, f"首步一致（{d_first:.2e}）—— 说明前向和一步反传都对")
    # 30 步的判据放宽到 1e-5：这是 Day 17 实测的经验值（30 步 1.49e-08，
    # Day 16 的 100 步是 2.98e-08）。卷积的重算更多，给足余量。
    ok &= _ok(d_param < 1e-5, f"30 步后参数差 {d_param:.2e} < 1e-5")
    ok &= _ok(L_my[-1] < L_my[0], f"loss 确实在降（{L_my[0]:.4f} → {L_my[-1]:.4f}）")

    print("    这张表怎么读：")
    print("      首步差 ~1e-16  → 前向和单步反向是对的（不是巧合）")
    print("      30 步差 ~1e-8   → 浮点误差被 SGD 放大了 8 个数量级，但绝对值仍可忽略")
    print("      这就是「单步一致、多步分叉、终值差远小于噪声」")
    return ok


# ═══════════════════════════════════════════════════════════════════
# ⑤ main
# ═══════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="Day 22 手写卷积，七关对拍")
    ap.add_argument("--train", action="store_true", help="额外跑关 7（真模型训练）")
    ap.add_argument("--all", action="store_true", help="等价于 --train")
    args = ap.parse_args()

    print(f"torch {torch.__version__}   CUDA 可用={torch.cuda.is_available()}")

    results = {
        "关1 朴素版 vs 官方": gate1_naive_vs_official(),
        "关2 三实现互拍": gate2_im2col_vs_both(),
        "关3 输出尺寸公式": gate3_output_shape(),
        "关4 手写反向": gate4_backward(),
        "关5 gradcheck": gate5_gradcheck(),
        "关6 参数账": gate6_parameter_accounting(),
    }
    if args.train or args.all:
        results["关7 真模型训练"] = gate7_replace_in_training()

    print(f"\n{'═' * 70}\n汇总\n{'═' * 70}")
    for k, v in results.items():
        print(f"  {'✅ 通过' if v else '❌ 失败'}   {k}")
    n_pass = sum(results.values())
    print(f"\n  {n_pass}/{len(results)} 关通过")

    if n_pass == len(results):
        print("\n  全过：卷积的前向与反向都与三种独立实现一致（1e-12 级）。")
        print("  ⚠️ 这是「单步一致」，不是「训练结果完全一样」—— 见关 7。")
        return 0
    print("\n  有未通过的关，别往下走。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
