"""Day 16 · 手写 softmax + 交叉熵（前向 + 反向，与 PyTorch 逐位对拍）

用法（PowerShell，在项目根目录跑）：
    cd "E:\\work Buddy\\强化学习\\rl-vla-journey"
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\utils\\losses.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\utils\\losses.py --train

════════════════════════════════════════════════════════════════════
这个文件怎么读（建议按这个顺序，别从头顺着看）

  1) softmax()            —— 3 行。先看这个，它是一切的基础
  2) log_softmax()        —— 5 行。为什么不能 log(softmax(x))？
  3) cross_entropy()      —— 3 行。前向就这三步
  4) MyCrossEntropy.backward()  —— 5 行。今天最核心的 5 行
  5) 最后再看 main() 里的六道关 —— 每道关都在验证一个具体的东西

  文件里「数学：xxx」的注释，就是公式翻译成的代码。
  六道关的标号 关1~关6，和讲义里的编号一一对应。
════════════════════════════════════════════════════════════════════
为什么非要手写一遍？

  Day 13 你用 nn.CrossEntropyLoss() 一行就训完了 MNIST，0.98 的准确率。
  那为什么还要拆开？

  因为那是**一行你不认识的代码**。它内部做了三件事：
      ① softmax     把 logits 变成概率
      ② log + 取负  把概率变成 loss
      ③ 反向时算梯度 —— 也就是那个 (p - onehot) / N

  第 ③ 步是整条链上最难的一块。等你做到 P3（VLA / 强化学习），
  会碰到很多 PyTorch 没有现成实现的目标函数（比如 PPO 的 clip 损失、
  策略梯度里的 log_prob 加权）。到那时你必须能自己写出它的 forward
  和 backward —— 而「自己写 backward」这件事，唯一的练法就是
  先把手写的东西和官方实现逐位对拍一遍，确认自己会。

  今天就是那个练习。
════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 让 `python src/utils/losses.py` 也能 import 到 src 下的模块
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════
# ① softmax：把一排任意实数，变成一排和为 1 的正数
# ═══════════════════════════════════════════════════════════════════
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """数学：softmax(x)_i = exp(x_i) / Σ_j exp(x_j)

    代码比公式多了一步「先减最大值」。这一步不改变数学结果：

        exp(x_i - m) / Σ exp(x_j - m)
      = exp(x_i)·exp(-m) / (Σ exp(x_j)·exp(-m))
      = exp(x_i) / Σ exp(x_j)                      ← exp(-m) 上下约掉了

    但它救了一个致命的数值问题，见下面关 3。
    """
    # 数学：m = max_j x_j，沿最后一维取，形状要能广播回去所以 keepdim=True
    m = x.max(dim=dim, keepdim=True).values

    # 减完之后，最大的那个元素变成 0，所以 exp 的最大值是 exp(0)=1
    # → 永远不可能溢出成 inf
    shifted = x - m

    # 数学：分母 Σ_j exp(x_j - m)
    e = shifted.exp()

    # 数学：softmax = 分子 / 分母
    # 分母 keepdim 保持形状，不然 (N,C) / (N,) 会广播错（这一步极易出错）
    return e / e.sum(dim=dim, keepdim=True)


# ═══════════════════════════════════════════════════════════════════
# ② log_softmax：为什么不能写成 torch.log(softmax(x))？
# ═══════════════════════════════════════════════════════════════════
def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """数学：log_softmax(x)_i = x_i - log Σ_j exp(x_j - m) - m

    为什么不写成 `softmax(x).log()`？

      因为 softmax 的输出可能下溢成 0。比如 x = [0, -100]：
          softmax  → [0.9999..., 3.7e-44]   勉强还没到 0
          x = [0, -1000] → [1.0, 0.0]        真的变成 0 了
          再 log(0) → -inf                     loss 直接爆掉

      而正确的 log_softmax 算的是 -1000，是个正常的有限值。
      所以 PyTorch 内部的 CrossEntropyLoss 走的是 log_softmax 这条路，
      不是 softmax → log。**先取对数再归一化，永远比先归一化再取对数安全。**

    实测（关 3 会复现）：logits 量级到 ±1000 时，
    `softmax(x).log()` 出现 -inf，而本函数仍是有限值。
    """
    # 数学：m = max_j x_j（防溢出，同 softmax）
    m = x.max(dim=dim, keepdim=True).values
    shifted = x - m

    # 数学：logsumexp = log Σ_j exp(x_j - m)
    # 这就是那个「log-sum-exp 技巧」：先减 max 再 exp 再 log，全程不溢出
    # 注意顺序：先 sum 再 log，不是先 log 再 sum
    lse = shifted.exp().sum(dim=dim, keepdim=True).log()

    # 数学：log_softmax = (x - m) - logsumexp
    # 展开就是 x - m - log Σ exp(x - m)，和上面公式一致
    return shifted - lse


# ═══════════════════════════════════════════════════════════════════
# ③ 交叉熵的纯前向版本（不参与 autograd，只用来算数）
# ═══════════════════════════════════════════════════════════════════
def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """数学：L = -(1/N) Σ_i log p_i[y_i]

    只有两步，非常短：
        ① log_softmax 把 logits 变成 log 概率
        ② 按 target 取出「正确那一类的 log 概率」，取负、取均值

    【关键认知：交叉熵自带 softmax】
    所以模型的最后一层**不要**再套 softmax。
    如果你在 MLP 末尾加了 Softmax 层，再喂给 CrossEntropyLoss，
    就是套了两层 softmax —— 梯度会被压扁，loss 永远降不到 0
    （Day 12 算过，两层 softmax 的 loss 下界是 ln3 ≈ 1.0986，
      MNIST 上表现为 loss 卡在 1.0 附近再也下不去，但完全不报错）。
    """
    # ① 数学：log p = log_softmax(logits)
    log_p = log_softmax(logits, dim=-1)

    # ② 取出正确类的 log 概率。
    # 这一行等价于 log_p[i, target[i]] for i in range(N)，但它是向量化的。
    # arange(N) 是「第 i 行」，target 是「那一行的第几列」，两个索引一对就是对角线上的值。
    n = logits.shape[0]
    picked = log_p[torch.arange(n, device=logits.device), target]

    # 数学：L = -mean(log p_y)
    # 注意是 mean 不是 sum —— 这个选择决定了反向里的那个 1/N 从哪来
    return -picked.mean()


# ═══════════════════════════════════════════════════════════════════
# ④ 今天的主角：手写 forward + backward
# ═══════════════════════════════════════════════════════════════════
class MyCrossEntropyFunction(torch.autograd.Function):
    """把 softmax + 交叉熵打包成一个「原子操作」，反向由我们自己写。

    【为什么要继承 torch.autograd.Function】

    正常写 Python 函数，PyTorch 不知道它内部做了什么，所以没法对它求导
    （它只会看到一个"未知操作"）。继承这个类，等于告诉 PyTorch：

        "这个操作的前向长这样，反向我手写好了，你直接用。"

    于是它可以像 nn.Linear 一样被接进计算图，backward() 会自动调用我们的版本。

    【两个必须遵守的契约 —— 违反会报错或静默出错】

    1. forward 是 @staticmethod，第一个参数固定叫 ctx
       ctx 就是「这一层的记事本」，前向时记下反向需要的东西。

    2. backward 必须返回一个**元组**，长度 = forward 的输入个数（不含 ctx）
       这里 forward(ctx, logits, target) 有 2 个输入
         → backward 必须返回 (grad_logits, grad_target)
         → target 是整数张量，不需要梯度，所以第二个位置填 None
       返回个数不对 → 报 "function returned an incorrect number of gradients"

    3. 想支持二阶导，就必须在 backward 里**从输入重新算**，不能用前向存下的中间量。
       详细解释见 backward 的注释。
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前向：和 cross_entropy() 做完全一样的事。

        【注意】这个函数体里运行在 no_grad 环境下 —— PyTorch 已经把
        logits 的梯度记录关掉了（否则会无限递归：求导 → 调 forward → 求导...）。
        所以这里不能指望 autograd，一切都要自己算。
        """
        # 数学：log p = log_softmax(logits)     —— 和上面 ③ 完全一样
        log_p = log_softmax(logits, dim=-1)

        n = logits.shape[0]

        # 数学：L = -(1/N) Σ log p_y
        loss = -log_p[torch.arange(n, device=logits.device), target].mean()

        # 【记下来，反向要用】存的是 logits（输入），不是 p（中间结果）。
        #
        # 为什么？因为想让二阶导也能算（gradgradcheck 会检查这一点）。
        #   - 如果存 p：p 是在 no_grad 环境里算出来的，它身上没有"我是怎么来的"这张图。
        #     backward 里拿它做运算，得到的东西同样没有图 → 二阶导恒为 0。
        #     而 gradgradcheck 会报 "analytical gradient is all zeros"（我第一版就栽在这）。
        #   - 如果存 logits：它在图里。backward 里对它重算 softmax，
        #     这一步会被记录下来 → 二阶导能一路接回去。
        #
        # 代价是 backward 要多算一次 softmax（约 1.5 倍开销）。
        # torch 官方的 CrossEntropyLoss 用 C++ 把 p 同时存下来，快在这一点 ——
        # 但那是性能优化，不是原理差别。
        #
        # 另外为什么要用 save_for_backward 而不是 ctx.logits = logits：
        #   save_for_backward 会做原地修改检查，如果你在 backward 前改了这个张量，
        #   它会抛 "one of the variables needed for gradient computation has been
        #   modified by an inplace operation" —— 帮你抓 bug。
        #   ctx.xxx = 什么都不检查，改了就静默算错。
        # 张量一律走 save_for_backward，非张量的普通 Python 值才挂 ctx。
        ctx.save_for_backward(logits, target)
        ctx.n = n

        return loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """反向：5 行，就是今天要背下来的东西。

        数学推导（讲义第 3 节有完整版，这里只放结论）：

            L = -(1/N) Σ_i log p_i[y_i]
            ∂L/∂z_ij = (1/N) · (p_ij - 1{j == y_i})

        写成矩阵就是：   ∂L/∂z = (p - onehot(y)) / N

        这个结果漂亮得反常 —— 它**只用到 p 和 y**，完全不需要 z 本身。
        这正是「softmax 的导数」和「log 的导数」互相约掉的结果。
        也正因为如此，交叉熵的梯度永远被限制在 [-1/N, 1/N] 的量级内，
        所以它几乎不会梯度爆炸（这是它比 MSE + sigmoid 好训的根本原因）。

        【grad_out 是什么 —— 今天最容易漏的一行】
        backward 不是「求 ∂L/∂z」，而是求「∂FinalLoss/∂z」。
        如果这个损失后面还接了别的运算（比如 loss*3、或者两个 loss 相加），
        上游会把「FinalLoss 对当前这个 loss 的梯度」通过 grad_out 传进来。

        数学：∂L_total/∂z = (∂L_total/∂L) · (∂L/∂z) = grad_out · (p - onehot)/N

        漏乘 grad_out 的后果：
          - 单独测 loss.backward() 时 grad_out 恰好 = 1.0，**结果完全正确**
          - 一旦上游有缩放（loss*3 / 梯度累积 / 多任务加权），梯度就少乘一个系数
          - 而它**不报错**，你只会觉得"这个模型怎么训不动"
        关 5 专门复现这个 bug。
        """
        logits, target = ctx.saved_tensors
        n = ctx.n

        # 数学：p = softmax(z)      —— 从 logits 重算，保证二阶导能接上图
        p = softmax(logits, dim=-1)

        # 数学：onehot(y)  —— 正确那一类为 1，其余 0
        # scatter_ 是原地写入：在第 1 维（类别维）按 target 指定的位置写 1.0
        # target.unsqueeze(1) 把 (N,) 变成 (N,1)，才能和 (N,C) 的索引对齐
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, target.unsqueeze(1), 1.0)

        # 数学：∂L/∂z = (p - onehot) / N
        #
        # 这里故意**不**写成 grad = p.clone(); grad[arange, target] -= 1
        # 那种原地写法有两个坑（关 6 会把它们跑出来给你看）：
        #   ① 忘了 clone 就会污染 p，第二次 backward 结果全错，而且第一次完全正常
        #   ② 原地索引赋值会切断二阶导的计算图
        # 用「先算 onehot 再减」是纯函数式写法，两个坑都不存在。
        grad = (p - onehot) / n

        # 数学：乘以链式法则传下来的 grad_out
        # 注意 grad_out 是 0 维标量（因为 loss 是标量），直接乘即可广播
        return grad * grad_out, None


# ═══════════════════════════════════════════════════════════════════
# ⑤ 包装成 nn.Module —— 这样它能直接替换 nn.CrossEntropyLoss()
# ═══════════════════════════════════════════════════════════════════
class MyCrossEntropy(nn.Module):
    """接口与 nn.CrossEntropyLoss() 完全一致：crit(logits, target) -> 标量。

    为什么再包一层 Module，而不是直接调 MyCrossEntropyFunction.apply？

      因为 train_mnist.py 里 crit 是当**对象**用的（`crit(logits, yb)`）。
      nn.CrossEntropyLoss 是 Module，所以它可调用、能被 print、
      能放进 nn.Sequential、能被 pickle 进 checkpoint。Function 这些都不行。

      这是 PyTorch 的一条惯例：
        Function 是"算子"（无状态、只管算）
        Module  是"组件"（有状态、能组合、能存盘）
      两者用一层 forward 连起来：Module.forward 里调 Function.apply。
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction != "mean":
            # 只实现 mean：sum 只是差一个 N 倍，none 要处理 grad_out 的形状，
            # 两者对理解核心没有增量。留着这个 raise 是为了避免
            # "传了 sum 但它默默按 mean 算"这种静默错误。
            raise ValueError(
                f"手写版只实现了 reduction='mean'（与 torch 默认一致），收到 {reduction!r}"
            )
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 传 int32 的 target 会在 forward 里索引时静默取错（负数会被当成倒数），
        # 所以在这里强制转成 long。torch 自己也会做这一步。
        target = target.long()
        return MyCrossEntropyFunction.apply(logits, target)

    def extra_repr(self) -> str:  # 让 print(crit) 显示成 MyCrossEntropy()
        return f"reduction={self.reduction}"


# ═══════════════════════════════════════════════════════════════════
# ⑥ 六道关（每关验证一个具体的东西）
# ═══════════════════════════════════════════════════════════════════
def _hdr(k: int, title: str) -> None:
    print(f"\n{'═' * 70}\n关 {k} · {title}\n{'═' * 70}")


def _ok(cond: bool, msg: str) -> bool:
    print(f"  {'✅' if cond else '❌'}  {msg}")
    return cond


def gate1_forward() -> bool:
    """关 1：前向的值和 F.cross_entropy 逐位一致。

    这一关要是不对，后面全都不用看了 —— 说明前向公式本身就错了。
    """
    _hdr(1, "前向数值 vs F.cross_entropy")
    torch.manual_seed(0)
    ok = True
    for i, (n, c) in enumerate([(8, 10), (128, 10), (1, 3), (64, 100)]):
        logits = torch.randn(n, c)
        target = torch.randint(0, c, (n,))
        mine = cross_entropy(logits, target)
        ref = F.cross_entropy(logits, target)
        diff = abs(mine.item() - ref.item())
        ok &= _ok(diff < 1e-6, f"N={n:<4} C={c:<4}  手写={mine.item():.10f}  "
                              f"torch={ref.item():.10f}  diff={diff:.2e}")
    return ok


def gate2_grad() -> bool:
    """关 2：handwritten backward 的梯度与 autograd 一致。"""
    _hdr(2, "反向梯度 vs autograd 自动微分")
    torch.manual_seed(1)
    logits = torch.randn(32, 10, requires_grad=True)
    target = torch.randint(0, 10, (32,))

    # 路线 A：用我们的 Function，梯度由手写的 backward 给出
    loss_mine = MyCrossEntropyFunction.apply(logits, target)
    g_mine, = torch.autograd.grad(loss_mine, logits)

    # 路线 B：用 torch 的 F.cross_entropy，梯度由 autograd 自动推出
    loss_ref = F.cross_entropy(logits, target)
    g_ref, = torch.autograd.grad(loss_ref, logits)

    diff = (g_mine - g_ref).abs().max().item()
    ok = _ok(diff < 1e-6, f"loss    手写={loss_mine.item():.10f}  "
                          f"torch={loss_ref.item():.10f}")
    ok &= _ok(diff < 1e-6, f"梯度 max|diff|={diff:.2e}  形状={tuple(g_mine.shape)}")

    # 顺手验证那个「漂亮性质」：每一行的梯度之和必须是 0
    # 因为 softmax 输出之和恒为 1，所以对 logits 的任何"整体平移"都不改变 loss，
    # 沿着 [1,1,...,1] 方向的导数当然为 0。
    row_sum = g_mine.sum(dim=1).abs().max().item()
    ok &= _ok(row_sum < 1e-7, f"每行梯度求和 max|Σ|={row_sum:.2e}（应严格为 0）")
    return ok


def gate3_overflow() -> bool:
    """关 3：数值稳定性 —— 朴素 exp 会溢出，减 max 不会。

    这是「为什么 softmax 里必须写 x - x.max()」的实证。
    float32 的 exp 溢出阈值是 88.7（e^88.7 ≈ 3.4e38 = float32 上限）。
    """
    _hdr(3, "数值稳定性：logits 量级放大到 1000 会怎样")
    logits = torch.tensor([[0.0, 991.0, 1000.0]])

    naive = logits.exp() / logits.exp().sum(-1, keepdim=True)
    stable = softmax(logits)
    print(f"  logits          = {[round(v, 1) for v in logits.tolist()[0]]}")
    print(f"  朴素 exp 分子   = {[float(v) for v in logits.exp().tolist()[0]]}   ← 溢出成 inf")
    print(f"  朴素 softmax    = {naive.tolist()[0]}   ← nan")
    print(f"  本文件 softmax  = {[round(float(v), 6) for v in stable.tolist()[0]]}")

    ok = _ok(not torch.isfinite(naive).all(), "朴素版确实产生了非有限值（inf/nan）")
    ok &= _ok(torch.isfinite(stable).all(), "先减 max 的版本全程有限")
    ok &= _ok(abs(stable.sum().item() - 1.0) < 1e-6, "稳定版仍然归一化到 1")

    # ── log(softmax) 下溢成 -inf：为什么不能先 softmax 再 log ────────
    logits2 = torch.tensor([[0.0, -1000.0, 5.0]])
    naive_log = torch.log(softmax(logits2))
    safe_log = log_softmax(logits2)
    print(f"\n  logits          = {[round(v, 1) for v in logits2.tolist()[0]]}")
    print(f"  softmax 后的值  = {[float(f'{v:.3e}') for v in softmax(logits2).tolist()[0]]}"
          f"   ← 中间那个被压成 0")
    print(f"  log(softmax(x)) = {[float(f'{v:.4f}') for v in naive_log.tolist()[0]]}   ← -inf")
    print(f"  log_softmax(x)  = {[round(float(v), 4) for v in safe_log.tolist()[0]]}")
    ok &= _ok(torch.isinf(naive_log[0, 1]).item(),
              "log(softmax) 在第二个位置给出 -inf（真实信息 −1005.0067 完全丢失）")
    ok &= _ok(torch.isfinite(safe_log).all().item(), "log_softmax 全程有限")

    # 用 torch 自己的实现当参照，不依赖我手算的数
    ref = F.log_softmax(logits2, dim=-1)
    d2 = (safe_log - ref).abs().max().item()
    ok &= _ok(d2 < 1e-4, f"log_softmax 与 F.log_softmax 一致（diff={d2:.2e}）")

    # 常规范围内两条路线必须一致 —— 说明减 max 没改变数学结果
    torch.manual_seed(2)
    z = torch.randn(4, 5) * 3
    d = (torch.log(softmax(z)) - log_softmax(z)).abs().max().item()
    ok &= _ok(d < 1e-6, f"常规范围内 log(softmax) 与 log_softmax 一致（diff={d:.2e}）")

    # ── 为什么这对你特别重要：fp16 的溢出阈值只有 11.09 ─────────────
    print("\n  【和你后面的训练直接相关】exp 的溢出阈值：")
    print("    float32: e^88.7 ≈ 3.4e38   → logit 差 88 才炸")
    print("    float16: e^11.1 ≈ 6.6e4    → logit 差 11 就炸！")
    logits16 = torch.tensor([[0.0, 5.0, 20.0]], dtype=torch.float16)
    naive16 = logits16.exp() / logits16.exp().sum(-1, keepdim=True)
    stable16 = softmax(logits16)
    print(f"    fp16 下 logits  = {logits16.tolist()[0]}")
    print(f"    fp16 朴素       = {naive16.tolist()[0]}   ← 差 20 就 nan")
    print(f"    fp16 稳定版     = {[round(float(v), 6) for v in stable16.float().tolist()[0]]}")
    print("                      ↑ 第二项真值 e^-15≈3e-7，fp16 最小正规数约 6e-5，")
    print("                        所以它下溢成 0 —— 这是精度损失，不是溢出，可以接受")
    ok &= _ok(not torch.isfinite(naive16).all().item() and
              torch.isfinite(stable16).all().item(),
              "fp16 下朴素版炸了、稳定版只是损失精度 —— 混合精度训练必须用稳定实现")

    return ok


def gate4_gradcheck() -> bool:
    """关 4：gradcheck —— 用有限差分逐元素验证 backward。

    关 2 是「和 autograd 比」，但两者的实现路径可能错得一样。
    gradcheck 是「和数值微分比」，完全独立于任何解析推导 ——
    它真的去扰动每一个输入元素，看 loss 变了多少。

    要求：输入必须是 float64，否则有限差分的误差会淹没真实信号。
    """
    _hdr(4, "gradcheck 有限差分逐元素验证（float64）")
    torch.manual_seed(3)
    logits = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    target = torch.randint(0, 4, (6,))

    # 只对 logits 求梯度，打包成一个单参数函数交给 gradcheck
    fn = lambda z: MyCrossEntropyFunction.apply(z, target)
    ok = bool(torch.autograd.gradcheck(fn, (logits,), eps=1e-6, atol=1e-5))
    _ok(ok, "全部 24 个元素的解析梯度与数值梯度一致（容差 1e-5）")

    # gradgradcheck：连二阶导都对 —— 说明 backward 里的运算本身也是可微的
    # 这一关第一版没过（二阶导全 0），原因是 forward 里存了 p 而不是 logits
    ok2 = bool(torch.autograd.gradgradcheck(fn, (logits,), eps=1e-6, atol=1e-4))
    _ok(ok2, "二阶导也对（gradgradcheck）—— 说明 backward 用了输入重算，没存中间量")
    return ok and ok2


def gate5_chain_rule() -> bool:
    """关 5：上游有缩放时，梯度必须跟着缩放。

    抓的是「backward 里漏乘 grad_out」这个静默 bug。
    它是今天最危险的一个 bug —— 不测就永远发现不了。
    """
    _hdr(5, "链式法则：loss * 3 之后梯度应该是 3 倍")
    torch.manual_seed(4)
    logits = torch.randn(16, 10, requires_grad=True)
    target = torch.randint(0, 10, (16,))

    base = MyCrossEntropyFunction.apply(logits, target)
    g1, = torch.autograd.grad(base, logits, retain_graph=True)

    # 上游乘 3：grad_out 会变成 3.0，我们的 backward 必须把它乘进去
    scaled = MyCrossEntropyFunction.apply(logits, target) * 3.0
    g3, = torch.autograd.grad(scaled, logits)

    # 正确的行为：g3 == 3 * g1
    err = (g3 - 3.0 * g1).abs().max().item()
    ok = _ok(err < 1e-6, f"max|g_scaled - 3·g_base| = {err:.2e}")
    if not ok:
        ratio = (g3.abs().max() / g1.abs().max()).item()
        print(f"     实测比值 = {ratio:.4f}（正确应为 3.0000；等于 1.0 就是漏乘 grad_out）")

    # 对照：torch 自己扛得住同样的测试（说明这个测试本身没写错）
    lg2 = logits.detach().clone().requires_grad_(True)
    ref = F.cross_entropy(lg2, target) * 3.0
    gr, = torch.autograd.grad(ref, lg2)
    err_ref = (gr - 3.0 * g1).abs().max().item()
    ok &= _ok(err_ref < 1e-6, f"同条件下 torch 的梯度也是 3 倍（diff={err_ref:.2e}）")
    return ok


def gate6_retain_graph() -> bool:
    """关 6：两种「忘了 clone」的写法，一个会被拦住，一个会静默算错。

    这一关的结论比我原本预想的有意思：PyTorch 对 save_for_backward 存的张量
    有版本检查，原地改它会直接抛 RuntimeError。但如果你用的是 ctx.p = p
    （很常见，因为代码看起来更短），这套检查完全不生效 —— 那就是静默错误。

    **同一个 bug，两种写法，一个报错一个不报。这是今天最值得记住的一条。**
    """
    _hdr(6, "retain_graph 幂等性 + 两种「忘记 clone」的写法对比")
    torch.manual_seed(5)
    logits = torch.randn(8, 5, requires_grad=True)
    target = torch.randint(0, 5, (8,))

    loss = MyCrossEntropyFunction.apply(logits, target)
    g_a, = torch.autograd.grad(loss, logits, retain_graph=True)
    g_b, = torch.autograd.grad(loss, logits, retain_graph=True)

    err = (g_a - g_b).abs().max().item()
    ok = _ok(err < 1e-7, f"本文件的实现两次 backward 完全一致，max|g_a - g_b| = {err:.2e}")

    # ── 写法 A：save_for_backward 存 p，backward 里原地改（没 clone）────
    class BadCE_saved(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z, y):
            lp = log_softmax(z, dim=-1)
            n = z.shape[0]
            loss = -lp[torch.arange(n), y].mean()
            ctx.save_for_backward(lp.exp(), y)          # 存 p
            ctx.n = n
            return loss

        @staticmethod
        def backward(ctx, go):
            p, y = ctx.saved_tensors
            p[torch.arange(p.shape[0]), y] -= 1.0        # ← 没 clone，直接改缓存
            return p / ctx.n * go, None

    lg = torch.randn(8, 5, requires_grad=True)
    l_bad = BadCE_saved.apply(lg, target)
    print("\n  写法 A：save_for_backward 存 p，backward 里原地改 p[...] -= 1")
    caught = False
    try:
        g1a, = torch.autograd.grad(l_bad, lg, retain_graph=True)
        g2a, = torch.autograd.grad(l_bad, lg, retain_graph=True)
        print(f"    没报错，两次差 = {(g1a - g2a).abs().max().item():.2e}")
    except RuntimeError as e:
        caught = True
        first = str(e).split("\n")[0]
        print(f"    ❗ RuntimeError: {first[:88]}...")
    ok &= _ok(caught, "写法 A 被 PyTorch 的版本检查当场拦住（报错，不是静默错）")

    # ── 写法 B：用 ctx.p = p 绕过版本检查 → 静默算错 ─────────────────
    class BadCE_ctx(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z, y):
            lp = log_softmax(z, dim=-1)
            n = z.shape[0]
            loss = -lp[torch.arange(n), y].mean()
            ctx.p = lp.exp()                             # ← 挂在 ctx 上，无版本跟踪
            ctx.y = y
            ctx.n = n
            return loss

        @staticmethod
        def backward(ctx, go):
            p = ctx.p
            p[torch.arange(p.shape[0]), ctx.y] -= 1.0     # ← 直接改，没人管
            return p / ctx.n * go, None

    lg2 = torch.randn(8, 5, requires_grad=True)
    l_bad2 = BadCE_ctx.apply(lg2, target)
    g1b, = torch.autograd.grad(l_bad2, lg2, retain_graph=True)
    g2b, = torch.autograd.grad(l_bad2, lg2, retain_graph=True)

    # 参照系必须用「同一个 lg2 上的正确实现」，否则比的是两份不同的随机数据
    l_ok2 = MyCrossEntropyFunction.apply(lg2, target)
    g_ok2, = torch.autograd.grad(l_ok2, lg2, retain_graph=True)

    d_ok = (g1b - g_ok2).abs().max().item()
    d12 = (g1b - g2b).abs().max().item()
    print("\n  写法 B：ctx.p = p 存 p，backward 里原地改 p[...] -= 1")
    print(f"    第 1 次与正确值差 = {d_ok:.2e}  ← 看起来完全正确！")
    print(f"    第 2 次与第 1 次差 = {d12:.2e}  ← 这里才露馅，但没人告诉你")
    ok &= _ok(d_ok < 1e-6, "写法 B 的第一次 backward 确实是对的（所以它骗得过你）")
    ok &= _ok(d12 > 1e-3, "写法 B 静默算错：第二次全错，全程无异常")

    # 确认好实现没被污染：softmax 输出的每一行和必须还是 1
    p_now = softmax(logits.detach())
    ok &= _ok(abs(p_now.sum(1) - 1.0).max().item() < 1e-6,
              "本文件实现的缓存未被污染（每行概率和仍为 1）")
    return ok


def gate7_real_training() -> bool:
    """关 7：接到真的模型上训 20 步，两条路线的参数轨迹必须一致。

    前面六关都是人造数据。这一关用真实的 MLP + 真实形状的数据跑训练，
    验证「换了损失函数，训练结果一个字都不变」—— 这才是能放心用它的证据。
    """
    _hdr(7, "真模型训练 100 步：MyCE 与 torch CE 的参数轨迹")
    from src.models.mlp import MLP

    def run(make_crit):
        torch.manual_seed(7)                      # 同一个初始化
        model = MLP(784, 64, 10, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        crit = make_crit()
        g = torch.Generator().manual_seed(7)

        # 「真实规则」：y = x.sum() 落在哪个固定分箱里（0~9）。
        # 用它而不是随机标签，是因为随机标签下模型学不到任何东西，
        # loss 只降 0.0x，看不出两条路线是否真的走了同样的路。
        #
        # 阈值必须**固定**，不能按当前 batch 算分位点 ——
        # 那样每批的"正确答案"都不一样，模型学的其实是噪声，
        # loss 会不降反升（我第一版就是这么写的，实测 2.31 → 2.36）。
        # x ~ N(0,1) 的 784 维求和 ≈ N(0, 28²)，所以 ±40 覆盖绝大多数样本。
        cuts = torch.linspace(-40.0, 40.0, 9)
        losses = []
        for _ in range(100):
            x = torch.randn(64, 784, generator=g)
            y = torch.bucketize(x.sum(dim=1), cuts)   # (64,) 取值 0..9，规则固定
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        params = torch.cat([p.detach().flatten() for p in model.parameters()])
        return losses, params

    L_mine, P_mine = run(lambda: MyCrossEntropy())
    L_ref, P_ref = run(lambda: nn.CrossEntropyLoss())

    d_last = abs(L_mine[-1] - L_ref[-1])
    d_param = (P_mine - P_ref).abs().max().item()
    print(f"  第 1 步 loss     {L_mine[0]:.8f}  vs  {L_ref[0]:.8f}")
    print(f"  第 {len(L_mine)} 步 loss   {L_mine[-1]:.8f}  vs  {L_ref[-1]:.8f}")
    ok = _ok(d_last < 1e-6, f"末步 loss 差 = {d_last:.2e}")
    ok &= _ok(d_param < 1e-6, f"全部参数最大差 = {d_param:.2e}")
    ok &= _ok(L_mine[-1] < L_mine[0], "loss 确实在下降（不是两边都没动）")
    return ok


# ═══════════════════════════════════════════════════════════════════
# ⑦ main
# ═══════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="Day 16 手写 softmax + 交叉熵，六关对拍")
    ap.add_argument("--train", action="store_true", help="额外跑关 7（真模型训练）")
    args = ap.parse_args()

    print(f"torch {torch.__version__}   CUDA 可用={torch.cuda.is_available()}")

    results = {
        "关1 前向数值": gate1_forward(),
        "关2 反向梯度": gate2_grad(),
        "关3 数值稳定": gate3_overflow(),
        "关4 gradcheck": gate4_gradcheck(),
        "关5 链式法则": gate5_chain_rule(),
        "关6 retain_graph": gate6_retain_graph(),
    }
    if args.train:
        results["关7 真模型训练"] = gate7_real_training()

    print(f"\n{'═' * 70}\n汇总\n{'═' * 70}")
    for k, v in results.items():
        print(f"  {'✅ 通过' if v else '❌ 失败'}   {k}")
    n_pass = sum(results.values())
    print(f"\n  {n_pass}/{len(results)} 关通过")

    if n_pass == len(results):
        print("\n  手写版与 PyTorch 内部实现逐位一致 —— 可以把它当成自己写的损失函数用了。")
        return 0
    print("\n  有未通过的关，别往下走：先修好再进入 Day 17。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
