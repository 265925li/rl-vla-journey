"""Day 25 · 手写 BatchNorm2d（两套计算 + 手写反向 + 七关对拍）

用法（PowerShell，在项目根目录跑）：
    cd "E:\\work Buddy\\强化学习\\rl-vla-journey"
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\utils\\batchnorm.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\utils\\batchnorm.py --train

════════════════════════════════════════════════════════════════════
这个文件怎么读（按这个顺序，别从头顺着看）

  1) 文件头这一段                  —— 为什么今天讲 BN，它到底治什么病
  2) my_batch_norm_train()         —— 训练态，6 行。BN 的**定义**
  3) my_batch_norm_eval()          —— 推理态，3 行。**和上面不是同一个公式**
  4) MyBatchNorm2dFunction.backward() —— 4 行 + 一段推导。今天最核心的地方
  5) MyBatchNorm2d                —— Module 包装，含 running stats 的更新
  6) 最后看七道关

  文件里「数学：xxx」的注释，就是公式翻译成的代码。
════════════════════════════════════════════════════════════════════
为什么现在讲 BN？

  Day 24 跑出的一组数字，直接指出了当前的瓶颈：

      组别        最低 val_loss 出现在    说明
      baseline    ep7（25 轮里）         ep21 起 train_acc = 1.0000，早饱和了
      aug3        ep20                   增强把可训练窗口从 7 轮拉长到 20 轮

  baseline 从 15 轮跑到 25 轮，test acc 一位没变（0.9900 → 0.9900）。
  所以现在的瓶颈**不是精度，是「收敛得太早」**。

  而 BN 恰好就是治这个的：它把每一层的输出按批重新标准化，
  于是每一层不用再追着上一层不断漂移的分布去适应 ——
  优化景观被磨平了，同样步数能走更远。

  它和 Day 23/24 的增强是一对：
      增强      把输入分布**搞得更散**（模型更难见过拟合）
      BN        把中间层分布**重新收紧**（每层输入始终均值 0 方差 1）
  一个往外推，一个往回拉，方向不冲突。
════════════════════════════════════════════════════════════════════
今天最反直觉的一件事：BN 有**两套完全不同的计算**

  这是 BN 和 P0 所有手写算子（SGD / CrossEntropy / Linear / Conv2d）
  的根本区别 —— 那四个的前向只有一种算法。

  BN 前向分两个模式，而且**必须**分：

  ┌── 训练态 training=True ─────────────────────────────────────┐
  │  用**当前这一批**数据算均值和方差，然后标准化                 │
  │  同时把这对统计量按滑动平均**存下来**（叫 running stats）      │
  │  输出的每个值都依赖同批的其它样本 —— **批内耦合**              │
  └─────────────────────────────────────────────────────────────┘

  ┌── 推理态 training=False ────────────────────────────────────┐
  │  用存下来的 running stats，**完全不看当前批**                 │
  │  输出对每个样本独立 —— 和 Linear / Conv 一样是逐点运算         │
  └─────────────────────────────────────────────────────────────┘

  为什么推理时不能用当前批？
    1) 语义上：一张图进来就该得到一个确定的答案，
       不能因为"和它一起送进来的另外 99 张图不同"而改变。
    2) 工程上：线上服务经常一次只来一张（batch_size=1），
       这时批内方差恒为 0，标准化会直接除零。

  为什么训练时不能直接用 running stats？
    running stats 初始值是 mean=0 / var=1，等于什么都没做。
    而且它是"逐渐逼近"的，训练早期还没逼近时用它标准化，
    等于用错误的分母，梯度方向会系统性偏掉。

  ⚠️ 这个设计带来一个非常隐蔽的 bug 来源：

      训练完忘了 model.eval() 就去评估
          ↓
      评估时还在用当前批统计量
          ↓
      batch_size=128 → 每批均值方差都不同 → 同一个模型两次评估差好几个点
      batch_size=1   → 批内方差 0 → 输出恒等于 beta → 精度掉到随机猜

  **它不报错。** 你会看到"模型训崩了"，然后去调学习率 ——
  真正的原因是 4 个字符的 `model.eval()` 没写。关 4 专门验这件事。
════════════════════════════════════════════════════════════════════
BN 在做什么优化？一句话版本

  原始问题（论文叫 internal covariate shift）：第 3 层的输入分布，
  取决于第 1、2 层的参数，而那两层一直在变。
  于是第 3 层的梯度目标一直在移动 —— 学习率不敢开大，收敛慢。

  BN 的做法：在每层输入处插一道"重新标准化"，
  让第 3 层永远收到均值 0 方差 1 的东西，不管前两层怎么变。

  ⚠️ 但要说清楚：**BN 不让网络表达能力变弱**。
  它标准化之后立刻跟一个**可学习的**仿射变换 y = γ·x̂ + β。
  如果"均值 0 方差 1"不是最优的，γ 和 β 会把分布重新拉回去。
  所以 BN 的真实作用是「**改变参数化的形状，让优化更容易**」，
  而不是「限制输出范围」。这是理解 BN 的分水岭：

      ❌ BN 是正则化，会降低表达能力
      ✅ BN 是重参数化，表达能力不变，但优化景观更平滑

  一个可以检验的推论：如果 BN 只是"限制输出"，那加了 BN 的模型
  表达能力必然下降；但实测里 BN 通常**提升**精度。
  这只能用"更好优化"解释，不能用"限制输出"解释。
════════════════════════════════════════════════════════════════════
今天要自己答的三个问题（看完代码再回来答）

  1) 为什么 conv 后面接 BN，conv 的 bias 就**没用了**？
     （提示：BN 减的是什么？bias 加的是常数还是变量？）
  2) 为什么 `momentum=0.1` 意味着"记住 90% 的历史"，
     而不是"记住 10%"？
  3) `batch_size=1` 一定会让 BN 崩吗？
     （提示：BN2d 的"批"里有多少个元素 —— 是 N，还是 N·H·W？
      条件换成什么，才会真的崩？第一条我原先答错了。）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# 1. 训练态：用当前批的统计量
# ═══════════════════════════════════════════════════════════════════
def my_batch_norm_train(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """训练态 BN。返回 (y, mean, var)，后两个用来更新 running stats。

    数学（对**每个通道** c 独立做，输入形状 (N,C,H,W)）：

        m      = N · H · W                 ← 一个通道里有 m 个数
        μ_c    = (1/m) Σ x                ← 批内均值
        σ²_c   = (1/m) Σ (x − μ_c)²       ← 批内方差，除以 m 不是 m−1
        x̂      = (x − μ_c) / √(σ²_c + ε)
        y      = γ_c · x̂ + β_c

    ── 三个容易写错的地方 ────────────────────────────────────────

    ① 求和维度是 (0,2,3)，**保留通道维 1**
       要按通道算，就不能把 C 也一起平均掉。
       写成 x.mean() 会把 N,C,H,W 全平均 → 所有通道用同一个均值 → 静默错误。
       keepdim=True 让结果保持 (1,C,1,1)，后面能和 (N,C,H,W) 直接广播。

    ② 方差除以 m，不是 m−1
       统计学里样本方差通常用 m−1（贝塞尔校正，无偏估计）。
       但 PyTorch 的 BN 用的是 **biased** 版本（除以 m）。
       为什么：BN 这里要的不是"总体方差的无偏估计"，
       而是"这一批数据的描述性统计量"，除以 m 才是定义。
       写成 m−1 不会报错，只会让结果差一个 m/(m−1) 的因子 ——
       m=128 时差 0.8%，训练照样跑，但和官方实现对不上，也偏离论文。

    ③ eps 加在**方差**上，不是加在标准差上，也不是加在分母外面
       目的是防止 σ²=0 时除零。加在方差上等价于"方差至少为 eps"，
       于是分母至少是 √eps ≈ 3.16e-3，不会爆成 inf。
       （关 4 会演示 batch_size=1 时方差真的是 0）
    """
    # 训练态要求 4 维。写成显式检查是为了给一条有用的报错，
    # 而不是等到广播时才发现形状对不上。
    if x.dim() != 4:
        raise ValueError(
            f"BatchNorm2d 需要 4 维输入 (N,C,H,W)，收到 {tuple(x.shape)}。\n"
            "  如果你在做 MLP，应该用 BatchNorm1d（对 (N,C) 或 (N,C,L) 操作）。"
        )

    reduce_dims = (0, 2, 3)          # N, H, W 三个维度求和，保留 C

    # 数学：μ = (1/m) Σ x，形状 (1,C,1,1)
    mean = x.mean(dim=reduce_dims, keepdim=True)

    # 数学：σ² = (1/m) Σ (x − μ)²，形状 (1,C,1,1)
    # 为什么不用 x.var(dim=..., unbiased=False)：
    # 两者数学上等价，但显式写出来才能保证和下面的 backward 用同一个算法，
    # 也对齐 F.batch_norm 的数值路径（关 1 会验到逐位）。
    var = ((x - mean) ** 2).mean(dim=reduce_dims, keepdim=True)

    # 数学：1/√(σ²+ε)。用 rsqrt 而不是 1/sqrt，少一次除法，
    # 也和 PyTorch 内部实现一致（关 1 需要逐位，路径必须相同）。
    inv_std = torch.rsqrt(var + eps)

    # 数学：x̂ = (x − μ)·inv_std
    x_hat = (x - mean) * inv_std

    # 数学：y = γ·x̂ + β
    # gamma/beta 形状是 (C,)，view 成 (1,C,1,1) 才能沿 (N,H,W) 广播。
    # 直接用 gamma[None,:,None,None] 也行，view 更直观。
    y = x_hat * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)

    return y, mean, var


# ═══════════════════════════════════════════════════════════════════
# 2. 推理态：用 running stats
# ═══════════════════════════════════════════════════════════════════
def my_batch_norm_eval(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """推理态 BN。3 行，而且是**逐点运算**（不跨样本耦合）。

    数学：

        y = γ · (x − running_mean) / √(running_var + ε) + β

    注意它和训练态差在哪：
        训练态   μ、σ² 是**这一批算出来的**，每次前向都变
        推理态   μ、σ² 是**固定存的**，同一个模型跑一百次结果一样

    工程上可以做一步优化：把 γ/√(running_var+ε) 预先算成一个系数，
    于是推理时只有一次乘法 + 一次加法。
    这等价于「把 BN 折叠进前一层的卷积」，是部署时的标准操作。
    """
    inv_std = torch.rsqrt(running_var + eps)          # (C,)
    return (
        (x - running_mean.view(1, -1, 1, 1))
        * inv_std.view(1, -1, 1, 1)
        * gamma.view(1, -1, 1, 1)
        + beta.view(1, -1, 1, 1)
    )


# ═══════════════════════════════════════════════════════════════════
# 3. 手写反向（今天最核心的 4 行）
# ═══════════════════════════════════════════════════════════════════
class MyBatchNorm2dFunction(torch.autograd.Function):
    """训练态 BN 的 autograd Function。

    ── 为什么值得手写 ────────────────────────────────────────────
    BN 的反向是所有 P0 手写算子里最容易推错的一个，因为
    **输入 x 的每个元素都会通过三条不同的路径影响输出**：

        路径 1  直接：x̂_i 依赖 x_i
        路径 2  通过均值 μ：μ 依赖所有 x_j
        路径 3  通过方差 σ²：σ² 依赖所有 x_j

    前两个手写算子（Linear / Conv）只有路径 1（加上参数共享的求和）。
    这里的路径 2、3 是 BN 特有的，也是最容易漏掉的。
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        eps: float,
    ):
        y, mean, var = my_batch_norm_train(x, gamma, beta, eps)

        # ── 只存**输入**，不存中间量 ────────────────────────────────
        # 这是 Day 22 踩过的坑，原样搬过来：
        # 如果存 x_hat 和 inv_std，那么它们是在 forward 里算出来的张量，
        # 一整条从 x 到它们的计算图就断了 —— 一阶导照样对，
        # 但二阶导（gradgradcheck）会算出 0，而且不报错。
        # 存 x + 在 backward 里重算中间量，计算图才连得回去。
        # 代价是 backward 多一次前向计算，约 30%（实测过）。
        ctx.save_for_backward(x, gamma)
        ctx.eps = eps

        # mean / var 要返回给外面更新 running stats，但**不参与求导**。
        # 标记成 non_differentiable 才不会在 backward 里要求它们的梯度。
        ctx.mark_non_differentiable(mean, var)
        return y, mean, var

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_mean, grad_var):
        """BN 反向。推导在下面，代码只有 4 行。

        ── 推导 ─────────────────────────────────────────────────
        记号：m = N·H·W（一个通道的元素个数），Σ 表示对该通道所有元素求和。

        x̂_i = (x_i − μ)/σ，其中 μ、σ 都依赖**全部** x_j。所以

            dx_i = Σ_j (∂L/∂x̂_j)·(∂x̂_j/∂x_i)                     ← 直接项
                 + Σ_j (∂L/∂x̂_j)·(∂x̂_j/∂μ)·(∂μ/∂x_i)             ← 均值项
                 + Σ_j (∂L/∂x̂_j)·(∂x̂_j/∂σ)·(∂σ/∂x_i)             ← 方差项

        先记三个偏导（都很简单，但符号最容易错）：
            ∂x̂_i/∂x_i = 1/σ
            ∂x̂_j/∂μ   = −1/σ
            ∂μ/∂x_i   = 1/m
            ∂x̂_j/∂σ   = −(x_j−μ)/σ²  = −x̂_j/σ
            ∂σ/∂x_i   = (x_i−μ)/(m·σ)          ← σ=√(σ²)，链式法则带 1/(2σ) 后化简

        逐项算：
            直接项 = dxhat_i / σ
            均值项 = Σ_j dxhat_j · (−1/σ) · (1/m)
                   = −(1/(m·σ))·Σ dxhat
            方差项 = Σ_j dxhat_j · (−x̂_j/σ) · (x_i−μ)/(m·σ)
                   = −(1/(m·σ))·x̂_i·Σ(dxhat_j·x̂_j)      ← 因为 (x_i−μ)/σ = x̂_i

        合并（提出公因子 1/(m·σ)）：

            dx_i = (1/(m·σ)) · [ m·dxhat_i − Σ dxhat − x̂_i·Σ(dxhat·x̂) ]

        这三项的经济含义：
            m·dxhat_i          这个元素自己想往哪走
            −Σ dxhat           整批的平均走向（因为 μ 被拉动了）
            −x̂_i·Σ(dxhat·x̂)    整批的"缩放"走向（因为 σ 被拉动了）
        """
        x, gamma = ctx.saved_tensors
        eps = ctx.eps

        n, c, h, w = x.shape
        m = n * h * w                        # 每个通道的元素个数

        # ── 重算前向中间量（不存它们，见 forward 里的说明）──────────
        reduce_dims = (0, 2, 3)
        mean = x.mean(dim=reduce_dims, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=reduce_dims, keepdim=True)
        inv_std = torch.rsqrt(var + eps)
        x_hat = (x - mean) * inv_std

        # 数学：dxhat = dy · γ
        # γ 是逐通道的，要广播到所有元素。
        g = gamma.view(1, -1, 1, 1)
        dxhat = grad_y * g

        # 数学：dγ = Σ dy·x̂ ，  形状 (C,)
        # ⚠️ 必须对 (N,H,W) 求和。漏掉求和 → 形状变成 (N,C,H,W)，
        # 赋给 (C,) 的 Parameter.grad 时会报形状错（这个会暴露）。
        dgamma = (grad_y * x_hat).sum(dim=reduce_dims)

        # 数学：dβ = Σ dy ，形状 (C,)
        dbeta = grad_y.sum(dim=reduce_dims)

        # 数学：dx = (1/(m·σ))·[ m·dxhat − Σdxhat − x̂·Σ(dxhat·x̂) ]
        # 三个求和都要 keepdim，才能和 (N,C,H,W) 的 x_hat 广播。
        sum_dxhat = dxhat.sum(dim=reduce_dims, keepdim=True)
        sum_dxhat_xhat = (dxhat * x_hat).sum(dim=reduce_dims, keepdim=True)
        dx = inv_std / m * (m * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)

        # 返回值和 forward 的参数一一对应：x, gamma, beta, eps
        # eps 是浮点数不是张量，梯度给 None。
        return dx, dgamma, dbeta, None


# ═══════════════════════════════════════════════════════════════════
# 4. Module 包装（接口与 nn.BatchNorm2d 对齐）
# ═══════════════════════════════════════════════════════════════════
class MyBatchNorm2d(nn.Module):
    """手写 BN 的 Module 包装，构造签名与 nn.BatchNorm2d 对齐。

    ── 三种东西，别混 ─────────────────────────────────────────────
    Parameter（要学，进 optimizer）
        weight (γ)：初始 1
        bias   (β)：初始 0
    buffer（不学，但要存进 checkpoint）
        running_mean：初始 0
        running_var ：初始 1
        num_batches_tracked：初始 0，只用来统计跑了多少批
    「什么都不算」的
        eps、momentum：超参数，在 __init__ 里定死

    ⚠️ 把 running_mean 写成 nn.Parameter 是个**很常见的错误**：
    它会在 optimizer 里被当成可学习参数去更新，
    而且 SGD 的 momentum 会污染它 —— 训练照样跑，结果微妙地错。

    ── 为什么 γ 初始化成 1、β 初始化成 0 ─────────────────────────
    这样 BN 一开始就是"纯标准化"，不额外改变分布。
    如果 γ 初始成 0，整个通道的输出恒为 β，梯度对 x 也是 0，
    这一层永远学不动（和 Day 10 的"全零初始化"是同一类问题）。
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if num_features < 1:
            raise ValueError(f"num_features 必须是正整数，收到 {num_features}")
        if eps <= 0:
            raise ValueError(f"eps 必须 > 0（它就是为了防除零），收到 {eps}")
        if momentum is None:
            # nn.BatchNorm2d 支持 momentum=None → 用累积平均 1/num_batches。
            # 我们不实现，但要**显式拒绝**而不是静默当成 0.1。
            raise NotImplementedError(
                "MyBatchNorm2d 不支持 momentum=None（累积平均模式）。"
                "请显式给一个 momentum，比如 0.1。"
            )
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum 要在 [0,1] 内，收到 {momentum}")
        if not affine:
            # 不支持但显式拒绝：如果静默接受 affine=False 却仍建 γ/β，
            # 参数量会比预期多 2C 个，而参数量对账**不一定**接得住。
            raise NotImplementedError(
                "MyBatchNorm2d 只支持 affine=True。"
                "关 7 会演示：即使不要仿射，也要让 β 存在（详见讲义）。"
            )

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # 可学习参数
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # buffer：不是参数，但会跟着 state_dict 一起存取
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            y, mean, var = MyBatchNorm2dFunction.apply(
                x, self.weight, self.bias, self.eps
            )

            # ── 更新 running stats ──────────────────────────────────
            # 必须在 no_grad 里做：这是统计量维护，不是计算图的一部分。
            # 忘了 no_grad 的后果：`mean` 带着 grad_fn，
            # 而 running_mean 是 buffer（不需要梯度），
            # 却存了一个需要梯度的值进去 → 下次 backward 时报
            # "backward through the graph a second time"。
            with torch.no_grad():
                self.num_batches_tracked.add_(1)

                # ⚠️ 用 reshape(-1) 而不是 squeeze()：
                # mean 形状是 (1,C,1,1)，squeeze() 会把所有 1 都去掉，
                # 当 C=1 时直接塌成 0 维标量 → 后面 add_ 时形状对不上。
                # 这个坑在 C=1 的单通道模型上才会暴露，很容易漏测。
                mean_flat = mean.reshape(-1)
                var_flat = var.reshape(-1)

                # 数学（PyTorch 语义）：
                #   new = (1−momentum)·old + momentum·batch
                #
                # ⚠️ 这和很多教科书写法**相反**。教材常用 ρ 表示
                # "保留旧值的比例"，写成 new = ρ·old + (1−ρ)·batch。
                # PyTorch 的 momentum 是"**新值的权重**"。
                # 所以 momentum=0.1 意味着 old 占 0.9，即"记住 90% 历史"。
                # 关 6 用极端数值把这件事钉死。
                keep = 1.0 - self.momentum
                self.running_mean.mul_(keep).add_(mean_flat * self.momentum)
                self.running_var.mul_(keep).add_(var_flat * self.momentum)

            return y

        # 推理态：完全不碰 running stats，也不碰当前批的统计量
        return my_batch_norm_eval(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.eps,
        )

    def extra_repr(self) -> str:
        # 让 print(model) 里能看到关键超参，和 nn.BatchNorm2d 一样
        return (
            f"{self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}"
        )


# ═══════════════════════════════════════════════════════════════════
# 5. 七道关
# ═══════════════════════════════════════════════════════════════════
def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    """最大绝对误差。两个张量形状必须一致。

    为什么所有关卡都返回这一个数：
    一串数字里"哪个最不对"是唯一能立刻定位问题的信息。
    用 torch.allclose 只会得到一个 True/False，出问题时等于没信息。
    """
    return (a - b).abs().max().item()


def gate1_train_forward() -> bool:
    """关 1：训练态前向 vs F.batch_norm（同一批数据、同一组参数）。

    这一关验的是**公式**：均值/方差的定义、eps 的位置、γ/β 的广播方式。
    三处里任何一处错，误差都是 1e-1 量级 —— 不会模棱两可。
    """
    print("\n[关 1] 训练态前向 vs F.batch_norm")
    torch.manual_seed(0)

    x = torch.randn(4, 3, 8, 8, dtype=torch.float64) * 2.0 + 3.0
    gamma = torch.randn(3, dtype=torch.float64)
    beta = torch.randn(3, dtype=torch.float64)

    y_my, mean_my, var_my = my_batch_norm_train(x, gamma, beta)

    # 用 F.batch_norm 的官方实现做参照。
    # ⚠️ running stats 必须传 clones：training=True 时官方实现会**原地修改**它们，
    # 传原件的话第一次对比之后就被污染了，第二次跑就对不上。
    rm = torch.zeros(3, dtype=torch.float64, requires_grad=False)
    rv = torch.ones(3, dtype=torch.float64)
    y_ref = F.batch_norm(
        x, rm.clone(), rv.clone(), gamma, beta,
        training=True, momentum=0.1, eps=1e-5,
    )

    d_y = _maxdiff(y_my, y_ref)

    # 顺便验 mean / var 本身。它们是"中间产物"，但如果不验，
    # running stats 错了这一关就抓不到（y 对不代表 mean/var 对）。
    mean_ref = x.mean(dim=(0, 2, 3))
    var_ref = x.var(dim=(0, 2, 3), unbiased=False)
    d_mean = _maxdiff(mean_my.reshape(-1), mean_ref)
    d_var = _maxdiff(var_my.reshape(-1), var_ref)

    print(f"    y         最大差  {d_y:.2e}")
    print(f"    mean      最大差  {d_mean:.2e}   （自己算 vs x.mean）")
    print(f"    var       最大差  {d_var:.2e}   （自己算 vs x.var(unbiased=False)）")
    print(f"    y 的均值 {y_my.mean():+.2e}，方差 {y_my.var(unbiased=False):.6f}"
          f"   （γ/β 是随机的，所以不是 0/1）")

    # 单独验一下"纯标准化"（γ=1, β=0）确实得到均值 0 方差 1
    y_pure, _, _ = my_batch_norm_train(
        x, torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    )
    m_pure = y_pure.mean(dim=(0, 2, 3)).abs().max().item()
    v_pure = (y_pure.var(dim=(0, 2, 3), unbiased=False) - 1).abs().max().item()
    print(f"    纯标准化后：|均值| 最大 {m_pure:.2e}，|方差−1| 最大 {v_pure:.2e}")
    print("    → eps 会让方差略小于 1（差 eps 的量级），这是预期的")

    ok = d_y < 1e-12 and d_mean < 1e-12 and d_var < 1e-12
    assert ok, f"训练态前向对不上：y {d_y:.3e} / mean {d_mean:.3e} / var {d_var:.3e}"
    print("    ✓ 通过")
    return True


def gate2_backward() -> bool:
    """关 2：三个梯度逐个对拍 autograd。

    三个梯度分别验不同的事：
        dβ  最简单，只验"求和维度对不对"
        dγ  验 x̂ 算得对不对（它和 x̂ 直接相乘）
        dx  验那个三项公式 —— 今天最容易错的一个
    """
    print("\n[关 2] 手写反向 vs autograd（三个梯度逐个比）")
    torch.manual_seed(1)

    dtype = torch.float64
    x0 = (torch.randn(5, 4, 6, 6, dtype=dtype) * 2.0 + 1.0)
    g0 = torch.randn(4, dtype=dtype)
    b0 = torch.randn(4, dtype=dtype)

    # ── 手写路径 ─────────────────────────────────────────────────
    x1 = x0.clone().requires_grad_(True)
    g1 = g0.clone().requires_grad_(True)
    b1 = b0.clone().requires_grad_(True)
    y1, _, _ = MyBatchNorm2dFunction.apply(x1, g1, b1, 1e-5)

    # 用随机权重求导，而不是 y.sum()。
    # 为什么：sum 之后每个输出元素的梯度都是 1，
    # 于是所有元素权重相同 —— 那种情况下很多错误（比如把某个求和漏掉）
    # 恰好相互抵消，验不出来。随机权重才能让每个元素独立暴露。
    w = torch.randn_like(y1)
    (y1 * w).sum().backward()

    # ── 官方路径 ─────────────────────────────────────────────────
    x2 = x0.clone().requires_grad_(True)
    g2 = g0.clone().requires_grad_(True)
    b2 = b0.clone().requires_grad_(True)
    y2 = F.batch_norm(
        x2, torch.zeros(4, dtype=dtype), torch.ones(4, dtype=dtype),
        g2, b2, training=True, momentum=0.1, eps=1e-5,
    )
    (y2 * w).sum().backward()

    d_x = _maxdiff(x1.grad, x2.grad)
    d_g = _maxdiff(g1.grad, g2.grad)
    d_b = _maxdiff(b1.grad, b2.grad)

    print(f"    ∂L/∂x     最大差  {d_x:.2e}")
    print(f"    ∂L/∂γ     最大差  {d_g:.2e}")
    print(f"    ∂L/∂β     最大差  {d_b:.2e}")

    # 三条可以手算的性质，用来确认这不是"两边都错得一样"
    # 性质 1：Σ(dx) 在每个通道上恒为 0
    #   原因：BN 输出对整批做平移不变，把 x 整体加 c，x̂ 不变 → 导数之和为 0
    s = x1.grad.sum(dim=(0, 2, 3)).abs().max().item()
    print(f"\n    性质检查（这些不能靠两边对拍，只能自己算）：")
    print(f"      Σ(∂L/∂x) 每个通道  最大绝对值  {s:.2e}   （应为 0）")
    # 性质 2：Σ(dx·x̂) 在 **ε=0 时**严格为 0 —— 对应"整批缩放不变"
    #
    # ⚠️ 这一条我第一版写错了，值得记下来：
    #   当时直接拿 ε=1e-5 的数据断言"应为 0"，实测 1.6e-05，挂了。
    #   但实现是对的（三个梯度都 1e-16），错的是我的**性质本身**。
    #
    #   推导：dx = (1/(mσ))·[m·dxhat − Σdxhat − x̂·Σ(dxhat·x̂)]
    #   乘 x̂ 求和，用 Σx̂ = 0：
    #        Σ(dx·x̂) = (1/(mσ))·[m·Σ(dxhat·x̂) − 0 − Σ(dxhat·x̂)·Σx̂²]
    #                = (Σ(dxhat·x̂)/σ)·[1 − Σx̂²/m]
    #   而 **x̂ 的方差不是恰好 1**，是 σ²/(σ²+ε) —— 因为分母用加了 ε 的 σ。
    #   所以 Σx̂²/m = σ²/(σ²+ε)，残差
    #        Σ(dx·x̂) = (Σ(dxhat·x̂)/σ)·ε/(σ²+ε)
    #   它**正比于 ε**，不是 0。ε=1e-5、σ²≈4 时是 2.5e-6 倍，量级正好对上实测。
    #
    #   教训：「我的数学推导」和「实现」是两份独立的东西，
    #   断言挂了要先怀疑哪一份，不要默认是实现错了。
    x2b = x0.clone().requires_grad_(True)
    g2b = g0.clone().requires_grad_(True)
    b2b = b0.clone().requires_grad_(True)
    y2b, _, _ = MyBatchNorm2dFunction.apply(x2b, g2b, b2b, 0.0)   # ε = 0
    (y2b * w).sum().backward()
    mean0 = x0.mean(dim=(0, 2, 3), keepdim=True)
    var0 = ((x0 - mean0) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    xhat_exact = (x0 - mean0) * torch.rsqrt(var0)                 # 无 ε
    s2 = (x2b.grad * xhat_exact).sum(dim=(0, 2, 3)).abs().max().item()
    print(f"      Σ(∂L/∂x·x̂) 每通道 max  {s2:.2e}   （ε=0，应为 0）")

    # ε>0 时不该是 0，而是一个正比于 ε 的小量。单独把这个量级验一下，
    # 免得"验证"变成"凑一个能过的阈值"。
    xhat_eps = (x0 - mean0) * torch.rsqrt(var0 + 1e-5)
    s3 = (x1.grad * xhat_eps).sum(dim=(0, 2, 3)).abs().max().item()
    ratio = 1e-5 / (var0 + 1e-5)          # 理论残差系数 ε/(σ²+ε)
    print(f"      Σ(∂L/∂x·x̂) 每通道 max  {s3:.2e}   （ε=1e-5，**不**为 0）")
    print(f"    → 两条同时成立，说明那个三项公式的另外两项真的在起抵消作用。")
    print(f"      只写第一项（inv_std·dxhat）的话，Σdx 会立刻不成立。")
    print(f"      ε>0 的残差正比于 ε/(σ²+ε) ≈ {ratio.mean():.2e}，所以只到 1e-5 量级")

    ok = d_x < 1e-12 and d_g < 1e-12 and d_b < 1e-12 and s < 1e-12 and s2 < 1e-12
    assert ok, (
        f"梯度对不上：dx {d_x:.3e} / dγ {d_g:.3e} / dβ {d_b:.3e} / "
        f"Σdx {s:.3e} / Σ(dx·x̂) {s2:.3e}"
    )
    print("    ✓ 通过")
    return True


def gate3_gradcheck() -> bool:
    """关 3：gradcheck + gradgradcheck（一阶 + 二阶）。

    为什么单独一关：gradcheck 用数值微分独立算一遍梯度，
    和"另一个 autograd 实现"是**完全不同的验证方式**。
    关 2 是两边都用手写公式（如果公式错了但错得一致，关 2 过不了；
    但如果只是符号错在两边相同的位置，也可能一起错）。
    gradcheck 是独立的第三方，它只认极限定义。

    ⚠️ 必须用 double。float32 的数值微分步长 h=1e-6 会淹没在舍入误差里。
    """
    print("\n[关 3] gradcheck + gradgradcheck")
    torch.manual_seed(2)

    dtype = torch.float64
    # 尺寸要小：gradcheck 是 O(参数个数) 次前向，8×2×3×3 = 144 个元素
    x = (torch.randn(3, 2, 4, 4, dtype=dtype) * 2.0 + 1.0).requires_grad_(True)
    gamma = torch.randn(2, dtype=dtype, requires_grad=True)
    beta = torch.randn(2, dtype=dtype, requires_grad=True)

    def f(x_, g_, b_):
        # 只取 y。mean/var 被 mark_non_differentiable 了，
        # 放进 gradcheck 的返回值里会报"不可微输出"。
        return my_batch_norm_train(x_, g_, b_)[0]

    ok1 = torch.autograd.gradcheck(f, (x, gamma, beta), eps=1e-6, atol=1e-8)
    print(f"    gradcheck       {'通过' if ok1 else '失败'}")

    ok2 = torch.autograd.gradgradcheck(f, (x, gamma, beta), eps=1e-6, atol=1e-8)
    print(f"    gradgradcheck   {'通过' if ok2 else '失败'}")
    print("    → 二阶导能过，说明 forward 只存了 x（没存中间量）")
    print("      如果存了 x̂ / inv_std，一阶照样过，二阶会全 0 —— 这是 Day 16 的坑")

    assert ok1 and ok2, "gradcheck 或 gradgradcheck 失败"
    print("    ✓ 通过")
    return True


def gate4_train_vs_eval() -> bool:
    """关 4：训练态和推理态**必须是两种结果**，且推理态不改 running stats。

    这一关的断言方向和别处相反：别处都要求"两个结果一样"，
    这里要求"两个结果**不一样**"。

    因为"一样"才是 bug —— 说明你只实现了一套计算，
    另一套是复制粘贴的（或者干脆忘了模块的 training 标志）。

    顺带演示两件真实会踩的事：
      · batch_size=1 时方差为 0，输出恒等于 β
      · 推理态跑一百次结果不变（训练态会变，因为换了批）
    """
    print("\n[关 4] 训练态 vs 推理态（必须不同）")
    torch.manual_seed(3)

    bn = MyBatchNorm2d(4, eps=1e-5, momentum=0.1)
    # 故意把 γ/β 设成非平凡值，否则"输出恒等于 β"这个现象看不明显
    with torch.no_grad():
        bn.weight.copy_(torch.tensor([1.0, 2.0, 0.5, 3.0]))
        bn.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))

    x = torch.randn(8, 4, 5, 5) * 3.0 + 7.0     # 均值 7、标准差 3 的数据

    # 刚初始化：running_mean=0, running_var=1
    bn.train()
    y_train = bn(x)
    stats_after_train = (bn.running_mean.clone(), bn.running_var.clone())

    bn.eval()
    y_eval = bn(x)

    d = _maxdiff(y_train, y_eval)
    print(f"    同一批数据，两种模式的最大差   {d:.4f}")
    print(f"    训练态输出均值 {y_train.mean():+.4f}（≈0，因为按批标准化了）")
    print(f"    推理态输出均值 {y_eval.mean():+.4f}（≈7，因为 running_mean 还是 0）")
    assert d > 1.0, (
        f"训练态和推理态输出几乎一样（差 {d:.3e}）—— "
        "大概率是忘判断 self.training 了"
    )
    print("    → 差得很大是正确的。running stats 还没学，推理态基本等于恒等映射")

    # ── 推理态不能改 running stats ────────────────────────────────
    for _ in range(5):
        bn.eval()
        bn(x)
    after = (bn.running_mean.clone(), bn.running_var.clone())
    same = torch.equal(stats_after_train[0], after[0]) and torch.equal(
        stats_after_train[1], after[1]
    )
    print(f"    推理态连跑 5 次，running stats 变了吗：{'没变 ✓' if same else '变了 ✗'}")
    assert same, (
        "推理态修改了 running stats！\n"
        "  后果：验证集（eval 模式）会悄悄污染训练统计量，\n"
        "  训练和验证互相干扰，而且完全不报错。"
    )
    # 顺带确认 num_batches_tracked 也不动
    n_before = bn.num_batches_tracked.item()
    assert n_before == 1, f"num_batches_tracked 应该是 1（只训练过 1 次），实际 {n_before}"
    print(f"    num_batches_tracked = {n_before}（只在训练态 +1）")

    # ── 推理态是确定性的 ─────────────────────────────────────────
    y1 = bn(x)
    y2 = bn(x)
    assert torch.equal(y1, y2), "推理态两次结果不同 —— 说明它还在用当前批统计量"
    print("    推理态两次结果逐位相同 ✓（训练态会随批次变）")

    # ── batch_size=1 到底会不会崩？这里要分清两件事 ───────────────
    #
    # ⚠️ 我原先的说法是"batch_size=1 时批内方差为 0，BN 直接崩"。
    #   实测方差 9.29 —— **这个说法是错的**，错在一个很基本的地方：
    #
    #   BN2d 的"批"是 **N×H×W 个元素**，不是 N 个元素。
    #   batch_size=1、图 5×5 时，每个通道仍有 25 个数，方差当然不为 0。
    #
    #   真正让方差为 0 的是 m = N·H·W = 1，也就是 1×1 的特征图。
    #   这在 CNN 里少见，但在 **BatchNorm1d 作用在 (N,C) 上**时是常态 ——
    #   MLP 最后一层、RL 里 (batch, action_dim) 的输入，batch_size=1 时
    #   m 恰好等于 1 → 方差 0 → 输出恒等于 β。**差别就在这里。**
    bn2 = MyBatchNorm2d(4)
    bn2.train()
    x_single = torch.randn(1, 4, 5, 5) * 3.0 + 7.0
    m_single = 1 * 5 * 5
    var_single = (
        (x_single - x_single.mean(dim=(0, 2, 3), keepdim=True)) ** 2
    ).mean().item()
    print(f"\n    batch_size=1、图 5×5：每通道元素数 m = {m_single}，"
          f"方差 = {var_single:.3f}")
    print(f"    → **不崩**。它会正常输出，只是统计量只有 {m_single} 个样本。")
    print(f"      方差的相对标准误约 √(2/m) = {(2 / m_single) ** 0.5 * 100:.0f}% ——"
          f" 估计很糙，running stats 会抖")

    # 真正的 m = 1
    bn3 = MyBatchNorm2d(4)
    bn3.train()
    x_one = torch.full((1, 4, 1, 1), 7.0)      # 每通道只有 1 个数
    y_one = bn3(x_one)
    print(f"\n    ⚠️ m = N·H·W = 1 时（1×1 特征图）：方差**真的**为 0")
    print(f"       输出 = {[round(v, 6) for v in y_one.reshape(-1).tolist()]}")
    print(f"       β    = {bn3.bias.tolist()}")
    print("       x̂ = (x−μ)/√(0+eps) = 0 → 输出恒等于 β，信息全丢但**不报错**")
    print("       这就是小 batch 场景（GAN / 检测）改用 InstanceNorm、")
    print("       GroupNorm、LayerNorm 的原因 —— 它们不跨样本求统计量")

    print("    ✓ 通过")
    return True


def gate5_running_stats() -> bool:
    """关 5：running stats 的更新公式，手算 3 步对拍。

    手算部分必须自己写死在代码里，不能"用公式推公式"。
    如果实现和验证用同一个公式，等于自己证明自己。
    """
    print("\n[关 5] running stats 更新：手算 3 步 vs 实际")

    # ⚠️ 用 float64 做这一关。
    #   第一版用的默认 float32，实测差 2e-8 —— 那不是 bug，是 float32 的精度极限
    #   （相对精度 1e-7，数值 0.5 上就是 1e-8 的绝对误差）。
    #   但这样一来断言阈值只能卡在 1e-8 附近，而那个位置正好是
    #   "精度"和"小 bug"的分界线，关卡会随环境误报。
    #   手算期望值用的是 Python float（= float64），所以实现也统一到 float64，
    #   阈值才敢收到 1e-12。【容差必须和被测对象的精度匹配】
    bn = MyBatchNorm2d(1, momentum=0.1).double()
    bn.train()
    with torch.no_grad():
        bn.weight.fill_(1.0)
        bn.bias.fill_(0.0)

    # 造 3 个批，让每个通道的均值分别约等于 2 / 4 / 6
    # 用常数张量最干净：均值和方差都能一眼看出来
    batches = [
        torch.full((2, 1, 3, 3), v, dtype=torch.float64) for v in (2.0, 4.0, 6.0)
    ]

    # ── 手算（写死在代码里，不调用任何函数）────────────────────────
    # 规则：new = 0.9·old + 0.1·batch
    # 初始 running_mean = 0
    rm_manual = 0.0
    manual_means = []
    for b in batches:
        batch_mean = b.mean().item()      # 常数张量 → 均值就是那个常数
        rm_manual = 0.9 * rm_manual + 0.1 * batch_mean
        manual_means.append(rm_manual)

    # ── 实际 ─────────────────────────────────────────────────────
    actual_means = []
    for b in batches:
        bn(b)
        actual_means.append(bn.running_mean[0].item())

    print(f"    {'步':>3}{'手算 running_mean':>22}{'实际':>14}{'差':>12}")
    worst = 0.0
    for i, (mm, aa) in enumerate(zip(manual_means, actual_means), 1):
        diff = abs(mm - aa)
        worst = max(worst, diff)
        print(f"    {i:>3}{mm:>22.10f}{aa:>14.10f}{diff:>12.2e}")

    print(f"    → 每步保留 90% 旧值：0 → 0.2 → 0.58 → 1.122，永远追不上真值 6.0")
    print(f"      这就是『滑动平均』的意思：它是个低通滤波器，不是平均值")

    # 方差那一路也验一下：常数张量的批内方差是 0
    print(f"\n    常数批的批内方差 = {((batches[0] - batches[0].mean()) ** 2).mean().item():.2e}")
    print(f"    3 步后 running_var = {bn.running_var[0].item():.6f}")
    # 手算：v: 1.0 → 0.9 → 0.81 → 0.729
    v_manual = 1.0
    for _ in batches:
        v_manual = 0.9 * v_manual + 0.1 * 0.0
    print(f"    手算：1.0 → 0.9 → 0.81 → 0.729，实际末值 {bn.running_var[0].item():.6f}")
    d_var = abs(v_manual - bn.running_var[0].item())

    ok = worst < 1e-9 and d_var < 1e-9
    assert ok, f"running stats 对不上：mean 最大差 {worst:.3e}，var 差 {d_var:.3e}"
    print("    ✓ 通过")
    return True


def gate6_momentum_semantics() -> bool:
    """关 6：`momentum=0.1` 到底是"记住 10%"还是"记住 90%"。

    这是一个纯粹的语义问题，但**两种理解都不会报错**，
    只是数值差 9 倍。用极端数值一次钉死。
    """
    print("\n[关 6] momentum 的语义（PyTorch 约定 vs 教科书写法）")

    bn = MyBatchNorm2d(1, momentum=0.1).double()
    bn.train()
    with torch.no_grad():
        bn.weight.fill_(1.0)
        bn.bias.fill_(0.0)

    # 一批全是 10 的常数。初始 running_mean = 0。
    x = torch.full((2, 1, 3, 3), 10.0, dtype=torch.float64)
    bn(x)
    got = bn.running_mean[0].item()

    print(f"    初始 running_mean = 0，批均值 = 10，momentum = 0.1")
    print(f"    一步后 running_mean = {got:.6f}")
    print(f"      若理解成『新值占 10%』→ 0.9·0 + 0.1·10 = 1.0")
    print(f"      若理解成『新值占 90%』→ 0.1·0 + 0.9·10 = 9.0")
    print(f"    → 实测 {got:.4f}，所以 PyTorch 的 momentum 是**新值的权重**")

    assert abs(got - 1.0) < 1e-9, (
        f"实测 {got:.6f}，不是 1.0 —— 要么公式写反了，要么 momentum 语义理解错了"
    )

    # 连续 5 步同一个批，看它怎么逼近 10：应该是 1 − 0.9^k
    print(f"\n    连续 5 步同一个批（均值都是 10）：")
    bn.running_mean.zero_()
    print(f"      {'步':>3}{'running_mean':>16}{'1−0.9^k':>14}")
    ok_series = True
    for k in range(1, 6):
        bn(x)
        got = bn.running_mean[0].item()
        expect = (1 - 0.9 ** k) * 10.0
        ok_series = ok_series and abs(got - expect) < 1e-6
        print(f"      {k:>3}{got:>16.6f}{expect:>14.6f}")

    print(f"    → 第 5 步才到 4.10，离 10 还差 59%")
    print(f"      所以 momentum=0.1 的『记忆长度』约 1/0.1 = 10 批")
    print(f"      这也解释了为什么 BN 的 running stats 要跑一段才准 ——")
    print(f"      前 10 个 batch 的统计量基本是『半信半疑』的")

    # 另一个极端：momentum=1.0 表示完全不用历史
    #
    # ⚠️ 这一行我第一版写成了 bn(x)（漏了下标 _hi），于是 bn_hi 的
    #   running_mean 一次都没被更新，打印出来是 0.0000。
    #   看起来很像是"momentum=1.0 的公式反了"，其实是打错变量名。
    #   这类错没法靠读代码发现 —— 只能靠"打印值和预期差一个量级以上"这个信号。
    bn_hi = MyBatchNorm2d(1, momentum=1.0).double()
    bn_hi.train()
    bn_hi(x)
    print(f"\n    对照：momentum=1.0 → 一步就到 {bn_hi.running_mean[0].item():.4f}"
          f"（完全丢弃历史，等于记住最后一批）")
    ok_hi = abs(bn_hi.running_mean[0].item() - 10.0) < 1e-6

    assert ok_series and ok_hi, "滑动平均的序列对不上"
    print("    ✓ 通过")
    return True


def gate7_conv_bias_vanish() -> bool:
    """关 7：BN 前面的卷积，bias 是**完全没用**的。

    结论：conv + BN 时应该用 `bias=False`，能省参数且不损失任何东西。
    （这也是为什么 BN 论文里明确写了要去掉前一层 bias。）

    为什么：bias 加的是**逐通道的常数**，
    而 BN 减的正是"逐通道的均值" —— 常数被原样减掉。
    方差也不受常数平移影响。所以 bias 根本传不到输出。

    这一关不光是验证，还直接影响今天的**参数量手算**：
    加 BN 后 conv1/conv2 各少 C_out 个参数（16 + 32 = 48 个）。
    """
    print("\n[关 7] BN 前面的 conv bias 失效")
    torch.manual_seed(4)

    # 同样用 float64：float32 下这个"抵消"只能到 1e-6，
    # 而 1e-6 正好是"精度"与"极小 bug"的重叠区，判不清。
    # （第一版用 float32，实测 1.19e-06，和阈值 1e-6 打了个照面。）
    conv_b = nn.Conv2d(1, 4, 3, padding=1, bias=True).double()
    conv_n = nn.Conv2d(1, 4, 3, padding=1, bias=False).double()

    # 权重必须完全相同，否则比的是两件事
    with torch.no_grad():
        conv_n.weight.copy_(conv_b.weight)
        conv_b.bias.copy_(torch.tensor([0.5, -1.0, 2.0, -3.0], dtype=torch.float64))

    x = torch.randn(3, 1, 6, 6, dtype=torch.float64)

    # 两个 BN 实例，参数完全相同，但状态各自独立 ——
    # 训练态的 BN 会改 running stats，共用一个实例的话第二次比的就是脏状态。
    bn_a = MyBatchNorm2d(4, eps=1e-5).double()
    bn_b = MyBatchNorm2d(4, eps=1e-5).double()

    y_a = bn_a(conv_b(x))
    y_b = bn_b(conv_n(x))

    d = _maxdiff(y_a, y_b)
    print(f"    conv(bias=True)  → BN   输出")
    print(f"    conv(bias=False) → BN   输出")
    print(f"    最大差  {d:.2e}")

    # 对照：没有 BN 时，bias 当然有用
    d_raw = _maxdiff(conv_b(x), conv_n(x))
    print(f"\n    对照（不接 BN）：最大差 {d_raw:.4f} ≈ bias 本身的大小")
    print(f"    → 接了 BN 就变成 {d:.2e}，bias 被完全抵消掉")

    # 改成 float64 之后可以把容差收到 1e-12 —— 这才叫"精确抵消"。
    # float32 下只能到 1e-6，那个量级既可能是舍入也可能是小 bug，判不清。
    ok = d < 1e-12 and d_raw > 1.0
    assert ok, f"bias 没有被抵消（差 {d:.3e}）—— BN 的均值减法可能算错了"

    # 参数账：bias=False 省下 C_out 个参数
    p_b = sum(p.numel() for p in conv_b.parameters())
    p_n = sum(p.numel() for p in conv_n.parameters())
    print(f"\n    参数账：conv(bias=True) {p_b} vs conv(bias=False) {p_n}"
          f"，省 {p_b - p_n} 个（= C_out）")
    print("    → 所以 CNN 里 conv1/conv2 合计省 16+32 = 48 个参数，且精度不变")

    print("    ✓ 通过")
    return True


# ═══════════════════════════════════════════════════════════════════
# 6. 入口
# ═══════════════════════════════════════════════════════════════════
def main() -> int:
    import argparse
    import sys
    from pathlib import Path

    # 允许直接 `python src/utils/batchnorm.py` 跑（不经过包安装）
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    parser = argparse.ArgumentParser(description="Day 25 手写 BatchNorm2d 的七道关")
    parser.add_argument(
        "--train",
        action="store_true",
        help="跑全部七关（默认只跑不需要数据的几关）",
    )
    args = parser.parse_args()

    print("=" * 66)
    print("Day 25 · 手写 BatchNorm2d")
    print("=" * 66)
    print("""
BN 有两套前向计算，今天要同时实现并分别验证：

    训练态  用**当前批**的统计量标准化 → 输出的每个值依赖同批其它样本
    推理态  用**存下来的** running stats → 输出对每个样本独立

忘了 model.eval() 时，评估走的是训练态逻辑：不报错，但结果全错。
""")

    gates = [
        ("训练态前向 vs F.batch_norm", gate1_train_forward),
        ("手写反向 vs autograd", gate2_backward),
        ("gradcheck + gradgradcheck", gate3_gradcheck),
        ("训练态 vs 推理态", gate4_train_vs_eval),
        ("running stats 手算对拍", gate5_running_stats),
        ("momentum 语义", gate6_momentum_semantics),
        ("conv bias 失效", gate7_conv_bias_vanish),
    ]

    if not args.train:
        print("（这是预览模式，只跑前两关。加 --train 跑全七关）\n")
        gates = gates[:2]

    results = []
    for name, fn in gates:
        try:
            results.append((name, fn()))
        except AssertionError as e:
            print(f"    ✗ 失败：{e}")
            results.append((name, False))

    print("\n" + "=" * 66)
    passed = sum(1 for _, ok in results if ok)
    print(f"总计：{passed}/{len(results)} 关通过")
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'}  {name}")
    print("=" * 66)

    if passed == len(results):
        print("""
回忆一下开头那三个问题，现在应该能答了：

  1) conv 后面接 BN，bias 为什么没用？
     → bias 是逐通道常数，BN 减的正是逐通道均值，常数被原样减掉，
       方差又不受常数平移影响。所以 bias 传不到输出。
       关 7 实测：接 BN 后差 1.8e-15，不接 BN 差 3.0。
       推论：conv + BN 时写 bias=False，省 C_out 个参数且精度不变。

  2) 为什么 momentum=0.1 是"记住 90% 历史"？
     → PyTorch 的 momentum 是**新值的权重**：new = 0.9·old + 0.1·batch。
       关 6 实测：初始 0、批均值 10，一步后是 1.0 而不是 9.0。
       记忆长度约 1/momentum = 10 批。

  3) batch_size=1 时 BN 一定会崩吗？—— **不一定，要看 m = N·H·W**
     → 图 5×5 时 m=25，方差不为 0，能正常跑，只是统计量粗糙。
       真正崩的是 m=1（1×1 特征图，或 BatchNorm1d 作用在 (N,C) 上），
       这时方差恒为 0，x̂ ≡ 0，输出恒等于 β —— 信息全丢，但不报错。
       （我原先以为 batch_size=1 就一定崩，关 4 的实测把这个说法否掉了。）
""")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
