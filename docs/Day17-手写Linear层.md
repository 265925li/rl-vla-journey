# Day 17 · 手写 Linear 层（前向 + 反向）

> **今天的硬指标**：手写 Linear 层的前向+反向，与 PyTorch 误差 < 1e-5。
> **实测结果**：单步三个梯度 `0.00e+00`，30 步参数轨迹 `1.49e-08`，真 MNIST 两轮 test 差 5 条样本。
>
> 这是 P0 四个硬指标里的最后一个算子。做完它，P0 的「手写算子」部分就全部完成了。

---

## 1. 为什么值得手写一个只有一句矩阵乘的层

一个 Linear 层，前向就是：

```python
y = x @ weight.t() + bias
```

反向三行：

```python
grad_x      = grad_out @ weight
grad_weight = grad_out.t() @ x
grad_bias   = grad_out.sum(0)
```

就这些。那为什么还要专门花一天？

**因为它写错了不会当场报错。**

对比一下两种算子的失败方式：

| | 写错了会怎样 |
|---|---|
| **交叉熵**（Day 16） | 形状对不上，`RuntimeError: shape mismatch` 当场炸。你立刻知道错了 |
| **Linear** | 有相当一部分写法**形状完全合法**。跑得通、loss 照样下降、准确率照样涨 —— 只是学的东西是错的 |

具体是什么情况？看这两行：

```python
grad_x = grad_out @ weight      # 正确
grad_x = grad_out @ weight.t()  # 错！但当 in_features == out_features 时形状一样
```

`grad_out` 是 `(N, out)`，`weight` 是 `(out, in)`。

- 正确写法：`(N,out) @ (out,in)` → `(N,in)` ✓
- 错误写法：`(N,out) @ (in,out)` → 只有当 `out == in`（方阵）时才合法

在方阵的层上，两种写法都跑得通。而 Transformer 里几乎所有的投影层都是方阵
（`hidden → hidden`），MLP 里的中间层也常常是方阵。

**所以这个 bug 在真实模型里非常容易发生，而且不会被任何「跑一下看看」抓住。**

唯一的抓法是：**和官方实现对拍**。这就是今天七道关的意义。

---

## 2. 前向：`y = x @ Wᵀ + b`

```python
def linear(x, weight, bias=None):
    y = x @ weight.t()
    if bias is not None:
        y = y + bias
    return y
```

### 2.1 为什么是 `W.T` 而不是 `W`

PyTorch 把 `weight` 存成 **`(out_features, in_features)`**，也就是**每一行是一个输出神经元的全部权重**。

这个布局的好处：`print(W[3])` 能直接看到第 3 个输出单元长什么样。

而矩阵乘要求 `(N, in) @ (in, out)`，所以必须转置。

> 你自己写 numpy 版的时候，通常习惯存成 `(in, out)`，那样就不需要转置。
> **两种约定都存在，但 PyTorch 选了前者，你必须跟着它。**
> 这就是为什么「自己写一遍」和「读别人代码」感觉差那么多 —— 约定不一致。

### 2.2 `x` 可以是任意维

```python
x = torch.randn(64, 784)         # (N, in)          → 输出 (64, 10)
x = torch.randn(2, 3, 784)       # (B, T, in)       → 输出 (2, 3, 10)
```

`@` 会自动处理前面的所有维度，把它们当 batch 维。只要最后两维是 `(..., in)` 就行。

这一点在反向里会带来麻烦 —— 见 §4。

---

## 3. 反向推导：三个梯度，三个形状

设 `y = xWᵀ + b`，`L` 是最终损失，`g = ∂L/∂y`（就是 `grad_out`）。

### 3.1 `∂L/∂b`

`b` 是 `(out,)`，在广播时被加到了 `y` 的每一行上。广播的反向规则是**求和**：

```
∂L/∂b = Σ_N g           形状 (out,)
```

直觉：一个 batch 里 64 个样本用的是**同一个** `b`，所以 64 个样本对它的梯度要累加起来。

### 3.2 `∂L/∂W`

从 `y_n = x_n Wᵀ` 出发。对单个元素：

```
y_{n,o} = Σ_i x_{n,i} · W_{o,i}

∂y_{n,o} / ∂W_{o,i} = x_{n,i}
```

所以：

```
∂L/∂W_{o,i} = Σ_n g_{n,o} · x_{n,i}
```

写成矩阵：

```
∂L/∂W = gᵀ x            形状 (out, N) @ (N, in) = (out, in) ✓
```

### 3.3 `∂L/∂x`

```
∂L/∂x_{n,i} = Σ_o g_{n,o} · W_{o,i}
```

写成矩阵：

```
∂L/∂x = g W             形状 (N, out) @ (out, in) = (N, in) ✓
```

**注意这里没有 `.T`。**

### 3.4 记不住公式？从形状反推

这是最可靠的记忆法：

```
要得到 (N, in)     →  g(N,out) @ W(out,in)      ← W 直接用
要得到 (out, in)   →  gᵀ(out,N) @ x(N,in)       ← g 要转置
```

**形状不对会报错，而背公式你会记混哪边有 `.T`。**

这也是今天的核心方法论：**不要背，让形状告诉你。**

---

## 4. 批维：`x` 是三维时怎么办

`∂L/∂W` 的公式里的矩阵乘只认 2 维。而 `x` 可能是 `(B, T, in)`。

解决办法：**摊平**。

```python
x_2d = x.reshape(-1, in_features)        # (B, T, in) -> (B*T, in)
g_2d = grad_out.reshape(-1, out_features) # (B, T, out) -> (B*T, out)

grad_weight = g_2d.t() @ x_2d             # (out, in)  ← 所有位置累加到同一份 W
grad_x      = g_2d @ weight               # (B*T, in)
grad_x      = grad_x.reshape(x.shape)     # 还原成 (B, T, in)
```

**为什么摊平是对的**：数学上，对一个 batch 里所有位置求和，就等于把它们摊平后一起乘。参数是共享的，所以每个位置都贡献一份梯度。

**为什么用 `reshape` 而不是 `view`**：`view` 要求内存连续，某些输入（比如转置过的）会报
`view size is not compatible with input tensor's size and stride`。`reshape` 会自动处理，
内存连续时它退化成 `view`，不连续时才真的复制。**没有理由用 `view`。**

关 4 专门测这个：`x` 是 `(2,3,8)` 时三个梯度的 diff 全是 `0.00e+00`。

---

## 5. 实现：避开 Day 16 踩过的三个坑

```python
class MyLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        y = x @ weight.t()
        if bias is not None:
            y = y + bias
        ctx.save_for_backward(x, weight)   # ← 存**输入**，不是输出
        ctx.has_bias = bias is not None    # ← 只记布尔值
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        in_f, out_f = x.shape[-1], grad_out.shape[-1]
        x_2d = x.reshape(-1, in_f)
        g_2d = grad_out.reshape(-1, out_f)

        grad_weight = g_2d.t() @ x_2d
        grad_x = (g_2d @ weight).reshape(x.shape)
        grad_bias = g_2d.sum(dim=0) if ctx.has_bias else None

        return grad_x, grad_weight, grad_bias
```

### 坑 1：存输入，不存输出

Day 16 写交叉熵时踩过这个：我在 `forward` 里存了 `p = softmax(logits)`，
也就是一个**由输入算出来的中间量**，然后在 `backward` 里对它做运算。

结果：`gradcheck` 过了，`gradgradcheck` 报「二阶梯度全为 0」。

原因：中间量已经**脱离计算图**。`backward` 里对它做的运算无法再被求导，
所以一阶导看起来对、二阶导整个塌掉。

对 Linear 来说这个坑更明显：`∂L/∂W` 需要 `x`，`∂L/∂x` 需要 `weight`，
**两个输入都要用**。存 `y` 的话什么都算不出来，会直接报错 —— 这反而是好事。

真正的危险是那些「一阶能对、二阶才错」的中间量。**判断标准**：
`backward` 里用到的每个张量，问一句「它是输入，还是我算出来的？」
如果是后者，就要考虑二阶导还能不能接上计算图。

今天这里存的是 `x` 和 `weight` 两个**输入**，所以关 3 的 `gradgradcheck` 能过。

### 坑 2：`save_for_backward` 而不是 `ctx.x = x`

`ctx.save_for_backward` 会做**版本检查**：如果你在 `backward` 之前原地改了它存的张量，
会当场抛 `RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation`。

`ctx.x = x` 什么都不检查，改了就静默算错。

**张量一律用 `save_for_backward`；非张量的普通 Python 值（int、bool）才挂 `ctx`。**

### 坑 3：`bias=None` 时返回值必须是 `None`

`forward` 有 3 个输入，`backward` 就必须返回 3 个值。当 `bias` 是 `None` 时，
对应位置必须返回 `None`，**不能返回 `torch.zeros(out_features)`**。

返回 zeros 看起来"更安全"，但那是错的：调用方会以为 `bias` 存在并拿到一个梯度，
而实际上根本不该有。

关 6 实测了这一点。真正的契约是**「个数对齐」**，不是「把 None 换成 zeros」。

---

## 6. 七关实测

```
汇总：7/7 关通过
```

| 关 | 验什么 | 实测 |
|---|---|---|
| 1 | 前向 vs `F.linear`（含 `bias=None`、三维输入） | `0.00e+00` |
| 2 | 反向：`x` / `weight` / `bias` **三个梯度逐个**比 | 三个全 `0.00e+00` |
| 3 | `gradcheck` + `gradgradcheck`（float64，31 个元素） | 一阶二阶都过 |
| 4 | `x` 是 `(2,3,8)` 三维时 | 三个梯度 `0.00e+00` |
| 5 | `linear` 输出乘 3，梯度应该是 3 倍 | `9.54e-07` |
| 6 | **形状陷阱**：方阵上的两种错误写法 | `gradcheck` 都抓到了 |
| 7 | 集成：MLP 用两种 Linear 各训 30 步 | 参数差 `1.49e-08` |

### 关 2 为什么能做到 `0.00e+00`

不是巧合。`MyLinearFunction.forward` 里的 `x @ weight.t()` 和 `F.linear` 内部调的是
**同一个 cuBLAS kernel**。既然算的是同一件事、用的是同一段代码，结果自然逐位相同。

这和 Day 16 的 `ce` vs `myce` 形成鲜明对比 —— 那里是 `0.00e+00` 只出现在第一步，
多步后立刻分叉。原因见 §8。

### 关 2 里那个「手算校验」是干什么的

前三个对比都是「和 PyTorch 比」。如果我和 PyTorch 犯的是**同一个错**
（比如双方都对 `W` 做了转置），那三个 diff 会全是 0，但结论是错的。

所以加了一条完全独立的验证：

```
当 grad_out 全为 1 时，∂L/∂x 应该等于 W 每一行的和
```

这是能用纸笔算出来的，不依赖任何框架。它过了，才说明方向没搞反。

> **方法论**：**至少要有两个互相独立的验证来源。**
> 「和参考实现比」和「和手算比」是两个来源；「和参考实现比两次」只是一个来源。

### 关 6：形状合法但语义错，到底能不能被抓到

两种错误写法，都在 `in == out == 4` 的方阵层上：

| 错误写法 | `gradcheck` | 前向输出 |
|---|---|---|
| `∂L/∂x = g @ W.T`（应为 `g @ W`） | ✓ 抓到 `Jacobian mismatch` | **`0.00e+00` 差** |
| `∂L/∂W = g @ x.T`（应为 `g.T @ x`） | ✓ 抓到 | 同上 |

**这就是「静默」的含义**：前向输出和正确实现一模一样，一点差别都没有。
只有把梯度和数值微分对拍，才会露馅。

而如果你只跑训练看 loss 曲线 —— loss 会下降，准确率会涨，你看不出任何异常。

---

## 7. 真实 MNIST：把手写层换进训练脚本

`MLP` 加了 `linear_cls` 参数，`train_mnist.py` 加了 `--linear`：

```powershell
& $PY scripts\train_mnist.py --linear my --epochs 2 --tag d17_my
```

日志里会打印：

```
线性层 my  →  MyLinear
```

**光看这一行不够，要实测数字。** 同一 seed、同样 2 epoch：

| epoch | train_loss (torch) | train_loss (my) | 差 | val_loss (torch) | val_loss (my) | 差 |
|---|---|---|---|---|---|---|
| 1 | 0.29972900 | 0.29972900 | **0** | 0.14888200 | 0.14888200 | **0** |
| 2 | 0.10761400 | 0.10760800 | `+6.00e-06` | 0.11054800 | 0.11121400 | `−6.66e-04` |

```
test acc   torch 0.9701   my 0.9696   差 5 条样本
```

**读法**：第 1 轮逐位相同（连验证集都一样）。第 2 轮开始分叉，
`train_loss` 只差 `6.00e-06`，但 `val_loss` 差 `6.66e-04` —— 放大了 111 倍。

为什么 `val_loss` 的分叉比 `train_loss` 大得多？

`train_loss` 是 **844 步的平均**，误差正负抵消；`val_loss` 是在**已经分叉的参数上**算出来的，
它是「结果」不是「平均」。**平均会抵消噪声，结果不会。**

---

## 8. 一个意外发现：今天的误差比 Day 16 小 5 倍

把两次「换实现」的结果放在一起：

| 换什么 | ep2 `val_loss` 差 | 换掉的深度 |
|---|---|---|
| Day 16：`ce` → `myce` | `3.6e-03` | **换了 `log_softmax` 的实现路径** |
| Day 17：`torch` → `my`（Linear） | `6.66e-04` | **只换了 `@` 的调用位置** |

**今天的误差小 5.4 倍。** 原因很具体：

- **Day 16**：PyTorch 的 `log_softmax` 用的是融合 kernel（`logsumexp` 一步算完，
  中间不落地）。我的版本是 `log(softmax(x))` —— 先 `exp`、再除、再 `log`，
  **三步各自的舍入误差会累加**。起始相对误差在 `1e-7` 量级。
- **Day 17**：`x @ weight.t()` 和 `F.linear` 内部调的是**同一个 cuBLAS kernel**。
  唯一差别是 `backward` 里我做了 `reshape` 再乘，矩阵乘的求和顺序略有不同。
  起始相对误差在 `1e-8` 量级。

起始误差差一个数量级，经过 SGD 放大后就是 5 倍。

**结论**：**「换实现」对数值的影响，取决于你换了多深的那一层。**

- 换调用位置（同一 kernel）→ 误差 `~1e-8`，几轮后仍可忽略
- 换整条数值路径（不同算法）→ 误差 `~1e-7`，几轮后可见
- 换数学等价但计算顺序完全不同的实现（如手动 logsumexp）→ 误差更大

这也解释了为什么「两个实现完全一致」这句话不能用。**正确的说法是**：

> **单步数值一致、多步因误差累积分叉、终值差远小于统计噪声。**

test acc 差 5 条样本 = 0.05 个百分点，标准误约 0.17 个百分点 —— 完全在噪声内。

---

## 9. 三个静默失败清单

今天这三个，**没有一个会抛异常**：

| # | 写法 | 症状 | 怎么抓 |
|---|---|---|---|
| 1 | `grad_x = grad_out @ weight.t()` | 方阵层上跑得通、loss 照样降 | `gradcheck` 或与官方对拍 |
| 2 | `grad_weight = grad_out @ x.t()` | 只有 `N == in` 时形状合法 | 同上 |
| 3 | `bias=None` 时返回 `zeros` 而非 `None` | 不报错，调用方拿到假梯度 | 关 6 的契约测试 |

还有两个会报错但值得记住的：

| 写法 | 报什么 |
|---|---|
| 三维输入时直接 `grad_out.t() @ x`（忘了 reshape） | `mat1 and mat2 shapes cannot be multiplied` |
| `x.view(-1, in)` 但 `x` 内存不连续 | `view size is not compatible with ... stride` |

---

## 10. 一句话带走

> **Linear 只有一行矩阵乘，但它写错的代价比交叉熵更大 ——
> 因为交叉熵写错会炸，Linear 写错会静默跑出一个次优模型。**
>
> 所以判断一个手写算子对不对，只有一个标准：**和官方实现对拍**。
> 而且要有**两个独立来源**（官方 + 手算），否则你可能和官方犯同一个错。

---

## 附：P0 四个硬指标完成情况

- [x] 手写 MLP 训练 MNIST，test acc > 97% → **98.00%**（Day 13）
- [x] 手写 softmax / 交叉熵前向+反向，误差 < 1e-5 → **1.86e-09**（Day 16）
- [x] 手写 Linear 层前向+反向，误差 < 1e-5 → **1.49e-08**（Day 17）
- [x] 本阶段提交 ≥ 15 次 → 已 34 次

**算子部分全部完成。** 剩下两天：Day 19 的通用 `train.py --config` 模板，
以及之后的 CNN（P1 第一周）—— 那是「让模型变强」，不是「让代码变对」。
