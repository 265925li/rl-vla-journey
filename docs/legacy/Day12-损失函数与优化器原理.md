# Day 12 · 损失函数与优化器原理

> 昨天你写下了这五行：
>
> ```python
> optimizer.zero_grad()
> logits = model(xb)
> loss = criterion(logits, yb)
> loss.backward()
> optimizer.step()
> ```
>
> 其中有两行是**黑盒**：`criterion(...)` 和 `optimizer.step()`。
> 今天把这两行拆开，看清楚里面到底算的是什么。拆完之后你会发现：
> **`CrossEntropyLoss` = 两行数学；`SGD` = 一行数学；`Adam` = 五行数学。**
>
> 这三样你自己都能写出来，而且能和 `torch` 的结果**逐位对上**。

---

## 第〇节 今天要回答的三个问题

| # | 问题 | 在哪里回答 |
|---|---|---|
| 1 | `CrossEntropyLoss` 内部做了什么？为什么前面**不能**再加 softmax？ | §1、§2 |
| 2 | `SGD` / `Momentum` / `Adam` 各自在改什么？差别有多大？ | §3、§4 |
| 3 | 我怎么证明自己手写的优化器和 `torch` 一致？ | §5 |

**今天不涉及模型结构、不涉及数据**。就是这两个函数。

---

## §1 CrossEntropyLoss 到底算了什么

### 1.1 一行公式

对一条样本，设最后一层的输出（logits）是 $z\in\mathbb{R}^{C}$，真标签是 $y$：

$$
\text{CE}(z, y) = -\log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right) = -z_y + \log\sum_j e^{z_j}
$$

括号里那一项就是 **softmax 的第 $y$ 个分量**。所以：

> **`CrossEntropyLoss` 内部自带 softmax。** 你交给它的必须是**未归一化的原始分数**（logits），不是概率。

对 batch 平均之后，就是你昨天看到的那一个数字。

### 1.2 手算一个样本

取 `logits = [2.0, 1.0, 0.1]`，真标签 `y = 0`：

```
softmax      = [0.659001, 0.242433, 0.098566]      每项 = e^{z_i} / Σe^{z_j}
log_softmax  = [-0.417029, -1.417029, -2.317029]   就是 log(上一行)
CE = -log_softmax[0] = 0.417029
```

实测 `nn.CrossEntropyLoss()(logits, y) = 0.417030`，三者一致。

注意 `log_softmax` 三个数之间**只差常数**（`-1.000000`、`-1.900000`），这正好反映了 softmax 的平移不变性——离答案只差一个与类别无关的常数。

### 1.3 为什么内部用 `log_softmax` 而不是 `log(softmax(...))`

数学上等价，数值上不等价。

`log(softmax(z))` 要先算出 $p$，再取 $\log p$。当某个 $z_j$ 远小于最大值时，$p_j$ 会被下溢成 **0**，然后 `log(0) = -inf`。

`log_softmax` 直接用 $z_j - \log\sum e^{z_i}$ 计算（内部还会先减掉最大值），全程不产生 0。

实测大规模对拍（5000 条 × 10 类）：

```
nn.CrossEntropyLoss   = 2.7237057686
两步写法 -log_softmax  = 2.7237062454      diff = 4.768e-07
索引写法（同上一行）    = 2.7237062454      diff = 4.768e-07
随机猜的理论值          = ln(10) = 2.302585
```

差 `4.768e-07` 是 float32 的精度极限，不是错误。**但记住这个差值**：以后你手写 CE 做对拍时，别指望 `0.000e+00`，`1e-7` 量级就说明对了。

### 1.4 `reduction` 参数

```python
nn.CrossEntropyLoss(reduction="mean")   # 默认：对 batch 求平均 → 0 维张量
nn.CrossEntropyLoss(reduction="sum")    # 求和 → 0 维张量
nn.CrossEntropyLoss(reduction="none")   # 不聚合 → (B,) 逐条 loss
```

实测（8 条样本）：

```
reduction=none  shape=(8,)  均值 1.587628
reduction=sum   shape=()    value 12.701021   ← = 1.587628 × 8
reduction=mean  shape=()    value 1.587628
```

`reduction="none"` 在需要**逐样本加权**时用（比如硬负样本挖掘），平时不用。

---

## §2 多加一层 softmax：loss 会被卡在 0.5514

这是 Day 11 讲义里那个"卡在 0.5649 上不去"的坑的**完整解释**。

### 2.1 实测

`logits = [2.0, 1.0, 0.1]`，`y = 0`：

```
正确：          CE(logits)            = 0.417030
多加一层：      CE(softmax(logits))   = 0.802121
```

### 2.2 那它会一直降到 0 吗

不会。假设模型越来越自信，`logits = [M, 0, 0]`：

| M | 第一次 softmax 后的 $p_y$ | 再 softmax 后再算 CE |
|---|---|---|
| 0 | 0.3333 | 1.098612 |
| 1 | 0.5761 | 0.871100 |
| 5 | 0.9933 | 0.559948 |
| 10 | 0.99995 | 0.551502 |
| 200 | ≈1 | **0.551445** |

**无论 M 多大，loss 都停在上表最后一行。** 正确用法在 M=200 时已经是 `0.000e+00`。

### 2.3 下界是怎么来的

当 M → ∞，第一次 softmax 把结果送成 `[1, 0, 0]`。第二次 softmax 看到的是 `[1, 0, 0]`：

$$
p_y = \frac{e^1}{e^1 + e^0 + e^0} = \frac{e}{e+2} = 0.576124
$$

$$
\text{loss}_{\min} = -\ln 0.576124 = \mathbf{0.551445}
$$

所以双层 softmax 的 loss 被夹在 `[0.5514, 1.0986]` 之间——**永远学不到 0.55 以下**，看起来就像"训练卡住了"。

> **一句话记法**：`CrossEntropyLoss` 的输入必须是**最后一层的原始输出**。
> 如果你在它前面看见一个 `softmax`、`log_softmax` 或 `F.normalize`，删掉。

---

## §3 优化器的更新公式

### 3.1 所有优化器都是同一句话

```python
for p in model.parameters():
    p -= lr * 某个方向     # 往哪走、走多远
```

**所有区别只在「那个方向怎么算」。**

### 3.2 SGD（随机梯度下降）

```python
p.data -= lr * p.grad
```

- **方向**：当前梯度（负梯度方向）
- **步长**：`lr × |g|` —— 梯度大就迈大步，梯度小就迈小步

这就是它的毛病：梯度接近 0 时几乎不动；梯度方向来回摆时来回震荡。

### 3.3 Momentum（动量）

```python
buf = momentum * buf + grad      # 记一个"速度"，会累积历史梯度
p.data -= lr * buf
```

- **方向**：梯度的**指数加权移动平均**，不是当次梯度
- 效果：方向一致的梯度被累加放大，来回摆的梯度被抵消

用同一组梯度序列跑 5 步，累计位移：

```
SGD       1.6000
Momentum  3.7424      ← 同一 lr 下走了 2.3 倍
Adam      1.3057
```

**这就是"动量让收敛快"的量化含义**——不是玄学，是同 lr 下位移更大。

### 3.4 Adam

占四个寄存器（`m`、`v`）和三个常数（`β1`、`β2`、`eps`）：

```python
m = β1 * m + (1 - β1) * g              # 一阶矩：梯度的均值
v = β2 * v + (1 - β2) * g * g          # 二阶矩：梯度平方的均值
m_hat = m / (1 - β1 ** t)              # 偏差校正（前几步 m 被 0 初始值拉小了）
v_hat = v / (1 - β2 ** t)              # 同上
p -= lr * m_hat / (sqrt(v_hat) + eps)  # 除以梯度的尺度
```

三件事，一件比一件重要：

1. **同时有均值和方差** —— 比 Momentum 多记了一个"梯度有多抖"
2. **偏差校正** —— `m` 从 0 起步，第一步只有 `(1-β1)·g = 0.1g`，除以 `0.1` 才还原
3. **逐元素除以 `sqrt(v)`** —— 每个参数按自己的梯度尺度归一化

第 3 条是 Adam 的本质，下一节单独讲。

---

## §4 Adam 的本质：每步只走 `lr`

### 4.1 实测：第一步位移与梯度大小无关

```python
p.grad = torch.tensor([g]);  opt.step()   # Adam, lr=0.1
```

| 梯度 g | 第一步位移 |
|---|---|
| 0.001 | -0.099999 |
| 1.0 | -0.100000 |
| 1000.0 | -0.100000 |

**不管梯度是千分之一还是 1000 倍，第一步都只走 `lr`。**

原因就在 `m_hat / sqrt(v_hat)`：第一步 `m_hat = g`、`v_hat = g²`，比值为 `g/|g| = sign(g)`，**梯度大小被约掉了**，只剩下方向。

### 4.2 好处与代价

| | 好处 | 代价 |
|---|---|---|
| 步长恒定 | 梯度很小的参数也能正常更新；不需要为每个参数手调 lr | 接近最优时仍然迈 `lr` 那么大的一步，会绕着最低点抖 |

所以 Adam 后期通常要**衰减 lr**（§6），否则 loss 停在某个值上下抖。

### 4.3 三种优化器在真实数据上的对比

数据 `3 类 × 1000 点`、`MLP(2,32,3,3)`、5 个 epoch、每格是该 epoch 的平均 loss：

```
优化器           lr       ep1      ep2      ep3      ep4      ep5
SGD           0.001   1.0954   1.0708   1.0471   1.0242   1.0020
SGD            0.01   0.9979   0.8040   0.6391   0.5020   0.3934
SGD            0.05   0.6803   0.2093   0.0939   0.0655   0.0540
Momentum .9   0.001   1.0355   0.8479   0.6818   0.5399   0.4247
Momentum .9    0.01   0.6499   0.0958   0.0485   0.0421   0.0395
Momentum .9    0.05   0.2855   0.0423   0.0362   0.0407   0.0373
Adam          0.001   0.8622   0.4371   0.1734   0.0882   0.0621
Adam           0.01   0.2119   0.0459   0.0365   0.0429   0.0384
Adam           0.05   0.1517   0.0849   0.0400   0.0509   0.0439
```

**怎么读这张表**：

- **同一个 lr=0.001**：SGD 几乎没动（1.0954→1.0020），Adam 已经到 0.0621。差 16 倍。这就是"Adam 好调"的含义——**它不挑 lr 的量级**。
- **各自挑到合适的 lr**：SGD 0.05 / Momentum 0.01 / Adam 0.01，末轮都在 `0.038 ~ 0.054`。**最终都能到同一个地方，只是快慢不同**。
- **Adam 调太大反而差**：`lr=0.05` 时 ep2 是 0.0849，比 `lr=0.01` 的 0.0459 差一倍。**"Adam 对 lr 不敏感"是相对的**，超过 0.01 一样震荡。
- **`Momentum lr=0.05` 的 ep3→ep5 在 0.036/0.041/0.037 之间抖** —— 这是"步子太大围着最低点转"的典型症状。

> **RL 实践的默认选择**：PPO / SAC 等主流算法几乎都用 **Adam**，`lr` 取 `3e-4 ~ 1e-3`。
> 因为 RL 的梯度噪声极大，Adam 的"归一化尺度"能自动吃掉这种噪声。
> 而 supervised 学习里 SGD+Momentum 在 CV 任务上常常泛化更好——但那是另一个话题。

---

## §5 手写优化器的三个坑

作业要求"直接操作 `param.data`"。这里必须先把这条路的风险说清。

### 5.1 `.data` 会绕过 Autograd 的版本检查

看这个例子：

```python
a = torch.tensor([2.0], requires_grad=True)
b = a * a                  # b = 4，数学上 db/da = 2a = 4
a.data *= 3                # 悄悄改成 6，autograd 完全不知情
b.backward()
print(a.grad)              # 实测：12.0
```

**算出来的梯度是 12.0，不是 4.0**，因为反向传播时读的是改过之后的 `a=6`。而**一句警告都没有**。

对比用 `no_grad` 改同样的值：

```
RuntimeError: one of the variables needed for gradient computation
has been modified by an inplace operation: [torch.FloatTensor [1]]
is at version 1; expected version 0 instead.
```

**差异的本质**：`no_grad` 只是"不记录梯度"，**值一变，版本计数器照样 +1**，反向传播时一查就发现"前向用的值被改过了"，立刻报错。而 `.data` 取到的是脱离 autograd 的裸张量，改了也不计数。

### 5.2 那为什么优化器里还能用 `p.data`？

因为**时机不同**。`optimizer.step()` 执行时：

- 前向已经结束
- `backward()` 也已经结束
- 计算图已经被释放

这时候不存在"还有谁要读这个值做反向"，所以改它安全。这也是 PyTorch 官方优化器内部一直在做的事。

实测两种写法在正常场景下结果相同：

```
.data 版  = [0.85, 2.15]
no_grad版 = [0.85, 2.15]
```

**推荐写法**（更安全，且不需要 `.data`）：

```python
with torch.no_grad():
    p -= lr * p.grad
```

### 5.3 Momentum 的第一步：默认参数下测不出来

`torch.optim.SGD(momentum=0.9)` 的内部实现是：

```python
if buf is None:
    buf = torch.clone(grad)              # ← 第一步：直接拷贝梯度
else:
    buf.mul_(momentum).add_(grad, alpha=1 - dampening)
```

手动从 0 初始化 `buf` 看起来会差一步，实测：

```
默认参数（dampening=0）         max |手写 - torch| = 0.000e+00
```

**完全对得上。** 因为 `0 × 0.9 + g = g`，和"直接拷贝"结果一样。

只有当 `dampening > 0` 时才露馅：

```
dampening=0.1                   max |手写 - torch| = 4.095e-02   ← 差了 4%
```

因为此时从 0 起算的第一步是 `0.9g`，而 torch 走的是 `1.0g`。

> **这是一个关于「测试覆盖」的教训**：
> 用默认参数对拍，你的实现可能有 bug 也发现不了。
> **对拍要覆盖到非默认分支**——这条和 Day 9 的"漏 1/N 在 N=1 时完全隐身"是同一族问题。

### 5.4 完整对拍表

同一组固定梯度、5 步、`lr=0.1`：

| 检查项 | max abs diff | 结论 |
|---|---|---|
| plain SGD | `0.000e+00` | 逐位一致 |
| Momentum（dampening=0） | `0.000e+00` | 一致 |
| Momentum（dampening=0.1） | `4.095e-02` | **必须特判首步** |
| weight_decay=1e-3 | `0.000e+00` | 一致（L2 项加在梯度上） |
| nesterov=True | `0.000e+00` | 一致 |
| Adam（50 步） | `5.960e-08` | float32 精度内一致 |

**判据**：`< 1e-6` 就算对上了。Adam 因为有 `sqrt` 和除法，误差比 SGD 大一个量级，这是正常的。

> 注：本节 `4.095e-02` 与 §7.2 的 `6.513e-02` 是**同一现象在两条不同梯度序列上的值**
> （本节用 `[1+0.3s, -0.5s, 0.2s-1]`，§7.2 用作业里的 `_grad_at`），量级一致，
> 不是其中一个是错的。**做对拍时如果 diff 落在 1e-2 量级，就说明"某个分支基本写对了但有一处偏差"**，
> 而不是"完全写错"（完全写错会是 1e0 量级，比如 §7.2 第一行的 9.7e-01）。

### 5.5 `weight_decay` 加在哪

```python
g = g + weight_decay * p        # 加在梯度上（L2 正则）
```

顺序很重要：**先加 wd 项，再进动量缓冲**。如果加在 `buf` 更新之后，就变成了"只对原始梯度做正则"的另一种算法（等价于 AdamW 的思路），和 `torch.optim.SGD(weight_decay=...)` 对不上。

---

## §6 学习率调度

### 6.1 两种最常见的衰减

同一个初始 `lr=0.1`、跑 20 步：

```
StepLR(step_size=5, gamma=0.5)
[0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025]
末值 0.006250，前 6 步累计降 50%

CosineAnnealingLR(T_max=20)
[0.1, 0.09938, 0.09755, 0.09455, 0.09045, 0.08536, 0.07939, 0.0727, 0.06545, 0.05782, 0.05, 0.04218]
末值 0.000000，前 6 步累计降 20.6%
```

- **StepLR**：阶梯式，每 N 步砍一半。简单粗暴，适合"知道大概什么时候该降"
- **Cosine**：平滑下降，前期几乎不动，中段快速降，末尾贴近 0。适合"不知道什么时候该降"

### 6.2 怎么用

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(E):
    for xb, yb in loader:
        ...
        optimizer.step()
    scheduler.step()          # ← 注意：这个放在 epoch 循环里，不在 batch 循环里
```

**顺序陷阱**：`scheduler.step()` 要放在 `optimizer.step()` 之后。新版本 PyTorch 会为顺序颠倒打 warning，老版本可能静默少走一次。

### 6.3 为什么 RL 里常提"线性衰减"

因为 RL 的采样分布一直在变（策略在变），训练后期梯度信号更嘈杂。线性把 lr 一路降到 0，比 Cosine 更容易"稳住最后的性能"。
PPO 的参考实现里常见 `lr = lr_max * (1 - progress)`，就是这个思路。

---

## §7 报错对照

### 7.1 `CrossEntropyLoss` 的写法

| 写法 | 结果 |
|---|---|
| `criterion(softmax(logits), y)` | **不报错**，loss 卡在 0.5514 以上下不去 |
| `criterion(logits, y_onehot)` | `RuntimeError: Expected target size ...`（CE 要类别下标，不要 one-hot） |
| `criterion(logits, y.float())` | `RuntimeError: expected scalar type Long`（target 必须是 int64） |

### 7.2 手写优化器的六种填错（本机实测）

| 填错 | 后果 | 被抓住的位置 |
|---|---|---|
| `p = p - lr*d`（普通赋值） | max diff **`9.699e-01`** | 关 1 无选项 |
| Adam 漏偏差校正（`m_hat = m`） | max diff **`3.218e-01`** | 关 3 Adam |
| Momentum 首步从 0 累加 | max diff **`6.513e-02`** | **只在 `dampening=0.1` 那一行**，默认参数那行是 PASS |
| StepLR 漏 `last_epoch += 1` | lr 序列第 1 步就不同 | 关 4 |
| Adam 漏 `self.t += 1` | 参数变 `inf`/`nan` | 关 3（见下面的盲区） |
| Adam 漏 `zero_grad` | 梯度跨步累加，loss 先降后炸 | Day 11 已实测 |

第一行值得单独说：

> `p = p - lr * g` 里的 `=` 是**重新绑定名字**，不是"修改原张量"。
> 张量是对象，`-=` 才是原地修改。这和 NumPy 里 `a = a - b` vs `a -= b` 是同一回事，
> 但在这里的后果更严重：模型里的那个参数对象根本没被碰到，loss 一动不动。

### 7.3 对拍本身的三个盲区（这一节比上面两节重要）

写检查代码时踩到的三个坑，都实测过：

**盲区一：`nan` 会让 `max()` 悄悄跳过它**

`Adam 漏 self.t += 1` 时，`1 - 0.9**0 = 0`，于是 `m / 0 = inf`，参数变 `nan`。
但下面这段检查**判了通过**：

```python
worst = max(diffs)          # diffs = [0.0, nan, nan, ...]
assert worst < 1e-6         # max([0.0, nan]) 返回 0.0！
```

因为 Python 的 `max` 用 `>` 比较，而 `nan > 0.0` 是 `False`，于是 nan 被当成"不够大"跳过了。
第一个元素 `diffs[0]` 恒为 0（起点相同），所以永远返回 0。

**修法**：对拍前先显式查有限性。

```python
for i, t in enumerate(trail):
    assert torch.isfinite(t).all(), f"第 {i} 步出现 inf/nan"
```

**盲区二：默认参数覆盖不到非默认分支**

`buf` 从 0 累加 vs 首步拷贝梯度，在 `momentum=0.9`（默认 `dampening=0`）下**差 0.000e+00**，
只有把 `dampening` 设为 0.1 才暴露 `6.513e-02`。

**这跟 Day 9 那个"漏 1/N 在 N=1 时完全隐身"是同一族问题**：测试不覆盖到那个分支，bug 就住在里面。

**盲区三：检查代码的场景太干净**

`self.buf[i] = g`（漏 `.clone()`）在当前的检查里**能通过**——因为检查代码每一步都重新
`p.grad = _grad_at(s)` 赋一个新张量，`buf` 即使指向旧的梯度对象也不会被再次读写。

但真实训练里，如果连续两次 `backward()` 之间没有 `zero_grad()`，梯度会在**同一个张量上累加**，
此时 `buf.mul_()` 会把梯度本身一起改掉。所以 `.clone()` 该写还得写——**检查通过不等于没有问题，
只等于"这个场景下没问题"**。

---

## §8 自问 6 题

1. `CrossEntropyLoss` 的输入应该是什么？如果输入已经过 softmax，loss 的下界是多少？
2. 为什么内部用 `log_softmax` 而不是 `log(softmax())`？给出一个会出错的输入。
3. 同一个 `lr` 下，SGD 和 Adam 的步长分别由什么决定？为什么 Adam 的第一步位移恒等于 `lr`？
4. Momentum 在数学上等价于对梯度做了什么运算？它为什么能加速？
5. `.data` 和 `no_grad` 都能改参数的值，它们的区别是什么？什么时候 `.data` 会算出错误梯度？
6. 为什么用默认参数和 `torch` 对拍，可能发现不了你实现里的 bug？`max([0.0, nan])` 返回什么？

---

## 附录 A · 一句话总结

| 概念 | 一句话 |
|---|---|
| `CrossEntropyLoss` | `-log_softmax(logits)[y]` 再对 batch 取均值；**自带 softmax** |
| 多加 softmax | loss 下界 `-ln(e/(e+2)) = 0.5514`，永远降不到 0 |
| SGD | `p -= lr * g` |
| Momentum | `buf = μ·buf + g`，再 `p -= lr·buf`；同 lr 下位移放大 2.3 倍 |
| Adam | 两个矩估计 + 偏差校正 + 逐元素除以 `sqrt(v)`；步长恒 ≈ `lr` |
| `.data` | 绕过版本检查，前向之后改值会静默算错梯度 |
| 对拍判据 | SGD `< 1e-6`，Adam `< 1e-6`（有除法，误差大一个量级也算对） |

## 附录 B · 本机实测环境

`torch 2.13.0+cu126` / Python 3.10.20 / GTX 1660 Ti（sm_75）

所有数字均由 `_probe_d12*.py` 脚本在本机实测得到，未引用外部资料。
