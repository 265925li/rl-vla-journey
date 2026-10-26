# Day 16 · 手写 softmax + 交叉熵（前向 + 反向对拍）

> **今天的目标不是"学会用 CrossEntropyLoss"——你 Day 13 就会用了。**
> 目标是把它拆成流水线、自己写一遍反向、然后跟 PyTorch 逐位对拍确认自己写对了。
>
> 交付物：`src/utils/losses.py`（七关对拍全过）+ 训练脚本能一键切换到你的实现。

---

## 0. 今天要干的四件事

| # | 做什么 | 产出 | 判据 |
|---|---|---|---|
| 1 | 把 softmax / log_softmax / CE 拆成三个纯函数 | `losses.py` 前 3 个函数 | 关 1：前向数值与 torch 一致 |
| 2 | **手写 backward** | `MyCrossEntropyFunction` | 关 2/4/5/6 都对得上 |
| 3 | 六关对拍 + 一关真模型训练 | `losses.py --train` | 7/7 通过 |
| 4 | 接进训练脚本，跑一次真 MNIST | `--loss myce` | test acc 与 `--loss ce` 在噪声内一致 |

---

## 1. 动机：一行代码为什么值得拆开

Day 13 你写了这一行：

```python
crit = nn.CrossEntropyLoss()
```

然后 MNIST 到了 98%。那为什么今天要花三小时拆它？

因为**它是你唯一不认识的一行**。整条训练链上，前面的 `Linear`、`ReLU`、
`optimizer.step()` 你都拆过了（Day 10/11/12），只有它还是黑盒。

它内部做了三件事：

```
logits (N,10)  ──softmax──▶  p (N,10)  ──log+取负──▶  loss (标量)
                    ↑                        ↑
              前向要算的                反向要算的 ← 只有这一块最难
```

而 P3 之后你会频繁遇到 PyTorch **没有**现成实现的目标函数：

| 场景 | 你要自己写的东西 |
|---|---|
| PPO（强化学习最常用的算法） | `min(r·A, clip(r)·A)` 的 forward + backward |
| 策略梯度 | `-(log_prob * advantage).mean()`，且 log_prob 要从采样里带梯度回传 |
| VLA 的动作回归 | 混合离散+连续的复合损失 |

到那时候没有 `nn.PPOLoss()` 可用。唯一的练法就是：**先把手写的东西和官方实现
逐位对拍一遍，确认自己真的会。**

今天就是那个练习。

---

## 2. softmax：从 logits 到概率

### 2.1 公式

模型最后一层吐出 10 个实数（叫 **logits**），它们可以是任意大小、任意正负：

```
logits = [ 2.1, -0.3, 5.7, ..., 0.8 ]
```

softmax 把它变成"10 个加起来等于 1 的正数"：

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

三条性质，后面全要用：

1. **每一项都 > 0**（指数函数永远为正）
2. **和恒为 1**（分子分母同源）
3. **保序**：z 大的那一项，概率也大（exp 单调递增）

第 3 条是它比"直接归一化"好的原因。你可能会想：为什么不直接 `z / z.sum()`？
因为 z 可能是负数，直接除会得到负的"概率"，而且 10 个数加起来也不保证是 1。
exp 把整个实数轴压到 (0, +∞)，这才是能当概率用的东西。

### 2.2 为什么要减 max —— 数值稳定

看公式，分子是 $e^{z_i}$。如果某个 $z_i = 1000$：

```
e^1000 = inf          ← float32 的上限是 e^88.7 ≈ 3.4e38
```

分子变成 `inf`，分母也变成 `inf`，`inf / inf = nan`。整个 loss 变 nan，
梯度变 nan，训练**当场死亡**，但报错信息只会告诉你"loss is nan"。

解决办法是分子分母同时乘 $e^{-m}$（m 取这行最大的那个）：

$$\frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{z_i-m}}{\sum_j e^{z_j-m}}$$

**数学上完全相等**（上下同乘一个数），但数值上：减完之后最大的那一项变成 $e^0 = 1$，
永远不可能溢出。这就是代码里那个 `x - x.max(...)`。

实测（关 3）：

```
logits          = [0.0, 991.0, 1000.0]
朴素 exp 分子   = [1.0, inf, inf]            ← 溢出
朴素 softmax    = [0.0, nan, nan]            ← 完了
本文件 softmax  = [0.0, 0.000123, 0.999877]  ← 正常
```

> **一句话记：`x - x.max()` 不改变数学，只改变数值安全性。**
> 写任何含 exp 的东西之前，先想一遍"最大能到多大"。

### 2.3 fp16 的溢出阈值只有 11.09 —— 和你后面的训练直接相关

| dtype | 最大有限值 | exp 溢出阈值 |
|---|---|---|
| float32 | 3.4e38 | **88.7** |
| float16 | 65504 | **11.1** ← 差 11 就炸 |

真实的神经网络里，logit 之间的差超过 11 太常见了（一个类判得很确信，另一个完全不像）。
所以**混合精度训练（AMP）时，稳定实现不是"可选优化"，是"必须"**：

```
fp16 下 logits  = [0.0, 5.0, 20.0]
fp16 朴素       = [0.0, 0.0, nan]      ← 差 20 就 nan
fp16 稳定版     = [0.0, 0.0, 1.0]      ← 只是损失精度，不炸
```

（第二项真值是 $e^{-15} \approx 3\times10^{-7}$，fp16 最小正规数约 $6\times10^{-5}$，
所以它下溢成 0 —— 这是精度损失，可以接受。）

**这条对你的意义**：Day 13 你已经确认 `1660 Ti` 只能用 fp16 训练。
那从 P1 开始用 `torch.cuda.amp` 时，只要损失函数里走的是 `log_softmax` 这条路，
就不会因为溢出而 nan。自己写损失函数时更要守住这一条。

---

## 3. 交叉熵：从概率到 loss

### 3.1 公式与直觉

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log p_{i,\,y_i}$$

拆开看，每一项是 $- \log(\text{正确类的概率})$：

| 正确类概率 $p_y$ | $-\log p_y$ | 含义 |
|---|---|---|
| 1.0 | 0 | 完全预测对，不罚 |
| 0.9 | 0.105 | 有点不确定，罚一点 |
| 0.5 | 0.693 | 罚得多一些（= ln2） |
| 0.1 | 2.303 | 罚得重（= ln10） |
| 0.001 | 6.908 | 完全错，往死里罚 |

只关心**正确那一类**的概率，其他 9 类怎么样都不管。
这就是分类任务的标准打法。

### 3.2 交叉熵自带 softmax —— 今天最重要的一个坑

`nn.CrossEntropyLoss` 内部是这么两步：

```
① log_softmax(logits)        ← 自己做了 softmax
② 按 target 取出正确的 log 概率，取负、取均值
```

**所以模型的最后一层千万不要再接 softmax。** 接了就变成：

```
logits → softmax → softmax → log → 取负
         ↑ 这一层        ↑ CE 内部又来一次
```

在真实 MNIST 上实测（用 Day 13 训好的模型）：

```
正确做法（logits 直接进 CE）  loss = 0.0307
末尾多加一层 softmax          loss = 1.4837     ← 差 48 倍
```

为什么卡在 1.48？算一下极限。设第一层 softmax 输出接近 one-hot `[1,0,...,0]`
（这已经是它能给出的最好答案了），第二层 softmax 作用在它上面：

$$\text{softmax}([1,0,\ldots,0])_{\max} = \frac{e}{e+9} = 0.23197$$

$$-\log(0.23197) = 1.4612$$

**这就是双层 softmax 的理论下界 1.4612**（10 类）。
实测 1.4837 略高一点，因为真实模型的输出不是完美 one-hot。

而正常情况的 loss 可以是 0.03。也就是说：**你无论怎么训，都到不了 0.03，永远卡在 1.46 以上。**

关键是——**它不报错。** 你会看到 loss 从 2.30（均匀预测）降到 1.5 就再也不动了，
然后怀疑是自己的模型太小、学习率不对、数据有问题，查三天。

> 症状识别：**分类任务上 loss 卡在 1.4~1.5 附近降不下去（10 类），
> 或者卡在 0.55 附近（3 类）** → 九成是末尾多套了一层 softmax。

### 3.3 为什么用 log_softmax 而不是 log(softmax)

这个问题很隐蔽，很多人写损失函数时就是这么写的：

```python
log_p = torch.log(torch.softmax(logits, dim=-1))   # ← 有隐患
```

看极端情况，`logits = [0, -1000, 5]`：

```
softmax 后的值  = [0.006693, 0.0, 0.9933]          ← 第二个被压成 0
log(softmax(x)) = [-5.0067, -inf, -0.0067]         ← log(0) = -inf
log_softmax(x)  = [-5.0067, -1005.0067, -0.0067]   ← 正常
```

$e^{-1005}$ 在 float32 里下溢成 0，再取 log 就是 $-\infty$。
而正确的答案是 $-1005.0067$ —— 一个完全正常的有限值。

**先归一化再取对数，会把"很小但非零"的信息彻底抹掉；
先取对数再归一化（log-sum-exp），全程都安全。**

这也是为什么 PyTorch 的 `CrossEntropyLoss` 内部走的是 `log_softmax` 而不是 `softmax → log`。
你手写时必须跟着它走，否则关 1 就会在极端输入上对不上。

---

## 4. 反向：那 5 行代码从哪来

### 4.1 推导

目标是求 $\dfrac{\partial L}{\partial z_{ij}}$，也就是"第 i 个样本、第 j 个类的 logit 变一点点，loss 变多少"。

先只看一个样本 i。设正确类是 $y$：

$$L_i = -\log p_{i,y}$$

**第一步**：用链式法则，把对 $z_{ij}$ 的导数转成对 $p_{i,y}$ 的导数。

$$\frac{\partial L_i}{\partial z_{ij}} = -\frac{1}{p_{i,y}} \cdot \frac{\partial p_{i,y}}{\partial z_{ij}}$$

**第二步**：求 softmax 的导数。这一步是今天的数学核心。

$p_{i,y} = \dfrac{e^{z_{iy}}}{\sum_k e^{z_{ik}}}$，对一个特定的 $z_{ij}$ 求导，要分两种情况：

- **当 $j = y$**（就是正确类本身在分子上）：

$$\frac{\partial p_{iy}}{\partial z_{iy}} = \frac{e^{z_{iy}} \cdot S - e^{z_{iy}} \cdot e^{z_{iy}}}{S^2} = p_{iy}(1 - p_{iy})$$

（$S = \sum_k e^{z_{ik}}$ 是分母。分子求导得 $e^{z_{iy}}$，分母求导得 $e^{z_{iy}}$，商法则出来就是这个。）

- **当 $j \neq y$**（分子里没有 $z_{ij}$，只有分母有）：

$$\frac{\partial p_{iy}}{\partial z_{ij}} = \frac{0 \cdot S - e^{z_{iy}} \cdot e^{z_{ij}}}{S^2} = -p_{iy}\,p_{ij}$$

两种情况可以合并成一个式子（用 $\delta_{yj}$ 表示"j 等于 y 时为 1，否则 0"）：

$$\frac{\partial p_{iy}}{\partial z_{ij}} = p_{iy}\,(\delta_{yj} - p_{ij})$$

**第三步**：代回第一步。

$$\frac{\partial L_i}{\partial z_{ij}} = -\frac{1}{p_{iy}} \cdot p_{iy}(\delta_{yj} - p_{ij}) = -(\delta_{yj} - p_{ij}) = p_{ij} - \delta_{yj}$$

**$p_{iy}$ 被约掉了。** 这就是整个推导最美的地方 —— 也是为什么交叉熵比
"MSE + sigmoid"好训得多的根本原因：sigmoid 的导数 $p(1-p)$ 里那个 $p$ 约不掉，
当 p 接近 0 或 1 时梯度会消失，而交叉熵的梯度**永远不会消失**。

**第四步**：加上 $1/N$（因为 $L$ 是 $N$ 个样本的平均）。

$$\boxed{\frac{\partial L}{\partial z} = \frac{1}{N}\,(p - \text{onehot}(y))}$$

矩阵形式，形状是 (N, 10)。每一行 = 该样本的预测概率，减去 one-hot 标签，再除以 N。

### 4.2 这个结果的两个性质（都是自检点）

**性质 1：每一行加起来必须是 0。**

$$\sum_j (p_{ij} - \delta_{yj}) = \sum_j p_{ij} - 1 = 1 - 1 = 0$$

直观解释：softmax 的输出和恒为 1，所以你把所有 logit **同时加同一个常数**，
loss 一点都不会变。既然往 $[1,1,\dots,1]$ 方向动不改变 loss，那这个方向的导数当然是 0。

实测关 2：`max|Σ| = 4.86e-09`，机器精度级的 0。

**这个性质是免费的自检**：如果你的实现某一行加起来不是 0，一定是 bug。
最常见的错是漏了 one-hot（只用了 p），这时行和 = 1 而不是 0。

**性质 2：梯度绝对值永远不会超过 1/N。**

- 正确类那一项：$p_{iy} - 1$，介于 $[-1, 0]$（因为 $p \in (0,1)$）
- 其他类：$p_{ij} - 0$，介于 $[0, 1]$

所以 $|p - \text{onehot}| \le 1$，整个梯度被 $1/N$ 卡死。

实测（N=64）：梯度最大绝对值 0.015538，而 $1/64 = 0.015625$。

**这就是交叉熵几乎不会梯度爆炸的原因** —— 它的梯度天生有界。
你在 P3 做 VLA 时会发现，有些自定义损失需要手动加 `clip_grad_norm`，
而交叉熵从来不需要。

---

## 5. 手写 backward：`autograd.Function` 的三条契约

正常写 Python 函数，PyTorch 不知道里面发生了什么，没法求导。
继承 `torch.autograd.Function` 等于告诉它："前向这样算，反向我写好了，你直接用。"

```python
class MyCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        ...
        ctx.save_for_backward(logits, target)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        ...
        return grad_logits, None
```

**契约 1：`@staticmethod` + 第一个参数固定叫 `ctx`。**

`ctx` 是"这一层的记事本"，前向时往里记，反向时从里取。
它不是一个特殊对象，就是个普通的上下文容器。

**契约 2：`backward` 返回**元组**，长度 = forward 的输入个数。**

```
forward(ctx, logits, target)   ← 2 个输入
   ↓
backward 必须返回 (grad_logits, grad_target)   ← 2 个元素
```

`target` 是整数张量，不需要梯度，所以填 `None`。
个数不对会报 `function returned an incorrect number of gradients`——**这个错很好抓**。

**契约 3：想支持二阶导，backward 里必须"从输入重算"，不能用前向存的中间量。**

这条是我今天亲自踩的坑。第一版我写的是：

```python
# ❌ 第一版：存 p
ctx.save_for_backward(p, target)     # p 是 forward 里算出来的
```

`gradcheck`（一阶）过了，但 `gradgradcheck`（二阶）报：

```
analytical gradient is all zeros
```

因为 `p` 是在 `no_grad` 环境里算出来的，**它身上没有"我是怎么来的"这张图**。
对图外的量再求导，当然是 0。

改成存输入：

```python
# ✅ 正确版：存 logits，backward 里重算 p
ctx.save_for_backward(logits, target)
...
def backward(ctx, grad_out):
    logits, target = ctx.saved_tensors
    p = softmax(logits, dim=-1)      # ← 重算，这一步会接回计算图
```

代价是 backward 多算一次 softmax（1.5 倍开销）。torch 官方的实现用 C++ 把 p
同时留着，快就快在这 —— 那是性能优化，不是原理差别。

**顺带一条：张量一律用 `save_for_backward`，不要 `ctx.p = p`。**

原因是前者会做**原地修改检查**。关 6 实测了两种写法，结论值得单独记：

```
写法 A：save_for_backward 存 p，backward 里原地改 p[...] -= 1
   → ❗ RuntimeError: one of the variables needed for gradient computation
                     has been modified by an inplace operation

写法 B：ctx.p = p 存 p，backward 里原地改 p[...] -= 1
   → 第 1 次与正确值差 = 0.00e+00   ← 看起来完全正确！
     第 2 次与第 1 次差 = 1.25e-01   ← 这里才露馅，全程无异常
```

**同一个 bug，两种写法：一个当场报错，一个静默算错。**
写法 B 更短、更"自然"，所以更常被写出来 —— 这就是它的危险之处。

### 5.1 最容易漏的一行：`grad_out`

```python
def backward(ctx, grad_out):
    ...
    return grad * grad_out, None     # ← 这个乘号
```

`backward` 求的不是 $\partial L/\partial z$，而是 $\partial \text{FinalLoss}/\partial z$。
如果这个损失后面还接了别的运算（`loss*3`、两个 loss 相加、梯度累积），
上游会把"FinalLoss 对当前这个 loss 的梯度"通过 `grad_out` 传进来。

**漏乘的后果：**

- 单独测 `loss.backward()` 时 `grad_out` 恰好 = 1.0，**结果完全正确**
- 一旦上游有缩放，梯度就少乘一个系数，**不报错**
- 你只会觉得"这个模型怎么训不动"

关 5 专门构造了 `loss*3` 的场景，实测：

```
✅ max|g_scaled - 3·g_base| = 0.00e+00
✅ 同条件下 torch 的梯度也是 3 倍
```

**这是今天最隐蔽的一个坑。** 写任何自定义 Function 都要问自己一句：
"如果上游乘了 3，我的 backward 会跟着乘 3 吗？"

---

## 6. 七关对拍结果

跑法（PowerShell，在**项目根目录**）：

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
& "E:\anaconda\envs\pytorch\python.exe" src\utils\losses.py --train
```

**实测输出（7/7 全过）：**

| 关 | 验什么 | 实测 |
|---|---|---|
| 1 | 前向数值 vs `F.cross_entropy` | 4 组形状，最大 diff `2.38e-07` |
| 2 | 反向梯度 vs autograd | `max|diff| = 1.86e-09`；行和 `4.86e-09` |
| 3 | 数值稳定性 | f32 差 88 / f16 差 11 会炸；稳定版全程有限 |
| 4 | gradcheck + gradgradcheck | 24 个元素逐位一致（双精度） |
| 5 | 链式法则（`loss*3`） | `max|g_scaled - 3·g_base| = 0.00e+00` |
| 6 | retain_graph 幂等 | 两次 backward 差 `0.00e+00` |
| 7 | 真模型训 100 步 | 参数最大差 `2.98e-08`，loss 2.333→2.208 |

**六关的定位不一样，不能只做一关：**

```
关 1  前向对不对？        ← 不对就不用往下看了
关 2  和 autograd 比      ← 但两者可能"错得一样"
关 4  和数值微分比        ← 完全独立于解析推导，这才是硬证据
关 5  上游缩放时还对吗？  ← 抓漏乘 grad_out
关 6  连续用会对吗？      ← 抓污染缓存
关 7  接到真模型上会怎样？← 前面全是人造数据，这一步才算落地
```

**关 2 和关 4 的区别值得单独说。** 关 2 是"我的解析梯度 vs PyTorch 的自动微分"。
但如果我的推导错在一个 torch 也犯的地方（几乎不可能，但原则上可能），
关 2 会通过。关 4 是"我的解析梯度 vs 有限差分"——它真的去扰动每个输入元素、
看 loss 变了多少，**跟任何解析推导完全无关**。这才是能让你放心的那个。

---

## 7. 真实验：单步一致，多步分叉

把 `--loss myce` 接进训练脚本，同一个 seed 跑两次：

```powershell
& "E:\anaconda\envs\pytorch\python.exe" scripts\train_mnist.py --loss ce   --epochs 2 --tag d16_ce
& "E:\anaconda\envs\pytorch\python.exe" scripts\train_mnist.py --loss myce --epochs 2 --tag d16_myce
```

**结果（第一轮完全相同，第二轮分叉）：**

| | 官方 `ce` | 手写 `myce` | 差 |
|---|---|---|---|
| ep1 train_loss | 0.299729 | 0.299729 | **0** |
| ep1 val_loss | 0.148882 | 0.148860 | +2.2e-05 |
| ep2 train_loss | 0.107614 | 0.107602 | +1.2e-05 |
| ep2 val_loss | 0.110548 | 0.114138 | **−3.6e-03** |
| ep2 val_acc | 0.967167 | 0.965333 | 11 条样本 |
| test acc | 0.9701 | 0.9698 | 3 条样本 |

这个现象值得停下来想一分钟。

### 7.1 为什么会分叉

第一天 train_loss 的**前 6 位小数完全一样**，说明单步运算是一致的（这印证了关 7：
参数最大差 2.98e-08）。但那个 1e-8 的差异不会消失，它会：

```
1e-8 的参数差
   ↓ 第 2 轮 469 个 batch，每个 batch 都放大一点点
3.6e-3 的 val_loss 差（放大了 36 万倍）
   ↓ argmax 是离散操作，一点扰动就可能改变分类
11 条验证样本判断不同
```

**SGD 训练是混沌系统。** 非线性（ReLU 的不可导点）+ 离散化（argmax）+ 几百次连乘，
微小误差会被指数级放大。这和天气预报算不准是同一类现象。

### 7.2 这解释了三条你可能早就疑惑的事

1. **为什么深度学习论文从来不报"精确到小数点后 4 位"的准确率** ——
   换个 GPU、换个 batch 顺序、换个 PyTorch 版本，结果就会差这么一点。
   这就是 Day 15 那条"0.9792 和 0.9793 之间没有区别"的物理原因。

2. **为什么实验必须多跑几个 seed 取平均。** 单次跑出来的 0.9701 和 0.9698，
   你没法说哪个更好 —— 它们可能就是同一个东西的两次抽样。

3. **为什么复现性要做"同样的代码同样的机器"** —— 跨机器时浮点运算顺序会变
   （尤其 GPU 的并行归约），误差起点就不一样了。

### 7.3 那到底算不算"对上了"？

算。判据是**差多少 vs 噪声有多大**：

```
test 集 10000 条，p ≈ 0.97
标准误 SE = sqrt(0.97 × 0.03 / 10000) ≈ 0.17 个百分点
实测差 = 0.03 个百分点  <  0.17
```

**差距只有标准误的 1/6，完全在噪声内。** 换个 seed 重跑，谁高谁低都可能反转。

正确的表述是：

> 「手写版和官方版**单步数值一致**（1e-8 级），因此训练轨迹在前几步重合；
> 之后因浮点误差累积而分叉，但最终指标差异远小于统计噪声。」

而不是「完全一样」。这个措辞差别，就是做过实验和没做过实验的区别。

---

## 8. 今天的代码改了什么

| 文件 | 改了什么 |
|---|---|
| `src/utils/losses.py` | **新增**，677 行。softmax / log_softmax / cross_entropy / `MyCrossEntropyFunction` / `MyCrossEntropy` + 七关 |
| `src/utils/config.py` | 加 `loss` 字段（`ce` / `myce`）+ 入口校验 |
| `scripts/train_mnist.py` | 加 `build_criterion(cfg)` + `--loss` 参数 + 日志打印实际用的是哪个 |

**三处设计都遵循同一个原则：接口一致，调用方零改动。**

- `MyCrossEntropy` 继承 `nn.Module`，签名 `(logits, target) -> 标量`，与 `nn.CrossEntropyLoss` 完全相同
- 所以 `train_one_epoch` / `evaluate` **一个字都没改**（它们只知道"crit 是个可调用对象"）
- 这和 Day 12 的 `MySGD` 替换 `torch.optim.SGD` 是完全相同的套路

**为什么 `MyCrossEntropy` 要包一层 Module，而不是直接调 `Function.apply`？**

```
Function = 算子（无状态、只管算）
Module   = 组件（可调用、能 print、能放进 nn.Sequential、能随 checkpoint 存）
```

`train_mnist.py` 里 `crit` 是当**对象**用的（`crit(logits, yb)`），所以必须是 Module。

**还有一处：`target.long()`。**

```python
def forward(self, logits, target):
    target = target.long()
    return MyCrossEntropyFunction.apply(logits, target)
```

传 int32 的 target 不会报错，但索引时负数的行为会静默出错。
这种"类型不对但不报错"的情况，一律在入口强制转换。

---

## 9. 五关（自己答，不用写下来）

**关 1 · 解释减 max**
`softmax` 里为什么要写 `x - x.max()`？它改变数学结果吗？
（提示：分子分母同时乘了什么）

**关 2 · 写出那 5 行**
不看讲义，把 `backward` 的 5 行写出来。写完对照 §4.2 的两个性质自检：
每行和是否为 0？绝对值是否 ≤ 1/N？

**关 3 · 说出一个下界**
如果用双层 softmax（模型末尾已经 softmax，再喂 CE），10 分类的 loss 最小能到多少？
这个数是哪个公式算出来的？

**关 4 · 找出那个静默 bug**
下面这个 `backward` 少了什么？漏了之后**什么情况下会表现正常**？

```python
def backward(ctx, grad_out):
    p, target = ctx.saved_tensors
    onehot = ...
    return (p - onehot) / ctx.n, None
```

**关 5 · 说明为什么分叉**
两次训练的 `train_loss` 第一轮第六位小数完全相同，第二轮差了 1e-5。
这算"实现不一致"还是"浮点误差放大"？你用哪两个数判断的？

---

## 10. 四句话带走

**第一句：交叉熵 = log_softmax + 按 target 取值 + 取负平均。**

三步，不能多也不能少。**模型末尾不要接 softmax** —— 接了会让 loss 卡在 1.46（10 类）
再也降不下去，而且不报错（实测差 48 倍）。

**第二句：softmax 必须减 max，log_softmax 必须用 log-sum-exp。**

不是优化，是正确性。float32 差 88 就炸，**float16 差 11 就炸** ——
你后面用 fp16 训练，这条是硬要求。

**第三句：交叉熵对 logits 的梯度 = (p − onehot) / N。**

只用到 p 和标签，不需要 z 本身。两个免费自检：**每行和为 0**、**绝对值 ≤ 1/N**。
梯度天生有界，所以它几乎不会梯度爆炸。

**第四句：自定义 Function 的三个静默陷阱。**

```
存中间量而不存输入     → 一阶对，二阶全 0
漏乘 grad_out          → 单独测对，上游一缩放就错
ctx.p = p 而不是 save_for_backward  → 原地改不报错，第二次才错
```

三个都**不会**在正常路径上报错。写任何自定义损失函数之前，先回头看一眼这三条。

---

## 11. 明日方向与欠账

### Day 17 建议

到这里，MNIST + MLP 这条线基本走到尽头了（98% 天花板、正则化剂量曲线、损失函数手写完毕）。
按 P0 计划，**Day 17 该做「手写 Linear 层 + 模块化重构」**：

- 手写 `Linear` 的前向 + 反向，与 `nn.Linear` 对拍
  （注意 `weight` 的 shape 约定：torch 用 `(out, in)`，数学书写 `(in, out)` —— **最常见的转置来源**）
- 也可以直接跳 **P1 第一周（CNN）** —— 那是唯一能突破 98% 的路

### 欠账（延续，别忘了）

| 欠账 | 来自 |
|---|---|
| `notebooks/w2_backprop_derivation.md` 仍全空（312 行留白） | Day 9 |
| `weekly/w01.md` 未写 | Day 7 |
| Day 11 讲义自问 6 题未答 | Day 11 |
| `train_step` 5 行盲写未验 | Day 11 |
| `outputs/` 需要重跑生成（目录被清空） | Day 16 事故 |

> 今天的 §9 五关是**新的欠账**，但量很小，睡前口述一遍就能过。
> 真正该还的是 `w2_backprop_derivation.md`——它和今天的内容是同一件事的两面
> （一个 numpy 手写、一个 PyTorch 手写），现在补它的成本比当时更低。
