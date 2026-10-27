# Day 8 学习文档 · Tensor 与自动微分

> 目标：**手算的导数和 autograd 算的导数对上**，误差 < 1e-6；并且能说清 `requires_grad` / 计算图 / `.backward()` 三者的关系。
> 对应清单：`P0-逐日执行清单.md` → Day 8（60min 学 / 90min 作业 / 30min 踩坑）
> 本机前置（已实测）：torch **2.13.0+cu126** · GTX 1660 Ti **6GB sm_75** · numpy 2.2.6
> 最后更新：2026-10-19

> ⏸ **Day 6 / Day 7 已按你的决定挂账**（清单里标了 ⏸）。本文**不依赖**这两天——Day 8 要的是纯 PyTorch 数学，和 gymnasium / CartPole 一点关系都没有。
> 唯一从 Day 7 借来的是那 60 分钟的 *Autograd Mechanics* 阅读，本文第一节到第四节能顶掉它。

⚠️ **本文所有输出都来自本机真实运行**，包括报错原文和误差数字。

---

# 〇、结果先行：三个数字

`f(x, y) = x²y + y³`，在 `(x=2, y=3)` 处：

| 来源 | ∂f/∂x | ∂f/∂y |
|---|---|---|
| **纸上手算** | `2xy = 12` | `x² + 3y² = 31` |
| **autograd** | `12.0` | `31.0` |
| **有限差分（第三种方法）** | `12.00000000` | `31.00000000` |

三列完全一致。**今天就要做到这件事：让这三列对上，并且你能解释为什么会一致。**

为什么非要三种方法？一种方法只能证明"我算出了一个数"，三种方法互相独立才能证明"这个数是对的"：

| 方法 | 它在证明什么 |
|---|---|
| 手算（解析式） | **你的数学没问题** |
| autograd | **框架的反向传播没问题** |
| 有限差分 | **前向函数本身没写错**（它完全不看解析式，只用 `f` 的值） |

少一个，你都有一个盲区。真实工作中 90% 的"梯度不对"事故，就是手算和 autograd 都错了——因为它们错得一样（比如你对 `f` 的理解错了）。**有限差分是唯一的独立裁判。**

---

# 一、Tensor 三要素：shape / dtype / device

NumPy 数组有两个要素（shape、dtype），PyTorch 张量有**三个**。多出来的那个是今天所有内容的入口。

```python
import torch
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(a.shape, a.dtype, a.device)
```

```
torch.Size([2, 2]) torch.float32 cpu
```

| 要素 | 是什么 | 为什么重要 |
|---|---|---|
| `shape` | 每一维多长 | 90% 的 bug 第一眼要看它 |
| `dtype` | 每个数占几位、有没有符号 | 决定精度和显存；`float32` 是默认，`float64` 用于对拍 |
| `device` | 这块内存在 CPU 还是 GPU | **CPU 和 GPU 的张量不能直接运算** |

**本机实测**（这条以后每天都会用到）：

```python
torch.cuda.is_available()          # True
torch.cuda.get_device_name(0)      # 'NVIDIA GeForce GTX 1660 Ti'
torch.cuda.get_device_capability(0)  # (7, 5)  ← 就是 sm_75
torch.cuda.is_bf16_supported()     # True（但见下面警告）
```

> ⚠️ **`is_bf16_supported()` 返回 True 不等于你能享受 bf16 加速。**
> bf16 的硬件加速来自 **Ampere（sm_80）起**的 Tensor Core，你这块是 **Turing sm_75**，没有。
> 所以 W2 之后凡是涉及混合精度，一律用 **`fp16` + `GradScaler`**，不要用 bf16。TF32 同理（也是 sm_80 起的）。
> 这是**硬件边界**，不是你配置错了——记住这条，以后看到教程说"用 bf16 省显存"就知道对你无效。

**跨 device 的经典报错**（现在只需认识，Day 15 才真用 GPU）：

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

---

# 二、`requires_grad`：谁需要梯度

PyTorch 的核心设计：**梯度不是自动给所有人的，得有人主动要。**

```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0)                 # 没写，默认 False
```

规则只有一条：**只有 `requires_grad=True` 的浮点张量，才会被记录进计算图、才会有梯度。**

## 2.1 三条硬规则（都有实测报错）

**① 整数张量不能要梯度**

```python
torch.tensor([1, 2, 3], requires_grad=True)
```

```
RuntimeError: Only Tensors of floating point and complex dtype can require gradients
```

导数是对连续量定义的，整数没有"无穷小变化"可言。**这也是为什么神经网络里所有参数都是浮点。**

**② 叶子张量（leaf）才有 `.grad`**

`torch.tensor(...)` 直接创建出来的是**叶子**；任何运算结果都**不是**叶子。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 2
print(x.is_leaf, y.is_leaf)   # True False
```

非叶子张量的 `.grad` **永远是 `None`**，而且会给你一条很长的警告：

```
UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed.
Its .grad attribute won't be populated during autograd.backward().
```

中间结果本来就只是过路的，PyTorch 不替你留（留了显存就爆了）。真要留，用 `y.retain_grad()`。

**③ 只有标量能直接 `.backward()`**

```python
a = torch.randn(3, requires_grad=True) * 2
a.backward()
```

```
RuntimeError: grad can be implicitly created only for scalar outputs
```

因为反向传播要一个"起点梯度"。标量的起点是 1.0，天然存在；向量没有，你得自己给（`a.backward(torch.ones(3))`）。**这就是为什么训练时 loss 一定是标量** —— 你会写 `loss = criterion(...)`，而不是 `losses = criterion(...)`。

---

# 三、计算图：不是流程图，是"反向的链"

张量每做一次运算，PyTorch 就在背后记一笔：**这个结果是由谁、用什么运算得来的**。这些记录串起来就是计算图。

**它只在 `requires_grad=True` 的路径上建立。** 全是常量的运算不建图。

## 3.1 把图打印出来看

```python
x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
y = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
f = x**2 * y + y**3

def walk(node, depth=0):
    if node is None:
        return
    print("  " * depth + type(node).__name__)
    for nxt, _ in node.next_functions:
        walk(nxt, depth + 1)

walk(f.grad_fn)
```

实测输出：

```
AddBackward0
  MulBackward0
    PowBackward0
      AccumulateGrad
    AccumulateGrad
  PowBackward0
    AccumulateGrad
```

对着公式读 `f = (x² · y) + y³`：

```
                 Add（最外层：加法）
                /                  \
        Mul（x² · y）            Pow（y³）
        /          \
    Pow(x²)         y
      |             |
      x             ↑
   AccumulateGrad ← 梯度最终"落"到叶子上的地方
```

三个要点：

1. **节点的名字是"反向操作"的名字**（`AddBackward0` 而不是 `Add`）——因为这张图的用途是往回走。
2. **`AccumulateGrad` 是终点**：梯度一路反传到叶子张量，就"累加"进它的 `.grad`。注意是**累加**（`+=`）不是赋值——这一个词就是今天第一个坑的根源（§八）。
3. 图是**动态**的：每次前向都重新建一张，用完就释放。所以 Python 能写 `if` / `for` 控制流，这是 PyTorch 相对早期 TensorFlow 的最大优势。

## 3.2 一个更直观的观察

```python
print(f)          # tensor(39., dtype=torch.float64, grad_fn=<AddBackward0>)
```

`grad_fn=<AddBackward0>` 这半截就是"我是加法算出来的，要往回想请走 AddBackward"。**只要一个张量打印出来带 `grad_fn`，它就在图里。**

---

# 四、`backward()` vs `autograd.grad()`：两条路，不一样

两个函数都能算梯度，但行为不同——**这是最容易踩的一个隐形坑**。

```python
x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
y = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
f = x**2 * y + y**3

g = torch.autograd.grad(f, (x, y), retain_graph=True)
print(g)          # (tensor(12.), tensor(31.))
print(x.grad)     # None  ← 注意！
```

| | `f.backward()` | `torch.autograd.grad(f, (x, y))` |
|---|---|---|
| 返回值 | `None` | **元组，直接给你梯度** |
| 写到哪 | **写进叶子的 `.grad`** | **什么都不写** |
| 适合 | 训练循环 | 对拍、检查、高阶导 |

实测确认：`autograd.grad` 之后 `x.grad` 仍是 `None`；再调一次 `f.backward()`，`x.grad` 才变成 `12.0`、`y.grad` 变成 `31.0`。

## 4.1 `retain_graph` 是什么

上面那行我加了 `retain_graph=True`。如果**不加**：

```python
torch.autograd.grad(f, (x, y))     # 图被释放了
f.backward()
```

```
RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors
after they have already been freed). Saved intermediate values of the graph are freed when you call
.backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the
graph a second time...
```

**原因：反传完图就扔了**（省显存）。想再走一次得提前说 `retain_graph=True`。

> 为什么训练循环里不用管这个？因为每轮 `loss = model(x)` 都会**重新建一张新图**，旧图用完就该扔。
> 只有"同一张图要反传两次"（对拍、求二阶导、GAN 里更新两个网络）才需要它。

## 4.2 `zero_grad()` 到底清的是什么

训练循环里那行 `optimizer.zero_grad()`，清的就是叶子张量的 `.grad`。因为 §三说过：**`AccumulateGrad` 是累加**。

---

# 五、三方对拍：手算 / autograd / 有限差分

## 5.1 手算

```
f(x, y) = x²y + y³

∂f/∂x = 2xy                  （y 当常数，x² 求导得 2x）
∂f/∂y = x² + 3y²             （x²y 对 y 求导得 x²；y³ 求导得 3y²）

在 (x=2, y=3)：
∂f/∂x = 2·2·3 = 12
∂f/∂y = 2² + 3·3² = 4 + 27 = 31
```

**这两行必须自己在纸上写出来**，不许翻公式表。Day 9 要手推两层网络的反传，今天这步是热身。

## 5.2 autograd

```python
x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
y = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
f = x**2 * y + y**3
gx, gy = torch.autograd.grad(f, (x, y))
print(f.item(), gx.item(), gy.item())
```

```
39.0   12.0   31.0
```

## 5.3 有限差分：完全不看解析式的"第三裁判"

原理就是导数的定义：

```
f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
```

**为什么用中心差分 `(x+h) 和 (x-h)` 而不是前向 `(x+h) 和 x`**：中心差分的误差是 `O(h²)`，前向是 `O(h)`。同样步长，中心差分精确得多。

实测：

```
有限差分 df/dx = 12.00000000   df/dy = 31.00000000
```

## 5.4 ⚠️ `h` 不能随便取：这是一个 U 型曲线

直觉上"h 越小越精确"——**错**。实测（拿复合函数 `exp(sin(xy))` 试，因为它是超越函数，截断误差才看得见）：

| h | 有限差分结果 | 误差 |
|---|---|---|
| 1e-1 | 1.7335366466 | 8.40e-03 |
| 1e-2 | 1.7418548555 | 8.41e-05 |
| 1e-3 | 1.7419381588 | 8.41e-07 |
| 1e-4 | 1.7419389919 | 8.41e-09 |
| **1e-5** | **1.7419390002** | **9.25e-11** ← 最好 |
| 1e-6 | 1.7419390003 | 3.70e-11 |
| 1e-8 | 1.7419390153 | 1.50e-08 |
| 1e-10 | 1.7419388154 | 1.85e-07 |

真实值 `1.7419390003029978`。

**两头都错，中间最好**：

- **h 太大** → 截断误差（把曲线当直线了，泰勒展开的高阶项没忽略干净），误差 ∝ `h²`
- **h 太小** → 舍入误差（`f(x+h)` 和 `f(x-h)` 差了 1e-10，浮点有效位不够，相减之后有效数字被吃光），误差 ∝ `1/h`

`float64` 有约 16 位十进制有效位，两者平衡点在 `h ≈ 1e-5` 附近。**默认用 `h=1e-6` 是安全的选择。**

> 这条对你会一直有用：做数值验证时看到"h 调小反而更不准"，不是你的代码坏了，是数学就这样。

## 5.5 `float32` 够不够

同一道复合题，autograd 的误差：

| dtype | ∂z/∂x 误差 | ∂z/∂y 误差 |
|---|---|---|
| `torch.float32` | 6.75e-08 | 1.16e-08 |
| `torch.float64` | **0.0** | **0.0** |

`float32` 其实**已经过了 1e-6 的门槛**。但今天的关卡是"对拍"，所以我统一用 **`float64`**：

- 误差直接归零，任何偏差都是真 bug，不会被精度噪声掩盖
- `float32` 的相对精度约 1e-7，你的容差若设在 1e-7 就分不清"真错"和"精度"

**规则**：**对拍用 float64，训练用 float32。** 两件事的目的不同。

---

# 六、复合函数：链式法则一次到位

第二阶段作业的题：`z = exp(sin(x·y))`，在 `(x=0.5, y=1.2)`。

链式法则拆开：

```
z = exp(u),  u = sin(v),  v = x·y

dz/dx = exp(u) · cos(v) · y = cos(xy)·y·exp(sin(xy))
dz/dy = exp(u) · cos(v) · x = cos(xy)·x·exp(sin(xy))
```

```python
z = torch.exp(torch.sin(x2 * y2))
zx, zy = torch.autograd.grad(z, (x2, y2))
```

实测：

```
z  = 1.7588188457669927
dz/dx = 1.7419390003029978   手算 cos(xy)*y*exp(sin(xy)) = 1.7419390003029978
dz/dy = 0.7258079167929158   手算 cos(xy)*x*exp(sin(xy)) = 0.7258079167929158
```

**逐位相同。** 这不是巧合——autograd 干的就是把你纸上做的链式法则，按图 §三 的结构自动执行一遍。

> **一个能省大量时间的判断**：只要你的手算用了链式法则，而 autograd 结果对不上，那 99% 是你某个中间步骤漏了一个因子（最常见：忘了 `sin` 求导还要乘里面的 `cos`，或者矩阵乘法忘了 transpose）。**去检查链子最弱的一环，别怀疑框架。**

---

# 七、矩阵版：`L = Σ (Av)²`，梯度是 `2Aᵀ(Av)`

标量练熟了，上真正训练里会出现的形状。

```python
A = torch.randn(4, 3, dtype=torch.float64)      # 常量
v = torch.randn(3, dtype=torch.float64, requires_grad=True)
L = ((A @ v) ** 2).sum()
(gv,) = torch.autograd.grad(L, v)
```

手推：

```
L = Σᵢ (Aᵢ·v)²          ← Aᵢ 是 A 的第 i 行
∂L/∂v = 2 · Aᵀ @ (A @ v)
```

拆解：设 `r = A @ v`（残差向量），`L = Σrᵢ²`。
- `∂L/∂r = 2r`（这是向量，逐元素平方求导）
- `∂r/∂v = A`（线性变换的雅可比就是矩阵本身）
- 按链式法则 **`∂L/∂v = Aᵀ · 2r = 2Aᵀ(Av)`**

关键点：**`Aᵀ` 不是手写 transpose 的巧合，是链式法则里"线性映射反传要用转置"**。这一条在 Day 9 手推反传时是核心，在 Day 17 手写线性层时是核心 —— 你会遇到三次。

实测：`torch.allclose(autograd 结果, 2Aᵀ(Av))` 为 **True**，误差在 1e-10 以内。

> 这个式子的形状直觉也要有：`v` 是 `(3,)`，所以梯度必须也是 `(3,)`。`Aᵀ` 是 `(3,4)`、`Av` 是 `(4,)` → 乘出来 `(3,)` ✓。**算完先看形状对不对，比看数值更快发现错误。**

---

# 八、三个坑（都有实测输出）

## 坑 1：忘记 `zero_grad()` —— 梯度会累加

```python
w = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
for i in range(3):
    loss = w * 2
    loss.backward()
    print(w.grad.item())
```

```
2.0
4.0
6.0      ← 不是 2, 2, 2
```

因为 `AccumulateGrad`（§三）真的在**累加**。第二次反传时 `.grad` 里还留着上一次的 2.0，于是变成 4.0。

清零后：

```python
for i in range(3):
    w2.grad = None      # 或 w2.zero_grad() / optimizer.zero_grad()
    (w2 * 2).backward()
```

```
2.0
2.0
2.0      ✓
```

**为什么这个坑致命**：梯度变成 2 倍、3 倍…… 训练**不会报错**，只是学习率等效变成 2 倍、3 倍，曲线难看、偶尔发散。你会以为是自己超参没调好。

> 顺带：`w.grad = None` 比 `w.grad.zero_()` 更好，前者真把内存放掉了，后者是把已有张量填 0。
> 但正常训练里用 `optimizer.zero_grad()` 就够了，它内部会选合适的方式。

## 坑 2：`.detach()` 和 `.no_grad()` —— 一个是刀，一个是房间

这两个经常被当成同一件事，**实际完全不同**。实测：

```python
a = torch.tensor([1.0, 2.0], requires_grad=True)
b = a.detach()
```

```
a.requires_grad = True    b.requires_grad = False
b 与 a 共用内存: True       ← 关键
改 b[0]=99 后 a = tensor([99., 2.], requires_grad=True)   ← 改到了原张量！
```

**`detach()` 只是把"这一根线"从图上剪断，数据一点没复制。** 你改 `b` 就等于改 `a`。

```python
with torch.no_grad():
    c = a * 2
print(c.requires_grad)     # False
```

`no_grad()` 是**开一个房间**：里面所有新产生的张量都不建图。它是上下文管理器，作用范围是代码块。

| | `.detach()` | `torch.no_grad()` |
|---|---|---|
| 作用对象 | **一个张量** | **一个代码块** |
| 数据 | **共享内存**，改它等于改原张量 | 新建的张量，独立 |
| 典型用途 | 取出数值当 target（如 target network） | 推理/评估，省显存加速 |

> **一个典型事故**：`target = online.detach()` 是**对的**；`target = online` 就完了 —— 训练时 target 会跟着 online 一起变，你还以为自己用了冻结的 target。这类 bug 不报错，只是"效果没达到预期"。Day 5 的「视图 vs 拷贝」讲的是同一类陷阱，只是这次在 PyTorch 里。

## 坑 3：`loss.item()` 为什么必须加

```python
loss = torch.tensor([1.5, 2.5], requires_grad=True).sum()
print(loss)          # tensor(4., grad_fn=<SumBackward0>)    ← 是 Tensor
print(loss.item())   # 4.0                                    ← 是 float
```

三个理由，按严重程度排：

**① 内存泄漏（最致命）**。如果你把 `loss` 直接 append 进列表：

```python
losses = []
for ...:
    losses.append(loss)              # 错：每个元素都攥着整张计算图
    losses.append(loss.item())       # 对：只是一个 float
```

图里有所有中间激活值。攒 10000 个 round，显存/内存一路涨到 OOM。实测对比：

```
存 tensor 的列表元素: ['Tensor', 'Tensor', 'Tensor']
存 float 的列表元素: [0.0, 2.0, 4.0]
sum(存 tensor 的): tensor(6., grad_fn=<AddBackward0>)   ← 连 sum 都在建图
```

注意最后一行：**你对一堆 loss 求和，它又在建新图**。这就是为什么日志里的 loss 曲线会越跑越慢。

**② 日志难看**。`tensor(4., grad_fn=<SumBackward0>)` 写进日志文件，后面想做数值分析还得正则清洗。

**③ 精度**。`float` 是 Python 原生类型，存 CSV / 画图 / 传 wandb 都不会有类型问题。

**规则：`loss` 用于反传，`loss.item()` 用于记录。** 两个都留，别混。

---

# 九、报错对照表（全是本机原文）

| 报错 | 真实原因 | 怎么改 |
|---|---|---|
| `Only Tensors of floating point and complex dtype can require gradients` | 给整数张量设了 `requires_grad` | 加 `.float()` 或创建时写 `dtype=torch.float32` |
| `grad can be implicitly created only for scalar outputs` | 对向量调了 `.backward()` | 先 `.sum()` / `.mean()`，或传 `gradient=` |
| `Trying to backward through the graph a second time...` | 同一张图反传了两次 | 加 `retain_graph=True`，或重新前向 |
| `The .grad attribute of a Tensor that is not a leaf Tensor is being accessed`（警告） | 读中间结果的 `.grad` | 读叶子；真要留中间梯度用 `.retain_grad()` |
| `Expected all tensors to be on the same device...` | 一个在 CPU 一个在 GPU | 统一 `.to(device)` |
| 梯度数值差好几倍 | 忘了 `zero_grad()` | 每轮先清零 |
| 梯度是 `None` | 该张量不在图里（`requires_grad=False` 或经过了 `.detach()`） | 检查是否被剪断 |
| 手算和 autograd 差一个因子 | 链式法则漏乘 | 逐段检查，用有限差分定位是哪一段错 |

**排查顺序（记这个）**：
1. `print(t.is_leaf, t.requires_grad, t.grad)` —— 先确认它在不在图里
2. `print(t.shape)` —— 形状对不对
3. 用有限差分算同一个点 —— 如果 autograd 和差分一致，是你的手算错了；如果不一致，是 `f` 写错了
4. 才去怀疑框架

---

# 十、作业：`scripts/autograd_check.py`

骨架已放进仓库，**4 处 TODO**：

| TODO | 位置 | 要填什么 |
|---|---|---|
| **TODO 1** | `hand_grad_f` | `∂f/∂x = 2xy`、`∂f/∂y = x²+3y²` 的代码形式（用 `xv`、`yv`） |
| **TODO 2** | `auto_grad` | `torch.autograd.grad(...)` 那一行 |
| **TODO 3** | `numeric_grad` | 中心差分：`(fn(x+h) - fn(x-h)) / (2h)` |
| **TODO 4** | 矩阵部分 | 手推的解析式 `2 · Aᵀ @ (A @ v)` |

骨架里已经写好：`check()` 对比函数（含 `torch.allclose` 断言）、复合函数、打印格式、失败时的定位提示。

跑法（**项目根目录**）：

```powershell
& "E:\anaconda\envs\pytorch\python.exe" scripts/autograd_check.py
```

全部通过时你会看到三张对拍表（标量 / 复合 / 矩阵），每行四个数字：手算、autograd、有限差分、是否一致。

**验收要求**（对应过关标准第 1 条）：所有误差 < `1e-6`。骨架里用 `float64` 就是为了让这个门槛不吃力。

---

# 十一、过关自问（口述）

1. `requires_grad`、计算图、`.backward()` 三者是什么关系？为什么给整数张量设 `requires_grad` 会报错？
2. `x` 是叶子，`y = x*2` 不是。`y.grad` 是什么？为什么 PyTorch 不给你留？
3. `f.backward()` 和 `torch.autograd.grad(f, x)` 有哪三处不同？`retain_graph=True` 什么时候必须加？
4. 为什么训练时 loss 必须是标量？
5. 有限差分的 `h` 取 1e-10 为什么会变差？（不要答"精度不够"，要说清是哪两种误差在打架）
6. `a.detach()` 之后改 `b`，为什么 `a` 也变了？这和 Day 5 的哪个概念是同一件事？
7. `losses.append(loss)` 和 `losses.append(loss.item())` 差在哪？为什么前者会让程序越来越慢？
8. `L = Σ(Av)²` 的梯度为什么是 `2Aᵀ(Av)`？`Aᵀ` 是从哪来的？

---

## 附：本机实测原始数据

```
=== f(x,y) = x^2*y + y^3 在 (2,3) ===
f = 39.0   (手算 4*3+27 = 39)
df/dx = 12.0   手算 2xy = 12
df/dy = 31.0   手算 x^2+3y^2 = 31
有限差分 df/dx = 12.00000000   df/dy = 31.00000000

=== z = exp(sin(x*y)) 在 (0.5,1.2) ===
z = 1.7588188457669927
dz/dx = 1.7419390003029978   手算 = 1.7419390003029978
dz/dy = 0.7258079167929158   手算 = 0.7258079167929158

=== 有限差分步长扫描（同一题，真实值 1.7419390003029978）===
h=0.1      err=8.40e-03
h=0.01     err=8.41e-05
h=0.001    err=8.41e-07
h=1e-04    err=8.41e-09
h=1e-05    err=9.25e-11     ← 最优
h=1e-06    err=3.70e-11
h=1e-08    err=1.50e-08
h=1e-10    err=1.85e-07

=== dtype 对精度的影响（复合题）===
torch.float32  dx err= 6.75e-08  dy err= 1.16e-08
torch.float64  dx err= 0.0       dy err= 0.0

=== 计算图（f = x^2*y + y^3）===
AddBackward0
  MulBackward0
    PowBackward0
      AccumulateGrad
    AccumulateGrad
  PowBackward0
    AccumulateGrad

=== 忘记 zero_grad ===
2.0 / 4.0 / 6.0        （应为 2 / 2 / 2）
清零后：2.0 / 2.0 / 2.0

=== 硬件 ===
NVIDIA GeForce GTX 1660 Ti · sm_75 · is_bf16_supported()=True（但无原生 bf16 Tensor Core）
```
