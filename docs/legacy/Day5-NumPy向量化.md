# Day 5 学习文档 · NumPy 向量化思维

> 目标：建立"能用张量批量算就不写 for 循环"的**直觉**，并且这个直觉是**被数字打出来的**，不是被讲出来的。
> 对应清单：`P0-逐日执行清单.md` → Day 5（40min 学 / 80min 作业 A / 60min 作业 B / 30min 笔记）
> 本机前置（已实测）：numpy **2.2.6** · 环境 `E:\anaconda\envs\pytorch` · CPU i7-10750H 6C12T
> 最后更新：2026-10-19

⚠️ **本文所有输出、所有加速比数字都来自本机真实运行**，没有任何估算值。你的机器上数字会有差异（CPU 不同、后台有没有开浏览器），但**量级不会变**。

---

# 〇、先上数字，再讲道理

今天不做无用功。先看一件事：同一个矩阵乘法，两种写法。

| M=K=N | 三层 for 循环 | `A @ B` | 倍数 |
|---|---|---|---|
| 200 | **3.10 – 3.42 s** | **0.26 – 0.52 ms** | **6,148 – 13,391×** |
| 500 | **47.5 – 53.8 s** | **1.48 – 2.12 ms** | **23,945 – 36,380×** |

（3 次取平均，`numpy 2.2.6`，i7-10750H。**写成区间是因为同一份代码跑了三遍**——为什么三遍都不一样，本身就是今天第一个知识点，下面就说。）

我在上一条消息里说"你会看到 50 倍以上的差距"——**这个预估太保守了，实际是四位数到五位数。**

你先盯着这张表看 10 秒，然后接受一个结论：

> **写 for 循环处理数组，不是"慢一点"，是"完全不同的量级"。**
> 500 规模的矩阵乘，`A @ B` 你眨眼都来不及，for 循环要等将近一分钟。而这只是 500。

## 为什么 200 那一行抖得这么厉害？

看三遍的明细：

| 规模 | for 循环 | `A @ B` | 倍数 |
|---|---|---|---|
| 200（第 1 遍） | 3.415 s | 0.000255 s | 13,391× |
| 200（第 2 遍） | 3.097 s | 0.000504 s | 6,148× |
| 200（第 3 遍） | 3.323 s | 0.000515 s | 6,446× |
| 500（第 1 遍） | 53.760 s | 0.001478 s | 36,380× |
| 500（第 2 遍） | 47.471 s | 0.001484 s | 31,981× |
| 500（第 3 遍） | 50.677 s | 0.002116 s | 23,945× |

**for 循环那边很稳**（3.4 → 3.1 → 3.3，53.8 → 47.5 → 50.7，全在 12% 内）。**抖的是分母**：`A @ B` 在 200 规模只要 0.3–0.5 毫秒，这个量级上，BLAS 启动线程池、第一次分配内存的**固定开销占了很大比例**，测出来本身就有 ±1 倍抖动。分母抖一倍，倍数就抖一倍。

**这就是"小规模基准测不出真性能"的原因**，也是清单要求你同时跑 200 和 500 两个规模的原因。500 那一行的倍数区间（2.4 万到 3.6 万）之所以还是有一倍多的跨度，是因为分母虽然涨到 1.5–2.1 毫秒，仍然处在"BLAS 固定开销还能咬一口"的量级。

> 记住这个方法论：**做性能对比，两个量级都要测**。只测小的，你测的是"启动开销"；只测大的，你等不起。

更狠的是：**倍数随规模增长**（200 是 6 千到 1.3 万倍，500 是 2.4 万到 3.6 万倍）。因为 for 循环是 O(n³) 次 Python 解释器调度，矩阵乘虽然也是 O(n³) 次浮点运算，但它每 8 个浮点塞进一个 SIMD 寄存器、并且不经过解释器。

这在 RL 里意味着什么：PPO 训练时一个 batch 是 `(4096, 64)`，GAE 计算要跑几十万次张量运算。**写错一处 for 循环，训练从 20 分钟变成 3 天**，而你不会收到任何报错——它只是慢，慢到你以为是算法有问题。

所以今天这一整天的分量，比它看起来重。

---

# 一、ndarray 到底是什么：一块内存 + 一个 shape

这是理解后面所有内容的地基。

```python
import numpy as np
a = np.arange(6)
print(a)          # [0 1 2 3 4 5]
print(a.shape)    # (6,)
print(a.dtype)    # int64
print(a.strides)  # (8,)      ← 走一步跨几个字节
```

`ndarray` = **一块连续内存** + **一套"怎么解读这块内存"的说明书**（shape / strides / dtype）。

关键点：`shape` 不是数据的属性，是**说明书上的属性**。同一块内存可以被解读成不同 shape：

```python
a = np.arange(6)
b = a.reshape(2, 3)
print(b)
# [[0 1 2]
#  [3 4 5]]
print(b.base is a)   # True ← 同一个内存块
```

`reshape` **没有复制数据**，它只是换了份说明书。这就是为什么 `reshape` 几乎不花时间，而 `.copy()` 要花时间。

> **记住这一条**：NumPy 里凡是"只换说明书"的操作都极快，凡是"要真搬数据"的操作才慢。
> 视图（view）= 换说明书；拷贝（copy）= 搬到新家。

---

# 二、广播机制：三条规则，右对齐

广播（broadcasting）是 NumPy 最省事也最容易踩坑的地方。规则只有三条：

> **① 把两个 shape 从右往左对齐**
> **② 逐维比较：相等 → 通过；其中一个为 1 → 把 1「撑开」成对方的长度；缺位 → 补 1**
> **③ 任何一维既不相等、也都不是 1 → 报错**

## 2.1 逐维相等 ✅

```python
a = np.arange(6).reshape(2, 3)      # (2,3)
b = np.array([10, 20, 30])          # (3,)
print(a + b)
```

输出（实测）：

```
a shape (2, 3) b shape (3,)
a + b =
 [[10 21 32]
  [13 24 35]]
```

右对齐：`b` 的 `(3,)` 对着 `a` 的最后一维 `3`，相等 → 通过。然后 `b` 在缺位的那一维上被**复制了 2 份**（逻辑上复制，内存里没真的复制）。

## 2.2 有一个是 1 ✅ 撑开

```python
c = np.array([[100], [200]])        # (2,1)
print(a + c)
```

```
c shape (2, 1)
a + c(2,1) =
 [[100 101 102]
  [203 204 205]]
```

`(2,3)` 和 `(2,1)`：右对齐后第一维都是 2，第二维 3 vs 1 → `1` 撑成 `3`。

**这就是"给每一行加一个不同的常数"的写法**，不用循环：

```python
a + c        # c 是列向量，每行加不同值
a + b        # b 是行向量，每列加不同值
```

## 2.3 既不相等也不为 1 ❌ 报错

```python
d = np.array([1, 2])     # (2,)
a + d
```

```
ValueError: operands could not be broadcast together with shapes (2,3) (2,)
```

**看清楚报错里的两个 shape**：`(2,3)` 和 `(2,)`。右对齐后是 `3` vs `2` —— 既不等也没有 1，炸。这个报错你以后会天天见，**辨认方法就是看这两个 shape 在哪一维对不上**。

## 2.4 广播**不能凭空造出新维度**

这是最容易误解的一点：

```python
x = np.arange(3)      # (3,)
y = np.arange(4)      # (4,)
x + y
# ValueError: operands could not be broadcast together with shapes (3,) (4,)
```

很多人以为"3 和 4 应该能算成 3×4 的表"。**不能。** 广播只会「撑开已有的 1」，**不会「把两个不同长度的东西拼成一个新维度」**。要做外积必须**手动插一个长度 1 的轴**：

```python
print(x[:, None].shape)            # (3,1)
print(y[None, :].shape)            # (1,4)
print(x[:, None] + y[None, :])
# [[0 1 2 3]
#  [1 2 3 4]
#  [2 3 4 5]]
```

`None` 就是 `np.newaxis`，作用是**在这一维插一个长度为 1 的轴**。这个写法你在 RL 代码里会看到无数次（比如算 advantage 时 `(B,1) - (B,)`）。

实测还有一个更绕但更典型的：

```python
a = np.random.standard_normal((5, 1, 3))
b = np.random.standard_normal((4, 1))
print((a + b).shape)     # (5, 4, 3)
```

`b` 的 `(4,1)` 右对齐后变成 `(1,4,1)`，和 `(5,1,3)` 逐维比较：5vs1 ✓ 撑开、1vs4 ✓ 撑开、3vs1 ✓ 撑开 → `(5,4,3)`。

## 2.5 `@` 和广播的关系（过关标准第 1 条）

`(B,L,D) @ (B,D,S) -> (B,L,S)` 的规则：

> **对于 >2 维的 `@`：最后两维做矩阵乘，前面所有维必须"广播兼容"。**

```python
B, L, D, S = 2, 3, 4, 5
Q = rng.standard_normal((B, L, D))
K = rng.standard_normal((B, D, S))
print((Q @ K).shape)          # (2, 3, 5)
```

拆开看：
- 最后两维 `(L,D) @ (D,S)` → 内维 `D` 必须相等，结果是 `(L,S)`
- 前面的 `(B,)` 和 `(B,)` → 相等，保留
- 拼起来 → `(B,L,S)`

**如果 B 不一样但其中一个是 1，也能算**（广播），比如 `(1,L,D) @ (B,D,S) -> (B,L,S)`。这就是"同一份权重作用到 B 个样本上"。

---

# 三、axis 语义：axis 指的是**被消掉的那一维**

这是今天第二重要的一句话。绝大多数人第一次学 axis 都靠死记"0 是行、1 是列"，然后在 3 维张量上全线崩溃。

> **`axis=k` 的意思是：沿第 k 维做归约，所以结果里第 k 维消失。**

```python
t = np.array([[1, 2, 3], [4, 5, 6]])   # (2,3)
print(t.sum(axis=0))    # [5 7 9]   shape (3,)
print(t.sum(axis=1))    # [ 6 15]   shape (2,)
print(t.sum())          # 21        标量
```

`axis=0` → 第 0 维（长度 2）消失，剩 `(3,)`。数值上就是"把两行摞起来加起来"，即**每列的和**。
`axis=1` → 第 1 维（长度 3）消失，剩 `(2,)`。即**每行的和**。

**不要再背"0 是列方向"。就问自己一句：这一维消掉之后，剩下的 shape 对不对？**

## 3.1 三维上更明显

```python
x = np.random.standard_normal((2, 3, 4))    # (B, L, D)
print(x.sum(axis=1).shape)     # (2, 4)   ← 中间那维没了
print(x.sum(axis=-1).shape)    # (2, 3)   ← 最后一维没了
```

`axis=1` 和 `axis=-1` 在这里**完全不同**，因为维度大于 2。

- `axis=-1` 永远是"最后一维"，**与张量有几维无关**。写 `axis=-1` 表示"在特征维上归约"是最稳的习惯——因为你在 `(B,L,D)` 和 `(B,D)` 上写的代码是同一行。
- `axis=1` 只在 2 维时才等于 `-1`。

实测确认：`x.sum(axis=2)` 与 `x.sum(axis=-1)` 结果完全一致（`np.allclose` 为 True）。

---

# 四、keepdims：为什么它总和 softmax 一起出现

先看不用 `keepdims` 会发生什么：

```python
t = np.array([[1, 2, 3], [4, 5, 6]])
s = t.sum(axis=1)        # shape (2,)
t / s
```

```
ValueError: operands could not be broadcast together with shapes (2,3) (2,)
```

"每行除以自己的和"这么自然的操作，**直接报错**。原因就是 §2.3：`(2,3)` 右对齐 `(2,)` → `3 vs 2` 炸。

`keepdims=True` 的意思是：**归约之后，把消掉的那一维保留成长度 1**。

```python
s = t.sum(axis=1, keepdims=True)    # shape (2,1)
t / s                               # shape (2,3)  ✓
```

`(2,3)` 和 `(2,1)` → `1` 撑开成 `3` → 每行除以自己的和。**这就是广播 + keepdims 的组合拳**，softmax、LayerNorm、batch mean 全都靠这个。

```python
logits = rng.standard_normal((2, 3, 5))
e = np.exp(logits - logits.max(axis=-1, keepdims=True))
sm = e / e.sum(axis=-1, keepdims=True)
print(sm.sum(axis=-1))
# [[1. 1. 1.]
#  [1. 1. 1.]]
```

三处 `axis=-1` 全部配 `keepdims=True` —— 少一个就直接炸或者算出错的形状。

> **`logits.max(axis=-1)` 那一步不能省**：`exp` 遇到大的 logits 会溢出成 `inf`，`inf/inf = nan`。
> 减掉最大值不改变 softmax 的结果（数学上分子分母同乘一个常数），但能把数值压回安全区。
> PyTorch 的 `F.softmax` 内部做的就是这件事。

---

# 五、视图 vs 拷贝：改一个，另一个跟着变

这是 NumPy 最容易写出"幽灵 bug"的地方。**规则：**

| 操作 | 视图还是拷贝 |
|---|---|
| 基本切片 `a[2:5]`、`a[::2]` | **视图** |
| `reshape`、`ravel`、`transpose`、`a.T` | **视图** |
| 花式索引 `a[[2,3,4]]` | **拷贝** |
| 布尔索引 `a[a > 0]` | **拷贝** |
| `flatten()` | **拷贝** |
| `np.ascontiguousarray()`（对不连续数组） | **拷贝** |

## 5.1 视图会改到原数组

```python
arr = np.arange(10)
v = arr[2:5]      # 视图
v[0] = 999
print(arr)
# [  0   1 999   3   4   5   6   7   8   9]
```

**你没碰过 `arr`，它变了。** 这是 NumPy 的设计（省内存），不是 bug。

## 5.2 花式索引不会

```python
arr2 = np.arange(10)
f = arr2[[2, 3, 4]]     # 拷贝
f[0] = 999
print(arr2)
# [0 1 2 3 4 5 6 7 8 9]   ← 没变
```

**为什么？** 因为花式索引选出来的元素在内存里**不连续**，NumPy 没法用"起点+步长"描述它，只能真复制一份。

## 5.3 视图可以连环套

```python
arr = np.arange(12)
s = arr[::2]        # [0 2 4 6 8 10]
t = s[1:]           # [2 4 6 8 10]
t[0] = -1
print(arr)
# [ 0  1 -1  3  4  5  6  7  8  9 10 11]
```

`s.base is arr` 为 True，`t.base is arr` **也为 True** —— `t` 的说明书直接指向 `arr` 的内存，中间那层视图被"拍平"了。

**怎么查？** 两个工具：

```python
np.shares_memory(arr, t)    # True
t.base is arr               # True
```

## 5.4 `+=` 改的是同一个对象

```python
z = np.ones(4)
w = z
w += 1
print(z)      # [2. 2. 2. 2.]   ← z 也变了
print(z is w) # True
```

`w = z` 只是给同一个对象起了第二个名字（Python 层面就是这样，不是 NumPy 的锅）。`w += 1` 是**就地修改**。
`w2 = z2 + 1` 则是新建一个数组，`z2` 不变。

> **在 RL 里这条会咬人**：经验的 buffer、参数快照、target network 的权重，全都是数组。
> 一句 `target = online` 会把你以为"冻结住"的 target 网络变成 online 的别名，训练曲线直接挂掉，而代码不报错。

## 5.5 转置不搬数据 → 会变慢

```python
A = rng.standard_normal((500, 500))
print(A.flags["C_CONTIGUOUS"])        # True
print(A.T.flags["C_CONTIGUOUS"])      # False
```

`A.T` 只是把 strides 反过来了（"按列读"），内存一点没动。所以 `A.T` 是**瞬间完成**的——但代价在后面：之后对 `A.T` 做运算时，CPU 缓存命中率暴跌，会慢好几倍。需要反复用时才值得 `np.ascontiguousarray(A.T)` 真复制一份。

---

# 六、einsum：用字母写维度

`einsum` 是 NumPy 里表达力最强的函数，**它让你直接写出"我要哪个维度乘哪个维度、剩下哪些维度"**，不用去想 transpose / reshape 的中间步骤。

## 6.1 基本语法

```
np.einsum("输入1下标,输入2下标->输出下标", 数组1, 数组2)
```

规则：
- **同一个字母** = 这个维度要**对齐相乘**
- **输出里出现的字母** = 保留下来
- **输入里有、输出里没有的字母** = 这一维被**求和消掉**
- `->` 可以省略，省略时保留"只出现一次"的字母并按字母序排列

## 6.2 先看几个熟悉的

```python
A = rng.standard_normal((2, 3))
B = rng.standard_normal((3, 4))

np.einsum("ij,jk->ik", A, B)    # 就是 A @ B
np.einsum("ij,jk", A, B)        # 省略 -> ，隐式输出 ik，等价
```

实测：两者 `np.allclose` 为 True，shape `(2, 4)`。

- `j` 在两个输入里都出现 → 对齐相乘，且**不在输出里** → 被求和消掉（这就是矩阵乘的"内维"）
- `i`、`k` 只在各自输入出现 → 保留

**转置**只要把输出字母顺序换一下：

```python
np.einsum("ij->ji", A)      # = A.T
np.einsum("ii->", M)        # = trace（对角线求和）
```

## 6.3 注意力打分（作业 B 的核心）

```python
# Q:(B,L,D)  K:(B,S,D)  ->  scores:(B,L,S)
scores = np.einsum("bld,bsd->bls", Q, K)
print(scores.shape)      # (2, 4, 6)
```

读法（**一个字母一个字母读**）：
- `b` 两边都有、输出也有 → **不动它**（batch 维）
- `d` 两边都有、输出**没有** → 对齐相乘后求和消掉（"每个 query 和每个 key 的向量点积"）
- `l` 只有 Q 有 → 保留
- `s` 只有 K 有 → 保留

一句话：**"对每个 b，把每个 l 和每个 s 的 d 维点积出来"**。这就是 attention 的 score 矩阵。

三种写法实测完全一致（`np.allclose` 全 True）：
```python
np.einsum("bld,bsd->bls", Q, K)
Q @ K.transpose(0, 2, 1)                    # 手写 transpose
# 以及一个三层 for 循环
```

**`...` 省略号形式**（维度多、懒得写全时用）：

```python
np.einsum("...ld,...sd->...ls", Q, K)    # 与 bld,bsd->bls 完全等价
```

`...` 表示"这里有多少维我不管，原样带过去"。写通用代码（不管前面有几个 batch 维）时用它。

## 6.4 ⚠️ einsum 不是性能优化手段

重要实测（`B=8, L=S=128, D=64`，5 次平均）：

| 写法 | 耗时 |
|---|---|
| `Q @ K.transpose(0,2,1)` | **0.62 ms** |
| `np.einsum("bld,bsd->bls", Q, K)` | 3.49 ms |
| `np.einsum(..., optimize=True)` | 3.91 ms |

**einsum 慢了 5.6 倍。** 而且 `optimize=True` 在这里**没有变快**（它优化的是多操作数的收缩顺序，只有 2 个操作数时没什么可优化的）。

原因：`@` 会走 BLAS 的批量矩阵乘（高度优化的汇编），而 `einsum` 的通用路径是朴素循环。

**所以正确的使用姿势是**：

> **einsum 是"表达工具"，不是"性能工具"。**
> 能用 `@` 表达的，用 `@`；`@` 表达起来要写一串 transpose 的，用 `einsum` 求正确和可读；
> 只有真成了瓶颈，才回头把热路径改成 `@` 或 `reshape`。

这和"向量化"不矛盾：**einsum 本身也是向量化的**（没有 Python 层 for 循环），只是没到 BLAS 那个级别的极致。

---

# 七、把今天四件事串起来：注意力打分

现在把 §2 的广播、§3 的 axis、§4 的 keepdims、§6 的 einsum 全部用在同一个例子上。**这个流程你在 P3 的 VLA 阶段会一字不差地再见到。**

```python
# 字母含义：
#   B = batch size          （一次处理几个样本）
#   L = query 序列长度       （当前有 L 个位置要"提问"）
#   S = key 序列长度         （有 S 个位置可供"被查询"）
#   D = 每个 token 的特征维  （embedding / head_dim）
Q = rng.standard_normal((2, 4, 8))     # (B,L,D)
K = rng.standard_normal((2, 6, 8))     # (B,S,D)

# ① 打分：einsum 一次算完，没有任何 for 循环
scores = np.einsum("bld,bsd->bls", Q, K)          # (B,L,S)

# ② 缩放：除以 sqrt(D)，防止 dot 值随 D 变大而爆炸
scaled = scores / np.sqrt(D)

# ③ 稳定化：减掉每行最大值（axis=-1 + keepdims）
scaled = scaled - scaled.max(axis=-1, keepdims=True)

# ④ softmax：exp 后按最后一维归一化
weights = np.exp(scaled)
weights = weights / weights.sum(axis=-1, keepdims=True)
```

实测输出：

```
weights shape (2, 4, 6) 每行和: [1. 1. 1. 1.]
每行和全 1: True
```

**"每行和为 1"** 就是 softmax 的验收标准 —— 自己写完一定要 `assert np.allclose(w.sum(axis=-1), 1.0)` 检查一次。

## 7.1 顺手加个因果掩码

语言模型不能"看到未来"，所以要遮掉上三角：

```python
mask = np.tril(np.ones((L, S), dtype=bool))
# [[1 0 0 0 0 0]
#  [1 1 0 0 0 0]
#  [1 1 1 0 0 0]
#  [1 1 1 1 0 0]]

scaled = np.where(mask, scaled, -np.inf)      # 遮掉的位置给 -inf
weights = softmax(scaled)                     # exp(-inf) = 0
```

实测：被遮掉的位置权重精确为 `0.`，第 0 行权重是 `[1. 0. 0. 0. 0. 0.]`（第一个 token 只能看到自己）。

**注意 `-np.inf` 而不是 `0`**：softmax 里"权重为 0"等价于 logit 为 `-inf`，写成 `0` 只会让它变成一个"普通候选"。

（`np.tril` 只对 2 维生效，所以 `(B,L,S)` 上要 `mask = np.tril(...)[None, :, :]` 让它在 batch 维上广播。）

---

# 八、报错对照表

| 报错 | 真实原因 | 怎么改 |
|---|---|---|
| `operands could not be broadcast together with shapes (2,3) (2,)` | 右对齐后 `3 vs 2` 对不上 | 加 `keepdims=True`，或 `(2,)` → `(2,1)` |
| `operands could not be broadcast together with shapes (3,) (4,)` | 想算外积 | `x[:,None] + y[None,:]` |
| `matmul: Input operand 1 has a mismatch in its core dimension 0` | `@` 的内维不相等 | 检查两个数组最后两维：`(L,D) @ (D,S)` 的 `D` 必须相等 |
| `ValueError: axes don't match array` | `transpose` 参数个数 ≠ 维数 | `(B,S,D)`→`(B,D,S)` 要写 `transpose(0,2,1)`，三个都要写 |
| `IndexError: too many indices for array` | 用 2 个下标索引 1 维数组 | 先 `print(x.shape)` 再说 |
| 结果形状对但数值是 `nan` | softmax 没减最大值，`exp` 溢出成 `inf` | `x - x.max(axis=-1, keepdims=True)` |
| 形状对、数值也对，但训练结果不对 | 该 `copy()` 的地方用了视图，或 `target = online` | 打 `np.shares_memory(a, b)` 排查 |

**通用排查动作（三秒定位）**：报错时**第一件事永远是 `print(x.shape)`**。NumPy 的 bug 有 90% 是 shape 问题，而报错信息里给你的那两个 shape 往往就是答案。

---

# 九、今天的作业（骨架已放在仓库里）

## 9.1 作业 A：`scripts/numpy_benchmark.py`

复现 §〇 那张表。骨架我已经写好，**结构、导入、argparse、日志、错误处理全部现成**，只留 3 处 TODO：

| TODO | 位置 | 要填什么 |
|---|---|---|
| **TODO 1** | `matmul_loop` 内层 | 三层循环的核心那一行：`C[i, j] += a * B[k, j]` |
| **TODO 2** | `time_it` 内 | 用 `time.perf_counter()` 包住 `fn(*args)`，收集 `runs` 次，返回平均 |
| **TODO 3** | `main` 循环里 | 调 `time_it` 测两个函数，算倍数，打一行表格 |

跑法（**在项目根目录** `E:\work Buddy\强化学习\rl-vla-journey` 下打开终端）：

```powershell
& "E:\anaconda\envs\pytorch\python.exe" scripts/numpy_benchmark.py
```

⚠️ **200 和 500 都跑、各 3 次，全程约 3–4 分钟**（本机实测两次分别是 3 分 53 秒和 2 分 32 秒），其中 `500` 那个 for 循环一次就 50 秒左右。**不要以为卡死了。** 想先验证逻辑对不对，用 `--sizes 100 --runs 1` 秒出结果。

结果会自动写进 `outputs/numpy_benchmark.txt`（复用 Day 2 的 logger 和 Day 3 的路径工具，**这就是"复用"第三次派上用场**）。

## 9.2 作业 B：`scripts/attention_scores.py`

按 §七 实现注意力打分。骨架同样给好，留 3 处 TODO：

| TODO | 要填什么 |
|---|---|
| **TODO 1** | `einsum` 的下标串：`"bld,bsd->bls"` |
| **TODO 2** | softmax 的分母：`e / e.sum(axis=-1, keepdims=True)` |
| **TODO 3** | 因果掩码：`np.where(mask, scaled, -np.inf)` |

**硬要求**（对应过关标准第 2 条）：**整个文件零 `for` 循环**。写完自己搜一遍 `for `（注意：文件头和提示注释里本来就有 "for 循环" 这几个字，搜到的注释不算，只看代码行）。

文件头已经写好了字母含义的注释模板，把 `?` 填成真实数字和含义。

## 9.3 笔记：`notebooks/w1_numpy_notes.md`

30 分钟，写 **3–5 条今天真觉得"原来如此"的点**。不要抄本文，写你自己被戳到的地方。

示例格式：
```
## 1. axis 不是"行/列"，是"被消掉的那一维"
我一直背"0 是列方向"，所以在 (B,L,D) 上就崩了。改成问"消掉后 shape 对不对"之后，
从来没有错过。验证：x.sum(axis=1).shape 在 (2,3,4) 上是 (2,4)。
```

---

# 十、过关自问（口述，别查文档）

对着空气说，说不出来就是没懂：

1. `(B,L,D) @ (B,D,S)` 为什么得到 `(B,L,S)`？哪一维必须相等？哪一维允许是 1？
2. `x` 是 `(2,3,4)`，`x.sum(axis=1)` 和 `x.sum(axis=-1)` 分别是什么 shape？为什么不一样？
3. 为什么 softmax 里 `keepdims=True` 一去掉就报错？报错长什么样？
4. `a[1:3]` 和 `a[[1,2]]` 哪个改原数组、哪个不改？为什么？
5. `einsum("bld,bsd->bls")` 里 `d` 去哪了？为什么它必须"消失"？
6. 你今天实测的加速比是多少？如果是 50 倍，你会得出什么错误的结论？

---

## 附：本机实测原始数据

**第 1 遍**（`--sizes 200,500 --runs 3`，早期版本，无 8×8 预检行）：

```
numpy 2.2.6 | sizes=[200, 500] | runs=3
   M=K=N |    for 循环(s) |     A @ B(s) |         倍数
---------------------------------------------------
     200 |        3.415 |     0.000255 |     13391x
     500 |       53.760 |     0.001478 |     36380x
---------------------------------------------------
200: 循环 3.42s vs 矩阵乘 0.26ms
500: 循环 53.76s vs 矩阵乘 1.48ms
```

**第 2 遍**（与仓库里最终骨架完全一致的代码）：

```
numpy 2.2.6 | sizes=[200, 500] | runs=3
   M=K=N |    for 循环(s) |     A @ B(s) |         倍数
---------------------------------------------------
小规模正确性检查通过（8x8 两版结果一致）
     200 |        3.097 |     0.000504 |      6148x
     500 |       47.471 |     0.001484 |     31981x
---------------------------------------------------
```

**第 3 遍**（作业全部填完后，用仓库里的 `scripts/numpy_benchmark.py` 跑出来的）：

```
numpy 2.2.6 | sizes=[200, 500] | runs=3
   M=K=N |    for 循环(s) |     A @ B(s) |         倍数
---------------------------------------------------
小规模正确性检查通过（8x8 两版结果一致）
     200 |        3.323 |     0.000515 |      6446x
     500 |       50.677 |     0.002116 |     23945x
---------------------------------------------------
完成。结果同时写在 outputs/numpy_benchmark.txt
```

> 你跑出来的数字**一定和这两个都不一样**，这很正常。要看的是**量级**，不是小数点后第三位。

**einsum vs `@`**（`B=8, L=S=128, D=64`，5 次平均）：

```
@ + transpose : 0.62 ms
einsum        : 3.49 ms
einsum优化    : 3.91 ms
```

**循环里 append vs 向量化**（20 万个数）：

```
append 建表 + np.array : 36.2 ms
np.arange(200000) * 2  : 0.89 ms      ← 反例：在循环里 append
预分配 buf[i] = ...    : 43.9 ms
```

最后两行值得单独说一句：**连"预分配 + 按索引赋值"这种看起来已经很优化的循环，也比纯向量化慢 40 倍以上。** 循环就是循环，省不回来。
