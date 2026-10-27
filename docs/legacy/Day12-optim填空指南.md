# Day 12 手写优化器 · 填空指南

> 配套作业：`rl-vla-journey/src/utils/optim.py`
> 什么时候看这份：打开 `optim.py` 之后发现自己不知道该往 TODO 里写什么。

---

## §1 先说清楚：你缺的不是算法，是"API 词汇量"

Momentum 的公式你肯定写得出来：

```
buf = momentum × buf + (1 - dampening) × g
p   = p - lr × buf
```

如果按字面翻译成 Python：

```python
buf = 0.9 * buf + 0.1 * g      # ← 这行其实能跑，结果也对
p   = p - lr * buf             # ← 这行能跑，但模型参数一点没变
```

所以真正让你卡住的不是"想不出公式"，而是两件很具体的事：

1. **`self.buf[i].mul_(...).add_(...)` 这种写法看不懂**——数学里没有带下划线的符号
2. **`p = p - ...` 为什么不对**——这是个 Python 的对象语义问题，不是数学问题

这两件都是**词汇问题**，认识一次就够了。下面先补词汇，再逐空讲。

---

## §2 四个必须认识的原地操作

PyTorch 的约定：**方法名结尾加下划线 = 直接改这个张量本身**（in-place）。

| 写法 | 等价于 | 实测结果（`x=[10,20,30]`, `g=[1,2,3]`） |
|---|---|---|
| `x.add_(g)` | `x = x + g` | `[11, 22, 33]` |
| `x.mul_(0.9)` | `x = x * 0.9` | `[9, 18, 27]` |
| `x.add_(g, alpha=0.1)` | `x = x + 0.1 * g` | `[10.1, 20.2, 30.3]` |
| `x.addcmul_(g, g, value=0.5)` | `x = x + 0.5 * g * g` | `[10.5, 22, 34.5]` |

可以链着写，相当于连续做两步：

```python
x.mul_(0.9).add_(g, alpha=0.1)
# 等价于 x = 0.9*x + 0.1*g  →  [9.1, 18.2, 27.3]（实测）
```

**为什么要用带下划线的版本？** 因为 `p.data -= ...` 改的是"模型里那个张量本身"，而 `p = p - ...` 只是让局部变量 `p` 指向了一个新张量，模型里的参数根本没动。实测：

```
p = p - 0.1*p.grad        → 模型 weight 仍是 [-0.0053, 0.3793]（原值，没变）
p.data -= 0.1*p.grad      → 模型 weight 变成 [-0.1053, 0.2793]（变了）
```

---

## §3 十一个空，按"报错顺序"排

⚠️ 打开文件后**不要从 ① 开始找**。第一次运行只会报 **⑤**，因为关 1 用的是默认参数，`weight_decay=0`、`momentum=0` 那些 `if` 分支根本不执行。按下面这个顺序填，每填一个重跑一次。

### ⑤ 更新参数

```python
p.data -= self.lr * direction
```

数学：`p = p - lr × 方向`。`direction` 是上面算好的方向（默认就等于梯度）。
**必须是 `p.data -=`**，`p = p - ...` 静默无效。

---

### ② momentum 首步

```python
self.buf[i] = g.clone()
```

首步缓冲是空的，直接把这次梯度**拷**进去。为什么是 `clone()` 不是 `= g`：

```
buf = g            → buf.mul_(0.9) 后，grad 也变成 [0.9, 1.8, 2.7]  ← 梯度被改坏
buf = g.clone()    → buf.mul_(0.9) 后，grad 仍是 [1.0, 2.0, 3.0]    ← 完好
```

因为 Python 里 `=` 对张量是"两个名字指向同一块内存"，不是复制。

---

### ③ 缓冲滚动

```python
self.buf[i].mul_(self.momentum).add_(g, alpha=1 - self.dampening)
```

数学：`buf = momentum × buf + (1 - dampening) × g`。
拆开就是两步：先 `mul_(momentum)`，再 `add_(g, alpha=1-dampening)`。

`alpha=` 是 `add_` 的关键字参数，意思是"加多少倍"，所以 `alpha=1-dampening` 就是加 `(1-dampening)×g`。

---

### ① weight_decay

```python
g = g + self.weight_decay * p
```

L2 正则把"参数大小"也算进损失，于是梯度上多出一项 `weight_decay × p`。
注意乘的是 **`p`（参数本身）**，不是梯度。

---

### ④ nesterov

```python
direction = g + self.momentum * self.buf[i]
```

普通 momentum 用 `buf` 当方向；nesterov 改成"**当前梯度 + momentum × 缓冲**"，相当于往前多看一步。

---

### 2-① 一阶矩

```python
self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)
```

数学：`m = b1 × m + (1-b1) × g`——梯度的滑动平均，就是"惯性"。
和 ③ 是同一个形状，只是少了 `dampening`。

---

### 2-② 二阶矩

```python
self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
```

数学：`v = b2 × v + (1-b2) × g²`。

**为什么传两次 `g`**：`addcmul_(a, b, value=k)` 算的是 `k × a × b`。传 `g, g` 就是 `g×g`，正好是"梯度平方"。

---

### 2-③④ 偏差校正

```python
m_hat = self.m[i] / (1 - self.b1 ** self.t)
v_hat = self.v[i] / (1 - self.b2 ** self.t)
```

`m` 和 `v` 都是从一个 0 张量起步的，前几步天然偏小（第 1 步只有 `0.1×g`），所以除以 `1 - b^t` 把它放大回来。`b ** t` 是 `b` 的 `t` 次方。

---

### 2-⑤ 更新参数

```python
p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
```

除以 `sqrt(v_hat)` 就是 Adam 的全部魔法：把梯度**大小**约掉，只留**方向**，所以每一步走的距离恒约等于 `lr`。`eps` 防止除零。

---

### 3 lr 衰减

```python
self.opt.lr *= self.gamma
```

第 5、10、15… 步各乘一次 0.5，序列是 `0.1, 0.05, 0.025 …`。

---

## §4 报错顺序表（已实测）

```
原样跑      → TODO 1-⑤
填 ⑤        → TODO 1-②
填 ②        → TODO 1-③
填 ③        → TODO 1-①
填 ①        → TODO 1-④
填 ④        → 关 2 全过 → TODO 2-①
填 2-①      → TODO 2-②
填 2-②      → TODO 2-③
填 2-③      → TODO 2-④
填 2-④      → TODO 2-⑤
填 2-⑤      → TODO 3
填 3        → 全部通过
```

**每一行都只报一个**，看到下一个编号就说明上一个填对了。

---

## §5 三个容易"静默出错"的地方

| 写错 | 现象 | 会不会报错 |
|---|---|---|
| `p = p - lr*direction` | loss 一动不动 | ❌ 不报 |
| `self.buf[i] = g`（漏 clone） | momentum 组的 max diff 变大 | ❌ 不报，但对拍能抓 |
| 漏 `self.t += 1` | 参数变 `nan` | 对拍里的 `isfinite` 检查会抓 |

第三种为什么致命：`1 - 0.9 ** 0 = 0`，除零 → `inf` → `nan`。
这也是为什么对拍代码里单独加了一条 `torch.isfinite()` 检查——**`max([0.0, nan])` 会返回 `0.0`**，不显式检查的话 `nan` 会被当成"不算最大"跳过。

---

## §6 还是写不出来的时候

**不要盯着整个 `step()` 看。** 把一个空单独拎出来，在项目根建一个 `play_optim.py`（已被 `.gitignore` 里的 `/play_*.py` 挡住，不会进仓库），只写三行：

```python
import torch

buf = torch.tensor([1.0, 2.0, 3.0])
g = torch.tensor([0.5, 0.5, 0.5])
print(buf.mul_(0.9).add_(g, alpha=0.1))     # 看结果对不对
```

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
& "E:\anaconda\envs\pytorch\python.exe" play_optim.py
```

**一次只验证一个表达式**，验对了再抄进 `optim.py`。
这比"在 300 行文件里改一行跑一次"快得多——因为报错信息不会告诉你哪一步错了，只会告诉你最后崩在哪。

---

## §7 这一天的标准答案长什么样

全部填对后的实测输出：

```
[关 1] plain SGD
    无选项                          max diff = 0.000e+00   OK
[关 2] Momentum
    momentum=0.9                    max diff = 0.000e+00   OK
    momentum=0.9 dampening=0.1      max diff = 0.000e+00   OK
    momentum=0.9 weight_decay=1e-3  max diff = 0.000e+00   OK
    momentum=0.9 nesterov=True      max diff = 0.000e+00   OK
[关 3] Adam
    Adam lr=0.01                    max diff = 5.960e-08   OK
    Adam lr=0.1                     max diff = 5.960e-08   OK
[关 4] StepLR
    20 步序列逐位一致   OK
[关 5] 训练轨迹
    loss[0]=1.138081  loss[49]=0.700545
    50 步最大偏差 = 1.192e-07   OK
```

Adam 的 `5.96e-08` 比 SGD 大一个量级是正常的——多了 `sqrt` 和除法，浮点误差会累积。判据是 `< 1e-6`，不用追到 0。
