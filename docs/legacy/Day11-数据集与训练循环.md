# Day 11 · 数据集与训练循环

> **今天只干一件事**：把「一个 batch 进网络、出来、参数被改一次」这条路走一遍，走到你能闭上眼睛把它画出来。
>
> 模型用 Day 10 自己写的 `MLP`，数据用现场造的 3000 个点。
> 不碰 MNIST、不碰 DataLoader 的多进程、不碰学习率调度——那些在 Day 13 之后。
>
> 配套作业：`rl-vla-journey/scripts/demo_train_loop.py`（4 处 TODO，按顺序报错）
> 讲义里所有数字都是本机实测（torch 2.13.0+cu126 / GTX 1660 Ti / 2026-10-20）。

---

## 〇、结果先行：今天跑完你应该看到这 5 个数

```
[0] 全量 X(3000, 2)  y(3000,)  每类 [1000, 1000, 1000]
    train X(2400, 2)   val X(600, 2)
[1] MLP(2, 32, 3, n_layers=3)  参数量=1251
[2] [形状] xb=(64, 2)  yb=(64,)  logits=(64, 3)  loss=()
[3] epoch  1/20  train loss=0.1784 acc=0.9529  |  val loss=0.0499 acc=0.9883
    epoch 20/20  train loss=0.0361 acc=0.9892  |  val loss=0.0328 acc=0.9850
[4] 两关都过：loss 降幅过半 + val acc > 0.95
```

就这五行。看起来平淡，但每一行背后站着一个「你不写就会错」的东西：

| 行 | 背后是什么 |
|---|---|
| `(2400, 2)` / `(600, 2)` | 训练集和验证集必须是**两批不同的数据**，否则 acc 是自欺欺人 |
| `参数量=1251` | Day 10 的账没记错（`2*32+32 + 32*32+32 + 32*3+3 = 96+1056+99`） |
| `loss=()` | 0 维标量，不是 `(1,)`。这个形状决定了 `.backward()` 能不能调 |
| `train loss` 降了 5 倍 | 五步写了、顺序对了 |
| `val acc 0.985` | 统计写对了（这一条今天最容易被写错，见 §8 的 E 行） |

---

## 一、Dataset 只要求两个方法

### 1.1 一个 dataset 到底是什么

一句话：**dataset 就是"能用方括号取"的东西**。Python 里这叫序列协议，具体要求只有两个：

```python
class BlobsDataset(Dataset):
    def __len__(self):        return self.X.shape[0]      # 一共多少条
    def __getitem__(self, i): return self.X[i], self.y[i]  # 第 i 条长什么样
```

写完这两行，`ds[7]` 能取、`len(ds)` 能算、`for x in ds` 能遍历。
`DataLoader` 全靠这两条工作——它自己不知道你的数据是图片、文本还是张量，**它只会调 `__getitem__`**。

> 这就是为什么 `Dataset` 这个基类看起来"什么都没干"。
> `torch.utils.data.Dataset` 里只定义了 `__getitem__` 抛 `NotImplementedError`——
> 它是个**契约**，不是个工具。你实现了它，就等于签了字。

### 1.2 官方的 TensorDataset 有多简单

`TensorDataset` 的源码核心就是三行：

```python
def __getitem__(self, index):
    return tuple(t[index] for t in self.tensors)
```

`tensors = (X, y)`，`t[index]` 就是 `X[index]` 和 `y[index]`。
你自己写的那 6 行和它**功能等价**——作业里 `check_dataset_interface` 会逐个元素对比，两者必须完全一致。

那为什么平时用官方的？因为它多了类型标注和边界检查。今天**先自己写一遍**，是为了以后它出问题时你知道该拆哪。

### 1.3 单条是 `(2,)`，不是 `(1, 2)`

这是第一个坑，而且是个形状直觉的坑：

```
ds[7]      → X: (2,)    y: ()      ← 单条样本，第 0 维是「特征数」
batch 之后 → xb: (64, 2) yb: (64,)  ← 第 0 维变成了「条数」
```

**为什么 `y[7]` 必须是 0 维标量**：`CrossEntropyLoss` 要求的形状是 `input (B, C)` + `target (B,)`。
如果你写成 `y[i:i+1]`（返回 `(1,)`），collate 之后 `yb` 会变成 `(64, 1)`，然后实测会撞上：

```
RuntimeError: 0D or 1D target tensor expected, multi-target not supported
```

注意这个文案**没提形状**——它说的是"我要 0 维或 1 维的 target"。第一次看到很容易以为是标签值有问题。
（作业里我加了一条断言，会在更早的地方拦住，并且直说 `y[0] 应该是 0 维标量`。）

这类报错在 Day 9 手推的时候是数学，今天它是你手指的事——**取下标，不要切片**。

---

## 二、DataLoader 做的三件事

```python
DataLoader(ds, batch_size=64, shuffle=True, generator=...)
```

它只干三件事，一件不多：

| 动作 | 具体做什么 | 什么时候需要关心 |
|---|---|---|
| **切批** | 把索引切成 `[0..63], [64..127], ...` | 每时每刻——`steps = ceil(N / bs)` |
| **打乱** | `shuffle=True` 时每个 epoch 重新洗牌一次 | 训练集要，验证集**不要**（没意义且不可复现） |
| **collate** | 把 64 个 `(2,)` 沿第 0 维拼成一个 `(64, 2)` | 拼接规则由 `collate_fn` 决定，默认能把张量堆起来 |

第 3 件事是**桥**：`Dataset` 一次给一条，模型一次吃一批。这座桥不显眼，但它就是
「`(2,)` 变成 `(64, 2)`」发生的地方，也就是张量形状第一次变化的地方。

### 2.1 实际切出来是这样（实测）

```
batch_size    每 epoch 步数      最后一个 batch      条数合计
         8           300               8      2400   OK
        64            38              32      2400   OK
       256            10              96      2400   OK
      2400             1            2400      2400   OK
```

盯着 `bs=64` 这一行看：`2400 / 64 = 37.5`，所以：

- **步数是 38**，不是 37（向上取整）
- **最后一个 batch 只有 32 条**，不是 64

### 2.2 "最后一批不满"会在哪咬你（实测）

算一个 epoch 的平均 loss，下面两种写法看起来都对：

```python
total_loss += loss.item()            # ❌ 每个 batch 等权
total_loss += loss.item() * bs       # ✅ 按条数加权（bs = xb.shape[0]）
```

在作业里的 `check_loss_weighting` 会把它俩都算一遍（同一个模型、同一份数据）：

```
按 bs 加权 = 1.882628   直接平均 = 1.883461
相对偏差   = 0.04%
```

0.04% 很小，但请记住两件事：

1. **它不会自己消失**。它不报错、不警告，只是让每一次记录的 loss 都偏一点点，
   最后你在"哪个超参更好"的比较上被它骗
2. **偏差大小会随打乱顺序变**。同一段代码换一种 `shuffle` 切法，我实测到的是 **0.35%** ——
   差了 9 倍。这条比第 1 条更有用：**加权平均和怎么切批无关，直接平均会跟着变**。
   一个指标如果"换个切法就换个值"，那它就是错的

> 唯一的例外：`drop_last=True` 时所有 batch 都满，两种写法等价。
> 但那时你扔掉了 32 条样本，代价换了个地方出现。

### 2.3 三个参数的取舍（今天知道有这么回事就行）

| 参数 | 默认 | 什么时候改 |
|---|---|---|
| `shuffle` | `False` | 训练集改成 `True`；验证/测试集保持 `False` |
| `drop_last` | `False` | 只有 BatchNorm 且最后一批太小会报错时才设 `True`（会丢掉余数样本） |
| `num_workers` | `0` | 数据加载成为瓶颈时（读图片）才调大。合成张量调大只会更慢 |

`num_workers=0` 的意思不是"不用 worker"，是"**在主进程里取数据**"。今天的数据在内存里，
`num_workers` 越大越慢——多进程要靠 pickle 把张量传过去。

---

## 三、张量形状全程追踪（过关标准 1）

### 3.1 实测输出

```
[形状] batch 0: xb=(64, 2)  yb=(64,)  logits=(64, 3)  loss=()
```

这是作业里的 `trace_shapes()` 打印的。它插在训练前，只跑 3 个 batch，开销可以忽略。

### 3.2 竖着看完整条链

```
X, y                       (3000, 2) / (3000,)       ← 全量
  │ split 80/20
Xtr, ytr                   (2400, 2) / (2400,)
  │ Dataset.__getitem__(7)
ds[7]                      (2,)      / ()            ← 一条
  │ DataLoader collate（64 条堆起来）
xb, yb                     (64, 2)   / (64,)         ← 一批
  │ model(xb)  = Linear(2→32) → ReLU → Linear(32→32) → ReLU → Linear(32→3)
logits                     (64, 3)                    ← 每类一个分数
  │ criterion(logits, yb)  = log_softmax + gather
loss                       ()                          ← 0 维标量
  │ .backward()
每个参数的 .grad            (64, 32) / (32,) / ...     ← 和参数本身同形
  │ .item()
python float               1.882628                    ← 出图、进日志、算平均
```

三条铁律，画在纸上比背下来有用：

1. **第 0 维永远是"条数"**，每一站都在变：3000 → 2400 → 1 → 64 → 64
2. **最后一维是"特征/类别"**，全程不变：2 → 2 → 3
3. **loss 是 0 维**。`()` 不是 `(1,)`——`(1,)` 是个有 1 个元素的向量，`()` 才是标量。
   `loss.backward()` 只接受 0 维张量，`(1,)` 会报 `grad can be implicitly created only for scalar outputs`

### 3.3 怎么用这张表

以后任何一个模型接不上，**先跑一遍 trace，找第一处对不上的站**，比读十遍代码快。
接不上的错误几乎都是这个形状：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x2 and 32x3)
                                              ↑        ↑
                                     xb 是 (64,2)   W 是 (32,3)
```

读法：`(64,2) @ (32,3)` 内维 2 ≠ 32 → 第一层 Linear 的 `in_features` 应该写 2（这是对的），
说明问题不在这一层，而在**进这一层之前被多塞了一维**（比如 batch 后变 `(64, 1, 2)`）。

---

## 四、五步循环，为什么是这个顺序（过关标准 2）

### 4.1 五步

```python
optimizer.zero_grad()          # 1. 清空上一批的梯度
logits = model(xb)             # 2. 前向：建图
loss = criterion(logits, yb)   # 3. 算标量，挂上 grad_fn
loss.backward()                # 4. 反向：沿图把梯度写进每个参数的 .grad
optimizer.step()               # 5. 用 .grad 改参数
```

### 4.2 为什么不能换顺序：十种写错的实测后果

作业里我留了 4 个 TODO，但**真正的坑在"填对了但填歪了"**。
下面每一条都是在本机实际跑出来的，不是推演：

| # | 写法 | 结果 | 报错吗 |
|---|---|---|---|
| A | `__getitem__` 返回 `y[i:i+1]` | 断言拦住：`y[0] 应该是 0 维标量，实际 1 维、形状 (1,)`（不拦的话 torch 会说 `0D or 1D target tensor expected`） | ✅ 报 |
| B | 漏 `.to(device)` | `RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cpu, different from other tensors on cuda:0` | ✅ 报 |
| C | 漏 `optimizer.step()` | loss `1.8826 → 1.8826`，**一位小数都没动** | ✅ 断言抓住 |
| D | 漏 `optimizer.zero_grad()` | loss `0.3345 → 4.3383`，先降后**炸** | ✅ 断言抓住 |
| E | 算准确率时 `argmax()` 漏了 `dim=1` | **train acc 恒为 0.0000，但 val acc 0.9883、退出码 0、两关都过** | ❌ **不报错** |
| F | `evaluate` 漏 `torch.no_grad()` | 结果完全一样，只是多占显存 | ❌ 不报错 |
| G | `evaluate` 漏 `model.eval()` | 结果**逐位相同**（这个模型没有 Dropout/BN） | ❌ 不报错 |
| H | `criterion` 前多加一层 `softmax` | loss `0.7303 → 0.5649`，卡在 0.55 下不去（理论下界 `-ln(e/(e+2)) = 0.5514`） | ✅ 断言抓住 |
| I | 漏 `loss.backward()` | 实测 `1.1096 → 1.1096`，和"完全不训练"一模一样 | ❌ 不报错 |
| J | `step()` 写在 `backward()` 前面 | 实测 `1.1096`，同样一动不动 | ❌ 不报错 |

**十条里有五条不报错。** 这不是今天特别倒霉，这是训练循环的常态——它没有编译期，
也没有类型检查，错的东西只是**静悄悄地算出一个看起来正常的数**。

### 4.3 逐条说清楚机制

**① `zero_grad()` 为什么必须在 `backward()` 之前**

PyTorch 的 `.grad` 是**累加**的（`AccumulateGrad`，Day 8 那张图的终点）。
不清零，第 38 个 batch 的梯度里带着前 37 个 batch 的总和。

实测 D 行：`lr=0.05` 在累积 38 倍梯度之后，等效学习率放大了近 40 倍，
loss 从 0.33 先降到最低点，然后**冲到 4.34**——比随机猜（`ln 3 = 1.099`）还差 4 倍。

这个坑的可怕之处：**小数据集 + 小 lr 时它也能"收敛"**，你就以为写对了。
只有看到 loss 突然爆掉，或者参数变成 `nan`，才会回头怀疑这一行。

> 顺带一个反转：梯度累加本身是**有用**的技术。
> 想用大 batch 但显存不够时，标准做法是「累加 4 个 batch 的梯度再 step 一次」，
> 效果等价于 batch 大 4 倍。所以 `zero_grad()` 不是多余的仪式，它是**在控制累加的次数**。

**② `step()` 为什么必须在 `backward()` 之后**

`step()` 读的是 `.grad`。`backward()` 之前，`.grad` 是 `None`（第一次）或上一批的值（之后）。
`optimizer.step()` 遇到 `grad is None` 的参数会**跳过**，不报错。

实测 I / J 两行：`1.1096 → 1.1096`，正好等于初始化那一步的 loss。**一次都没更新过。**

**③ `loss` 为什么要 `.item()`**

`loss` 是个张量，它的 `grad_fn` 指向整张计算图（前向的每个中间张量都还活着）。
把 `loss` 累加进一个列表，等于把这 38 张图**全部拎着不放**。
`.item()` 把它变成一个 Python `float`，计算图立刻可以被回收。

写对的样子：

```python
total_loss += loss.item() * bs     # ← float
```

写错的样子（不报错，内存只涨不降）：

```python
total_loss += loss                 # ← tensor，图被留住
```

> 为什么这一条今天不再给实测数字：小模型上显存计数受 CUDA 缓存分配器干扰，
> 我测出来的数字前后不一致（同一段代码两次差几十倍）。**测不准的就不写进讲义**——
> 这是比"给个大概数字"更重要的纪律。机制是确定的，量级等 P3 上真模型时会自己跳出来。

---

## 五、`train()` / `eval()` / `no_grad()` 是三件不同的事

三样东西经常被混为一谈，其实它们管的东西完全不同：

| 东西 | 管什么 | 影响数值吗 |
|---|---|---|
| `model.train()` / `model.eval()` | 切换子模块的**行为模式**：Dropout 是否随机置零、BatchNorm 是否更新统计量 | 有 Dropout/BN 时**影响巨大**，没有时**零影响** |
| `torch.no_grad()` | 关闭**计算图构建**：前向结果不带 `grad_fn` | 不影响数值，只影响显存和速度 |
| `optimizer.zero_grad()` | 清空 `.grad` | 影响下一步，不影响本步输出 |

### 5.1 实测：这个模型上 train / eval 完全没区别

```
train() 与 eval() 输出是否完全相同 : True   最大差 = 0.000e+00
```

原因：这个 MLP 里只有 `Linear` 和 `ReLU`，两者都是**逐样本确定性**的，没有随机性。

所以 G 行的错误（漏 `model.eval()`）在这个模型上**没有任何后果**——
这也是它危险的地方：你现在写错，一切正常；等 Day 13 之后加上 `Dropout`，同一个写法会让
验证集准确率莫名其妙地抖动，而你要花半天才能想到是这一行。

**今天是故意提前写的。** 不是因为现在有用，是因为现在**不写以后就忘了**。

### 5.2 实测：`no_grad()` 的代价（这个测得出）

模型 `MLP(2, 256, 3)`，一次性前向 6000 条样本：

| | 单次耗时 | 峰值显存 |
|---|---|---|
| `with torch.no_grad()` | 0.73 ms | 28.32 MB |
| 不加 | 0.76 ms | 34.18 MB |

- **时间差 0.03 ms** —— 在这个规模上你根本感觉不到
- **显存差 5.86 MB（+21%）** —— 全是前向中间结果的计算图

这个 21% 是**固定比例**，不随模型变小。换成 P3 那个级别的模型（一次前向要几 GB），
它就是几 GB 的差别 —— 也就是"能跑"和"OOM"的差别。

> 这也解释了为什么评估循环里三个东西**必须同时出现**：
> `model.eval()` + `with torch.no_grad():` + 不调 optimizer。
> 少任何一个都不会报错，只会在某个你没想到的时刻咬你。

---

## 六、batch_size 实验：epoch 不是进度单位

### 6.1 同 seed、同 epoch 数、同 lr，只改 batch_size（实测）

```
    bs   步/epoch    总更新次数    首个 epoch    末个 epoch   val acc
     8       300        6000       0.059       0.035    0.9817
    64        38         760       0.164       0.036    0.9850
  2400         1          20       1.883       0.081    0.9850
```

loss 轨迹（每个点是一个 epoch 结束时的训练集 loss）：

```
bs=8     0.059 0.038 0.041 0.040 0.040 0.039 0.039 0.038 0.040 0.037 ... 0.035
bs=64    0.164 0.050 0.043 0.041 0.039 0.038 0.038 0.038 0.037 0.037 ... 0.036
bs=2400  1.883 0.675 0.374 0.274 0.222 0.189 0.166 0.150 0.137 0.127 ... 0.081
```

### 6.2 三条结论，第二条最重要

**① 曲线形状变了，而且是相反方向。**

- `bs=8`：曲线**平且抖**。第一个 epoch 就走了 300 步，早就到底了，剩下的只是噪声
- `bs=2400`：曲线**平滑单调下降**，但**慢**。一个 epoch 才更新一次

平滑是因为大 batch 的梯度估计更准（64 条和 2400 条平均，方差差 6 倍）。
但一个 epoch 只走 1 步 —— **光滑不等于进步快**。

**② 同样的 epoch 数，更新次数差 300 倍。**

```
        bs=8   →  300 × 20 = 6000 次参数更新
      bs=2400  →    1 × 20 =   20 次参数更新
```

所以 `bs=2400` 那行末 loss 0.081 远高于 `bs=8` 的 0.035，**不是因为它学得差**，
是因为它只学了 20 次。

> **epoch 不是统一的进度单位。**
> "我训了 20 个 epoch" 在 bs=8 和 bs=2400 下是完全不同的两件事。
> 以后看论文、比结果，先问 batch_size 和步数，再问 epoch。

**③ 但最终准确率几乎一样（0.9817 / 0.9850 / 0.9850）。**

数据量固定时，batch_size 主要影响**速度**和**收敛路径的抖动**，不太影响**能达到的准确率**。
大 batch 真正的代价不是精度，是每 epoch 更新次数少；真正的收益是梯度噪声小、能上更大的学习率。

### 6.3 那实际该选多少

| 你的处境 | 选法 |
|---|---|
| 数据在内存里、模型很小（今天） | 能塞满显存的最大值，然后用**步数**而不是 epoch 计数 |
| 想做 RL / 有随机性（P3） | 小 batch（几十到几百），因为策略梯度本身就是高方差的 |
| 想复现别人的结果 | 抄他的 batch_size，因为 lr 通常是对那个 batch_size 调的 |

`lr` 和 `batch_size` 是**绑在一起**的：batch 变大 → 梯度更准 → 可以调大 lr。
后面把 lr 从 0.05 改成 0.5 而 batch 不动的实验，会让你亲眼看到这句话的代价。

---

## 七、盲写：`train_step` 的 5 行

### 7.1 为什么是这 5 行

上一份文档（`从看懂到写出.md` §9）说：每天留一次 20 行盲写，从 `train_step` 开始。
现在正好到这一天，因为它**同时满足三个条件**：

1. **短**：5 行有效代码，不涉及复杂 API
2. **顺序敏感**：换一个顺序就有一个静默 bug（§4 的 C/D/I/J 全是这一行的事）
3. **要写几百遍**：P1–P3 的每个实验都从它开始

### 7.2 怎么盲写（具体动作，不是口号）

```
第一步（今天，开着代码）  把 5 行抄进 notebooks/，每行后面写一句「为什么」
第二步（合上代码）        白纸重写 5 行
第三步（对照）            逐字比。差在变量名 → 无所谓；差在顺序 → 记下来
第四步（明天）            不看任何东西，再写一次。写不出来就回到第二步
```

**判据**：如果第三天还能默出来，这 5 行就归你了。如果默不出来，
说明你记的是"这五行大概长什么样"，没记住**为什么是这顺序**——回到 §4.2 的表格。

### 7.3 检查清单（写完自己过一遍）

```
□ optimizer.zero_grad() 在第一行
□ model(xb) 的返回值叫 logits 而不是 probs（它还没过 softmax）
□ criterion(logits, yb)，不是 criterion(softmax(logits), yb)
□ loss.backward() 在 optimizer.step() 之前
□ 返回值里 loss 是 .item() 过的 float
□ xb / yb 已经 .to(device)
```

这 6 条里任何一条不对，跑起来都不一定报错——这就是为什么需要清单，
而不是"我觉得我记住了"。

---

## 八、报错对照表

按「先怀疑什么」排序，别从上往下读：

| 你看到的 | 第一嫌疑 | 怎么确认 |
|---|---|---|
| `Expected all tensors to be on the same device` | 数据没 `.to(device)`，模型 `.to(device)` 了 | 在 `train_step` 里 `print(xb.device, model.fc0.weight.device)` |
| `mat1 and mat2 shapes cannot be multiplied` | 进 Linear 之前的形状多了一维 | 跑 `trace_shapes`，看第一处对不上的站 |
| `Expected target size (B, 3), got (B, 1)` | `__getitem__` 里把 `y[i]` 写成了 `y[i:i+1]` | `ds[0][1].shape` 应该是 `()` |
| `grad can be implicitly created only for scalar outputs` | `loss` 不是 0 维（可能被 `.mean(1)` 过） | `loss.shape` 应该是 `()` |
| `ValueError: num_samples should be a positive integer` | `batch_size` 传成了 0 或负，或 dataset 是空的 | `len(ds)` 先看一眼 |
| `TypeError: 'Tensor' object is not callable` | 忘了 `criterion = nn.CrossEntropyLoss()` 的括号 | 看赋值那行有没有 `()` |
| **loss 一动不动** | 漏 `step()` / 漏 `backward()` / `step()` 在 `backward()` 前 | §4.2 的 C / I / J 三行 |
| **loss 先降后炸** | 漏 `zero_grad()`，梯度在累加 | `print(p.grad.norm())` 会看到它越滚越大 |
| **loss 卡在 0.55 附近** | `criterion` 前多加了 softmax | 概率本身被当成了 logits 再 softmax 一次，3 类的理论下界是 `0.5514`；对照随机猜的 `ln 3 = 1.0986` |
| **train acc 恒为 0 而 val acc 正常** | `argmax()` 漏了 `dim=1` | `logits.argmax(dim=1).shape` 应该是 `(64,)` 而不是 `()` |
| **val acc 每次跑都不一样** | `evaluate` 里漏了 `model.eval()`（将来有 Dropout 时） | 或忘了 `torch.no_grad()`，但那个不改数值 |

---

## 九、今天的验收

### 过关标准（打不上勾就不进 Day 12）

```
□ 作业四关按顺序报错，一路填到「两关都过」
□ 不看讲义，把 §3.2 那条形状链默画出来（6 站，含每站的形状）
□ 能不看代码说出五步的顺序，以及漏掉每一步分别会发生什么
□ 跑一次 --compare，能解释「为什么 bs=2400 的末 loss 是 0.081 而 bs=8 是 0.035」
```

### 两条口述（说给我听，各 3 分钟）

1. **"为什么不 `zero_grad()` loss 会炸"** —— 要说清 `.grad` 是累加的（Day 8 的 `AccumulateGrad`），
   以及"等效学习率被放大 38 倍"这个机制
2. **"为什么 eval 要同时写 `model.eval()` 和 `no_grad()`"** —— 要说清这两件事管的**不是同一件事**
   （一个管子模块行为、一个管计算图），少哪个都不会报错

### 一个反问（先自己想，答不上来也没关系）

> 如果把验证集也 `shuffle=True`，val acc 会变吗？

答案：**acc 不变**（预测对错是逐条的，和顺序无关）。但 **loss 会变一点点**——
如果你漏了按 bs 加权，`shuffle` 一改，最后那批不满的 32 条换了样本，epoch/val loss
就会跟着变（§2.2 实测：同一个模型 0.04%，换一种切法 0.35%）。
另外你会**失去完全复现性**：浮点加法顺序变了，第 6 位小数上会有差异。

---

## 附：作业怎么跑

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"

# 主流程（约 3 秒）
& "E:\anaconda\envs\pytorch\python.exe" scripts\demo_train_loop.py

# batch_size 对比实验（约 15 秒，其中 bs=8 那轮要跑 6000 步）
& "E:\anaconda\envs\pytorch\python.exe" scripts\demo_train_loop.py --compare

# 想改超参自己试
& "E:\anaconda\envs\pytorch\python.exe" scripts\demo_train_loop.py --epochs 30 --lr 0.2
```

> ⚠️ 报告写在 `outputs/day11_train_loop.txt`，**每次运行覆盖**。
> 想留住某次结果，把文件复制走，或者用 `paths.run_dir()` 改成带时间戳的目录（Day 2 学过）。
>
> 提示：`--lr 0.2` 配合 `bs=64` 会明显不稳。这不是 bug，是 §6.3 那句话的反面实验。

---

## 自问 6 题（能全答再进 Day 12）

1. `Dataset` 最少要实现哪两个方法？为什么 `DataLoader` 不知道你的数据是什么也能工作？
2. 一个 dataset 有 1000 条、`batch_size=128`，一个 epoch 有几步？最后一个 batch 多少条？
   为什么累加 loss 时必须乘 `xb.shape[0]`？
3. 五步循环里，`zero_grad()` 和 `step()` 分别在 `backward()` 的哪一侧？各自越界的后果是什么？
4. `model.eval()` 和 `torch.no_grad()` 各管什么？为什么今天漏写 `model.eval()` 完全没后果？
5. 同样 20 个 epoch，`bs=8` 和 `bs=2400` 各更新多少次参数？
   为什么后者末 loss 更高却不是"学得更差"？
6. `logits.argmax()` 和 `logits.argmax(dim=1)` 在 `(64, 3)` 的张量上分别返回什么形状？
   写错的那个为什么会让 train acc 恒为 0 而脚本照常退出？
