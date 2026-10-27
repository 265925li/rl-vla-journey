# Day 10 · nn.Module 与参数管理

> 今天的目标不是"会用 PyTorch"，是**知道 PyTorch 在你背后做了什么**。  
> 因为 P1 之后你要写自定义网络（DQN 的 Q 网络、PPO 的策略网络），一旦注册机制搞错，  
> 会出现"代码能跑、loss 不降、不报错"这种最难查的问题。
>
> 全文所有数字都是本机实测（torch 2.13.0+cu126，i7-10750H）。

---

## 〇、先看一个数字，它会改变你对"初始化"的看法

40 层网络，每层 256 宽，输入是标准正态（方差 1.0），只用三种初始化跑前向：

| 初始化            | 第 1 层    | 第 5 层    | 第 10 层   | 第 20 层   | 第 40 层       |
| -------------- | -------- | -------- | -------- | -------- | ------------ |
| `nn.Linear` 默认 | 1.13e-01 | 8.15e-05 | 1.01e-08 | 8.34e-17 | **6.83e-32** |
| Xavier         | 3.43e-01 | 3.15e-02 | 9.22e-04 | 1.38e-06 | **1.05e-12** |
| Kaiming        | 6.82e-01 | 6.69e-01 | 1.07e+00 | 9.73e-01 | **4.34e-01** |

**看清楚第一行和第三行。**

> 说明：这张表用 40 个 `Linear+ReLU`、bias 保留默认初始化。  
> 你的作业脚本里 bias 被显式置零，所以跑出来的数字会略有不同（第 40 层约为  
> `5.4e-32 / 1.3e-12 / 4.7e-01`），**量级和结论完全一致**。

同一份网络结构、同一份输入，只换初始化方式：

- 默认初始化：到第 40 层，信号只剩 `1e-32`。`float32` 能表示的最小正规数约 `1e-38`，所以这已经**接近数值 0**。
- Kaiming：到第 40 层，方差还是 `0.43`，**和输入一个量级**。

反向传播走的是同一条路，梯度也会按同样的比例衰减。所以第一行的网络**根本训不起来**——不是慢，是梯度传不回去。

> 这不是理论。你在 P2 写 DQN 时如果自己搭网络忘了初始化，遇到"loss 卡住不动"，第一个要怀疑的就是这个。

今天要讲的四件事，就是从"为什么会衰减成这样"倒推出来的：

1. `nn.Module` 到底在背后帮你做了什么（注册机制）
2. 怎么核对"你的网络真的有那么多参数"
3. 初始化为什么会决定训练生死
4. `state_dict()` 为什么是唯一正确的存模型方式

---

## 一、`nn.Module` 不是"一个类"，是"一张注册表"

初学者常把 `nn.Module` 理解成"PyTorch 给我留的父类，继承一下才能用"。这个理解会让你在某个时刻踩坑。

**准确的说法是**：`nn.Module` 是一个**带自动注册功能的容器**。它重写了 `__setattr__`，所以你每次给它赋值，它都会先看一眼"你塞进来的是什么东西"，然后决定放到哪个抽屉里。

三个抽屉：

| 抽屉            | 放什么            | 怎么进去                              | 会被 `parameters()` 返回吗 |
| ------------- | -------------- | --------------------------------- | --------------------- |
| `_parameters` | `nn.Parameter` | `self.w = nn.Parameter(...)`      | 会（可训练）                |
| `_modules`    | `nn.Module` 子类 | `self.fc = nn.Linear(...)`        | 会（递归进去取参数）            |
| `_buffers`    | 想存但不想训练的张量     | `self.register_buffer("name", t)` | 不会，但会进 `state_dict`   |
| `__dict__`    | 其他一切           | `self.plain = torch.zeros(5)`     | **不会**                |

**这就是为什么 `self.fc = nn.Linear(3, 4)` 不需要你写任何"注册"代码**——不是省略，是 `__setattr__` 帮你做了。

### 实测：四种东西的待遇完全不同

```python
class Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 4)                     # 子模块
        self.w = nn.Parameter(torch.zeros(2))         # 参数
        self.plain = torch.zeros(5)                   # 普通张量
        self.register_buffer("buf", torch.zeros(7))   # 缓冲区
        self.wrong_list = [nn.Linear(4, 4)]           # 普通 list
```

跑出来：

```
子模块 _modules            : ['fc']
可训练参数 named_parameters : ['w', 'fc.weight', 'fc.bias']
缓冲区 named_buffers       : ['buf']
参数总数 sum(p.numel())     = 18
```

数一下：`fc.weight` 是 `4×3=12`，`fc.bias` 是 `4`，`w` 是 `2`，合计 `18`。✅

而 `self.plain`（5 个元素）和 `self.wrong_list` 里那个 `nn.Linear(4,4)`（20 个参数）**一个都没进去**。

**注意 `named_parameters()` 的顺序**：`['w', 'fc.weight', 'fc.bias']` —— 自己的参数排在子模块前面。这个顺序在调试时有用（你会看到"这个参数属于哪一层"）。

### `Parameter` 和普通 tensor 的区别

```python
self.w = nn.Parameter(torch.zeros(2))    # requires_grad=True，自动注册
self.plain = torch.zeros(5)              # requires_grad=False，不注册
```

`nn.Parameter` 就是"贴了标签的 tensor"，标签内容是"我是模型参数，请优化我"。在这个环境里它自动 `requires_grad=True`。

---

## 二、最危险的坑：`list` 和 `ModuleList`

这是今天**唯一一个会造成静默失败**的地方，值得单独一节。

```python
class Wrong(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.layers = [nn.Linear(4, 4) for _ in range(n)]      # ❌ 普通 list

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class Right(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])   # ✅

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
```

实测：

```
Wrong  参数总数 = 0      子模块数 = 0
Right  参数总数 = 60     子模块数 = 1
```

**两个类的 `forward` 都能正常跑、输出形状正确、不抛任何异常。** 但 `Wrong` 的参数总数是 **0**。

后果链条：

```
参数总数 0
  → optimizer = Adam(model.parameters(), lr=1e-3) 拿到一个空列表
  → step() 什么都不更新
  → loss 一条直线，不降
  → 不报错、不警告
```

> **这是最难查的一类 bug**：不是因为报错信息难懂，而是**根本没有报错信息**。  
> 你上次在 Day 9 遇到的"死代码"也是同一类——不报错，悄悄不干活。

**三个正确写法**（按推荐度）：

| 写法                           | 适用                            |
| ---------------------------- | ----------------------------- |
| `nn.Sequential(a, b, c)`     | 顺序执行，不需要在 forward 里穿插别的操作     |
| `nn.ModuleList([...])`       | 需要在 forward 里自己控制流程（循环、跳层、条件） |
| `self.add_module("name", m)` | 需要动态命名                        |

`nn.ModuleList` 本身**不是** `nn.Module` 的子类语义上的"层"，它是个"会注册的列表"——`__setattr__` 看到它就把里面的模块逐个登记。它没有 `forward`，不能直接调用。

---

## 三、`parameters()` / `named_parameters()` / `state_dict()`

这三个是最常混的，用一句话分清：

| 方法                   | 返回什么                               | 什么时候用       |
| -------------------- | ---------------------------------- | ----------- |
| `parameters()`       | 参数张量的**迭代器**（只有张量，无名字）             | 交给优化器       |
| `named_parameters()` | `(名字, 张量)` 的迭代器                    | 调试、打印、按规则分组 |
| `state_dict()`       | **有序字典**，含参数 + buffer（张量，detach 后） | 保存 / 加载     |

实测一个两层 MLP（`5 → 8 → 3`）：

```
0.weight       shape=(8, 5)     numel=40
0.bias         shape=(8,)       numel=8
2.weight       shape=(3, 8)     numel=24
2.bias         shape=(3,)       numel=3
```

**名字是 `0.` 和 `2.`，不是 `1.`** —— 因为用的是 `nn.Sequential(Linear, ReLU, Linear)`，ReLU 没有参数，但它占用了一个索引位置。所以层序号是 `0`（Linear）、`1`（ReLU）、`2`（Linear）。

> 这个现象在你写 `state_dict` 相关代码时会咬人：**名字里的数字是"在 Sequential 里的位置"，不是"第几个 Linear"**。

### `state_dict()` 为什么比 pickle 整个模型好

实测 `state_dict()` 的内容：

```
keys 共 6 个：
  0.weight    (8, 8)   dtype=torch.float32
  0.bias      (8,)     dtype=torch.float32
  2.weight    (8, 8)   dtype=torch.float32
  2.bias      (8,)     dtype=torch.float32
  4.weight    (8, 8)   dtype=torch.float32
  4.bias      (8,)     dtype=torch.float32
```

**全部是张量，一个超参数都没有**——层数是几、宽度是多少、用的什么激活函数，`state_dict` 一概不知道，也不关心。

这就是它和 `pickle` 的本质区别：

|                | 存什么        | 依赖什么          | 出问题的场景               |
| -------------- | ---------- | ------------- | -------------------- |
| `state_dict()` | 只有数字       | 结构由**代码**提供   | 代码改了结构 → 需要手动对齐      |
| `pickle`       | 整个对象（含类路径） | **类定义必须原样存在** | 类改名、挪文件、改构造参数 → 加载失败 |

举个实测的例子。只把属性名从 `fc1` 改成 `fc_first`：

```python
V2().load_state_dict(V1().state_dict())
```

```
RuntimeError: Error(s) in loading state_dict for V2:
    Missing key(s) in state_dict: "fc_first.weight", "fc_first.bias".
    Unexpected key(s) in state_dict: "fc1.weight", "fc1.bias".
```

报错信息说得很清楚：**权重是"数据"，靠名字对上位置。名字变了，数据就找不到自己的位置。**

这个性质是好是坏？**是好的**：

- 权重能跨代码版本复用（只要你保证名字对）
- 权重能被别的语言/框架读（就是一堆数字）
- 加载时**严格检查**，不会静默错位

而 pickle 的问题正相反：它把"这个对象是什么类的实例"也存进去了。你的类改个名、换个目录、加个必填参数，旧文件就永久废了。而且 pickle 反序列化会执行任意代码——**不要加载来路不明的 `.pkl`**。

> 一句话：**`state_dict()` 存的是"这盘菜的原料配比"，pickle 存的是"连盘子一起冻住的整盘菜"。** 换厨房（改代码）之后，前者还能用，后者只能扔。

---

## 四、参数计数：手算必须和程序一致

过关标准第一条就是这个。它的意义不在于"数得准"，而在于**你能不能在写代码之前就知道这个网络多大**。

### 公式

对 `in_dim → hidden → ... → hidden → out_dim`，共 `n_layers` 层（含输出层）：

```
第 1 层     : in_dim × hidden + hidden
中间层      : 每层 hidden × hidden + hidden        （共 n_layers - 2 层）
输出层      : hidden × out_dim + out_dim
```

那个 `+ hidden` 是 bias。**别漏了 bias**，这是最常见的计数错误。

### 实测核对

| 配置                   | 手算      | 程序数     |    |
| -------------------- | ------- | ------- | -- |
| `5 → 8 → 3`，2 层      | 75      | 75      | OK |
| `5 → 8 → 3`，3 层      | 147     | 147     | OK |
| `784 → 128 → 10`，3 层 | 118,282 | 118,282 | OK |
| `20 → 16 → 10`，4 层   | 1,050   | 1,050   | OK |

验算第二行（`5 → 8 → 8 → 3`，3 层）：

```
5×8 + 8 = 48          （第一层）
8×8 + 8 = 72          （中间层，只有 1 层）
8×3 + 3 = 27          （输出层）
48 + 72 + 27 = 147    ✅
```

### 为什么这件事值得练

宽度翻倍时参数如何增长（实测，输入 20 维、一个隐藏层）：

| 配置                    | 参数总数    | 相对上一行     |
| --------------------- | ------- | --------- |
| `20 → 64 → 64 → 10`   | 6,154   | —         |
| `20 → 128 → 128 → 10` | 20,490  | **3.33×** |
| `20 → 256 → 256 → 10` | 73,738  | **3.60×** |
| `20 → 512 → 512 → 10` | 278,538 | **3.78×** |

**宽度翻倍，参数接近 4 倍**，因为中间层是 `h × h`（平方），而第一层 `in × h` 是线性的。输入维度越大，倍数越被稀释（输入 784 时只有 2.1–2.5 倍）。

> 记住这条经验：**调 `hidden` 是平方代价，调 `n_layers` 是线性代价。**  
> 你这台机器只有 6GB 显存，这条规则以后会天天用到。

---

## 五、初始化：从方差传播推出来，不是背下来的

现在回到第〇节那张表——为什么默认初始化会让信号衰减 32 个数量级。

### 5.1 单层推导

一层线性层：

```
z_j = Σ_i w_ij · x_i + b_j
```

先假设 `w` 和 `x` 都零均值、相互独立、`b=0`。那么：

```
Var(z_j) = Var( Σ_i w_ij x_i )
         = Σ_i Var(w_ij x_i)          ← 独立性：和的方差 = 方差的和
         = Σ_i Var(w) · Var(x)        ← 零均值下 Var(wx) = E[w²]E[x²] = Var(w)Var(x)
         = fan_in · Var(w) · Var(x)
```

**这三个等号用的是三个假设，都要成立**：零均值、独立、无 bias 影响。现实中 `w` 初始化是对称的（零均值）、`x` 在浅层大致零均值——所以这个近似在实践里够用。

于是：

> **要让信号不衰减（`Var(z) = Var(x)`），必须 `fan_in · Var(w) = 1`，即 `Var(w) = 1/fan_in`。**

这就是所有初始化方法的核心，剩下的只是"要不要因为激活函数再调一个系数"。

### 5.2 拿 `nn.Linear` 的默认初始化验一下

实测 `nn.Linear(256, 256)`：

```
fan_in = 256, 理论边界 bound = 1/sqrt(fan_in) = 0.062500
实测 weight.min = -0.062499   max =  0.062499
实测 weight.std =  0.036138   理论 std = bound/sqrt(3) = 0.036084
```

`nn.Linear` 默认用的是 `kaiming_uniform_(weight, a=√5)`，落在 `U(-bound, bound)`，`bound = 1/√fan_in`。均匀分布的标准差是 `bound/√3`，和实测的 `0.0361` 吻合。

那么每层的方差传递是：

```
Var(z) = fan_in · Var(w) · Var(x)
       = 256 · 0.001306 · 1.0
       = 0.3343
```

**`0.3343 ≈ 1/3`——每过一层，信号被缩到三分之一。** 40 层就是 `(1/3)^40 ≈ 1e-19` 量级。这和第〇节实测的 `6.8e-32` 是同一件事（实测还叠加了 ReLU 把负半轴清零的效果，所以比纯 1/3 衰减更快）。

**结论：`nn.Linear` 的默认初始化为了让激活值"不要太大"，主动让信号衰减。浅层网络无所谓，深层就是灾难。**

### 5.3 Xavier 与 Kaiming：只差一个系数

| 方法             | 方差                       | 目标         | 适合                            |
| -------------- | ------------------------ | ---------- | ----------------------------- |
| Xavier（Glorot） | `2 / (fan_in + fan_out)` | 前向和反向方差都保住 | `tanh` / `sigmoid` 这类**对称**激活 |
| Kaiming（He）    | `2 / fan_in`             | 前向方差保住     | `ReLU` 这类**把一半清零**的激活         |

**"为什么 Kaiming 多一个 2"**：ReLU 把一半的输入变成 0，输出的方差直接减半。所以要 `2/fan_in` 而不是 `1/fan_in`，正好补偿。Xavier 假设激活函数是对称的（`tanh` 在 0 附近近似恒等），所以不乘 2。

实测里 Xavier 配 ReLU 还是会衰减（`1.05e-12`），因为**它做的是错配的假设**——这就是"为什么用 ReLU 就该用 Kaiming"的定量证据。

### 5.4 你今天要做的实验

三种初始化 × 40 层，打印每层方差。观察的点：

1. `default` 是不是**单调下降**，下降比例是不是稳定在 1/3 左右
2. `kaiming` 的方差是不是**稳定在 0.5 附近**，会不会越来越大
3. `xavier` 配 ReLU 会不会衰减，但比 default 慢

如果你把 `ReLU` 换成 `Tanh`，会发现 `xavier` 稳住了、`kaiming` 反而偏大——**这正好验证了 5.3 那张表的"适合"一栏。**

---

## 六、报错对照表

| 报错 / 现象                                                              | 原因                             | 怎么修                                                |
| -------------------------------------------------------------------- | ------------------------------ | -------------------------------------------------- |
| `参数总数 = 0`，loss 不降、不报错                                               | 用 `list` 存了层                   | 换成 `nn.ModuleList` / `nn.Sequential`               |
| `RuntimeError: Missing key(s) in state_dict: "fc1.weight"...`        | 属性名和存档不一致                      | 改回属性名，或 `load_state_dict(..., strict=False)` 后手动核对 |
| `RuntimeError: Unexpected key(s) in state_dict`                      | 存档里有当前模型没有的层                   | 结构变了，需要迁移权重                                        |
| `TypeError: cannot assign 'torch.FloatTensor' as parameter 'weight'` | 把普通 tensor 赋给了已有 Parameter 的名字 | 用 `nn.Parameter(...)` 包一层                          |
| `AttributeError: 'MLP' object has no attribute 'fc'`                 | `super().__init__()` 没调        | `__init__` 第一行必须 `super().__init__()`              |
| forward 里 `x` 形状不对                                                   | `in_dim` 和实际输入不匹配              | 在 forward 开头 `print(x.shape)` 一次                   |
| 显存不够 / OOM                                                           | width 太大（平方代价）                 | 先降 `hidden`，别先降 `n_layers`                         |

**最后一行值得记住**：显存和参数都是 `hidden²` 主导，先砍宽度收益最大。

---

## 七、过关自问

先自己答，答不出再看对应小节。

**Q1.** `self.fc = nn.Linear(3, 4)` 和 `self.fc = [nn.Linear(3, 4)]`，哪个会被 `parameters()` 看到？为什么？

**Q2.** 一个 `784 → 256 → 256 → 10` 的 MLP，参数总数是多少？写出算式。  
（提示：三层，中间有一层 hidden）

**Q3.** 为什么用 ReLU 要用 Kaiming 而不是 Xavier？那个系数 `2` 是从哪来的？


**Q4.** `state_dict()` 里为什么不存"层数、宽度、激活函数"？这样做的好处和代价各是什么？

**Q5.** `nn.Sequential(Linear, ReLU, Linear)` 的 `state_dict` key 为什么是 `0.weight` 和 `2.weight`，没有 `1.`？

**Q6.** 如果训练时 loss 完全不降、也不报错，你会**先**检查哪两件事？（这题是实操题，不看笔记）

---

## 附：本机实测原始数据

```
===== 注册机制 =====
子模块 _modules      : ['fc']
可训练参数           : ['w', 'fc.weight', 'fc.bias']
缓冲区               : ['buf']
参数总数             = 18   (fc.weight 12 + fc.bias 4 + w 2)
wrong_list 在 _modules 中: False

===== 参数计数核对 =====
in=5    h=8    out=3   layers=2  手算=75      程序=75      OK
in=5    h=8    out=3   layers=3  手算=147     程序=147     OK
in=784  h=128  out=10  layers=3  手算=118282  程序=118282  OK
in=20   h=16   out=10  layers=4  手算=1050    程序=1050    OK

===== nn.Linear 默认初始化 =====
fan_in = 256, bound = 1/sqrt(fan_in) = 0.062500
实测 weight.min = -0.062499   max = 0.062499
实测 weight.std =  0.036138   理论 = 0.036084
单层方差传递 = 256 * 0.001306 = 0.3343

===== 40 层网络（256 宽，输入方差 0.9992，batch 512）=====
初始化          第1层       第5层      第10层      第20层      第40层
default      1.133e-01   8.147e-05   1.009e-08   8.335e-17   6.832e-32
xavier       3.433e-01   3.148e-02   9.220e-04   1.380e-06   1.048e-12
kaiming      6.820e-01   6.693e-01   1.073e+00   9.733e-01   4.342e-01

===== list vs ModuleList =====
Wrong(list)        参数总数 = 0      子模块数 = 0
Right(ModuleList)  参数总数 = 60     子模块数 = 1

===== load_state_dict 改名 =====
V1 keys = ['fc1.weight', 'fc1.bias']
V2 改名 fc_first 后:
  Missing key(s):    "fc_first.weight", "fc_first.bias"
  Unexpected key(s): "fc1.weight", "fc1.bias"

===== 宽度增长（输入 20）=====
20->64->64->10     6,154       -
20->128->128->10   20,490      3.33x
20->256->256->10   73,738      3.60x
20->512->512->10   278,538     3.78x
```
