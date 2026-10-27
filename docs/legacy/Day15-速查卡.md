# Day 15 速查卡

> 一页看完，5 分钟。复习时看这个，不用翻 800 行的讲义。

---

# 第一部分 · 今天的四个模块

## 数据流：谁调用谁

```
configs/mnist_mlp_reg.yaml
        │
        │  load_config(path)              ← 读 yaml + 填默认值 + 校验 + 路径绝对化
        ▼
    Config 对象                          ← 一个 dataclass，所有参数都在这里
        │
        │  train_mnist.main()             ← 唯一的入口，按固定顺序装配
        ▼
   ┌──────────────────────────────────────────────────┐
   │ ① set_seed(cfg.seed)                              │  必须在建模型之前
   │ ② build_loaders(cfg)   → train/val/test 三个 loader│
   │ ③ build_model(cfg)     → MLP（用 cfg.dropout）     │
   │ ④ build_optimizer(cfg) → MySGD/SGD/Adam（用 wd）   │
   │ ⑤ build_scheduler(cfg) → MyStepLR / StepLR / None  │
   │ ⑥ for epoch in 1..N:                              │
   │      train_one_epoch()  ← model.train() 打开 Dropout│
   │      evaluate()         ← model.eval()  关掉 Dropout│
   │      早停判断 → 可能 break                          │
   │ ⑦ 载入 best.pt → 测一次测试集                       │
   │ ⑧ 存 config.yaml / train.log / history.json / curve.png│
   └──────────────────────────────────────────────────┘
```

`run_ablation.py` 是**外面再套一层**：它用 `subprocess` 反复调 `train_mnist.py`，
每组一个新进程（保证随机状态互不污染），最后把四个 `history.json` 汇总成一张表。

---

## 模块 1 · `src/models/mlp.py` — 模型

**它是什么**：Day 10 手写的 MLP，今天多了一个开关。

**接口**

```python
MLP(in_dim, hidden, out_dim, n_layers, act="relu", dropout=0.0)
```

**结构**（每个 hidden 单元）

```
Linear → 激活 → [Dropout]     ← 方括号里的可选，dropout>0 才插入
```

最后输出层只有 `Linear`，不接激活、不接 Dropout。

**今天改了什么**

| 改动 | 为什么 |
|---|---|
| 加 `dropout` 参数 | 正则化入口 |
| 在激活**之后**插 `nn.Dropout` | ReLU 后的 0 本来就是死的，丢它等于白丢 |
| 加 `0 <= dropout < 1` 校验 | `dropout=1.0` 会让输出恒为 0，**不报错但训练不动** |

**你必须记住的一点**

`nn.Dropout` **没有参数**。所以 `203530` 这个参数量加了 dropout 之后**一个字都没变**。
参数量不变，但模型的有效容量变了 —— **参数量不是衡量复杂度的唯一指标**。

**`count_params_manual()` 今天一个字都没改**，因为它干的是纯结构计算，
而 Dropout 不占参数。这也顺便验证了 Day 10 那个公式是对的。

---

## 模块 2 · `src/utils/config.py` — 配置

**它是什么**：把 yaml 变成一个「有字段、有类型、有默认值、能校验」的对象。

**接口**

```python
cfg = load_config(Path("configs/mnist_mlp.yaml"), overrides={"lr": 0.01})
save_config(cfg, out_dir / "config.yaml")     # 存一份"这一跑到底用了什么"
```

**今天的三个新字段**

| 字段 | 默认 | 含义 |
|---|---|---|
| `dropout` | `0.0` | 丢弃概率（**0 = 不加**，保持 Day 13 行为不变） |
| `early_stop_patience` | `0` | 连续几轮不改善就停（**0 = 关闭**） |
| `weight_decay` | 已有 | Day 12 就加了，今天才第一次真的用上 |

**默认值的原则**：新增字段的默认值必须让「老配置的行为完全不变」。
所以两个都是 0 —— 不填就等于没这个功能。

**校验加在哪**：`__post_init__`（字段赋值完成后自动调用）。

```
为什么在这里：配错的参数如果拖到训练第 1000 步才炸，你已经浪费半小时 GPU。
**配置的错，要在读配置的那一秒就炸。**
```

今天新加的三条：`dropout ∈ [0,1)`、`weight_decay >= 0`、`patience >= 0`。

**顺手修的 bug（重要）**

```python
# 修之前：data_dir: str = "data" 是相对路径，相对于【当前工作目录】
# 修之后：load_config() 末尾把它按【项目根】解析成绝对路径

for name in ("data_dir", "output_dir"):
    p = Path(getattr(cfg, name))
    if not p.is_absolute():
        setattr(cfg, name, str(ROOT / p))
```

**为什么必须修**：

```
在项目根跑        → <项目根>/data/             ✅
在 scripts/ 里跑  → <项目根>/scripts/data/     ❌ 又下了 64 MB
```

这不是理论风险，你的磁盘上真的多了一份。**任何"相对于当前目录"的路径，
早晚会在某个诡异的运行方式下出错。** 一律从 `__file__` 反推。

---

## 模块 3 · `scripts/train_mnist.py` — 训练主程序

**它是什么**：把「配置 → 数据 → 模型 → 训练 → 存盘」串起来的唯一入口。

**命令行参数**

| 参数 | 说明 |
|---|---|
| `--config` | 配置路径，默认 `configs/mnist_mlp.yaml` |
| `--epochs` `--lr` `--batch-size` `--seed` `--device` `--optimizer` | 覆盖 yaml |
| `--dropout` | 覆盖 `cfg.dropout` |
| `--weight-decay` | 覆盖 `cfg.weight_decay` |
| `--patience` | 覆盖 `cfg.early_stop_patience` |
| `--es-metric` | **早停看哪个指标**：`loss`（默认）/ `acc` |
| `--tag` | 输出目录后缀，**多跑几次必须用不同 tag，否则看不出谁是谁** |

> 所有覆盖参数的 `default` 都是 `None`。`None` = "用户没传"，就不覆盖 yaml。
> 写成 `default=10` 就永远分不清"用户想要 10"和"用户没说话"。

**四个核心函数**

| 函数 | 干什么 | 关键点 |
|---|---|---|
| `train_one_epoch()` | 跑完一个 epoch | `model.train()` + 五行循环；按 **样本数加权**累加 |
| `evaluate()` | 算 val/test 指标 | `@torch.no_grad()` + `model.eval()` |
| `weight_norm()` | `sqrt(Σp²)` | **验证 wd 有没有生效的观测量** |
| `plot_curves()` | 三张图：loss / acc / gap | gap 图是过拟合的体温计 |

**今天加的两个指标**

```python
"gap":   round(tr_acc - va_acc, 6),      # 过拟合的直接读数
"wnorm": round(weight_norm(model), 6),   # wd 有没有真的生效
```

两个都是**确定性值**，所以不影响 `history.json` 的复现性对比。
（Day 13 定下的规矩：**只有墙钟时间不能进 history**，否则 diff 永远有差异。）

**早停的实现（今天的重点）**

```python
best_val_loss = float("inf")
bad_by_acc = 0
bad_by_loss = 0

for epoch in ...:
    ...
    improved_by_acc = va_acc > best_acc          # ← 必须在更新 best_acc 之前算！
    if improved_by_acc:
        best_acc = va_acc
        torch.save(model.state_dict(), best_path)     # 存"最好那一版"

    improved_by_loss = va_loss < best_val_loss - 1e-6 # ← 1e-6 是容差
    bad_by_acc  = 0 if improved_by_acc  else bad_by_acc  + 1
    bad_by_loss = 0 if improved_by_loss else bad_by_loss + 1
    if improved_by_loss:
        best_val_loss = va_loss

    if cfg.early_stop_patience > 0:
        bad = bad_by_acc if args.es_metric == "acc" else bad_by_loss
        if bad >= cfg.early_stop_patience:
            stopped_at = epoch
            break
```

**三个必须看懂的点**

1. **`improved_by_acc` 要在存盘前算。** 写成先 `best_acc = va_acc` 再比，
   后者恒为 `False`，早停第 2 轮就触发。**不报错，只是行为错。**
2. **`1e-6` 容差不能省。** 不加的话浮点抖动（`0.0789 → 0.078899`）会被当成"有进步"，
   计数器永远归零，**早停永远不触发**。
3. **停下来了必须用 `best.pt`，不能用当前权重。** 早停触发那一轮权重已经变差了。

**输出（每个 run 一个目录）**

```
outputs/<时间戳>_mnist_<tag>/
    config.yaml     这一跑的参数（半年后复现全靠它）
    train.log       完整日志
    history.json    每轮指标 —— 复现性对比的就是这个文件
    curve.png       loss / acc / gap 三张图
    best.pt         验证集最好的那一版权重
```

---

## 模块 4 · `scripts/run_ablation.py` — 对照实验调度器（今天新增）

**它是什么**：不是模型代码，是**做实验的代码**。以后做 PPO 变体对比用的是同一套。

**四组设计**

```python
GROUPS = [
    {"name": "A_baseline",     "dropout": 0.0, "wd": 0.0},   # 参照物
    {"name": "B_dropout",      "dropout": 0.2, "wd": 0.0},   # 只改 dropout
    {"name": "C_weight_decay", "dropout": 0.0, "wd": 5e-4},  # 只改 wd
    {"name": "D_both",         "dropout": 0.2, "wd": 5e-4},  # 两个都改
]
```

**三个铁律**

| 铁律 | 为什么 |
|---|---|
| 必须有 A 当参照物 | 没有 A，"98.1% 算好吗"这个问题答不了 |
| B/C 各只改一个变量 | 只跑 A 和 D，就算变好也不知道是谁的功劳 |
| 四组同一个 seed | 否则差异里混进了"这次运气好不好" |

**为什么用 `subprocess` 而不是循环调用**

PyTorch 的随机状态、CUDA 上下文在同一个进程里会**残留**。
第二组用的"随机初始化"其实已经被第一组影响过了，四组就不独立。
开新进程 = 全新随机状态。代价是多花几秒 import，值得。

**跑法**

```powershell
& "E:\anaconda\envs\pytorch\python.exe" scripts\run_ablation.py --epochs 10
& "E:\anaconda\envs\pytorch\python.exe" scripts\run_ablation.py --only B_dropout   # 只跑一组，调试用
```

**产出**

```
outputs/<时间戳>_ablation/
    summary.md     四组并排的 markdown 表（可直接贴到周报）
    compare.png    左 val_loss、右 gap，四组各一条线
```

**一个设计细节**：每跑完一组就立刻 `write_table()` 落盘一次。
后面某组崩了，前面的结果不会白跑。

---

# 第二部分 · 过拟合与正则化

## 1. 一句话定义

> **过拟合 = 模型把训练集里的噪声当成了规律。**
> 症状：训练集上表现极好，新数据上不行。

数字版定义：

```
参数 203530，训练样本 54000 → 平均每张图"分到" 3.77 个参数
训练 loss 掉到 0.0096（几乎为零）→ 它不是在"学会规律"，是在"记住这 54000 张图"
```

---

## 2. 判据：看什么数字

| 指标 | 怎么读 | 陷阱 |
|---|---|---|
| `gap = train_acc − val_acc` | 持续扩大 = 警报 | **只是警报，不是判决书** |
| **`val_loss` 绝对值** | **有没有开始往上走 = 判决书** | 这才是唯一的最终判据 |
| `train_loss / val_loss` 倍数 | 8 倍以上要警觉 | 我们的实测是 8.05 倍 |

**核心判断原则**

```
gap 扩大 + val_loss 还在降  →  良性过拟合，别治
gap 扩大 + val_loss 开始升  →  恶性过拟合，该治了
```

**三个必须知道的读数陷阱**

1. **第 1 轮的 gap 一定是负数**，别看。
   `train_acc` 是"整个 epoch 的平均"（从 10% 爬到 95%），
   `val_acc` 是"epoch 结束那一刻"（95%）。**时间口径不同，不能直接比。**
2. **val_acc 的第 4 位小数是假的。** 6000 条验证集，1 条 = **0.0167%**。
   `0.9792`（5875 条）和 `0.9793`（5876 条）差的只是 **1 条样本**。
3. **gap 小 ≠ 模型好。** 一个瞎猜的模型 gap 恒为 0。
   判断好坏永远看 `val_loss` / `val_acc` 的绝对值。

---

## 3. 三个武器

### 3.1 Dropout

**运算**（训练时，对每个神经元输出 `x`）

```
以概率 p     → 0
以概率 1-p   → x / (1-p)
```

**为什么必须除以 `(1-p)`**

```
x=10, p=0.5
不除：E[输出] = 0.5×0 + 0.5×10    = 5     ← 期望值砍半，训练/推理不一致
除以：E[输出] = 0.5×0 + 0.5×(10/0.5) = 10  ← 期望值不变 ✅
```

保持期望值不变，所以**推理时什么都不用做**（叫 inverted dropout）。

**为什么有效**：每步都在一个不同的子网络（`2^512` 种可能）上训练，
没有任何一个神经元可以"垄断"某个特征，模型被迫学更冗余、更鲁棒的特征。

**三个坑**

| 坑 | 后果 |
|---|---|
| 忘了 `model.eval()` | 推理还在丢人 → 准确率掉，**且每次掉得不一样**，难查 |
| 开 dropout 后 `train_loss > val_loss` | 正常！训练时任务变难、推理时满血。**gap 变负是好事** |
| `dropout=1.0` | 输出恒为 0，准确率 10%，**不报错** → 必须入口校验 |

**今天实测**：`dropout=0.2` → `train_acc` 0.9979 → 0.9893（训练确实变难了），
但 `test acc` 0.9800 → 0.9775（**掉 0.25 pt**，对我们这个容量过剩程度来说太重了）。

### 3.2 Weight Decay（L2 正则）

**运算**

```
L_total = CE(logits, y) + (λ/2)·Σw²
∂L/∂w   = ∂CE/∂w + λw
w      ← w − lr·(∂CE/∂w + λw) = w − lr·∂CE/∂w − lr·λ·w
                                        └────────┘
                                   每步额外缩小 lr·λ 倍
```

**算一遍**（`lr=0.1, λ=5e-4`）

```
每步系数 = 1 − 0.1×5e-4 = 0.99995
422 步/epoch → 0.99995^422 ≈ 0.9791
10 轮        → 0.9791^10  ≈ 0.809   ← 光靠衰减压到 81%
```

实测 `wnorm` 28.851 → 16.845，**降 41.6%**，比上面算的 81% 更狠。
主因是 **`momentum=0.9` 把 `λw` 这一项也一起累积了** ——
动量对梯度的等效放大是 `1/(1-0.9) = 10 倍`，所以 wd 的实际力度被放大了一个量级。

> **这是个值得记住的交互**：`weight_decay` 和 `momentum` **会互相放大**。
> 你调 wd 的时候如果同时开着 momentum，实际效果不是 yaml 里那个数字。

**为什么有效**：`w` 越小，输入扰动引起的输出变化越小 = 函数越平滑 = 越不容易拟合噪声。

**三个坑**

| 坑 | 说明 |
|---|---|
| SGD 下 = L2，**Adam 下不等价** | Adam 把 `λw` 也除以二阶矩，大权重的衰减被压小。**AdamW 才是解耦的** |
| λ 太大会欠拟合 | 权重压过头 → 记不住东西 → train/val 双降。**这时 gap 很小，但不是好事** |
| 剂量窗口很窄 | 今天的实测就是证据（见下） |

**今天的剂量曲线（倒 U 型）**

| wd | test acc | best val_loss | wnorm |
|---|---|---|---|
| 0 | 0.9800 | 0.0773 | 28.851 |
| 5e-5 | 0.9804 | 0.0766 | 26.127 |
| **1e-4** | **0.9812** | **0.0733** | 23.998 |
| 5e-4 | 0.9782 | 0.0904 | 16.845 |

峰在 1e-4（`val_loss` 降 **5.2%**），5e-4 就过量了。
但 `0.9812 − 0.9800 = 0.12 pt`，而测试集标准误 `SE ≈ 0.14 pt` → **不显著**。

### 3.3 Early Stopping（早停）

**它做什么**：验证指标连续 `patience` 轮没改善就停。

**判据怎么选 —— 今天的核心发现**

同一份数据、同 seed、同 `patience=3`：

| 判据 | 停在哪 | 判据的理由 | test acc |
|---|---|---|---|
| `acc` | ep9 | ep6 之后 acc 再没超过 0.9792 | 0.9782 |
| `loss` | 跑满 10 轮 | ep9/ep10 的 loss 还在创新低 | **0.9800** |

**结论相反。** acc 说"第 6 轮后没进步了"，loss 说"一直在进步"。

**谁对**：loss。因为 acc 分辨率太粗（1 条 = 0.0167%），
`0.9782 / 0.9785 / 0.9790` 之间的差距本来就只有几条样本，是噪声。

**但最终交付看 acc** —— 只是**别拿它当早停判据**。

**三个坑**

| 坑 | 说明 |
|---|---|
| 去掉 `1e-6` 容差 | 浮点抖动被当"有进步" → 计数器永远归零 → **早停永不触发** |
| 停了不回滚权重 | 触发那一轮权重已经变差，必须 `load_state_dict(best.pt)` |
| `improved` 在更新 `best` **之后**算 | 恒为 `False` → 第 2 轮就停。**不报错，只是行为错** |

**早停什么时候值得用**

```
训练 100 轮、第 30 轮开始过拟合  →  省 70 轮，非常值
训练 10 轮、第 9 轮才刚饱和      →  省 1 轮，毫无意义（今天就是这种情况）
```

**早停的价值和"过拟合出现的早晚"成正比。**

**附带发现（很值钱）**：验证集上只差 **1 条**样本的两版权重，
在测试集上差了 **18 条**。因为你在 6000 条上挑"最好那版"，
挑出来的很大成分是"恰好在这 6000 条上顺手" ——
这叫**对验证集过拟合**（选择偏差）。所以测试集**只能用一次**。

---

## 4. 一句话总纲

> **正则化是刹车，不是方向盘。**
> 它只能让模型变笨，不能让模型变聪明。
> 所以：只在"模型确实在记噪声"时踩，而且必须知道**该踩多深**。
> 踩过头（wd=5e-4）比不踩（wd=0）还糟。

**"要不要用"是个错问题，"用多少"才是真问题。**

---

## 5. 六个静默失败清单（今天全部踩过或验证过）

| # | 写法 | 现象 | 为什么不报错 |
|---|---|---|---|
| 1 | `dropout=1.0` | 准确率恒 10% | 数学上合法，只是没信息 |
| 2 | 推理时忘 `model.eval()` | acc 抖动下降 | 每次丢的人不一样 |
| 3 | 早停 `improved` 算在 `best` 更新之后 | 第 2 轮就停 | 布尔值恒 False |
| 4 | 早停去掉 `1e-6` 容差 | 永不触发 | 浮点抖动被当成改善 |
| 5 | `data_dir: "data"` 相对路径 | 换目录跑就多下一份数据 | 相对路径总能解析成功 |
| 6 | 只看 gap 不看 val_loss | 结论完全搞反 | 两个指标都在动 |

**共同点：这些错误没有一个会抛异常。**
所以「跑通了」和「跑对了」是两件事 —— Day 21 的埋 bug 测试考的就是这个。

---

## 6. 命令速查

```powershell
# 环境（永远用绝对路径，不要敲裸 python）
$PY = "E:\anaconda\envs\pytorch\python.exe"

# 快速试一组（约 20 秒）
& $PY scripts\train_mnist.py --epochs 2 --tag smoke --dropout 0.2

# 四组对照（约 6 分钟）
& $PY scripts\run_ablation.py --epochs 10

# 扫 wd 剂量
& $PY scripts\train_mnist.py --weight-decay 1e-4 --tag wd1e4

# 早停两种判据对比
& $PY scripts\train_mnist.py --patience 3 --es-metric acc  --tag es_acc
& $PY scripts\train_mnist.py --patience 3 --es-metric loss --tag es_loss

# 环境自检（动过环境后必跑）
# ⚠️ 这个脚本不在项目仓库里，要在【工作区根】敲：E:\work Buddy\强化学习
& $PY "E:\work Buddy\强化学习\00-setup\check_env.py"
```

> 上面除了最后一条，其余命令都在**项目根目录** `E:\work Buddy\强化学习\rl-vla-journey` 下敲。
> 输出落在 `outputs\<时间戳>_mnist_<tag>\`。
> **多跑几次一定用不同 `--tag`**，否则分不清哪个目录是哪次。
>
> 记不清在哪敲的时候，先看路径：`scripts\...` 就是项目根，
> `00-setup\...` 就是工作区根。
