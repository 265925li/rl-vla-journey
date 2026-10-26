# Day 18 · 模块化重构与通用训练模板

> **P0 的最后一个硬指标**：写出通用 `train.py --config config.yaml` 模板。
> 但「通用」不是「能传配置文件」——那样连 Day 3 的 `train.py` 都算通用。
> 今天要说清楚它的真正含义，并且用**逐字节对拍**证明重构没改变任何行为。

---

## 1. 为什么今天必须拆

Day 13 的 `train_mnist.py` 是 546 行，什么都往里塞：参数表、造模型、
训练循环、画图、写 json。**当时那样写是对的** —— 先跑通，再整理。

但到了「要加第二个模型」的时候，它的代价立刻显形。三处硬编码：

```python
def build_model(cfg):
    return MLP(in_dim=784, hidden=cfg.hidden, out_dim=10, ...)
    #           ↑ 写死 784          ↑ 写死 10 类
    # 想加 CNN？→ 改这个函数

# 训练循环里：
xb = xb.view(xb.size(0), -1)
    # ↑ 写死要摊平。CNN 恰恰不能摊平，要保留 28×28 的空间结构
    # 想加 CNN？→ 改训练循环（这个函数被所有实验共用，风险最高）

# 参数表里：
ap.add_argument("--dropout", ...)      # 每加一个配置字段，就得回来加一行
ap.add_argument("--weight-decay", ...)
```

**三处都是「改一片」而不是「改一处」**，而且第二处要动的是
被所有实验共用的训练循环。这就是今天重构的理由，跟"代码好看"没关系。

---

## 2. 拆成了什么

```
src/engine/
├── registry.py    112 行   名字 → 构建函数的表（零项目内依赖）
├── builtin.py      98 行   注册 mnist 和 mlp —— 唯一知道具体实现的文件
├── builders.py     96 行   优化器 / 损失 / 调度器 / 指标
├── loops.py       112 行   训练循环、评估循环
├── trainer.py     313 行   Trainer 类，把上面串成一条流水线
└── cli.py         184 行   参数解析 + main()
src/utils/plotting.py  70 行   画曲线（不参与训练，所以放 utils）

scripts/train.py         51 行  ┐
scripts/train_mnist.py   40 行  ┘ 两个入口，**共用同一个 main()**
```

**注意总行数：546 变成 1008+91 = 1099 行，涨了一倍。**

这是重构的常态，不是失败。多出来的绝大部分是注释和分层说明。
重构的价值不在省行数，在于**把「改一处牵动全身」变成「改一处只动一处」**。
如果有人告诉你重构之后代码变短了，先问一句：功能是不是也变少了。

### 分层的硬约束

```
cli.py  →  trainer.py  →  registry.py
                ↓              ↑
           builtin.py ─────────┘
                ↓
        models/ · data/ · utils/
```

一条必须守住：**`trainer.py` 不许认识任何具体模型。**

```python
# trainer.py 里只有这句
self.model = build_model(cfg.model, cfg)
# 它不知道 mlp 是什么、cnn 是什么、MNIST 有几类
```

为什么守这条：P1 加 CNN 时，只要 `trainer.py` 一行没改，
就能确定训练流程（早停、存盘、记录格式）的行为完全没变。
**这是"改动面可控"的具体形式**，不是架构洁癖。

---

## 3. 「通用」的实质：注册表

这是今天最值钱的一节。先看两种写法：

**❌ 看起来通用，其实不通用**

```python
ap.add_argument("--config", ...)          # 配置文件路径可传
ap.add_argument("--dropout", ...)         # 但字段要一个个加
ap.add_argument("--weight-decay", ...)    # 新字段 → 回来加一行
...
def build_model(cfg):
    if cfg.model == "mlp": return MLP(...)
    elif cfg.model == "cnn": return CNN(...)   # 新模型 → 回来改这里
```

**✅ 真的通用**

```python
--set key=value      # 任意字段零改动可覆盖
MODELS[name](cfg)    # 任意模型零改动可接入
```

判断标准就一条：**换实验的时候，cli.py 和 trainer.py 要不要改？**

### 注册表长什么样

```python
# src/engine/builtin.py —— 唯一知道具体实现的文件
@register_model("mlp")
def build_mlp(cfg) -> nn.Module:
    model = MLP(...)
    # 参数量对账（见 §6.2）
    return model
```

P1 加 CNN 时：

```python
@register_model("cnn")
def build_cnn(cfg) -> nn.Module:
    return CNN(channels=cfg.channels, ...)
```

然后 `configs/cnn_mnist.yaml` 里写 `model: cnn`，**主程序一个字不用改。**

### 一个必须挡掉的坑：重复注册

```python
if name in MODELS:
    raise ValueError(f"模型 {name!r} 已经被注册过了")
```

为什么不能静默覆盖：两个实验都想用 `"cnn"` 这个名字，后 import 的赢，
你**会以为自己跑的是第一个**。这种错不报、不崩、日志也看不出，
是最难查的一类。

### 报错信息要带「有哪些可选」

```python
raise KeyError(f"未知模型 {name!r}。已注册的有：{sorted(MODELS)}。")
# 而不是 KeyError: 'cnn'
```

后半句省掉的是「回头 grep 才知道正确拼法」那一步。

---

## 4. `--set` 兜底：新字段零改动

参数表里只保留高频快键，其余全靠 `--set`：

```powershell
& $PY scripts\train.py --set dropout=0.2 --set hidden=512
```

实现只有 15 行，但它是**「通用模板」和「把参数搬个家」之间唯一实质的区别**。

两个细节：

**① 值用 `yaml.safe_load` 解析，不手写类型猜测**

```python
"512"  → int 512        "0.2"  → float 0.2
"true" → bool True      "adam" → str "adam"
```

手写「先试 int、再试 float、再试 bool」有一堆边界情况
（`"1e-4"` 是 float 还是 str？`"None"` 呢？）。yaml 已经定义好了，直接复用。

**② 必须 `split("=", 1)` 而不是 `split("=")`**

```python
"--set output_dir=a=b".split("=")      # 3 段 → 报错或截断
"--set output_dir=a=b".split("=", 1)   # 2 段 → key="output_dir", value="a=b"
```

---

## 5. 怎么证明「重构没改变行为」

这是今天最重要的一节。**重构最大的风险不是写错，是"写错了但看起来正常"。**

### 5.1 验收标准要先定，不能事后凑

我事先就定死：**同 seed 跑同一条命令，`history.json` 里的四个确定性字段
必须逐字节相同。**

| 字段 | 为什么算它 |
|---|---|
| `history`（每轮 8 个指标） | 训练轨迹。任何一个数值变化都会暴露 |
| `test` | 最终结果 |
| `best_val_acc` | 存盘决策的依据 |
| `final_wnorm` | 权重演化。能抓到"梯度算得不一样"这种深层差异 |

`config` 段**故意不比** —— 今天新增了 4 个字段，它必然不同。
把"预期会变的"和"预期不变的"分开，才叫验收标准；否则差异出来你分不清是 bug 还是预期。

### 5.2 实测结果

```
  OK    history
  OK    test
  OK    best_val_acc
  OK    final_wnorm

确定性字段完全一致： True

config 段新增的字段（预期会不同）：
  + dataset                'mnist'
  + model                  'mlp'
  + flatten                True
  + early_stop_metric      'loss'
```

### 5.3 第二个证据：早停路径

`es_metric` 今天从命令行挪进了 config，那条分支动过了，必须单独验：

```
epoch  9/10  train loss=0.0131 acc=0.9969  |  val loss=0.0787 acc=0.9790
  [早停] 按 acc 判据已连续 3 轮没改善（patience=3）
[早停] 实际跑了 9/10 轮，省下 1 轮 ≈ 9.3s
best val acc = 0.9792
test acc = 0.9782
```

对照 Day 15 台账的记录：「acc 判据 9 轮停，`0.9782`」——**一字不差。**

> 为什么这条比第一条更重要：第一条验的是"主干没坏"，
> 第二条验的是"**今天动过的那一条**没坏"。动过哪里就验哪里，
> 否则很容易出现"主干全对，但新加的分支是错的"。

### 5.4 一个「如果没做这步会怎样」的反面案例

假设重构时把 `xb.view(xb.size(0), -1)` 换成了 `xb.flatten(1)`。
这两个**结果完全相同**，测试会全过。但假设换成了 `xb.reshape(-1)`：

```python
xb.reshape(-1)   # ✗ (B, 1, 28, 28) → (B*784,)，batch 维被压掉了
```

它会立刻报形状错误 —— 这种反而是好事。真正危险的是假设把
`total_loss += loss.item() * bs` 写成 `total_loss += loss.item()`：

```python
# 末批不满时，简单平均与加权平均的差异
# 实测：val 600 条、bs=512 → 偏差 64%
```

这个错**不报、不崩，loss 曲线照样在降**，只是数字全都不对。
`history` 逐字节对比是唯一能抓到它的手段 —— 单看"训练能跑通"永远发现不了。

---

## 6. 顺手修掉的三个真问题

重构的价值不只在"搬代码"，在于搬的过程中会**被迫重新审视每一行**。
今天顺手修了三处：

### 6.1 `es_metric` 只活在命令行，没进 config

原来它是 `args.es_metric`，只存在于 argparse 里。后果很具体：

**「按 loss 早停」和「按 acc 早停」两次跑出来的 `config.yaml` 一模一样。**
半年后翻输出目录，你分不清哪个是哪个 —— 而这两个判据在 Day 15 的实测里
给出了**完全相反**的结论（9 轮 0.9782 vs 10 轮 0.9800）。

修法：加 `early_stop_metric` 字段，进 dataclass、进校验、进 config.yaml。

> 一般化的规矩：**凡是会影响训练结果的开关，必须落进 `config.yaml`。**
> 只活在命令行里的参数，等于没记录。

### 6.2 参数量对账放错了层

原来在 `main()` 里硬编码 `IN_DIM=784, NUM_CLASSES=10` 来对账。
搬进通用 Trainer 之后就不成立了 —— CNN 的参数量公式完全不同，
Trainer 不该知道任何一个具体模型的公式。

修法：对账移进 `build_mlp()` 内部。**模型自己负责自己的自检**，
Trainer 保持"不认识任何具体模型"。

这个检查抓的是「不报错但结果变差」的一类错：少加一层、`hidden` 传错、
忘了 bias、误加了参数。

### 6.3 `config.py` 里不能写死 `ALLOWED_MODELS`

诱惑很大：写个 `ALLOWED_MODELS = ("mlp", "cnn")`，模型名拼错就能在读配置时炸。
**但 P1 加 CNN 时就必须改 `config.py`** —— 而 `config.py` 是所有实验
都依赖的文件，改它风险最高。

修法：`config.py` 只查"是不是非空字符串"，名字是否存在交给 registry。

代价：「模型名拼错」的报错从「读配置时」推迟到「`Trainer.build()` 时」。
可接受，因为它仍在训练的第一秒内，不是第 1000 步；而且 registry 的报错
带可选列表，比 config 能给的更有用。

> 这是一个**权衡**，不是"哪种更对"。写下来的价值在于：
> 半年后你会问"为什么这里不校验"，答案在这儿。

---

## 7. 静默陷阱清单

| # | 陷阱 | 症状 | 怎么挡 |
|---|---|---|---|
| 1 | 注册表允许重复注册 | 以为跑 A，实际跑 B | `raise ValueError` |
| 2 | 模型名拼错，报错只有 `KeyError` | 回头 grep 才知拼法 | 报错里带 `sorted(MODELS)` |
| 3 | `--set` 用 `split("=")` | 值含 `=` 时静默截断 | `split("=", 1)` |
| 4 | `--set` 的值不解析类型 | `"512"` 变成字符串，与 int 比较时行为怪 | `yaml.safe_load` |
| 5 | argparse 参数写 `default=10` | 分不清"用户要 10"和"用户没说"，yaml 被永远覆盖 | 全部 `default=None` |
| 6 | `flatten` 写成字符串 `"False"` | 非空字符串恒为真 → 静默当 True | `isinstance(x, bool)` 校验 |
| 7 | 重构后只跑"能不能训" | 训练照跑、loss 照降，数值已错 | **逐字节对拍** |
| 8 | 两个入口各写一份参数表 | 行为漂移，`train.py` 和 `train_mnist.py` 结果不同 | 共用同一个 `main()` |

第 6 条的实测：yaml 里写 `flatten: "False"`（带引号），
`safe_load` 解析成字符串 `"False"`，`if flatten:` 判为真 → **静默当成 True**。
CNN 最需要它的时候，输入被摊平了，卷积核看到的是一维数组。

---

## 8. 验收五关（你自己跑）

**关 1 · 注册表非空**
```powershell
& $PY scripts\train.py --list
```
应看到 `mnist` 和 `mlp`。空表说明 `builtin.py` 没被 import。

**关 2 · 快照参数不训练**
```powershell
& $PY scripts\train.py --dry-run --set hidden=512
```
应在 1 秒内打印全部生效字段，`hidden` 是 512。

**关 3 · 逐字节复现**
```powershell
& $PY scripts\train_mnist.py --linear my --epochs 2 --tag verify
```
应得到 `test acc=0.9696`、`best val=0.9667`、`wnorm=19.743476`，
和重构前一致（`outputs/*_mnist_d17_my/history.json`）。

**关 4 · 早停新通道**
```powershell
& $PY scripts\train_mnist.py --patience 3 --es-metric acc --tag es
```
应在第 9 轮停、`test acc=0.9782`。

**关 5 · 旧调用方没断**
```powershell
& $PY scripts\run_ablation.py --epochs 1
```
四组都应跑起来。**这一关最容易漏** —— 重构时只想着新代码，
忘了别人还在用旧文件名。

---

## 9. 一句话带走

> **重构的验收标准是「输出一个比特都没变」，而不是「代码看起来更整齐了」。**
>
> 而要做到这一点，前提是你在重构**之前**就有了可对拍的东西 ——
> 今天能逐字节验收，靠的是 Day 13 那个决定：
> **耗时不能写进 `history.json`**。那个决定当时看不出价值。

另外记住那个反直觉的事实：**重构让代码从 546 行涨到 1099 行。**
重构的收益不在行数，在于改动面。
