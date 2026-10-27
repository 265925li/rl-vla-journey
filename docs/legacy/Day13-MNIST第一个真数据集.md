# Day 13 · MNIST：第一个真数据集

> 今天是 P0 里第一次用真数据。前 12 天所有数据都是自己造的（blobs、随机张量），
> 从今天起，错误不再是"我公式推错了"，而是"某个环节悄悄不对"。
>
> **今天只追求跑通，不追求成绩。** 成绩是 Day 15 的事。
>
> 配套代码（都是写好的完整版，带逐行注释）：
> - `src/utils/seed.py` — 四套随机数怎么固定
> - `src/data/mnist.py` — MNIST → batch 只做三件事
> - `scripts/vis_mnist.py` — 先看图，再训练
> - `scripts/train_mnist.py` — 主训练脚本（`--config` 入口）
> - `configs/mnist_mlp.yaml` — 这次实验的全部参数

---

## §1 今天要拿到什么

一条命令，四个产物：

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
& "E:\anaconda\envs\pytorch\python.exe" scripts\train_mnist.py
```

```
outputs/2026-10-22_170822_mnist_run/
    config.yaml     这次跑用了什么参数
    train.log       完整日志
    history.json    每个 epoch 的 loss/acc（复现性对比的就是它）
    curve.png       两条曲线（train / val 画在一起）
    best.pt         验证集最好的那一版权重
```

实测结果（本机 1660 Ti，10 个 epoch）：

| 项目 | 数值 |
|---|---|
| 数据 | train 54000 / val 6000 / test 10000 |
| 每 epoch 步数 | `54000 / 128 = 421.875` → **422 步** |
| 参数量 | **203530**（784→256→10） |
| 单 epoch 耗时 | **8.5–9.9 秒** |
| 总耗时 | **94.5 秒** |
| best val acc | **0.9793** |
| test acc | **0.9800** |

十轮完整曲线：

| epoch | train loss | train acc | val loss | val acc |
|---|---|---|---|---|
| 1 | 0.2997 | 0.9137 | 0.1489 | 0.9553 |
| 2 | 0.1076 | 0.9677 | 0.1105 | 0.9672 |
| 3 | 0.0757 | 0.9766 | 0.0955 | 0.9712 |
| 4 | 0.0532 | 0.9843 | 0.0840 | 0.9748 |
| 5 | 0.0412 | 0.9874 | 0.0904 | 0.9727 |
| 6 | 0.0304 | 0.9907 | 0.0789 | **0.9792** |
| 7 | 0.0223 | 0.9937 | 0.0792 | 0.9782 |
| 8 | 0.0179 | 0.9949 | 0.0824 | 0.9785 |
| 9 | 0.0131 | 0.9969 | 0.0787 | 0.9790 |
| 10 | 0.0096 | 0.9979 | 0.0773 | **0.9793** |

这张表里有三处值得停下来看，§6 逐个解释。

---

## §2 一条命令的完整路径

```
命令行参数  ──┐
              ├─→ Config 对象 ──→ 数据 → 模型 → 优化器 → 循环 → 产物
yaml 文件   ──┘
```

具体到代码，`main()` 的顺序是**有讲究的**，不能随手调换：

```
① 读配置（yaml + 命令行覆盖）      ← 参数错了在这里就炸，别等训练到一半
② 定设备（cuda 不可用就降级 cpu）   ← 别在别的机器上一上来就崩
③ 建输出目录 + 日志                ← 目录建不出来要马上知道
④ 设随机种子                       ← 必须在建模型之前！
⑤ 建数据集 / DataLoader
⑥ 建模型 + 优化器 + 调度器
⑦ 循环训练 → 存 checkpoint → 画图
⑧ 最后才看测试集（只一次）
```

**④ 为什么必须在 ⑥ 之前**：模型的初始权重是随机抽出来的。种子设晚了，
模型已经用"上一个随机流"初始化完了，你设的种子只影响后面的数据顺序。

**⑧ 为什么最后才看测试集**：见 §3.2。

---

## §3 数据：MNIST 到底是什么形状

### 3.1 一张图经过 transform 变成什么

```
磁盘：idx-ubyte（二进制）
  ↓ datasets.MNIST(...)
PIL 图像  mode=L  尺寸 28×28  取值 0~255 的整数
  ↓ transforms.ToTensor()
张量      (1, 28, 28)  float32   取值 0.0 ~ 1.0
```

`ToTensor()` 干了两件事，两件都得知道：

| 变化 | 从 | 到 |
|---|---|---|
| 加一维 | `(28, 28)` | `(1, 28, 28)` |
| 除以 255 | `0~255` uint8 | `0~1` float32 |

多出来的第 0 维是**通道**。PyTorch 的图像张量约定是 `(C, H, W)`，灰度图
只有一个通道，所以 `C=1`。等 DataLoader 把它们拼成一批，就是 `(B, 1, 28, 28)`。

实测确认（`scripts/vis_mnist.py` 的输出）：

```
单条样本：x.shape=(1, 28, 28)  dtype=torch.float32  取值 [0.00, 1.00]  y=0
```

**这一步一定要亲眼看过。** 如果 `取值 [0, 255]` 而 dtype 是 `uint8`，
说明 transform 没生效 —— 模型仍然能训，只是准确率会莫名其妙低一截，
而且**不报任何错**。

### 3.2 为什么必须切验证集

很多人图省事，直接拿 test 当 val。后果不是报错，而是**你的判断失效**：

> 你按测试集的表现调超参数，调到 98%，然后报"测试准确率 98%"。
> 这个数字已经被你看过几十次，它不再代表"没见过的数据"。
> 真正上线时表现会明显低于预期。这叫**数据泄漏**（leakage）。

正确做法（`src/data/mnist.py` 里就是这么写的）：

```
MNIST train 60000  ──┬─→ train 54000   （训练用）
                      └─→ val    6000   （调参用）
MNIST test  10000  ────→ test         （只在最后看一次）
```

**切的时候要打乱**，不能直接取前 6000 条：

```python
g = make_generator(cfg.seed)
perm = torch.randperm(n, generator=g)
val_idx = perm[: cfg.val_size].tolist()
train_idx = perm[cfg.val_size :].tolist()
```

MNIST 训练集是按标签大致排过序的。直接切前 6000 条，你会得到一堆 0 和 1，
验证集根本代表不了整体 —— 而这个错误同样**不会报错**。

### 3.3 一条 → 一批

```
Dataset 里的一条      (1, 28, 28)          ← 一张图
        ↓ DataLoader 拼 128 条
一个 batch            (128, 1, 28, 28)     ← 128 张图
        ↓ xb.view(xb.size(0), -1)   ← 摊平（§4）
                      (128, 784)           ← 128 行，每行 784 个像素
```

实测打印（训练脚本只在第 1 个 batch 打一次）：

```
形状检查  xb=(128, 1, 28, 28)  yb=(128,) (torch.int64)
```

两个细节：

- **`yb` 是 `(128,)` 而不是 `(128, 1)`**。`CrossEntropyLoss` 要的是
  「每条样本一个类别下标」，形状必须 `(B,)`。写成 `(128,1)` 会报
  `Expected target size ...`。
- **`yb.dtype` 是 `torch.int64`**。CE 的 target 必须是整数类型，
  `float32` 会报 `expected scalar type Long`。你在 Dataset 里拿到的是
  Python `int`，是 DataLoader 的 collate 自动帮你转成了 int64。

---

## §4 把 784 个像素喂给 MLP

MLP 不认识"图像"，它只认识"一行数字"。所以每批数据进模型之前要摊平：

```python
xb = xb.view(xb.size(0), -1)    # (128, 1, 28, 28) → (128, 784)
```

`view` 的三个位置：

| 参数 | 含义 | 为什么这么写 |
|---|---|---|
| `xb.size(0)` | 第 0 维保持 128 | **批大小不能变**，这是"几条样本" |
| `-1` | 让 PyTorch 自己算 | `1×28×28 = 784`，写死 784 也行，但换数据集时要改 |
| — | 结果是 `(128, 784)` | 每行是一条样本的全部像素 |

**写成 `xb.view(-1)` 会怎样**：整个 batch 塌成一维 `(128×784,)` = `(100352,)`，
再 `.unsqueeze(0)` 变成 `(1, 100352)`，模型会把 10 万个数字当成**一条**样本，
报 `mat1 and mat2 shapes cannot be multiplied (1x100352 and 784x256)`。

### 参数量：手算和程序必须对上

```
784 × 256 + 256  = 200960     第一层（784→256）+ 偏置
256 × 10  + 10   =   2570     输出层（256→10）+ 偏置
                   ───────
                   203530
```

训练脚本里做了这个对拍：

```python
actual = sum(p.numel() for p in model.parameters())
manual = count_params_manual(IN_DIM, cfg.hidden, NUM_CLASSES, cfg.n_layers)
assert actual == manual, f"参数量对不上：运行时 {actual} vs 手算 {manual}"
```

实测 `203530`，两边一致。

**为什么要断言**：`MLP(in_dim=784, ...)` 里如果有人把 `n_layers` 理解错
（比如以为 2 层是"2 个隐藏层"），实际结构就变了。参数量是结构的指纹，
对不上就说明你脑中的模型和真正的模型不是一个东西 —— 比看 loss 曲线更早发现问题。

---

## §5 训练循环：和 Day 11 一模一样

`train_one_epoch()` 的循环体就是那五行，一个字没变：

```python
opt.zero_grad()          # ① 擦掉上一步的梯度
logits = model(xb)       # ② 前向：(128,784) → (128,10)
loss = crit(logits, yb)  # ③ 打分：一个标量 ()
loss.backward()          # ④ 追责：每个参数算出梯度
opt.step()               # ⑤ 微调：按梯度挪一步
```

今天新增的只有三样，都是为了"看得见"：

**① 第一次打印形状**（`step == 0` 时打一次，不是每步都打）

```python
if step == 0 and log is not None:
    log.info(f"形状检查  xb={...} yb={...} ({yb.dtype})")
```

**② 按样本数加权累加**

```python
bs = xb.size(0)
total_loss += loss.item() * bs          # 不是 += loss.item()
total_correct += (logits.argmax(dim=1) == yb).sum().item()
n += bs
```

Day 11 算过：不加权时，末批占比越大错得越狠（那里实测过 0.04% 到 64%）。

**③ 只在验证集创新高时存盘**

```python
if va_acc > best_acc:
    best_acc = va_acc
    torch.save(model.state_dict(), best_path)
```

存"最好那一版"而不是"最后一版"。看上表 epoch 5：val acc 从 0.9748 掉到
0.9727，最后又爬回 0.9793。如果只看最后一版，你就丢掉了中间可能更好的权重。

---

## §6 读数字：三个最容易看错的地方

### 6.1 第一个 epoch：train acc 0.9137 < val acc 0.9553

第一反应通常是"验证集比训练集还高？是不是数据搞反了？"

没搞反。**train acc 是这一整个 epoch 的流水账平均值，包含了开头还很差的那些 batch。** 实测（`_probe_d13.py`）：

| 统计方式 | 数值 |
|---|---|
| 训练过程中累加出来的 train acc | **0.9137** |
| 前 20 个 batch 的平均 | **0.6172** |
| 后 20 个 batch 的平均 | **0.9592** |
| epoch 结束后用最终权重重算 train acc | **0.9636** |
| 同一时刻的 val acc | 0.9553 |

看第三、四行：**模型在 epoch 结束时其实已经到 0.9636 了**，比 val 的 0.9553 还高
（正常现象，因为 val 是没见过的数据）。

那个 0.9137 只是"把 422 个 batch 的历史全部平均"的结果，开头那批只有 0.6172。
**它描述的是"这一轮的过程"，不是"这一轮结束时的水平"。**

所以比较 train / val 时，正确做法是 epoch 结束后**再跑一遍训练集**。
训练脚本里没这么做（省时间），但你知道这个偏差存在就够了。

### 6.2 最后一轮：train 0.9979 vs val 0.9793

差的这 1.86 个点叫**过拟合**，不是 bug：

```
train loss  0.0096   ← 训练集上几乎全对
val   loss  0.0773   ← 换了没见过的人写数字，错得更多
```

模型有 203530 个参数，MNIST 训练集只有 54000 张图。参数量是样本量的 3.8 倍，
"背下来"是完全做得到的。这也是 Day 15 要处理的问题（加正则、加数据增强、
或者换更强的结构）。

**判断过拟合的信号不是"val acc 低"，而是「train 和 val 的差距在扩大」**：

| epoch | train acc | val acc | 差距 |
|---|---|---|---|
| 1 | 0.9137 | 0.9553 | −4.16（反向，见 6.1） |
| 3 | 0.9766 | 0.9712 | +0.54 |
| 6 | 0.9907 | 0.9792 | +1.15 |
| 10 | 0.9979 | 0.9793 | **+1.86** |

### 6.3 best val 不等于 last val

epoch 5 的 val acc 掉到 0.9727，比 epoch 4 的 0.9748 还低；epoch 10 又爬到 0.9793。
验证曲线**不是单调上升的**，它会抖。

所以：

- **不要看最后一行就下结论**，至少看 3 轮的趋势
- **不要用一次运行的 val acc 差异（比如 0.9748 vs 0.9727）判断"改动有没有效"**，
  这个量级的差别很可能只是随机波动
- 这也是为什么 §7 的复现性重要：**波动和改动的区别，只能靠"同配置跑两次"来定**

---

## §7 复现性：四套随机数 + 一个最常被漏的地方

### 7.1 随机性有四个来源

| 来源 | 用什么设 | 影响什么 |
|---|---|---|
| Python `random` | `random.seed(s)` | `random.shuffle`、`random.random` |
| numpy | `np.random.seed(s)` | 数据增强、噪声采样 |
| PyTorch CPU/GPU | `torch.manual_seed(s)` | **模型初始化**、dropout、DataLoader 的 shuffle |
| cuDNN 算法选择 | `cudnn.deterministic=True` | 同一份输入的浮点加法顺序 |

**只设 torch 而不管 numpy** 是最常见的漏法：某天你加了个用 numpy 抽样的
数据增强，复现性就悄悄断了，而且不会有任何提示。

`src/utils/seed.py` 里的 `set_seed()` 把四个都固定了。此外还关掉了
`cudnn.benchmark=True` —— 它会"试几种算法挑最快的"，这个挑选过程本身
带随机性（同一份输入，第一次跑用算法 A，第二次可能用 B，末位数字就变了）。

### 7.2 最常被漏的地方：DataLoader 的 generator

这是实测数据（`_probe_d13.py` 第一段，连续建两个 DataLoader，取第一个 batch 的标签）：

```
不打乱、无 generator    A=[0, 1, 2, 3, 4]        B=[0, 1, 2, 3, 4]        → 相同
打乱、  无 generator    A=[14, 9, 16, 17, 8]     B=[8, 14, 4, 12, 2]      → 不同
打乱、  有 generator(0) A=[13, 1, 9, 3, 7]       B=[13, 1, 9, 3, 7]       → 相同
```

关键在第二行：**在同一个进程里连续建两个 DataLoader，打乱顺序居然不一样。**
不传 `generator` 时 DataLoader 用的是**全局**随机流 —— 全局流每被取一次就往前走
一步，第二个 DataLoader 取的自然是后面的数。

等你把这段代码搬到训练脚本里，就变成：

> 今天跑一次，明天跑一次，每次开头 `set_seed(0)` 都调了，
> 但两次结果就是不完全一样。

原因往往是"两次运行的随机流消耗顺序不同"：你在中间加了一个 `print` 里调用了
`torch.rand`、多加了一层 dropout、或者调整了模型初始化的顺序，全局流的位置就变了，
后面所有数据顺序跟着错位。

解法就是给 DataLoader **自己的一条流**：

```python
generator=make_generator(cfg.seed)
```

### 7.3 验证方法：diff 两个 history.json

```powershell
& "E:\anaconda\envs\pytorch\python.exe" scripts\train_mnist.py --tag repro_a
& "E:\anaconda\envs\pytorch\python.exe" scripts\train_mnist.py --tag repro_b
```

然后对比两个 `history.json`。

<!-- REPRO_RESULT -->
实测（10 epoch 跑两次，`--tag repro_a` / `--tag repro_b`）：

```
c05d7cd285df346cf4d7d2b37223ff54  ..._repro_a/history.json
c05d7cd285df346cf4d7d2b37223ff54  ..._repro_b/history.json
→ diff 无任何输出，逐字节相同
```

十轮 40 个指标（train/val 的 loss 与 acc）+ 测试结果全部一致：

```
两次都是  epoch 10/10  train loss=0.0096 acc=0.9979 | val loss=0.0773 acc=0.9793
两次都是  test loss=0.0656  test acc=0.9800
```

**但这里有一个我踩到的设计问题，值得单独说。**

我第一版把每轮耗时也写进了 `history.json`：

```python
rec = {..., "sec": round(time.perf_counter() - t_ep, 2)}   # ← 错
```

于是两次跑出来 diff 是这样的：

```
<   "sec": 8.53
>   "sec": 8.41
---
<   "sec": 9.88
>   "sec": 9.87
```

40 个指标全对，但文件永远"有差异"。**如果我只看到"diff 不为空"就下结论说
复现性坏了，那就是被自己的记录方式骗了。** 反过来更危险：有人会习惯性忽略
这些差异行，等到真有指标不一致时也照样忽略过去。

修法不是"diff 时记得跳过 sec"，而是**根本不要让不确定的东西进这个文件**：

> `history.json` 的职责是"证明两次跑得一样"，
> 所以它里面只能放**确定性**的内容。墙钟时间天然不确定，它属于日志。

现在每轮耗时打在 `train.log` 里，`history.json` 只留参数 + 指标 + 测试结果。
这样 diff 判据就非常干脆：**没有任何输出 = 通过。**

这也是一条通用的经验：**你用来做判断的那个文件，必须只包含"你要判断的东西"。**
混进一点无关的波动，判断就失效了。

### 想自己复现这段验证

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
& "E:\anaconda\envs\pytorch\python.exe" scripts\train_mnist.py --tag my_a
& "E:\anaconda\envs\pytorch\python.exe" scripts\train_mnist.py --tag my_b

$a = (Get-ChildItem outputs\*_mnist_my_a\history.json).FullName
$b = (Get-ChildItem outputs\*_mnist_my_b\history.json).FullName
Compare-Object (Get-Content $a) (Get-Content $b)
```

`Compare-Object` 没有输出 = 两份文件完全一样。

**然后试着把 `src/data/mnist.py` 里的 `generator=make_generator(cfg.seed)` 删掉，
再跑一次。** 你会发现指标开始不一样了 —— 这就是 §7.2 那个坑的实证。

---

## §8 配置系统：为什么参数要从 yaml 读

```powershell
& "E:\anaconda\envs\pytorch\python.exe" scripts\train_mnist.py --lr 0.05 --epochs 3
```

优先级：**命令行 > yaml > `Config` 类里的默认值**。

三种来源各有用处：

| 来源 | 用途 |
|---|---|
| `Config` 默认值 | 兜底。yaml 里没写的键自动补上，配置文件不用写全 |
| `configs/*.yaml` | 一组完整实验的"配方"。换数据集就换一个 yaml |
| 命令行 | 临时试一下。`--epochs 2` 快速验证，不用改文件 |

`main()` 里那个 `overrides` 的写法值得看一眼：

```python
overrides = {k: v for k, v in {...}.items() if v is not None}
```

**参数默认值设成 `None` 而不是 `10`**，是为了区分"用户想要 10"和"用户没说话"。
如果默认写 `10`，那么 `--epochs` 不传时也会把 yaml 里的值覆盖成 10，yaml 就失效了。

### 为什么每次都要存一份 config.yaml

```python
save_config(cfg, out / "config.yaml")
```

半年后你回来看这个 98% 的结果，日志里只有 loss 曲线。想知道"当时 lr 是多少、
hidden 是多大、用了哪个优化器"，只能靠这个文件。

**没有它，你的历史实验就只是一堆无法复现的数字。**

---

## §9 报错对照（本机实测）

| 写法 | 真实报错 | 属于 |
|---|---|---|
| 忘摊平，直接 `model(xb)` | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (3584x28 and 784x256)` | 形状 |
| `view(-1)` 把整批塌成一维 | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x100352 and 784x256)` | 形状 |
| 忘 `.to(device)` | `RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cpu ...` | 设备 |
| target 转成 `float` | `RuntimeError: expected target dtype to be Long or Byte, but got Float` | dtype |
| `load_state_dict` 结构不符 | `Error(s) in loading state_dict for MLP: Unexpected key(s) ... size mismatch for layers.0.weight` | 结构 |
| yaml 里键名拼错 | **不报错**，只打印 `⚠️ 配置里有不认识的键`，然后**用默认值跑** | 静默 |
| 忘 `x/255`（transform 没生效） | **不报错**，准确率低一截 | 静默 |
| 拿 test 当 val | **不报错**，你还以为自己很准 | 静默 |
| val loader 也 `shuffle=True` | **不报错**，数值不受影响但失去可复现性 | 静默 |

第一条值得多看两眼：`(3584x28)` 是怎么来的？

`nn.Linear(784, 256)` 认的是「最后一维 = 784」。你喂进去 `(128, 1, 28, 28)`，
它取最后一维 28 去对 784 —— 对不上，于是把前面几维先压平再试：
`128 × 1 × 28 = 3584`，凑成 `(3584, 28)`。

**报错里的数字不是它"以为"的形状，而是它"试图凑出来"的形状。**
看懂这一点，你就能从错误信息反推自己漏了哪一步

**规律和老话一样：报错的都是形状/设备/dtype 这类"机械错误"；
真正毁掉实验的是下面那四个不报错的。**

---

## §10 自问

1. 一条 MNIST 样本从磁盘到进入模型，形状依次是什么？（答：`(28,28)` → `(1,28,28)` → `(128,1,28,28)` → `(128,784)`）
2. 为什么不能拿 test 当 val？用一句话说清"防的是什么"。
3. `train acc 0.9137 < val acc 0.9553` 是 bug 吗？如果不是，那个 0.9137 到底在描述什么？
4. 参数量 203530 是怎么算出来的？如果 `hidden` 从 256 改成 512，变成多少？
5. `set_seed()` 固定了四套随机数里的三套，第四套（cuDNN）为什么要单独关一个开关？
6. 你只设了 `set_seed(0)`，两次跑结果还是不一样。最先怀疑哪里？
7. 为什么 `--epochs` 的默认值是 `None` 而不是 `10`？
8. 为什么存 `best.pt` 而不是最后一个 epoch 的权重？
