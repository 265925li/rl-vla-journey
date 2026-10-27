# rl-vla-journey

从零开始的强化学习（RL）与视觉-语言-动作模型（VLA）学习仓库。

这个仓库的定位是**学习轨迹 + 作品集**：每个阶段的代码、复盘、实验结果都留在这里，
最后它会被用来向导师展示「我做过什么」，而不是「我看过什么」。

---

## 一、这个仓库怎么用

四条纪律，比任何技术细节都重要：

1. **每个算法自己手写一遍。** `stable-baselines3` 只用来做对照 baseline，不用来替代理解。
2. **一次只引入一个新概念。** 同时改三处，出错时你无法归因。
3. **曲线不好时，最后才怀疑超参数。** 先查环境 bug、数据、reward 定义。
4. **写完一个模块就提交并 push，不要攒。** 见下。

第 4 条是 `2026-10-27` 换来的：那天工作区被回滚到 10-23 的快照，
本地 git 里 Day 17–21 的 6 个提交整个消失、reflog 里也没有。
**因为之前 push 过，GitHub 上完好，`git fetch` + `merge --ff-only` 一步全恢复。**
但当天未提交的代码和讲义（手写卷积、CNN、讲义、台账）全部要重写。

两条结论：

- **「本地有」≠「存下来了」。** 只有 push 到远端的提交才算。
- **一个模块做完就 commit。** 第一版是「代码 + 讲义 + 台账一起做完再提交」，
  一被回滚就全没了。改成做完一块提交一块，最多只丢一块。

---

## 二、目录结构

| 目录 | 用途 | 说明 |
|---|---|---|
| `src/engine/` | **训练引擎** | 注册表、Trainer、训练循环、命令行。**换模型/数据集只改这里** |
| `src/utils/` | 工具模块 | 日志、路径、config、seed、**手写的优化器 / 损失函数 / Linear 层 / 卷积** |
| `src/models/` | 模型定义 | `mlp.py` · `cnn.py` |
| `src/data/` | 数据加载与增强 | `mnist.py` · `augment.py`（**手写的仿射增强**，Day 23）· `batch_aug.py`（**整批增强，collate 层**，Day 24） |
| `configs/` | 实验配置 | YAML 文件。超参写在配置里，不硬编码进代码 |
| `scripts/` | 入口脚本 | 训练、对照实验、环境自检。**`check_env.py` 在这里** |
| `notebooks/` | 探索性代码 | **正式代码不许放这里**。只用来快速验证想法 |
| `docs/` | **学习资产** | 讲义、进度台账、速查卡。**进版本控制** ← 见下方 |
| `outputs/` | 训练输出 | 日志、图表、checkpoint。**已在 .gitignore 中，不进版本库** |
| `weekly/` | 每周复盘 | 每周一份，写清这周做了什么、卡在哪、下周改什么 |

**`src/engine/` 的分层**（Day 18 重构的产物）：

```
cli.py  →  trainer.py  →  registry.py    名字 → 构建函数
                ↓              ↑
           builtin.py ─────────┘         只有这里知道 mlp / mnist 是什么
                ↓
        models/ · data/ · utils/
```

一条硬约束：**`trainer.py` 不认识任何具体模型。** 它只会说「给我一个叫
`cfg.model` 的东西」。这样 P1 加 CNN 时，只要 `trainer.py` 没改，
就能确定训练流程（早停、存盘、记录格式）的行为完全没变。

**为什么 `docs/` 要单独存在、而且必须进 git？**
`2026-10-26` 工作区被清空过一次：代码因为进了 git 全部完好，
而放在仓库外面 `学习计划/` 目录里的 Day 11–15 讲义、速查卡、逐日清单**全部丢失**，
一份都找不回来。从那天起讲义一律写进 `docs/` 并提交 ——
**只有 `git clone` 能救回来的东西，才算真的存下来了**。
详见 [`docs/README.md`](docs/README.md)。

**为什么 `outputs/` 要排除在 Git 之外？**
因为它是**产物**不是**源码**。几百 MB 的 checkpoint 提交上去，仓库会变得无法 clone，
而且这些东西随时能重新生成——只有源码才是不可再生的。

**为什么 `notebooks/` 和 `src/` 要分开？**
Notebook 的执行顺序是隐藏状态，同一个 cell 跑三次结果可能不同，无法复现。
一旦某段代码重要到需要被复用，就把它搬进 `src/`，写成函数。

---

## 三、怎么在这台机器上跑 Python

**重要：这台机器上 `conda activate pytorch` 在 PowerShell 里是无效的**
（conda 4.10.3 从未对 PowerShell 初始化过，执行会报「系统找不到指定的路径」）。

所以**不要**照搬网上教程的「先 activate 再 python」。用完整路径直调：

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
& "E:\anaconda\envs\pytorch\python.exe" scripts\env_info.py
```

两个坑：

- 路径含空格（`work Buddy`），**必须加引号**
- 命令开头的 **`&` 不能省**——路径带引号时，少了它 PowerShell 会当字符串处理

**为什么不能直接敲 `python`？**
因为 PATH 里的 `python` 指向另一个独立安装的 Python 3.12（`C:\Users\...\Programs\Python\Python312`，
**没有装 torch**）。用它跑任何脚本都会 `ModuleNotFoundError`。

同理，装包时也必须指定解释器路径，否则包装进那个野生 Python，在 conda 环境里永远 import 不到：

```powershell
& "E:\anaconda\envs\pytorch\python.exe" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 包名
```

**怀疑环境有问题时**，先跑这个：

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
& "E:\anaconda\envs\pytorch\python.exe" scripts\check_env.py
```

期望末行输出 `结果：16/16 项通过 —— 环境就绪，可以开工。`

它检查五组共 16 项：解释器（是否在 `envs\pytorch` 内、版本）、依赖库、GPU（CUDA 版
torch、算力等级、cuDNN、**真跑一次 fp16 矩阵乘**）、功能（autograd 对拍、中文字体
能否渲染）、项目结构（目录 + 配置能否解析、`data_dir` 是否已绝对化）。

它同时会打印实际使用的解释器路径，所以也是「我到底在不在正确环境里」的检测器。

---

## 四、现在能跑什么

**P0 已全部完成**：环境、配置系统、反向传播、训练循环、优化器、MNIST 端到端
（test acc 98.00%）、过拟合与正则化、三个手写算子、通用训练模板。

**P1 进行中**：Day 22 加了 CNN，MNIST 从 98.00% → **98.89%**
（同参数量 20.7 万，差距来自结构而非容量），手写卷积七关对拍通过。

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
$PY = "E:\anaconda\envs\pytorch\python.exe"

# 0) 环境自检（16 项）
& $PY scripts\check_env.py

# ── 通用训练入口（Day 18）─────────────────────────────────────
& $PY scripts\train.py --list                      # 有哪些模型/数据集可选
& $PY scripts\train.py --dry-run                   # 打印最终生效配置，不训练
& $PY scripts\train.py --epochs 3 --tag quick      # 覆盖单个字段
& $PY scripts\train.py --set dropout=0.2 --set hidden=512  # 覆盖任意字段

# ── 手写算子的对拍 ────────────────────────────────────────────
& $PY src\utils\optim.py --train      # Day 12  优化器五关
& $PY src\utils\losses.py --train     # Day 16  损失函数七关
& $PY src\utils\layers.py --train     # Day 17  线性层七关
& $PY src\utils\conv.py --train       # Day 22  卷积七关

# ── 模型自测（结构 / 形状流 / 感受野）─────────────────────────
& $PY src\models\mlp.py               # Day 10  MLP 三关
& $PY src\models\cnn.py               # Day 22  CNN 自测

# ── 数据增强（Day 23 / 24）───────────────────────────────────
& $PY src\data\augment.py --train      # 增强七关（含真 MNIST 统计对拍）
& $PY src\data\batch_aug.py --train    # 整批增强六关（含与逐样本的逐位对拍）
& $PY scripts\sweep_augment.py         # 5 组剂量扫描 + 早停离线分析（约 35 分钟）
& $PY scripts\sweep_augment.py --epochs 3 --only 0,3   # 先试水

# ── MNIST 训练 ───────────────────────────────────────────────
& $PY scripts\train_mnist.py                              # MLP 10 epoch，约 95 秒
& $PY scripts\train_mnist.py --loss myce  --epochs 2      # 换手写损失函数
& $PY scripts\train_mnist.py --linear my  --epochs 2      # 换手写 Linear 层
& $PY scripts\train_mnist.py --dropout 0.2 --weight-decay 1e-4 --patience 3

# ── CNN 训练（Day 22）────────────────────────────────────────
& $PY scripts\train.py --config configs\cnn_mnist.yaml --tag cnn            # 98.89%
& $PY scripts\train.py --config configs\cnn_mnist.yaml --set conv=my `
      --tag cnn_myconv                                                     # 换手写卷积

# ── 加数据增强（Day 23，25 轮成绩见 Day 24）───────────────────
& $PY scripts\train.py --config configs\cnn_aug.yaml --tag cnn_aug          # 15 轮 99.15%
& $PY scripts\train.py --config configs\cnn_aug.yaml --epochs 25 `
      --tag cnn_aug25                                                     # 25 轮 99.36%
& $PY scripts\train.py --config configs\cnn_mnist.yaml `
      --set augment_shift=2 --set augment_angle=10 --epochs 15

# ── 对照实验与教学脚本 ────────────────────────────────────────
& $PY scripts\run_ablation.py --epochs 10   # 四组正则化对照（约 6 分钟）
& $PY scripts\demo_train_loop.py            # Day 11 训练循环四关
```

`scripts\train.py` 和 `scripts\train_mnist.py` 调的是**同一个 `main()`**，
参数完全通用；后者只是为兼容旧命令保留的薄壳。

输出落在 `outputs\<时间戳>_<dataset>_<tag>\`，内含 `history.json`（逐轮指标，
**不含耗时** —— 混进耗时就没法用它 diff 两跑证明复现性）、`best.pt`、
`curve.png`、`config.yaml`（**这一跑的全部生效参数**）。

**多跑几次一定用不同 `--tag`**，否则分不清哪个目录是哪次。

**配置的三级优先级：命令行 > yaml > 代码里的默认值。**

```powershell
& $PY scripts\train.py --epochs 3 --lr 5e-4 --seed 42 --device cpu
& $PY scripts\train.py --set act=tanh --set n_layers=3    # 冷门字段也不用改代码
```

配置里的值会经过校验（`optimizer` / `scheduler` / `act` / `loss` / `linear` /
`early_stop_metric` / `flatten` 类型 / `dropout` 范围等），**配错在读配置的
那一刻就报错**，不会拖到训练第 1000 步才发现。

**加一个新模型**（Day 22 已经真用过一次）：在 `src/engine/builtin.py` 里加

```python
@register_model("cnn")
def build_cnn(cfg):
    return CNN(...)
```

然后在 yaml 里写 `model: cnn`。`trainer.py` / `cli.py` 一个字都不用改。

这不是设想 —— Day 22 加 CNN 时的实测结果是：
`grep -n cnn src/engine/trainer.py src/engine/cli.py` → **0 行命中**。
甚至 `loops.py` 里那个 `flatten` 开关都是 Day 18 就留好的。

> 一个细节：`registry.py` 里确实有 6 行提到 "cnn"，但**全在注释里**
> （Day 18 写的「想换 CNN？→ 在 builtin.py 里加一行注册」）。
> 所以「不影响行为」的判据是**代码行有没有变**，不是文件里有没有出现过这个词。

---

## 五、环境与硬件实况

| 项目 | 规格 |
|---|---|
| GPU | NVIDIA GTX 1660 Ti · 6 GB · sm_75 (Turing) · **无 Tensor Core** |
| CPU | i7-10750H · 6 核 12 线程 |
| 内存 | 16 GB |
| Python | 3.10.20（conda 环境 `pytorch`，位于 `E:\anaconda\envs\pytorch`） |
| PyTorch | 2.13.0+cu126 |

**能力边界（记牢，避免做无用功）：**

- 可行：DQN / PPO / SAC 跑 CartPole、MuJoCo；LeRobot + ACT / Diffusion Policy；SmolVLA（约 450M）微调
- 勉强：7B 模型 4-bit 推理
- **不可行：7B 模型 LoRA / QLoRA 微调**（6 GB 显存必 OOM，要租云端卡）
- 不支持：FlashAttention-2 / TF32 / 原生 bf16

**精度建议**：训练用 `fp16 + GradScaler` 或直接 `fp32`。
Turing 上 `torch.cuda.is_bf16_supported()` 返回 `True` 是**软件模拟**，用了反而更慢。

**缓存位置**（已迁移到 E 盘，避免撑爆 C 盘）：
`HF_HOME` / `PIP_CACHE_DIR` / `TORCH_HOME` / `WANDB_DIR` 均指向 `E:\ml-cache\`。

---

## 六、提交信息约定

用前缀标明这次提交在干什么：

| 前缀 | 含义 |
|---|---|
| `feat:` | 新增功能或算法实现 |
| `fix:` | 修 bug |
| `chore:` | 杂项（目录、配置、依赖） |
| `docs:` | 文档、复盘 |
| `exp:` | 实验记录、结果 |
| `refactor:` | 重构，不改变行为 |

示例：

```
feat: implement DQN with replay buffer and target network
fix: correct TD target to use target network params
exp: PPO on HalfCheetah, 5 seeds, mean return 3120 +- 180
```

---

## 七、进度

**P0 逐日台账（Day 1–21，含所有实测数字和踩过的坑）见 [`docs/P0-进度台账.md`](docs/P0-进度台账.md)。**

| 阶段 | 周期 | 状态 |
|---|---|---|
| P0 工具箱 | W1–W3（Day 1–21） | **✅ 完成** |
| P1 RL 地基 | W4–W8 | **进行中** |
| P2 策略优化 | W9–W14 | 未开始 |
| P3 VLA 入门 | W15–W20 | 未开始 |
| P4 速览 + mini 课题 | W21–W24 | 未开始 |
| P5 收口 | W25–W26 | 未开始 |

**P0 硬指标（5/5 全部完成）**

- [x] 手写 MLP 训练 MNIST，不用高级封装，test acc **> 97%** → 实测 **98.00%**（Day 13）
- [x] 手写 softmax / 交叉熵前向+反向，与 PyTorch 梯度误差 **< 1e-5** → 实测 **1.86e-09**（Day 16）
- [x] 手写 Linear 层前向+反向，误差 **< 1e-5** → 实测 **1.49e-08**（Day 17）
- [x] 通用 `train.py --config config.yaml` 模板 → Day 18，**重构前后确定性字段逐字节相同**
- [x] 本阶段提交 **≥ 15 次** → 已 36+ 次

**三个手写算子（softmax / 交叉熵 / Linear）全部与官方实现对拍通过。**
