# rl-vla-journey

从零开始的强化学习（RL）与视觉-语言-动作模型（VLA）学习仓库。

这个仓库的定位是**学习轨迹 + 作品集**：每个阶段的代码、复盘、实验结果都留在这里，
最后它会被用来向导师展示「我做过什么」，而不是「我看过什么」。

---

## 一、这个仓库怎么用

三条纪律，比任何技术细节都重要：

1. **每个算法自己手写一遍。** `stable-baselines3` 只用来做对照 baseline，不用来替代理解。
2. **一次只引入一个新概念。** 同时改三处，出错时你无法归因。
3. **曲线不好时，最后才怀疑超参数。** 先查环境 bug、数据、reward 定义。

---

## 二、目录结构

| 目录 | 用途 | 说明 |
|---|---|---|
| `src/utils/` | 工具模块 | 日志、路径、config、seed、**手写的优化器与损失函数** |
| `src/models/` | 模型定义 | `mlp.py` |
| `src/data/` | 数据加载 | `mnist.py` |
| `configs/` | 实验配置 | YAML 文件。超参写在配置里，不硬编码进代码 |
| `scripts/` | 入口脚本 | 训练、对照实验、环境自检。**`check_env.py` 在这里** |
| `notebooks/` | 探索性代码 | **正式代码不许放这里**。只用来快速验证想法 |
| `docs/` | **学习资产** | 讲义、进度台账、速查卡。**进版本控制** ← 见下方 |
| `outputs/` | 训练输出 | 日志、图表、checkpoint。**已在 .gitignore 中，不进版本库** |
| `weekly/` | 每周复盘 | 每周一份，写清这周做了什么、卡在哪、下周改什么 |

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

P0 已完成到「手写算子」，**MNIST 端到端训练已经跑通**（test acc 98.00%）。

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
$PY = "E:\anaconda\envs\pytorch\python.exe"

# 0) 环境自检（16 项）
& $PY scripts\check_env.py

# 1) 手写算子的对拍（Day 12 优化器五关 / Day 16 损失函数七关）
& $PY src\utils\optim.py --train
& $PY src\utils\losses.py --train

# 2) MNIST 训练（默认 10 epoch，约 95 秒）
& $PY scripts\train_mnist.py
#   用手写损失函数替换 nn.CrossEntropyLoss，证明可以无缝换掉
& $PY scripts\train_mnist.py --loss myce --epochs 2 --tag myce
#   正则化实验（dropout / weight decay / 早停）
& $PY scripts\train_mnist.py --dropout 0.2 --weight-decay 1e-4 --patience 3

# 3) 四组正则化对照实验（约 6 分钟，子进程隔离随机状态）
& $PY scripts\run_ablation.py --epochs 10

# 4) 训练循环的四关教学脚本（Day 11，含"填错会怎样"的反面案例）
& $PY scripts\demo_train_loop.py
```

输出落在 `outputs\<时间戳>_mnist_<tag>\`，内含 `history.json`（逐轮指标，**不含耗时**
—— 混进耗时就没法用它 diff 两跑证明复现性）、`best.pt`、`curves.png`、`config.yaml`。

**多跑几次一定用不同 `--tag`**，否则分不清哪个目录是哪次。

**配置的优先级：命令行 > yaml > 代码里的默认值。** 改参数不用改代码：

```powershell
& $PY scripts\train_mnist.py --epochs 3 --lr 5e-4 --seed 42 --device cpu
```

配置里的值会经过校验（`optimizer` / `scheduler` / `act` / `loss` / `dropout` 范围等），
**配错在读配置的那一刻就报错**，不会拖到训练第 1000 步才发现。

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

**P0 逐日台账（Day 1–16，含所有实测数字和踩过的坑）见 [`docs/P0-进度台账.md`](docs/P0-进度台账.md)。**

| 阶段 | 周期 | 状态 |
|---|---|---|
| P0 工具箱 | W1–W3（Day 1–21） | **进行中，16/21** |
| P1 RL 地基 | W4–W8 | 未开始 |
| P2 策略优化 | W9–W14 | 未开始 |
| P3 VLA 入门 | W15–W20 | 未开始 |
| P4 速览 + mini 课题 | W21–W24 | 未开始 |
| P5 收口 | W25–W26 | 未开始 |

**P0 硬指标**

- [x] 手写 MLP 训练 MNIST，不用高级封装，test acc **> 97%** → 实测 **98.00%**
- [x] 手写 softmax / 交叉熵前向+反向，与 PyTorch 梯度误差 **< 1e-5** → 实测 **1.86e-09**
- [ ] 手写 Linear 层前向+反向，误差 < 1e-5 → Day 17
- [ ] 通用 `train.py --config config.yaml` 模板 → Day 19
- [x] 本阶段提交 **≥ 15 次** → 已 30+ 次
