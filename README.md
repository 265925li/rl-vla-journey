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
| `src/utils/` | 工具模块 | 日志、路径、config 读取等被多个脚本复用的代码 |
| `configs/` | 实验配置 | YAML 文件。超参写在配置里，不硬编码进代码 |
| `scripts/` | 一次性脚本 | 画图、环境探针、数据转换等，跑完即弃的入口 |
| `notebooks/` | 探索性代码 | **正式代码不许放这里**。只用来快速验证想法 |
| `outputs/` | 训练输出 | 日志、图表、checkpoint。**已在 .gitignore 中，不进版本库** |
| `weekly/` | 每周复盘 | 每周一份，写清这周做了什么、卡在哪、下周改什么 |
| `src/` | 核心代码 | 算法实现、环境封装、训练循环 |

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
cd "E:\work Buddy\强化学习"
& "E:\anaconda\envs\pytorch\python.exe" 00-setup\check_env.py
```

期望末行输出 `结果：16/16 项通过`。它同时会打印实际使用的解释器路径，
所以也是「我到底在不在正确环境里」的检测器。

---

## 四、环境与硬件实况

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

## 五、提交信息约定

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

## 六、进度

详细的 26 周计划见 `../学习计划/`，P0 阶段的 21 天逐日清单见 `../学习计划/P0-逐日执行清单.md`。

| 阶段 | 周期 | 状态 |
|---|---|---|
| P0 工具箱 | W1-W3 | 进行中 |
| P1 RL 地基 | W4-W8 | 未开始 |
| P2 策略优化 | W9-W14 | 未开始 |
| P3 VLA 入门 | W15-W20 | 未开始 |
| P4 速览 + mini 课题 | W21-W24 | 未开始 |
| P5 收口 | W25-W26 | 未开始 |
