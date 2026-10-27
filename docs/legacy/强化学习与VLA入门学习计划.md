# 强化学习 & VLA 入门学习计划

> **适用画像**：每周 20–30 小时 · 数学强 / 编程弱 · 目标是先广泛探索、找到自己的方向
> **本机实况**：GTX 1660 Ti **6GB / sm_75（无 Tensor Core）** · i7-10750H · 16GB 内存 · C 盘仅剩 7.8GB
> **总周期**：约 26 周（6 个月），累计投入约 550–700 小时
> **核心策略**：**用数学换速度，用代码补短板，每 6 周一个可交付物，每阶段末一个方向决策点**

> ⚠️ **硬件已实测（2026-10-16）**，与本计划原先的"有 GPU 可用"假设有出入。
> **6GB 显存 + Turing 架构意味着 P3 阶段的 7B 模型微调本地跑不动**，具体边界见文末附录。
> RL 部分（P0–P2）完全不受影响。

---

## 一、总体设计思路

你的情况有一个很关键的错配：**理论不会卡你，代码会卡死你**。

很多数学好的同学学 RL 会掉进同一个坑——公式全懂，PPT 能讲，但代码跑不起来，于是永远停在"看懂"层面，做不了研究。而科研和就业真正筛的是"能不能把一个想法跑出结果"。

所以这份计划和你可能在别处看到的路线图有三点不同：

1. **前 3 周不碰 RL，只补工程**。这是投资回报率最高的 3 周。
2. **每个算法都要求手写一遍**，不许只用 `stable-baselines3` 调包。库只用来当对照 baseline。
3. **每 6 周设置一个决策点**。你的目标是"找方向"，那就必须把决策显式地排进计划，而不是学到一半才想起来问自己"我要做啥"。

### 三条并行主线

| 主线 | 定位 | 节奏 |
|---|---|---|
| **理论线**（你的优势区） | 数学推导、算法原理、论文中的理论部分 | 快速推进，不恋战，够用就往下走 |
| **代码线**（你的瓶颈） | 手写算法、跑实验、调 bug、复现 | 慢推细嚼，宁可少学一个算法也要写透 |
| **论文线**（从第 6 周开始） | 读论文、建品味、找问题 | 每周 1–2 篇，三遍读法，长期积累 |

### 阶段总览

| 阶段 | 周次 | 主题 | 关键交付物 | 决策点 |
|---|---|---|---|---|
| **P0** | W1–3 | 工具箱：Python 工程 + PyTorch + 环境 | 独立跑通并调试一个训练脚本 | — |
| **P1** | W4–8 | RL 理论地基：MDP → TD → DQN | 手写 DQN 打穿 CartPole/LunarLander | — |
| **P2** | W9–14 | 策略优化：PG → PPO → SAC | 手写 PPO 跑 MuJoCo + 复现一篇工作 | **★ 决策点 A** |
| **P3** | W15–20 | VLA / 具身智能入门 | LeRobot 跑通 + LoRA 微调 OpenVLA | — |
| **P4** | W21–24 | 相邻方向速览 + mini 课题 | 一个完整项目 + 技术报告 | **★ 决策点 B** |
| **P5** | W25–26 | 收口：作品集 + 方向调研 | GitHub 作品集 + 汇报材料 | — |

---

## 二、P0｜工具箱（W1–W3，约 60–75 小时）

> 📌 **本阶段已拆到天**：见 `P0-逐日执行清单.md`（21 天逐日任务 + 过关标准）与 `P0-每日打卡看板.html`（可勾选、自动保存进度）。
> 下面这一节是 P0 的**目标与验收定义**；每天具体做什么以逐日清单为准。

### 这一阶段为什么存在

你数学强，理论部分不会卡住你；真正会让你在第 5 周就放弃的，是"torch 和 CUDA 版本对不上""shape 报错看不懂""代码改了不知道哪一步坏的"。这三周是给后面 23 周铺路。

### 知识 / 技能清单

**1. 开发环境**
- Miniconda 或 `uv`（推荐，更快更干净）+ 虚拟环境，**一个项目一个环境**
- VS Code + Jupyter 插件
- 如果你的 GPU 在实验室服务器上：**务必学会 SSH + VS Code Remote / Jupyter 端口转发**，这个技能要用 6 个月
- CUDA / 驱动 / PyTorch 版本匹配（新手第一大坑，装完立刻验证 `torch.cuda.is_available()`）

**2. Python 工程基本功**
- 类型注解、`argparse`、`logging`（不要到处 `print`）、`pathlib`、异常处理
- **代码模块化**：拆成 `envs/` `algos/` `utils/` `train.py` `config.yaml`
- **Git**：`add / commit / branch / merge`，**每天至少提交一次**。你未来的 GitHub 就是你的科研履历
- 写 README、写 requirements.txt

**3. NumPy 向量化思维（RL 代码性能的命门）**
- 广播机制、`einsum`、批量矩阵运算
- 避坑：Python 的 `for` 循环在 RL 里是性能杀手，要养成"能用张量批量算就不写循环"的直觉
- 练习：写一个 3 层 for 循环的矩阵乘法，再写成 `A @ B`，对比耗时

**4. PyTorch 三件套**
- Tensor 与自动微分：**手推一次两层网络的反向传播，再和 `autograd` 的结果对数**
- `nn.Module` 自定义网络、`optim`、`nn.functional`
- 手写完整训练循环（不用 Lightning / Trainer）
- checkpoint 保存加载、`.to(device)`、`torch.no_grad()`

**5. 调试与实验管理**
- 断点调试、**打印张量形状**（80% 的深度学习 bug 是 shape 不匹配）
- 固定随机种子（`torch.manual_seed` / `np.random.seed`）——复现的前提
- TensorBoard 或 Weights & Biases 记录曲线

### 代码任务（必须全部完成）

- [ ] 手写 MLP 训练 MNIST，不使用任何高级封装，测试准确率 > 97%
- [ ] 手写 softmax / 交叉熵 / 一个线性层的前向+反向，与 PyTorch 梯度对比误差 < 1e-5
- [ ] 写一套通用的 `train.py --config config.yaml` 模板，之后所有实验复用
- [ ] 建 GitHub 仓库（建议名 `rl-vla-journey`），本阶段提交 ≥ 15 次

### 资源
- PyTorch 官方教程：60 Minute Blitz / Autograd Mechanics / Custom nn Modules
- 3Blue1Brown《神经网络》系列（补反向传播直觉）
- 廖雪峰 Git 教程，或《Pro Git》前 3 章

### 验收标准
> 随便给你一个陌生的训练脚本，你能逐步说清每个张量的形状和数据是怎么流动的，并能定位一处人为埋入的 bug。

---

## 三、P1｜RL 理论地基（W4–W8，约 100–125 小时）

### 主教材

| 用途 | 材料 |
|---|---|
| **主教材（中文，重数学推导，正好吃你的优势）** | 赵世钰《强化学习的数学原理》（B 站有完整课程） |
| **经典教材** | Sutton & Barto, *Reinforcement Learning: An Introduction*，第 1–9 章 |
| **补充直觉** | David Silver 课程（选看，不用全刷） |
| **中文速通** | 蘑菇书 EasyRL（GitHub 开源） |

### 知识清单

**1. MDP 与贝尔曼方程（务必吃透）**
- 状态 / 动作 / 奖励 / 折扣因子、回报、状态值函数、动作值函数
- **贝尔曼期望方程** 与 **贝尔曼最优方程**——必须能手推
- 最优策略的存在性、压缩映射与不动点定理（你数学好，这里可以真正搞懂"值迭代为什么收敛"）

**2. 动态规划（已知模型）**
- 策略迭代、值迭代、截断策略迭代
- 代码：GridWorld 手写三种 DP，画出最优策略箭头图

**3. 无模型预测（模型未知）**
- 蒙特卡洛（MC）、TD(0)、n-step TD、TD(λ)、eligibility trace
- 偏差–方差权衡：**用数学讲清楚为什么 MC 方差大无偏、TD 有偏方差小**

**4. 无模型控制**
- Sarsa、Q-Learning、ε-greedy、GLIE 条件
- on-policy 与 off-policy 的本质区别（不是定义，是"为什么 Q-learning 能用旧数据"）

**5. 值函数逼近与深度 RL**
- 致命三要素（Deadly Triad）：函数逼近 + 自举 + off-policy → 可能发散
- **DQN**：经验回放、目标网络、梯度裁剪，分别解决什么问题
- 改进链：Double DQN → Dueling → Prioritized Replay → Noisy Net → Rainbow

**6. 探索与利用**
- ε-greedy、UCB、Thompson Sampling、熵正则化

### 代码任务

- [ ] GridWorld：手写策略迭代 + 值迭代，输出最优策略和值函数热力图
- [ ] 手写 MC / TD(0) 估计状态值，画收敛曲线对比
- [ ] 手写表格 Q-Learning 解 FrozenLake 和 CliffWalking
- [ ] **手写 DQN（纯 PyTorch，约 200 行）**，打穿 CartPole，再上 LunarLander
- [ ] 加上 Double DQN + Dueling，与朴素 DQN 对比曲线，用 W&B 记录

### 验收标准
- 能手推贝尔曼最优方程，并解释值迭代收敛的原因
- 能不背课文地讲清"DQN 为什么要经验回放和目标网络"
- CartPole 上稳定拿到接近满分（500 步）

### 本阶段常见坑
1. **不要一上来读 PPO 论文**——P1 完全不需要，会浪费一周
2. **不要只看视频不写代码**：每看 1 小时课，至少写 2 小时代码
3. 曲线不涨时按顺序查：reward 尺度 / ε 衰减 / 学习率 / 种子 / 网络初始化
4. 不要在 P1 就上 MuJoCo——连续控制留到 P2

---

## 四、P2｜策略优化与主流算法（W9–W14，约 120–150 小时）★决策点 A

### 知识清单

**1. 策略梯度基础**
- **策略梯度定理推导**（对数技巧 / 似然比技巧，至少会两种推导）
- REINFORCE、高方差问题、baseline、reward-to-go（因果性）

**2. Actor-Critic 家族**
- A2C / A3C、优势函数 A(s,a)、**GAE(λ)**
- 再一次的偏差–方差权衡：这次你会真正理解为什么 GAE 是标配

**3. 信赖域方法（重点中的重点）**
- TRPO：KL 约束、单调改进定理、共轭梯度
- **PPO**：clipped surrogate objective——**必须能解释 clip 到底在约束什么，以及为什么它比 TRPO 好用**
- 从 TRPO 到 PPO 的动机链条要能讲出来

**4. 连续控制**
- DDPG → **TD3**（三个 trick：双 critic、延迟更新、目标动作平滑）
- **SAC**：最大熵框架、重参数化技巧、自动温度调节 α

**5. 工程工具链**
- Gymnasium API（注意 `terminated` 与 `truncated` 的区别，这是常见 bug 源）
- **Stable-Baselines3** —— 只作为对照 baseline，不作为学习主体
- MuJoCo（HalfCheetah / Hopper / Walker2d）
- RL Baselines3 Zoo（学超参配置怎么调）
- 向量化环境 `VectorEnv`、随机种子与可复现性

### 代码任务

- [ ] 手写 REINFORCE 解 CartPole，亲身体验方差有多大
- [ ] 手写 A2C + GAE，与 REINFORCE 对比
- [ ] **手写 PPO（本阶段核心）**：先 CartPole，再 MuJoCo，曲线对标 SB3
- [ ] 手写 SAC，解 Hopper 或 Walker2d 达到合理分数
- [ ] **复现一篇工作**，推荐二选一：
  - 《The 37 Implementation Details of PPO》（博客 + 代码，极其适合练复现）
  - 《What Matters in On-Policy RL》或 SAC 的实现细节消融

### 里程碑：第 14 周末 —— ★决策点 A

到这一步你已经真正知道 RL 是什么了。此时做一次**显式决策**，三选一：

| 路线 | 内容 | 适合谁 |
|---|---|---|
| **A. 算法 / 理论本身** | 离线 RL、多智能体、探索机制、收敛性保证 | 走这条路最吃你的数学优势，可以冲理论/算法类论文 |
| **B. RL for LLM** | RLHF → DPO → GRPO、过程奖励模型、推理模型、Agent RL | 当前最热、工业界岗位最多，Python 工程量大 |
| **C. 具身智能 / VLA** | 机器人策略学习，RL + 模仿学习 + 多模态 | 直接进入 P3，软硬件结合，天花板高 |

**决策依据（重要）**：别看"哪个方向火"，回头看 W9–W14 这六周里，**你写哪部分代码时最不觉得累**。这个体感信号通常比理性分析更可靠——因为它反映的是你能不能长期忍受这件事的日常。

选定后，P3 仍然建议走 VLA（它是 C 的必经路，也是 A/B 都能用上的技能），但从 P4 开始分叉。

---

## 五、P3｜VLA 与具身智能入门（W15–W20，约 120–150 小时）

### 前置补齐（W15，约 20 小时）

**Transformer 必须彻底搞懂**，否则后面全是黑箱。

- 手写 self-attention / multi-head attention / 位置编码，**不用 `nn.Transformer`**，然后和 PyTorch 结果对齐
- 理解 KV Cache、prefill / decode 阶段（VLA 推理延迟优化的基础）
- 资源：Andrej Karpathy《Let's build GPT》（必看）、《The Annotated Transformer》

### 知识清单

**1. 视觉语言基础**
- ViT、**CLIP**（对比学习）、SigLIP、DINOv2
- 视觉 token 与语言 token 如何在同一个空间对齐

**2. 模仿学习 / 行为克隆（VLA 的核心训练范式）**
- BC、DAgger、协变量偏移（covariate shift）问题
- 动作表示：离散 token vs 连续动作 chunk
- **Diffusion Policy**、**ACT**（Action Chunking Transformer）

**3. VLA 模型谱系（按时间顺序读，理解演进动机）**

| 时间 | 模型 | 解决了什么 |
|---|---|---|
| 2022 | RT-1 | 证明大规模机器人数据 + Transformer 可行 |
| 2023 | RT-2 | 把动作当作 token，用 VLM 直接输出，泛化大幅提升 |
| 2024 | **OpenVLA** | 开源 7B，社区可微调，成为事实基线 |
| 2024 | Octo | 开源通用策略，多本体 |
| 2024–25 | π0 / π0-FAST | 流匹配 / 自回归，动作连续生成质量提升 |
| 2025 | GR00T 等 | 通用机器人基础模型 |

**关键问题（你能讲清这些就算入门）**：动作 tokenization 怎么做、跨本体（cross-embodiment）泛化、推理频率够不够、数据规模与数据从哪来。

**4. 评测与仿真**
- 基准：LIBERO、SimplerEnv、RoboMimic、CALVIN、RLBench
- 仿真器：MuJoCo、Isaac Sim / Isaac Lab、Genesis

**5. RL 与 VLA 的结合（交叉地带，最容易出成果）**
- VLA 做初始化 + RL 微调（RL finetuning）
- 奖励设计、sim2real 基础（域随机化）
- 人类偏好 / RLHF 思路迁移到机器人

### 代码任务

- [ ] 手写 multi-head attention，与 PyTorch 输出对齐
- [ ] 手写 BC，在 MuJoCo 专家数据上训练（用 SB3 采数据即可）
- [ ] **跑通 LeRobot 官方全流程**：数据集 → 训练 → 推理 → 仿真部署
- [ ] **微调一个 LeRobot 策略**：优先 **SmolVLA（约 450M）** 或 ACT / Diffusion Policy，6GB 显存完全够
- [ ] 在 LIBERO 或 SimplerEnv 上做评测，写一份对比报告
- [ ] 加分项：**OpenVLA 7B 只做 4-bit 推理**，不微调（显存不够）；如需微调，租云端卡
- [ ] 加分项：用 Diffusion Policy 复现一个任务

### 资源
- **HuggingFace LeRobot**（框架 + 教程 + 数据集，**最佳入门起点，从这里开始**）
- 论文：RT-1、RT-2、OpenVLA、Diffusion Policy、ACT、π0
- 仓库：`Awesome-VLA`、`openvla/openvla`、`octo-models/octo`
- 课程：CS336（语言模型从零）、CS 223A（机器人学）选看

### 验收标准
> 能画出一张完整的 VLA 技术路线图，并讲清 RT-1 → RT-2 → OpenVLA → π0 每一代**解决了上一代的什么问题**。

---

## 六、P4｜相邻方向速览 + mini 课题（W21–W24，约 100–120 小时）

### 速览（每个方向 1–2 天，只求"知道是什么、难在哪、瓶颈在哪"）

- [ ] **世界模型**：Dreamer v3、Genie、World Models——用学到的模型做规划
- [ ] **多智能体 RL**：MADDPG、QMIX、MAPPO——核心难题是信用分配与非平稳性
- [ ] **LLM 后训练**：RLHF → DPO → GRPO、过程奖励模型、推理模型（o1/R1 类）
- [ ] **离线 RL**：CQL、IQL、Decision Transformer（把 RL 变成序列建模）
- [ ] **效率与安全**：样本效率、安全 RL、可解释性

这一步的价值不是学会，而是**排除法**——看清哪些东西你完全不感兴趣。

### mini 课题（本计划最重要的产出）

选一个**小而具体**的问题，规模控制在 3 周内能做完。推荐方向：

1. **复现 + 消融**：复现一篇论文，然后把其中一个模块换掉，看指标变化
2. **VLA 改造**：在 LIBERO 上把某个 VLA 的某模块替换成另一种设计，量化对比
3. **RL 微调 BC**：用 RL 微调一个行为克隆策略，量化提升幅度
4. **RL 对齐小实验**：复现 GRPO 在小模型上的对齐效果

**产出清单**
- [ ] 可运行代码仓库（README 写清：做了什么、怎么跑、结果如何）
- [ ] 一份 5–10 页技术报告（动机 / 方法 / 实验 / 结论 / **失败分析**）
- [ ] 一次 15 分钟口头汇报（对着室友讲也行，能讲清就是真懂）

### ★决策点 B

此时你手上有了：一套可复用的工程能力 + 一个完整项目 + 对 3–5 个方向的实际体感。

**用这些去谈，而不是空着手去谈。**
- 找导师：带着作品和报告去，而不是问"老师我该做什么"
- 判断深造：如果你在这个 mini 课题里体会到"调 bug 调三天也愿意"，那就是适合做研究的信号
- 如果发现都不喜欢：立刻换赛道也是这个计划的目的之一，你只花了 6 个月就完成了排除

---

## 七、P5｜收口（W25–W26）

- [ ] **GitHub 作品集**：整理出 3 个仓库，每个 README 讲清"我做了什么、结果如何"
- [ ] **方向调研报告**：RL / VLA 现状、国内外主要课题组、主要瓶颈、你的判断
- [ ] 主动联系 2–3 位导师（邮件里放上作品集链接和一页研究兴趣说明）

---

## 八、每周节奏模板（按 25 小时设计）

| 时段 | 时长 | 内容 |
|---|---|---|
| 周一至周五 | 每天 2h | **理论**：听课 / 读书 / 推公式，做结构化笔记 |
| 周一至周五 | 每天 1h | **代码**：写当天知识点对应的实现 |
| 周六 | 5h | **集中编码**：本周主任务（手写算法 / 跑实验 / 消融） |
| 周日 | 3h | **复盘**：读 1–2 篇论文 + 写周报 + 整理代码和实验记录 |

**论文线从 W6 启动**，每周至少 1 篇，用三遍读法：
1. 标题 + 摘要 + 图 + 结论（5 分钟，判断值不值得读）
2. 方法 + 实验设置 + 主要表格（30 分钟，搞懂做了什么）
3. 细节 + 附录 + 自己复现关键结果（按需）

每篇论文记入一个 `paper-list.md`：一句话总结 + 它解决的问题 + 它能用在我哪里。

---

## 九、笔记与输出系统

| 系统 | 工具 | 规则 |
|---|---|---|
| 笔记 | Obsidian / Notion | 每条笔记分类：**概念 / 公式推导 / 代码 / 踩坑** |
| 代码 | 一个 monorepo | 按 `01-dp/ 02-td/ 03-dqn/ 04-ppo/ ...` 分目录，每个算法独立可跑 |
| 输出 | 博客 / 知乎 | **每两周写一篇学习小结**。输出是检验理解最便宜的方式 |
| 实验 | W&B 或 TensorBoard | 每个实验记录 config + seed + 曲线，可复现 |

---

## 十、避坑清单（针对"编程弱"的专项提示）

1. **不要在没跑通玩具问题前上大项目**。CartPole 都没稳，别碰 MuJoCo。
2. **一次只引入一个新概念**。跑通，再加下一个。同时改三处，出错你归不了因。
3. **报错先读完整 traceback**：最后一行是错误类型和原因，往上数第三行才是出错位置。
4. **复制别人的代码必须逐行能解释**，否则等于没写。看不懂就问 AI，追问到懂为止。
5. **不要开三条线并行**。RL 和 VLA 分阶段走，不要同时学。
6. **不要追求"看完所有课"**。本计划的课程只是骨架，**代码任务才是主体**。
7. **曲线不好别急着调超参**。先检查：环境有没有 bug、数据对不对、reward 定义是否符合预期。超参是最不该先怀疑的东西。
8. **每天提交 Git**。不是为了好看，是为了你敢改代码。

---

## 十一、资源总表

### RL 理论
- 赵世钰《强化学习的数学原理》B 站课程（**中文首选，重推导**）
- Sutton & Barto, *Reinforcement Learning: An Introduction*（官网免费 PDF）
- 蘑菇书 EasyRL（GitHub，中文速通）
- David Silver 课程（B 站有搬运）
- UC Berkeley **CS285**（Deep RL，进阶必看）

### RL 实践
- OpenAI **Spinning Up in Deep RL**（代码 + 讲解，PPO 部分极好）
- HuggingFace **Deep RL Course**（免费，从 DQN 到 PPO）
- **CleanRL**（单文件实现，读源码最佳材料）
- Stable-Baselines3 + RL Baselines3 Zoo（对照 baseline）

### VLA / 具身智能
- HuggingFace **LeRobot**（框架 + 教程 + 数据集，**入门起点**）
- 论文：RT-1 / RT-2 / OpenVLA / Diffusion Policy / ACT / π0
- 仓库：`Awesome-VLA`、`openvla/openvla`、`octo-models/octo`
- 基准：LIBERO、SimplerEnv、RoboMimic

### Transformer / 大模型基础
- Andrej Karpathy《Let's build GPT》《Let's build a Tokenizer》
- 《The Annotated Transformer》
- 李宏毅机器学习 / 生成式 AI 课程（中文，直觉讲得好）

### LLM 后训练
- HuggingFace TRL 文档（PPO / DPO / GRPO 实现）
- 论文：InstructGPT / DPO / GRPO(DeepSeekMath)

---

## 十二、阶段自检表

每阶段结束时，逐条打勾。**任何一条打不上，不要进入下一阶段。**

**P0 结束**
- [ ] 我能在 30 分钟内从零配好一个 PyTorch 环境并验证 GPU 可用
- [ ] 我能看代码说出每个张量的形状和数据流
- [ ] 我有至少 15 次 Git 提交记录

**P1 结束**
- [ ] 我能手推贝尔曼最优方程
- [ ] 我能不背课文地解释 DQN 的每个组件为什么存在
- [ ] 我手写的 DQN 在 CartPole 上稳定满分

**P2 结束**
- [ ] 我能解释 PPO 的 clip 在约束什么
- [ ] 我手写的 PPO 在 MuJoCo 上曲线合理
- [ ] 我完整复现过一篇工作
- [ ] **我做出了决策点 A 的选择**

**P3 结束**
- [ ] 我能手写 multi-head attention
- [ ] 我跑通了 LeRobot 全流程
- [ ] 我微调过 SmolVLA / ACT 一类的策略并在基准上评测过
- [ ] 我能画出 VLA 技术演进路线图

**P4 结束**
- [ ] 我有一个完整可运行的项目 + 一份报告
- [ ] 我对 3–5 个方向有实际体感

---

*计划是为了被执行而存在的。如果某一阶段实际耗时超出 50%，优先砍内容（少学一个算法），而不是降低验收标准。*

---

# 附录：本机硬件实况与能力边界

> 实测时间：2026-10-16 · 全部为只读检测，未修改任何配置

## A1. 硬件清单

| 部件 | 型号 / 规格 | 评价 |
|---|---|---|
| 机型 | HUAWEI FRD-WX9 笔记本 | 标压本，散热和 TDP 有上限 |
| CPU | Intel i7-10750H · 6 核 12 线程 · 2.6GHz | 够用，数据加载不会成为瓶颈 |
| 内存 | **16 GB**（当前可用 5.2 GB） | ⚠️ 偏紧，MuJoCo 并行环境一开就吃满 |
| 显卡 | NVIDIA GeForce GTX 1660 Ti · **6 GB** · **sm_75 (Turing)** · 80W | 能跑，但见下方边界 |
| 硬盘 | Samsung MZVLB512HBJQ · 512GB NVMe SSD（单盘三区） | 速度没问题，容量告急 |
| 驱动 | 610.74（支持到 CUDA 13.3） | 很新，无需动 |

**分区空间**

| 分区 | 总量 | 可用 |
|---|---|---|
| C: (Windows) | 120.0 GB | **7.8 GB** ⚠️ |
| D: (Data) | 196.8 GB | 29.9 GB |
| E: (Data) | 146.5 GB | 69.6 GB |

## A2. 软件环境实况

**Anaconda 位于 `E:\anaconda`**（在 E 盘，这点很好，不会加剧 C 盘压力）

| 环境 | 路径 | Python | PyTorch | 结论 |
|---|---|---|---|---|
| `pytorch` | `E:\anaconda\envs\pytorch` | 3.10.20 | **2.13.0+cu126** | ✅ **唯一可用，CUDA 已通** |
| `py311` | `E:\anaconda\envs\py311` | 3.11.8 | 2.3.1+**cpu** | ❌ 无 CUDA，弃用 |
| `base` | `E:\anaconda` | 3.9.7 | 2.8.0+**cpu** | ❌ 无 CUDA，别用它跑训练 |
| `pingyeling` | `C:\Users\...\.conda\envs\` | — | — | ⚠️ **幽灵条目**，目录已不存在 |

**实测通过的 CUDA 信息**（`pytorch` 环境）
```
torch.__version__    : 2.13.0+cu126
torch.version.cuda   : 12.6
cuda.is_available()  : True
device               : NVIDIA GeForce GTX 1660 Ti
compute_capability   : sm_75
total_memory         : 6.0 GB
```

### ⚠️ 三个必须先处理的坑

**1. PATH 里的 `python` 不是 conda 的**
`python` → `C:\Users\...\Programs\Python\Python312\python.exe`（独立安装，**没装 torch**）。
如果直接 `pip install gymnasium`，会装进这个野生 Python，然后你在 conda 环境里怎么都 import 不到。
**规则：永远用完整路径调用 `E:\anaconda\envs\pytorch\python.exe`，即 `& "E:\anaconda\envs\pytorch\python.exe" 脚本.py`。**
> ⚠️ 补充（2026-10-16 实测）：`conda activate pytorch` 在**本机 PowerShell 里无效**（conda 4.10.3 未对 PowerShell 初始化，报"系统找不到指定的路径"）。所以不要依赖 activate，用完整路径最稳；或改用 Anaconda Prompt。详见 `P0-逐日执行清单.md` 开头的"怎么跑 Python"。

**2. `pytorch` 环境原本缺依赖（2026-10-16 已补齐并验证）**
原装：`torch 2.13.0`、`torchvision 0.28.0`、`numpy 2.2.6`、`tensorboard 2.21.0`
**现已补装**：`gymnasium 1.4.0`、`stable_baselines3 2.9.0`、`scipy 1.15.3`、`pandas 2.3.3`、`matplotlib 3.10.9`、`wandb 0.30.0`、`mujoco 3.16.0`、`box2d 2.3.10`、`tqdm`
**验证结果：自检脚本 16/16 全部通过**（见 `00-setup/check_env.py`）

**3. conda 4.10.3 是 2021 年的版本，且未对 PowerShell 初始化**
自带的是老的 classic solver，装现代包会极慢甚至解不出依赖。`.condarc` 用了清华源（很好），但 `default_channels` 里还留着 Anaconda 早已废弃的 `pkgs/free` 和 `pkgs/pro`，可能报 404。

另外 `CUDA_PATH` 指向 `CUDA 11.3`，与 torch 自带的 cu126 不一致——这是历史残留，日常训练无影响，但**编译自定义 CUDA 算子时会出问题**（部分机器人与 VLA 包需要）。

## A3. 能力边界（这张表决定了你能做什么、不能做什么）

| 任务 | 能否本地跑 | 说明 |
|---|---|---|
| DQN / PPO / SAC 打 CartPole | ✅ 轻松 | 模型只有几层 MLP |
| PPO / SAC 打 MuJoCo（HalfCheetah 等） | ✅ 可以 | 显存占用 < 1GB，主要压力在 CPU 和内存 |
| 并行环境（VectorEnv × 8） | ⚠️ 谨慎 | 瓶颈是 16GB 内存，建议先 ×4 |
| LeRobot + ACT / Diffusion Policy | ✅ 可以 | 小型 CNN/Transformer，几百 MB |
| **SmolVLA（约 450M）微调** | ✅ 可以 | 6GB 能装下，开混合精度 |
| 7B 模型 **4-bit 推理** | ✅ 勉强 | 权重约 4GB，做 demo 可以 |
| **7B 模型 LoRA / QLoRA 微调** | ❌ **不行** | 4-bit 权重 ~4GB + 激活 + 优化器，6GB 必 OOM |
| FlashAttention-2 | ❌ 不支持 | 需要 sm_80+（Ampere） |
| TF32 / 原生 bf16 | ❌ 不支持 | Turing 无 Tensor Core；torch 报 `bf16_supported=True` 是**软件模拟**，反而更慢 |

**训练精度结论**：Turing 上就用 **fp16 + GradScaler**，或干脆 fp32（RL 的网络小，fp32 也不慢）。**不要用 bf16。**

**需要 7B 以上微调时的替代方案**：Colab Pro / Kaggle（每周免费 30h T4×2）/ AutoDL / 学校集群。本地负责开发调试，云端负责大任务。注意 T4 也是 sm_75，同样没 Tensor Core；要快就选 A100/L40S/4090。

## A4. C 盘空间去向（2026-10-16 已执行清理）

**清理前**：C 盘已用 112 GB，只剩 **7.8 GB**。占用大头：

| 目录 | 占用 |
|---|---|
| `C:\Windows` | 28.19 GB |
| `AppData\Roaming` | **22.35 GB** |
| `AppData\Local` | 20.38 GB |
| `C:\Program Files` | 11.90 GB |
| `C:\Program Files (x86)` | 7.92 GB |
| `C:\ProgramData` | 4.15 GB |

**已清理（只清纯缓存，未碰任何个人文件与系统目录）**

| 项目 | 清理前 | 清理后 | 释放 |
|---|---|---|---|
| 用户 Temp（含遗留的 VS 安装包缓存 3.27 GB） | 4.03 GB | 0.07 GB | 3.96 GB |
| pip 下载缓存 | 1.42 GB | 0.00 GB | 1.42 GB |
| **C 盘可用空间** | **7.8 GB** | **13.2 GB** | **+5.4 GB** |

**明确未动**：`C:\Windows\Temp`、Prefetch、Windows Update 缓存（系统级，避免影响稳定性）；conda 包缓存（E 盘，5.14 GB，用户要求保留）；回收站（1.66 GB，为历史个人删除记录）；微信 / QQ 聊天记录；JetBrains / 金山 / 百度等应用数据。

**仍可回收的空间（需你决定）**
- 回收站 1.66 GB —— 清空即释放，但你之前删的文件会永久消失
- `AppData\Roaming` 里 Tencent 7.94 GB、金山 2.92 GB、百度 1.80 GB、QQ 1.63 GB —— 需在各客户端内改存储路径迁移到 D/E 盘，**不要直接删**
- JetBrains 缓存 4.23 GB —— 可在 IDE 内通过 `Invalidate Caches` 安全清理
- E 盘 conda 包缓存 5.14 GB —— `conda clean --all`（本次未执行）

> 以上仅为扫描数据与执行记录。凡涉及个人文件的删除或迁移，都需要你明确指定后再执行。

## A5. 环境状态与剩余待办

### ✅ 已完成（2026-10-16）

```bash
# 1) 补装 P0/P1/P2 全部依赖（已执行，用清华源）
pip install gymnasium stable_baselines3 scipy matplotlib pandas wandb tqdm
pip install "gymnasium[mujoco]"     # MuJoCo 3.16.0
pip install "gymnasium[box2d]"      # Box2D 2.3.10 + pygame-ce（有预编译包，无需 C++ 编译器）

# 2) 端到端验证（16/16 通过）
E:\anaconda\envs\pytorch\python.exe 00-setup\check_env.py
```

验证通过项：torch 2.13.0+cu126 · CUDA 可用（GTX 1660 Ti sm_75）· numpy/scipy/pandas/matplotlib · gymnasium 1.4.0 · mujoco 3.16.0 · CartPole-v1 / LunarLander-v3 / HalfCheetah-v5 实例化与交互 · SB3 2.9.0（使用 GPU）· wandb · tensorboard · DQN 完整训练+评估

> 注意：依赖安装**未触发任何已有包的降级或升级**，torch 2.13.0 保持原样。

### ⏳ 剩余待办（低优先级，不影响开始学习）

```bash
# 1) 删掉失效的幽灵环境条目（该环境目录已不存在，删的是元数据记录）
conda env remove -n pingyeling
```

**conda 4.10.3 偏旧**：老式 classic solver 装现代包会慢，但**当前环境已够用，不急着升**。真要升级时建议用 E 盘装 Miniforge 而非原地升级 Anaconda，风险更小。

**日常使用铁律**：不要敲裸的 `python` / `pip`（PATH 里的 `python` 是另一个独立安装的 3.12，两者不通用）。用完整路径 `E:\anaconda\envs\pytorch\python.exe`，或改用 Anaconda Prompt。**`conda activate pytorch` 在本机 PowerShell 里无效**，别依赖它。

### P3 阶段再装（别提前占空间）

```bash
pip install lerobot transformers accelerate peft bitsandbytes
```

## A6. 结论

- **P0 / P1 / P2（工程 + RL 全部内容）在这台机器上完全可行**，6GB 显存对 RL 算法是绰绰有余的。
- **P3 的 VLA 部分需要把目标从"微调 7B"下调到"微调 SmolVLA / ACT 级别"**，7B 只做推理。这个调整不影响你理解 VLA 的核心机制——SmolVLA 和 OpenVLA 的架构思路是一致的，只是规模不同。
- **16GB 内存比 6GB 显存更可能成为瓶颈**，跑实验前关掉微信、QQ、浏览器。
- **C 盘 7.8GB 是本机最紧迫的问题**，不是配置问题而是稳定性问题——Windows 更新和 conda 临时文件都可能因此失败。
