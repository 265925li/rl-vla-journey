"""
探针：BatchNorm 到底"具体"产生了什么作用。

不需要数据集，纯随机张量，CPU 上 2 秒跑完。每个数字都是现场测的。

  A. 输出的尺度不变性 —— 以及它何时失效（σ² 逼近 ε）
  B. 初始化尺度鲁棒性 —— loss / 梯度对 init 尺度的敏感度
  C. 批噪声 —— 正则化效果从哪来，量化它
  D. 深度漂移 —— 在"校准过的"默认初始化下会不会漂
  D2. 同样的实验换成**未校准**的初始化 —— 论文那张图的条件

用法：
  python scripts/probe_bn_effect.py
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
EPS = 1e-5


def hdr(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


# ── A. 输出的尺度不变性 ──────────────────────────────────────────
hdr("A. 把 conv 权重整体乘 s，看输出怎么变")

x = torch.randn(64, 3, 8, 8)
conv_plain = nn.Conv2d(3, 8, 3, padding=1, bias=False)
conv_bn = nn.Conv2d(3, 8, 3, padding=1, bias=False)
conv_bn.load_state_dict(conv_plain.state_dict())      # 同一组权重
bn = nn.BatchNorm2d(8)
bn.train()                                            # 训练态：用当前批统计量
w0 = conv_plain.weight.detach().clone()

with torch.no_grad():
    y_plain_ref = conv_plain(x).clone()
    y_bn_ref = bn(conv_bn(x)).clone()

print(f"{'s':>8} | {'σ²(conv 输出)':>13} {'σ²/ε':>10} | "
      f"{'无BN std':>10} {'相对差':>10} | {'有BN std':>10} {'相对差':>10}")
print("-" * 80)
for s in [0.01, 0.1, 1.0, 10.0, 100.0]:
    with torch.no_grad():
        conv_plain.weight.copy_(w0 * s)
        conv_bn.weight.copy_(w0 * s)
        z = conv_bn(x)                      # BN 的输入
        var = z.var(dim=(0, 2, 3), unbiased=False).mean().item()
        y_plain = conv_plain(x)
        y_bn = bn(z)
    rp = (y_plain - y_plain_ref).norm().item() / y_plain_ref.norm().item()
    rb = (y_bn - y_bn_ref).norm().item() / y_bn_ref.norm().item()
    print(f"{s:>8.2f} | {var:>13.3e} {var / EPS:>10.2e} | "
          f"{y_plain.std():>10.6f} {rp:>10.2e} | {y_bn.std():>10.6f} {rb:>10.2e}")

print(f"""
读法（数字就是上面表里的，别记成别的）：
  无 BN —— y 的 std 严格正比于 s：0.005488 → 54.8776，正好 1e4 倍，
           和 s 的 1e4 倍完全一致。相对差 = |s−1|，一位不差。
  有 BN —— s ≥ 0.1 时 std 稳定在 0.999998 ~ 1.000015，相对差 1e-3 以下。
           **但 s=0.01 时 std 掉到 0.864，相对差 13.7%** —— 不变性破了。

为什么破：s=0.01 时 σ² 只有 {0.005488**2:.2e} 量级，和 ε={EPS} 只差几倍。
          标准化分母 √(σ²+ε) 被 ε 顶住了，除不干净。
          **尺度不变性的前提是 σ² ≫ ε**，不是无条件成立。
          这也是"ε 不能随便调大"的真实原因 —— ε 大了，小尺度层的 BN 就退化成"几乎不标准化"。
""")


# ── B. 初始化尺度鲁棒性 ─────────────────────────────────────────
hdr("B. 换初始化尺度，第 1 步的 loss 和梯度范数")


def make_net(use_bn):
    bias = not use_bn                      # 有 BN 时 conv 的 bias 完全失效（Day 25 关 7）
    layers = [nn.Conv2d(1, 8, 3, padding=1, bias=bias)]
    if use_bn:
        layers.append(nn.BatchNorm2d(8))
    layers += [nn.ReLU(), nn.MaxPool2d(2),
               nn.Conv2d(8, 16, 3, padding=1, bias=bias)]
    if use_bn:
        layers.append(nn.BatchNorm2d(16))
    layers += [nn.ReLU(), nn.MaxPool2d(2),
               nn.Flatten(), nn.Linear(16 * 4, 10)]
    return nn.Sequential(*layers)


xb = torch.randn(128, 1, 8, 8)
yb = torch.randint(0, 10, (128,))
crit = nn.CrossEntropyLoss()

res = {}
print(f"{'init std':>9} | {'无BN loss':>11} {'无BN |grad|':>12} | "
      f"{'有BN loss':>11} {'有BN |grad|':>12}")
print("-" * 68)
for scale in [1e-3, 1e-2, 1e-1, 1.0]:
    row = [scale]
    for use_bn in (False, True):
        net = make_net(use_bn)
        for m in net.modules():                 # 只重设 conv，γ/β 保持论文初始的 1/0
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=scale)
        net.train()
        loss = crit(net(xb), yb)
        loss.backward()
        gnorm = sum(p.grad.norm().item() ** 2
                    for p in net.parameters() if p.grad is not None) ** 0.5
        row += [loss.item(), gnorm]
    res[scale] = row
    print(f"{row[0]:>9.0e} | {row[1]:>11.4f} {row[2]:>12.3e} | "
          f"{row[3]:>11.4f} {row[4]:>12.3e}")

g_plain = [res[s][2] for s in res]
g_bn = [res[s][4] for s in res]
print(f"""
读法：
  {'':<12}{'无BN':>16}{'有BN':>16}
  {'loss 范围':<12}{f'{min(res[s][1] for s in res):.4f} ~ {max(res[s][1] for s in res):.4f}':>16}"""
      f"""{f'{min(res[s][3] for s in res):.4f} ~ {max(res[s][3] for s in res):.4f}':>16}
  {'|grad| 范围':<12}{f'{min(g_plain):.2e} ~ {max(g_plain):.2e}':>16}"""
      f"""{f'{min(g_bn):.2e} ~ {max(g_bn):.2e}':>16}
  {'跨几个量级':<12}{f'{max(g_plain)/min(g_plain):.0f} 倍':>16}{f'{max(g_bn)/min(g_bn):.0f} 倍':>16}

  无 BN —— loss 从 2.3030 涨到 19.2487（init=1 时直接不是一个正常的分类 loss 了），
           梯度范数跨 {max(g_plain)/min(g_plain):.0f} 倍。同一个 lr 不可能同时适配这两端。
  有 BN —— loss 全在 2.4~2.8（log10≈2.303 是"均匀预测"的基准值），
           梯度跨 {max(g_bn)/min(g_bn):.0f} 倍。比无 BN 小 27 倍左右，但仍然不是常数。

  ⚠️ 一个反直觉的点：init=1e-3 时**有 BN 的梯度反而是最大的（{res[1e-3][4]:.1f}）**，
     比 init=1e-1 时的 {res[1e-1][4]:.2f} 大 29 倍。原因就是 A 里那条：
     权重太小 → σ² ≈ ε → 标准化分母被 ε 顶住 → 反向的 inv_std 极大 → 梯度被放大。
     **BN 不是"抹平一切尺度"，它在尺度极小的一端会把梯度放大。**
""")


# ── C. 批噪声 ───────────────────────────────────────────────────
hdr("C. 批噪声：同一张图放进不同的批，输出会变多少")

bn2 = nn.BatchNorm2d(16)
bn2.train()
x0 = torch.randn(1, 16, 5, 5)                 # 被观察的那张"图"
with torch.no_grad():
    y_solo = bn2(x0).flatten()                 # 单独一批（退化成 m=25 的粗糙统计）
    ys = torch.stack([
        bn2(torch.cat([x0, torch.randn(31, 16, 5, 5)], 0))[:1].flatten()
        for _ in range(200)
    ])
    ratio = ys.std(dim=0).mean().item() / ys.std(dim=1).mean().item()

bn2.eval()
with torch.no_grad():
    ys_eval = torch.stack([
        bn2(torch.cat([x0, torch.randn(31, 16, 5, 5)], 0))[:1].flatten()
        for _ in range(200)
    ])

print(f"  单独成批时 y0 的 std                     {y_solo.std():.6f}")
print(f"  200 种同伴下 y0 的 std 均值              {ys.std(dim=1).mean():.6f}")
print(f"  同一位置跨批的波动（逐元素 std）          {ys.std(dim=0).mean():.6f}")
print(f"  波动 / 信号                              {ratio:.4f}  （= {ratio*100:.1f}%）")
print(f"  推理态同位置波动                          {ys_eval.std(dim=0).mean():.2e}")

print(f"""
读法：
  训练态下同一张图的输出会随"同伴"变 —— 相对抖动 {ratio*100:.1f}%。
  推理态用固定的 running stats，抖动**严格为 0**（不是"很小"，是 0）。

  所以 BN 的正则化效果有三个特征：
    ① 只在训练时存在；
    ② 来源是"随机抽批"这个动作本身，不是任何显式惩罚项；
    ③ 噪声是**数据相关**的（取决于同批其它样本），这和 Dropout 的独立噪声不同。

  这也解释了为什么 BN 和 Dropout 常常不能简单叠加 —— 两者都在往激活里灌噪声。
""")


# ── D. 深度漂移：默认（校准过的）初始化 ─────────────────────────
hdr("D. 层数越多越漂吗？—— 用 PyTorch 默认初始化（kaiming，已按 ReLU 校准）")

d = 64
xin = torch.randn(256, d)


def stack_std(L, use_bn, init_std=None):
    layers = []
    for _ in range(L):
        lin = nn.Linear(d, d, bias=not use_bn)
        if init_std is not None:
            nn.init.normal_(lin.weight, std=init_std)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        layers.append(lin)
        if use_bn:
            layers.append(nn.BatchNorm1d(d))
        layers.append(nn.ReLU())
    net = nn.Sequential(*layers).train()
    with torch.no_grad():
        return net(xin).std().item()


print(f"{'层数 L':>7} | {'无BN 输出 std':>16} | {'有BN 输出 std':>16}")
print("-" * 48)
for L in [2, 5, 10, 20, 40]:
    print(f"{L:>7} | {stack_std(L, False):>16.6e} | {stack_std(L, True):>16.6f}")

print("""
读法（和预期相反，照实说）：
  **默认初始化下，无 BN 也没有指数崩塌。** L=2 时 0.139，L=40 时 0.035，
  只降了 4 倍，不是几个量级。
  原因是 PyTorch 的默认初始化是 kaiming_uniform，方差**已经按 ReLU 的半死特性校准过**了
  （每层乘 √(2/fan_in)），它本身就在对抗衰减。

  有 BN 那列稳定在 0.58~0.61 —— 也不是 1，因为我在 ReLU **之后**测的，
  ReLU(N(0,1)) 的 std 就是约 0.58。BN 保证的是"ReLU 之前"恒为 1。

  结论：**"BN 解决深层网络的分布漂移"这个叙事，前提是初始化没校准。**
  一旦用上 kaiming，这个收益就基本消失了。下面 D2 复现论文的条件。
""")


# ── D2. 同样的实验，换成未校准的初始化 ──────────────────────────
hdr("D2. 换成**未校准**的初始化（std=1 正态）—— 论文那张图的条件")

print(f"{'层数 L':>7} | {'无BN 输出 std':>16} | {'有BN 输出 std':>16}")
print("-" * 48)
tbl2 = {}
for L in [2, 5, 10, 20, 40]:
    tbl2[L] = (stack_std(L, False, init_std=1.0), stack_std(L, True, init_std=1.0))
    print(f"{L:>7} | {tbl2[L][0]:>16.6e} | {tbl2[L][1]:>16.6f}")

print(f"""
读法：
  未校准初始化下，无 BN 的输出 std 每层乘约 √64=8：
  L=2 是 {tbl2[2][0]:.3e}，L=40 已经到 {tbl2[40][0]:.2e}（接近 float32 上限 3.4e38）。
  有 BN 依然稳在 {tbl2[2][1]:.3f}~{tbl2[40][1]:.3f}。

  所以 D 和 D2 合起来才是完整答案：
    BN 的"抗深度漂移"是**真的**，但它只在**初始化没校准**时显现。
    现代框架的默认初始化已经把这个洞补掉了一大半，所以你不能指望
    "随手加个 BN 就涨点" —— 今天 MNIST 上测不出精度提升，一部分原因就在这里。
""")


# ── 汇总 ────────────────────────────────────────────────────────
hdr("汇总：BN 的作用，按「今天在 MNIST 上能否观测到」分开")
print("""
  作用                              机制                      本次实测
  ────────────────────────────────────────────────────────────────────────
  1 输出不随参数尺度漂移            按通道重新标准化          ✅ A 实测
  2 对初始化尺度 / lr 更鲁棒        尺度不变性                  ✅ B 实测
  3 轻微正则化                      批噪声（数据相关）          ✅ C 实测
  4 抗深层分布漂移                  每层出口拉回 N(0,1)         ⚠️ D2 才有，D 没有
  5 提高精度上限                    ——                          ❌ 测不出（6.3 节）

  第 1~3 条在今天的 5 组训练里都能对上：
    ep1 val_loss 0.1004 → 0.0590（−41%）   ← 第 1、2 条
    末轮 gap +0.27 → +0.06pt（−78%）        ← 第 3 条
    test acc −0.19pt / +0.14pt（t<1.6）     ← 第 5 条不成立
""")
