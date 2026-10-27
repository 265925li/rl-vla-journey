"""Day 22 · CNN：卷积神经网络（MNIST）

用法（在项目根目录）：
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\models\\cnn.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\models\\cnn.py --show-shapes

════════════════════════════════════════════════════════════════════
为什么 MLP 到 98% 就到顶，而 CNN 能到 99%

  MLP 的第一步是 x.view(N, -1)：28×28 的图被摊成 784 个数。
      第 1 个数和它右边那个数、和它下面那个数，**关系全部消失**。
      剩下的只是 784 个互相独立的特征。

  这意味着 MLP 学到的只能是「第 137 个位置的值大不大」，
  而不是「这里有个拐角」。想让它认出一个竖笔画，
  得在 28 个不同位置**各学一遍**（每个位置一套独立权重）。

  卷积的做法完全不同：
      同一套 3×3 权重在图上滑过所有位置。
      → 学到的是「左暗右亮 = 竖边」这种**与位置无关**的模式
      → 一个模式学会一次，全图通用（这叫平移不变性）
      → 参数从 784×784 降到 9

  这就是「参数共享」四个字的全部含义。
════════════════════════════════════════════════════════════════════
本模型的结构（每一层的尺寸都写清楚，算错哪一步会当场暴露）

      输入 (B, 1, 28, 28)
        │
      conv1: Conv2d(1, 16, k=3, p=1)     → (B, 16, 28, 28)   参 160
        │   ReLU
      pool1: MaxPool2d(2)                → (B, 16, 14, 14)
        │
      conv2: Conv2d(16, 32, k=3, p=1)    → (B, 32, 14, 14)   参 4640
        │   ReLU
      pool2: MaxPool2d(2)                → (B, 32,  7,  7)
        │
      flatten                            → (B, 1568)          ← 32*7*7
        │
      fc1:   Linear(1568, 128)           → (B, 128)          参 200832
        │   ReLU
      fc2:   Linear(128, 10)             → (B, 10)           参 1290
        │
      输出 logits (B, 10)                                     合计 206922

  ⚠️ 注意 206922 这个数 —— 和 MLP 的 203530 **几乎一样**。
     但 CNN 的精度更高。这说明：
     「参数量相同，效果不同」→ 差距来自**结构**，不是容量。
     这是今天最该带走的一条。
════════════════════════════════════════════════════════════════════
两个必背的点

  1) **CNN 不能摊平**。yaml 里必须写 flatten: false。
     如果 flatten 忘了关，Trainer 会先把 (B,1,28,28) 摊成 (B,784)，
     Conv2d 收到 2 维输入会直接报错（这个错会当场暴露，算幸运的）。

  2) **池化层和激活层没有参数**。所以参数量公式里它们不出现。
     它们的唯一作用是把图变小 / 加非线性。
════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ACT_MAP = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


# ═══════════════════════════════════════════════════════════════════
# 1. CNN 主体
# ═══════════════════════════════════════════════════════════════════
class CNN(nn.Module):
    """两层卷积 + 两层全连接的小 CNN。

    ── 参数说明 ─────────────────────────────────────────────────
    in_channels : 输入通道数。MNIST 是灰度图，所以是 1。
                  RGB 图是 3。**写错不会报错**，只会让第一层权重形状不对，
                  然后 PyTorch 在第一次 forward 时报通道不匹配 —— 这个错会暴露。
    channels    : 两个卷积层的输出通道数，默认 (16, 32)。
                  为什么逐层翻倍？因为每过一次池化，空间尺寸减半、
                  单个位置的"信息密度"变了，惯例是通道翻倍来补偿。
    hidden      : 全连接层的宽度。
    dropout     : 加在 fc1 之后。注意 P0 的实测结论 ——
                  MNIST + 这个规模加正则化会让精度**下降**（Day 15），
                  默认 0.0 是有依据的，不是偷懒。
    conv_cls    : 卷积层的类。传 MyConv2d 就能换成手写实现（Day 22 关 7 验过）。
    bn_cls      : BatchNorm 的类，**传 None 表示不加**（Day 25）。
                  传 nn.BatchNorm2d 或手写 MyBatchNorm2d 都可以 ——
                  两个签名是对齐的，这是 Day 17 的接口对齐原则。
                  注意构造签名收的是「类」而不是「字符串」：
                  「my / torch」这种字符串解析放在 builtin.py 里，
                  因为模型层不该知道配置系统长什么样。
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, int] = (16, 32),
        hidden: int = 128,
        out_dim: int = 10,
        dropout: float = 0.0,
        act: str = "relu",
        conv_cls: type[nn.Module] = nn.Conv2d,
        bn_cls: type[nn.Module] | None = None,
        image_size: int = 28,
    ) -> None:
        super().__init__()
        if act not in ACT_MAP:
            raise ValueError(f"act 只能是 {list(ACT_MAP)}，收到 {act!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout 要在 [0, 1) 内，收到 {dropout}")

        self.in_channels = in_channels
        self.channels = channels
        self.hidden = hidden
        self.out_dim = out_dim
        self.image_size = image_size
        self.has_bn = bn_cls is not None
        c1, c2 = channels
        act_cls = ACT_MAP[act]

        # ── 卷积部分 ──────────────────────────────────────────────
        # padding=1 配 kernel=3 是 "same" 卷积：输出尺寸 == 输入尺寸。
        # 所以 conv 层不改尺寸，只有 pool 层把尺寸减半。
        # 这让尺寸链变得很好算：28 → pool → 14 → pool → 7。
        #
        # Day 25：接 BN 时 conv 的 bias 要关掉。
        # 不是"省参数"这种小事 —— 关 7 实测证明了它**完全不起作用**：
        # BN 减的是逐通道均值，而 bias 就是逐通道常数，被原样减掉。
        # 留着的话参数量对账会多算 C_out 个（16+32=48 个），
        # 更重要的是会在 state_dict 里留下一批永远不更新的死参数。
        conv_bias = bn_cls is None
        self.conv1 = conv_cls(
            in_channels, c1, kernel_size=3, padding=1, bias=conv_bias
        )
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = conv_cls(c1, c2, kernel_size=3, padding=1, bias=conv_bias)
        self.pool2 = nn.MaxPool2d(2)
        self.act1 = act_cls()
        self.act2 = act_cls()

        # ── BN（Day 25）────────────────────────────────────────────
        # 顺序是 conv → BN → act，不是 conv → act → BN。
        # 为什么：ReLU 会把负半轴全部压成 0，先激活再 BN 的话，
        # 那个通道的分布被砍成"一半是 0"，标准化之后 0 的位置仍是 0，
        # BN 想做的事（把分布拉回均值 0 方差 1）被 ReLU 破坏掉一半。
        # 放 conv 之后、act 之前，BN 收到的是完整的高斯样分布，效果最好。
        #
        # 不加 BN 时用 Identity 占位，而不是 `if self.has_bn:` 分支 ——
        # 这样 forward 只有一条代码路径，不会出现"加了 BN 却没走那条分支"
        # 这种事，而且 print(model) 里能看到它的位置。
        self.bn1 = bn_cls(c1) if bn_cls is not None else nn.Identity()
        self.bn2 = bn_cls(c2) if bn_cls is not None else nn.Identity()

        # ── 展平后的尺寸 ──────────────────────────────────────────
        # 从 image_size 一路算下来，而不是写死 1568。
        # 为什么值得算：换个 image_size（比如 CIFAR 的 32）就得重算，
        # 写死的话会报形状不匹配，而报错位置在 fc1，离错因（image_size）隔了三层。
        feat_size = image_size // 4          # 两次 MaxPool2d(2)：/2/2 = /4
        flat_dim = c2 * feat_size * feat_size
        self.flat_dim = flat_dim

        # ── 分类头 ────────────────────────────────────────────────
        head: list[nn.Module] = [
            # 展平：把 (B, c2, f, f) 变成 (B, c2*f*f)
            # 用 nn.Flatten() 而不是 x.view()，是为了让它成为模型的一部分，
            # print(model) 能看见它，也不会忘记加。
            nn.Flatten(),
            nn.Linear(flat_dim, hidden),
            act_cls(),
        ]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(hidden, out_dim))
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入必须是 4 维 (B, C, H, W)。
        # 这里显式检查是为了给一条有用的报错 —— 默认的报错信息
        # 只会说 "Expected 4-dimensional input"，不会告诉你
        # 最可能的原因是 yaml 里 flatten 忘了关。
        if x.dim() != 4:
            raise ValueError(
                f"CNN 需要 4 维输入 (B, C, H, W)，收到 {tuple(x.shape)}。\n"
                "  最常见原因：配置文件里 flatten 是 true（默认值就是 true）。\n"
                "  CNN 必须写 flatten: false —— 摊平以后空间结构就没了。"
            )

        # 顺序固定为 conv → BN → act（见 __init__ 里的说明）。
        # bn1/bn2 在没加 BN 时是 Identity，这里不用分支 —— 一条路径。
        x = self.act1(self.bn1(self.conv1(x)))   # (B, c1, 28, 28)
        x = self.pool1(x)                        # (B, c1, 14, 14)
        x = self.act2(self.bn2(self.conv2(x)))   # (B, c2, 14, 14)
        x = self.pool2(x)                        # (B, c2,  7,  7)
        return self.head(x)                      # (B, out_dim)

    def feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """把每一层的中间结果都返回，用来做可视化 / 排查。

        为什么这个函数有教学价值：
          卷积层的输出是 (B, C, H, W)，可以当成 C 张灰度小图来看。
          训练好的 conv1 通常能看出「某个通道专门响应竖边、某个响应横边」。
          这是 CNN 可解释性最直观的一步 —— 比任何理论解释都有说服力。
        """
        out: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            out["input"] = x
            h = self.act1(self.bn1(self.conv1(x)))
            out["conv1"] = h
            h = self.pool1(h)
            out["pool1"] = h
            h = self.act2(self.bn2(self.conv2(h)))
            out["conv2"] = h
            h = self.pool2(h)
            out["pool2"] = h
            out["logits"] = self.head(h)
        return out


# ═══════════════════════════════════════════════════════════════════
# 2. 手算参数量（和 P0 的 expected_params 同一套思路）
# ═══════════════════════════════════════════════════════════════════
def expected_params_cnn(
    in_channels: int = 1,
    channels: tuple[int, int] = (16, 32),
    hidden: int = 128,
    out_dim: int = 10,
    image_size: int = 28,
    with_bn: bool = False,
) -> int:
    """按公式手算参数量，用来和 model.parameters() 对账。

    公式（三处来源）：
        卷积层 : C_out · (C_in · kh · kw + 1)     ← 每个输出通道一套窗口 + 一个 bias
        池化层 : 0                                 ← 无参数
        全连接 : in · out + out                    ← 和 MLP 一样

    为什么这个对账必须做：
      卷积层少写一个 padding、通道数写错、池化层数变了，
      都**不会让训练崩**，只会让精度莫名地低一点。
      参数量对不上是唯一能立刻发现"结构不是我想要的"的信号。

    ── Day 25：加了 BN 之后公式变了两处，方向相反 ─────────────────
        conv 的 bias     +1  →  0      BN 把它抵消掉了（关 7 实测 1.8e-15）
        BN 自己的参数     —  →  2·C   γ 和 β 各 C 个

    所以一层 16 通道的 conv：
        不带 BN   C_out·(C_in·9 + 1)
        带 BN     C_out·(C_in·9 + 0) + 2·C_out
    后者**多出 C_out 个参数**。这个方向值得记一下：
    **BN 不是"省参数"的优化，是"多花一点参数换优化性质"。**

    如果这里忘了改（照旧算 bias、又不算 γ/β），误差是
    `48 − 96 = −48` 个（c1+c2=48，2·(c1+c2)=96），
    对账会当场报出来 —— 这正是关 7 和这道检查配合的价值。
    """
    c1, c2 = channels
    conv_bias = 0 if with_bn else 1     # 接 BN 时 conv 不建 bias
    bn_per_ch = 2 if with_bn else 0     # γ 和 β，每个通道两个
    total = 0
    # conv1: in_channels → c1, kernel 3x3
    total += c1 * (in_channels * 3 * 3 + conv_bias) + bn_per_ch * c1
    # conv2: c1 → c2, kernel 3x3
    total += c2 * (c1 * 3 * 3 + conv_bias) + bn_per_ch * c2
    # 两次 MaxPool2d(2) 后：image_size / 4
    feat = image_size // 4
    flat = c2 * feat * feat
    # fc1 + fc2
    total += flat * hidden + hidden
    total += hidden * out_dim + out_dim
    return total


# ═══════════════════════════════════════════════════════════════════
# 3. 感受野（今天最该理解的概念之一）
# ═══════════════════════════════════════════════════════════════════
def receptive_field_trace(image_size: int = 28) -> list[tuple[str, int, int, int]]:
    """逐层追踪「一个神经元在原始图上看多大范围」。

    返回 [(层名, 该层尺寸, 感受野, 有效步长), ...]

    ── 公式 ─────────────────────────────────────────────────────
        rf_out    = rf_in + (k − 1) · jump_in
        jump_out  = jump_in · stride

    直觉：每过一层 k×k 卷积，视野向外扩 k−1 圈；
          但扩的是"上一层格子的间距"（jump），不是原图的一个像素。

    ── 为什么这个概念重要 ────────────────────────────────────────
      两层 3×3 卷积的感受野 = 5×5，参数量是 2·9 = 18（单通道）
      一层 5×5 卷积的感受野 = 5×5，参数量是 25
      → 同样的视野，**堆小卷积更省参数**，而且多了一次非线性。

      这是 VGG 的核心论点，也是"为什么大家都用 3×3"的原因。
      但注意：感受野增长是**线性**的，层数多了才够覆盖大图 ——
      这就是为什么深层网络 + 大分辨率图需要几十层。
    """
    rows = [("input", image_size, 1, 1)]
    size, rf, jump = image_size, 1, 1

    def conv(k=3, s=1, pad=1):
        nonlocal size, rf, jump
        size = (size + 2 * pad - k) // s + 1
        rf = rf + (k - 1) * jump
        jump = jump * s

    def pool(k=2, s=2):
        nonlocal size, rf, jump
        size = (size - k) // s + 1
        rf = rf + (k - 1) * jump
        jump = jump * s

    conv(3, 1, 1); rows.append(("conv1 k3 p1", size, rf, jump))
    pool(2, 2);    rows.append(("pool1 2x2", size, rf, jump))
    conv(3, 1, 1); rows.append(("conv2 k3 p1", size, rf, jump))
    pool(2, 2);    rows.append(("pool2 2x2", size, rf, jump))
    return rows


# ═══════════════════════════════════════════════════════════════════
# 4. 自测
# ═══════════════════════════════════════════════════════════════════
def check_structure() -> bool:
    print("\n[1] 结构与参数")
    model = CNN()
    n = sum(p.numel() for p in model.parameters())
    manual = expected_params_cnn()
    print(f"    参数量  程序 {n:,}   手算 {manual:,}   {'OK' if n == manual else 'MISMATCH'}")
    assert n == manual, f"参数量对不上：{n} vs {manual}"

    # 逐层拆一下，让"参数几乎全在全连接层"这件事看得见
    conv_p = sum(p.numel() for p in model.conv1.parameters()) + \
             sum(p.numel() for p in model.conv2.parameters())
    head_p = sum(p.numel() for p in model.head.parameters())
    print(f"    卷积部分 {conv_p:>7,}  ({conv_p / n * 100:5.1f}%)")
    print(f"    全连接   {head_p:>7,}  ({head_p / n * 100:5.1f}%)")
    print(f"    → 卷积只占 {conv_p / n * 100:.1f}%，但它负责提取全部特征")

    out = model(torch.randn(4, 1, 28, 28))
    print(f"    前向 (4,1,28,28) → {tuple(out.shape)}")
    assert out.shape == (4, 10)
    return True


def check_shapes() -> bool:
    print("\n[2] 逐层形状流")
    model = CNN()
    fm = model.feature_maps(torch.randn(2, 1, 28, 28))
    for name, t in fm.items():
        print(f"    {name:<8} {str(tuple(t.shape)):<20}")
    print("    ⚠️ 注意 pool 层和激活层不改变通道数，只改空间尺寸（pool）")
    print("       每过一层 pool，通道数不变、尺寸减半 —— 这是 CNN 的标准节奏")
    return True


def check_flatten_trap() -> bool:
    """故意演示 flatten 忘关会怎样 —— 这是今天最容易犯的配置错误。"""
    print("\n[3] 反面案例：flatten 忘了关")
    model = CNN()
    x = torch.randn(4, 1, 28, 28)
    x_flat = x.view(4, -1)          # 模拟 Trainer 里 flatten=true 的行为
    print(f"    正确输入 (4,1,28,28) → {tuple(model(x).shape)}")
    try:
        model(x_flat)
        print("    ⚠️ 摊平后居然跑通了？不应该")
        return False
    except Exception as e:
        msg = str(e).split("\n")[0]
        print(f"    摊平输入 (4,784) → 报错：{msg}")
        print("    这个错**会暴露**，算幸运的。但它报在 conv1，")
        print("    而真正的原因在 configs/*.yaml 里 —— 报错位置和错因隔了一个文件。")
    return True


def check_receptive_field() -> bool:
    print("\n[4] 感受野追踪")
    print(f"    {'层':<14}{'输出尺寸':>10}{'感受野':>10}{'有效步长':>12}")
    for name, size, rf, jump in receptive_field_trace(28):
        print(f"    {name:<14}{size:>10}{rf:>10}{jump:>12}")

    # 验一下 VGG 的核心论点：两层 3x3 vs 一层 5x5
    # 从 jump=1、rf=1 出发，连过两个 3x3（stride=1）：
    #   第 1 层：rf = 1 + (3−1)*1 = 3
    #   第 2 层：rf = 3 + (3−1)*1 = 5        ← 到 5×5 为止，不再是 7×7
    # 这里我第一版多叠加了一次，写成了 7 —— 而下面的 assert 当场抓住了。
    rf_2x3 = 1 + (3 - 1) * 1
    rf_2x3 = rf_2x3 + (3 - 1) * 1
    p_2x3 = 2 * 3 * 3
    p_1x5 = 5 * 5
    print(f"\n    两层 3×3：感受野 {rf_2x3}×{rf_2x3}，参数 {p_2x3}")
    print(f"    一层 5×5：感受野 5×5，参数 {p_1x5}")
    print(f"    → 同样视野，堆小卷积省 {p_1x5 - p_2x3} 个参数"
          f"（{p_1x5} → {p_2x3}），还多一次非线性")
    print("    → 这就是「为什么大家都用 3×3」")
    return rf_2x3 == 5


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 22 CNN 自测")
    ap.add_argument("--show-shapes", action="store_true", help="只看形状流")
    args = ap.parse_args()

    print("=" * 68)
    print("Day 22 · CNN 自测")
    print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    print("=" * 68)

    ok = True
    ok &= check_structure()
    if not args.show_shapes:
        ok &= check_shapes()
        ok &= check_flatten_trap()
        ok &= check_receptive_field()

    print("\n" + "=" * 68)
    if ok:
        print("全部通过。下一步：")
        print("  & \"E:\\anaconda\\envs\\pytorch\\python.exe\" scripts\\train.py \\")
        print("        --config configs\\cnn_mnist.yaml --tag cnn")
    else:
        print("有检查未通过，别看训练结果")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
