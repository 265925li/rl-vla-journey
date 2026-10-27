"""Day 23 · 手写数据增强：随机平移 + 随机旋转

════════════════════════════════════════════════════════════════════
为什么需要数据增强：一句话版本

  昨天（Day 22）的 CNN 在 ep7 之后 val_loss 从 0.0413 涨到 0.0572（+39%），
  同时 train_loss 还在降、gap 从 +0.64 涨到 +1.22。
  三条同时出现 = **真正意义上的过拟合**，不是良性抖动。

  过拟合的根源是「模型容量 > 数据能提供的约束」。
  治它有两条路：

      路子 A（降容量）  加 dropout、加 weight_decay、砍小模型
                        → Day 15 实测：在 MNIST 上这招会让精度**下降**
                           （wd 最优 1e-4 只到 0.9812，提升不显著；dropout 更差）
      路子 B（加约束）  不改模型，改数据 —— 让每张图"看起来不一样"，
                        但语义不变（还是那个数字）
                        → 这就是数据增强

  **Day 15 的教训在这里的价值**：既然正则化那条路在 MNIST 上走不通，
  就只剩增强这条路。这不是"多试一个技巧"，是排除法剩下的唯一选项。
════════════════════════════════════════════════════════════════════
为什么手写，不用 torchvision.transforms.RandomAffine

  三个理由，按重要性排：

  1) **它是唯一能"看见"增强到底改了什么的方式。**
     RandomAffine(degrees=10, translate=(0.1,0.1)) 这一行你没有信息可用：
     它到底移动了几个像素？旋转方向是顺时针还是逆时针？
     拼接在 ToTensor 前后，数值范围会差 255 倍。这些全靠猜。
     自己写之后，"translate 0.1" 到底是什么，是你自己定义的。

  2) **torchvision 的几何变换作用在 PIL 图像上（uint8, 0~255）。**
     旋转后插值出来的小数会被**量化回整数**，信息损失一次。
     我们在 float 张量上做，损失小得多。

  3) **参数要能进 config.yaml、要能扫剂量。**
     Day 15 的血泪教训：只测一个点就下结论，结论反了一半。
     要扫剂量就得让"平移几个像素""旋转几度"是一等公民，
     而不是藏在某个 transform 的元组里。

════════════════════════════════════════════════════════════════════
核心只有两个公式

  ① 仿射变换矩阵 theta（2×3）—— 决定"输出图的每个位置去输入的哪里取值"

         theta = [[ cos, -sin,  tx ],
                  [ sin,  cos,  ty ]]

     第 0 行管 x（列方向），第 1 行管 y（行方向），第 2 列是平移。
     tx / ty 的单位是**归一化坐标**，不是像素。换算见 _to_norm()。

  ② 归一化坐标：像素 → [-1, 1]

         x_norm = (2 * col + 1) / W - 1

     为什么要 +1 再减 1：因为 PyTorch 用 align_corners=False，
     坐标指向**像素中心**（第 col 列的中心在 (col+0.5)/W 处），
     不是像素左上角。这一点搞错，整数像素平移就不再精确（见关 2）。
════════════════════════════════════════════════════════════════════
三个必须记住的坑（都在下面的自测里验过）

  坑 1  **整批一个参数 ≠ 每个样本一个参数。**
         theta 的形状是 (N, 2, 3)，第 0 维就是 batch。
         如果你写成 (1, 2, 3) 再广播，整批 128 张图会被**同一个**角度旋转 ——
         等价于"把整个 batch 当成一张图转了一下"，增强效果几乎为零，
         而且不报错。关 3 专门验这个。

  坑 2  **增强只能用在训练集。**
        给验证集也增强，val_loss 会人为变高、抖动变大，
         早停就会乱判（Day 15 已经吃过"判据选错结论相反"的亏）。
         这体现在设计上：build_loaders() 里只包装 train，val/test 原样返回。

  坑 3  **随机数生成器必须是持久的。**
         如果在 Dataset.__init__ 里抽一次参数存下来，那么每个 epoch
         看到的是**同一批**增强图 —— 那只是换了 6 万张固定的新图，
         训练几个 epoch 之后模型就把它记住了，等于没增强。
         正确做法：__getitem__ 里现抽，generator 挂在 self 上（关 5 验）。

════════════════════════════════════════════════════════════════════
用法

    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\data\\augment.py --train

    from src.data.augment import AugmentedDataset, random_affine
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════
# 1. 仿射矩阵：把「平移几个像素、转几度」变成 2×3 的 theta
# ═══════════════════════════════════════════════════════════════════
def make_theta(
    tx: torch.Tensor,
    ty: torch.Tensor,
    angle_deg: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """把「像素单位的平移 + 角度单位的旋转」编译成 theta (N, 2, 3)。

    参数（全部是形状 (N,) 的 float32 张量，N = batch size）：
        tx        向右为正，**像素**为单位
        ty        向下为正，**像素**为单位
        angle_deg 旋转角度（度）。正方向见下面的注释
        image_size  正方形边长（MNIST = 28）

    ── 为什么三个参数都是张量而不是标量 ─────────────────────────────
    因为坑 1：每个样本要**独立的**变换。如果传进来是标量，
    整批就用同一个变换，增强退化成"整批转一下"，白做。
    做成张量之后，"每个样本不同"是接口层面保证的，不靠调用方自觉。

    ── 旋转的正方向（我用一个亮点实测过，见关 6）───────────────
    这张矩阵作用在归一化坐标上，且图像坐标是「x 向右、y **向下**」。
    实测结论：angle=+90° 时，**右侧的亮点会转到上方**。
    视觉上这是逆时针。所以「正角度 = 逆时针」这个直觉在这里成立，
    但它是两个负号抵消出来的结果，不是天然如此 —— 别靠记，
    靠关 6 那个断言。

    ── 平移量为什么要除以 W/2 ────────────────────────────────────
    归一化坐标的跨度是 2（从 -1 到 +1）对应 W 个像素，
    所以 1 像素 = 2 / W 个归一化单位。
    取负号是因为 theta 是「输出坐标 → 输入坐标」的映射：
    想让图**向右**移动 t 像素，输出的第 c 列要去输入的 c−t 列取值，
    所以偏移是 −t。写成 +t 的话图会朝反方向跑（关 2 会抓住）。
    """
    n = tx.numel()
    # 归一化：像素 → [-2/W, 2/W] 量级。不转 float 会触发整数除法（Py 3 是 / 所以没事，
    # 但显式写 float 是为了让人一眼看出这里在做归一化）
    half = float(image_size) / 2.0
    nx = -tx / half
    ny = -ty / half

    rad = angle_deg * (math.pi / 180.0)
    cos = torch.cos(rad)
    sin = torch.sin(rad)

    theta = torch.zeros(n, 2, 3, dtype=tx.dtype, device=tx.device)
    theta[:, 0, 0] = cos      #  x' = cos·x − sin·y + nx
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = nx
    theta[:, 1, 0] = sin      #  y' = sin·x + cos·y + ny
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ny
    return theta


def identity_theta(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """恒等变换的 theta。用来验"整套流水线不引入任何额外误差"。"""
    theta = torch.zeros(n, 2, 3, dtype=dtype, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    return theta


# ═══════════════════════════════════════════════════════════════════
# 2. 采样：theta → 输出图
# ═══════════════════════════════════════════════════════════════════
def warp(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """按 theta 对 x 做一次双线性重采样。

    x     : (N, C, H, W)
    theta : (N, 2, 3)

    两步，都很值得知道：

      affine_grid   把 theta 展开成一张"采样网格" (N, H, W, 2)。
                    对输出图的每个格子 (row, col)，算出它应该去输入的哪个
                    归一化坐标取数。注意最后那个 2 是 (x, y) 顺序 ——
                    **是 (x, y) 不是 (row, col)**，这是最容易搞反的地方。

      grid_sample   拿这张网格去输入上取值。落在整像素上就是原值；
                    落在像素之间就用双线性插值（4 个邻居加权）。

    ── 两个参数的取舍 ──────────────────────────────────────────────
    padding_mode="zeros"
        MNIST 的图边缘本来就是黑的（背景 0），所以旋转后四角填 0
        和"这张图本来就长这样"是一致的。这也是 MNIST 特别适合做
        几何增强的原因之一 —— 换成 CIFAR 那种满幅照片，
        填 0 会造出黑边，反而成了模型能抓住的假特征。

    align_corners=False
        坐标指向像素中心。**必须是 False**：
        如果设 True，坐标指向像素左上角，整数像素平移会整体偏半个像素，
        关 2 的验收就会从 1e-6 变成 ~1e-2 的误差 —— 而 1e-2 已经和
        "theta 符号写反了"是同一个量级，你就分不清是参数选错还是实现错了。
        容差要卡在"只可能是舍入"的量级，才有诊断价值。
    """
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def shift_exact(
    x: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor, pad: int | None = None
) -> torch.Tensor:
    """按**整数**像素平移，越界补 0。**不经过任何插值，所以它是精确的。**

    这是关 2 的"标准答案"：warp() 用插值做到的事，这个函数用手工切片做到。
    两者在整数平移上必须逐位相同 —— 因为插值点正好落在像素中心，
    双线性插值的 4 个权重是 (1,0,0,0)，等于直接取值。

    实现只要想清楚一件事：**pad 之后原图的 (0,0) 跑到了 (pad,pad)**。
        想要 out[r, c] = x[r − ty, c − tx]
        在 padded 里 x[r−ty, c−tx] 位于 (r−ty+pad, c−tx+pad)
        所以切片起点是 (pad−ty, pad−tx)，长度 H×W

    为什么逐样本循环而不是向量化：batch=128 时循环 128 次切片，
    每次都是零拷贝的 view，总耗时可以忽略；而向量化版本（比如用
    一次性 roll + mask）可读性差得多，且很容易在某一边界上写错 1。
    数据加载不是瓶颈，可读性优先。
    """
    n, c, h, w = x.shape
    if pad is None:
        pad = int(max(tx.abs().max().item(), ty.abs().max().item()))
    if pad == 0:
        return x.clone()

    # F.pad 的 pad 参数顺序是 (左, 右, 上, 下)，从最内维往外
    xp = F.pad(x, (pad, pad, pad, pad))
    out = x.new_zeros(x.shape)
    for i in range(n):
        sx = int(tx[i].item())
        sy = int(ty[i].item())
        out[i] = xp[i, :, pad - sy : pad - sy + h, pad - sx : pad - sx + w]
    return out


# ═══════════════════════════════════════════════════════════════════
# 3. 随机增强：从"最大幅度"抽一个实际幅度
# ═══════════════════════════════════════════════════════════════════
def random_affine(
    x: torch.Tensor,
    max_shift: float = 0.0,
    max_angle: float = 0.0,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """对一批图做随机的「平移 + 旋转」。每个样本抽自己的参数。

    参数：
        x         (N, C, H, W)
        max_shift 最大平移幅度，单位**像素**。0 = 不平移
        max_angle 最大旋转幅度，单位**度**。0 = 不旋转
        gen       随机数生成器。传 None 就用全局流（不推荐，见下）

    ── 为什么 max_shift 用像素而不是比例 ──────────────────────────
    torchvision 用比例（translate=(0.1, 0.1)）。比例的好处是换分辨率不用改，
    坏处是**你看不出它到底动了多少**：28×28 上 0.1 = 2.8 像素，
    224×224 上 0.1 = 22 像素，同一个配置在两种情况下含义完全不同。
    用像素就直白得多，而且我要扫剂量（0/1/2/3 像素），
    比例写法会让"1 像素"没法表达。

    ── gen 为什么必须从外面传进来 ────────────────────────────────
    增强的随机性和"数据打乱顺序"的随机性必须是**两条独立的流**。
    共用一条的话，你哪天改了 shuffle 的参数，增强的抽样序列也跟着变 ——
    复现性以极难排查的方式断掉（seed.py 里对此有同样的警告）。
    """
    n = x.shape[0]
    if max_shift <= 0 and max_angle <= 0:
        return x

    tx, ty, ang = sample_params(n, max_shift, max_angle, gen, dtype=x.dtype)
    theta = make_theta(tx, ty, ang, image_size=x.shape[-1])
    return warp(x, theta)


def sample_params(
    n: int,
    max_shift: float,
    max_angle: float,
    gen: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """抽 n 组（平移 x、平移 y、旋转角），每组三个数。

    ── 为什么是「一次抽 3n 个再 reshape」，而不是「分三次各抽 n 个」──────

    这是 Day 24 唯一改了数值行为的地方，值得说透。

    旧写法（Day 23）：

        tx  = rand(n)      # 消耗 n 个随机数
        ty  = rand(n)      # 消耗 n 个
        ang = rand(n)      # 消耗 n 个

    新写法：

        raw = rand((n, 3)) # 一次消耗 3n 个，行优先排列

    单个样本 n=1 时两者**完全一样**：都拿到同一串的三个连续数
    (r0, r1, r2)。所以 Day 23 的最优组（±2 像素 + ±10°，逐样本调用）
    的随机序列一个数都没变 —— 这点已由实跑逐位验证。

    但 n>1 时两者不同，而且差异正是关键：

        旧：样本 i 拿到 (r[i], r[n+i], r[2n+i])      ← 依赖 n
        新：样本 i 拿到 (r[3i], r[3i+1], r[3i+2])    ← 与 n 无关

    旧写法的每个样本拿到的参数**取决于批大小**。这意味着：
    把 batch_size 从 128 改成 64，同一张图从第 128 个位置变成
    第 64 个位置，它拿到的平移量就变了 —— 改批大小会**悄悄改变增强强度**，
    而 batch_size 本来是个"只影响速度不影响结果"的量。
    这种耦合在扫剂量时会致命：你以为只改了批大小，其实改了剂量。

    新写法把"第 i 个样本的参数"钉死成 r[3i..3i+2]，于是：
        · 批大小无关
        · 一次抽 384 个 == 逐样本抽 128 次各 3 个（因为 MT19937 是流式的，
          rand(n) 与 n 次 rand(1) 得到同一串 —— 已实测确认）

    第二条正是 Day 24 能做到「整批增强与逐样本增强逐位对拍」的全部前提。
    Day 22 换算法能对拍到 0.00e+00，Day 23 换数值路径只能对拍到 1e-15，
    今天换的是**执行时机**，本来以为又要退一步只比统计量，
    把随机消耗也改成批大小无关之后，对拍回到了逐位。
    """
    if n <= 0:
        raise ValueError(f"n 必须为正，收到 {n}")

    raw = torch.rand((n, 3), generator=gen, dtype=dtype)
    # [0,1) → [-1,1) → 乘各自幅度
    scale = torch.tensor([max_shift, max_shift, max_angle], dtype=dtype)
    p = (raw * 2.0 - 1.0) * scale
    return p[:, 0], p[:, 1], p[:, 2]


# ═══════════════════════════════════════════════════════════════════
# 4. Dataset 包装：把增强挂到训练集上（且只在训练集上）
# ═══════════════════════════════════════════════════════════════════
class AugmentedDataset(Dataset):
    """在原数据集外面套一层，每次 __getitem__ 现场抽一组变换参数。

    ── 这个类只做对了一件事，但它是唯一重要的事 ────────────────────
    随机数生成器 self.gen 挂在实例上，**跨调用持续消耗**。
    所以第 1 个 epoch 的第 0 张图和第 2 个 epoch 的第 0 张图
    拿到的是**不同**的变换 —— 这才是"每个 epoch 看到 6 万张新图"。

    如果在 __init__ 里把 60000 组参数一次性抽好存下来，
    那么每个 epoch 看到的增强图完全一样，训练两轮就记住了，
    增强变成"把数据集换了一套固定副本"，白做。这个错不报错、不掉精度
    得很明显，只是"增强好像没什么用" —— 最难查的那类。

    ── 为什么不直接用 torchvision.transforms──
    见文件开头。核心是我们需要"像素单位的剂量"这个可配置、可扫描的量。

    ── ⚠️ num_workers 的陷阱 ─────────────────────────────────────
    DataLoader 开子进程时，每个 worker 会 fork 一份 self.gen。
    由于 fork 时状态相同，**所有 worker 会生成完全相同的随机序列** ——
    增强的多样性会掉到 1/worker数。
    本项目 num_workers=0（Windows 上开多进程容易出怪问题），所以没这个风险。
    哪天把 num_workers 调大，必须改成用 worker 自己的 torch.initial_seed()
    去初始化 gen，否则这个 bug 完全不报错。
    """

    def __init__(
        self,
        base: Dataset,
        max_shift: float = 0.0,
        max_angle: float = 0.0,
        seed: int = 0,
    ) -> None:
        if max_shift < 0 or max_angle < 0:
            raise ValueError(
                f"增强幅度不能为负（0 表示关闭该项），收到 shift={max_shift} angle={max_angle}"
            )
        self.base = base
        self.max_shift = float(max_shift)
        self.max_angle = float(max_angle)
        # 独立的生成器：和 DataLoader 的 shuffle 流、模型的初始化流都无关
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)
        self.active = self.max_shift > 0 or self.max_angle > 0

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        if not self.active:
            # 幅度都是 0 时不碰数据。显式短路而不是让 warp 走一遍恒等，
            # 是为了让「关闭增强」和「Day 22 的原始管线」在数值上完全等价 ——
            # 否则 baseline 就被悄悄改了一点点，没法拿来对比。
            return x, y
        # x 是 (C, H, W)，增强函数要 (N, C, H, W)，所以加一维再用 squeeze 去掉
        x = random_affine(
            x.unsqueeze(0), self.max_shift, self.max_angle, self.gen
        ).squeeze(0)
        # 标签一个字都不动 —— 平移/旋转不改变"这是几"
        return x, y


# ═══════════════════════════════════════════════════════════════════
# 5. 自测：七关
# ═══════════════════════════════════════════════════════════════════
def _digits(n: int = 4, seed: int = 0) -> torch.Tensor:
    """造几张"像 MNIST"的测试图：黑底上一个亮块。

    为什么不用 randn：随机噪声图上任何位置都亮，看不出"平移到底动没动"。
    用一个亮块，它的 argmax 位置就是可断言的坐标。
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, 1, 28, 28)
    for i in range(n):
        r = int(torch.randint(6, 22, (1,), generator=g).item())
        c = int(torch.randint(6, 22, (1,), generator=g).item())
        x[i, 0, r : r + 4, c : c + 4] = 1.0
    return x


def gate1_identity() -> bool:
    """关 1：恒等变换不能改变任何东西。

    ⚠️ 这一关我第一版写成「要求 diff 严格等于 0」，结果失败了（1.91e-06）。
    这不是 bug，是 float32 的极限 —— 归一化坐标 (2i+1)/28 − 1 里
    1/28 不是二进制有限小数，fp32 存不下，于是采样点偏了不到一亿分之一，
    双线性插值的权重就从精确的 (1,0,0,0) 变成了 (0.999998, 2e-7, ...)。

    **怎么证明"是精度问题而不是逻辑错"：把精度提上去再测一次。**
    如果 fp64 下误差小两个数量级以上，就说明算法是对的，差的是表示精度。
    如果 fp64 下还是差这么多，那就是坐标顺序/符号这类逻辑错 —— 换精度救不了。
    这个"用提升精度来给误差归因"的动作，比盯着 1.91e-06 猜半天有用得多。
    """
    print("\n[关 1] 恒等变换（fp32 有 1e-6 级舍入，属正常）")
    x = _digits(4)
    out = warp(x, identity_theta(4))
    diff = (out - x).abs().max().item()

    # fp64 复核：只换精度、不换代码路径
    out64 = warp(x.double(), identity_theta(4, dtype=torch.float64))
    diff64 = (out64 - x.double()).abs().max().item()

    print(f"    fp32 最大差  {diff:.2e}")
    print(f"    fp64 最大差  {diff64:.2e}")
    print("    → fp64 下误差掉两个数量级以上 = 只是舍入，不是算法错")
    assert diff < 1e-5, f"fp32 误差 {diff:.2e} 超出舍入量级，大概率是逻辑错"
    assert diff64 < 1e-12, f"fp64 下仍有 {diff64:.2e}，这不是精度问题，去查坐标顺序"
    print("    ✓ 通过")
    return True


def gate2_integer_shift() -> bool:
    """关 2：整数像素平移，插值版与切片版必须对上（fp32 容差 1e-5）。

    这是今天最硬的一关。它同时验证了三件事：
        · theta 的平移符号没写反（写反了图会往反方向跑）
        · 归一化的 /half 换算对（换算错，移动量就不对）
        · align_corners=False 是对的（设 True 会有系统性偏移）
    三处里任何一处错，误差都是**像素级**的（≈1.0），不可能是 1e-6。

    ── 一个和 Day 22 不同的结论 ──────────────────────────────────────
    Day 22 的 im2col 与 F.conv2d 能做到 **0.00e+00 逐位相同**，
    因为两者走的是同一条 GEMM 路径。
    这里做不到，而且**不该**要求做到：grid_sample 是另一条路径
    （自己算归一化坐标 → 再插值），中间必然经过 fp32 的舍入。

    实测有个很有意思的现象：误差大小**和你平移了多少像素无关**。
    (0,0) 恒等变换反而差得最多（1.91e-06），(3,1) 却精确到 0。
    原因是误差来自「归一化坐标能不能被二进制精确表示」，
    而不是来自「移动了几个像素」。这也再次说明它是舍入，不是逻辑。
    """
    print("\n[关 2] 整数像素平移：插值版 vs 切片版（标准答案）")
    x = _digits(8, seed=1)
    tx = torch.tensor([1, -1, 2, -2, 3, -3, 0, 2], dtype=torch.float32)
    ty = torch.tensor([0, 2, -1, 3, 1, -2, 0, -3], dtype=torch.float32)

    got = warp(x, make_theta(tx, ty, torch.zeros(8), image_size=28))
    want = shift_exact(x, tx.to(torch.int64), ty.to(torch.int64), pad=4)
    diff = (got - want).abs().max().item()

    # fp64 复核：同样的 theta、同样的代码，只换精度
    got64 = warp(
        x.double(),
        make_theta(tx.double(), ty.double(), torch.zeros(8, dtype=torch.float64), 28),
    )
    diff64 = (got64 - want.double()).abs().max().item()

    print(f"    {'tx':>4}{'ty':>4}   fp32 最大差")
    for i in range(8):
        d = (got[i] - want[i]).abs().max().item()
        print(f"    {int(tx[i]):>4}{int(ty[i]):>4}   {d:.2e}")
    print(f"    总体  fp32 {diff:.2e}   fp64 {diff64:.2e}")
    print("    → fp64 掉到 1e-15 量级 = 平移量、符号、坐标顺序全部正确")
    assert diff < 1e-5, (
        f"整数平移对不上（fp32 {diff:.3e}）。按这个顺序查：\n"
        "      1. theta 第 3 列的符号（应该是 -tx/half，写成 + 会反向）\n"
        "      2. align_corners 是否 False（True 会偏半个像素）\n"
        "      3. 归一化是否除以了 W/2 而不是 W"
    )
    assert diff64 < 1e-12, f"fp64 下仍差 {diff64:.2e} —— 这不是精度问题，是逻辑错"

    # 顺带验方向：tx>0 时图应该往**右**跑
    bright_before = x[0, 0].sum(dim=0).argmax().item()   # 第一张图的亮点列号
    bright_after = got[0, 0].sum(dim=0).argmax().item()
    print(f"    方向检查  tx=+1 像素 → 亮点列号 {bright_before} → {bright_after}"
          f"（应 +1）")
    assert bright_after == bright_before + 1, "平移方向反了：tx 正应该向右"
    print("    ✓ 通过（fp64 下精确，方向正确）")
    return True


def gate3_per_sample() -> bool:
    """关 3：batch 内每个样本必须是**各自独立**的变换。"""
    print("\n[关 3] 每样本独立参数（坑 1）")
    x = torch.zeros(6, 1, 28, 28)
    x[:, 0, 12:16, 12:16] = 1.0          # 6 张一模一样的图，亮点都在中心

    # 正确做法：tx 是 6 个不同的值
    tx = torch.arange(6, dtype=torch.float32) - 2.5    # -2.5 .. 3.5
    out_ok = warp(x, make_theta(tx, torch.zeros(6), torch.zeros(6), 28))
    # 取亮点位置用「按列求和再 argmax」，不要用 argmax(dim=1)。
    # 我第一版写成 out[i,0].argmax(dim=1)[1]，结果是 "第 2 行里最亮的列号" ——
    # 一旦平移把亮块挪出第 2 行，那一行全是 0，argmax 返回 0，
    # 六张图全报出列号 0，看着像"位移没生效"，其实是断言本身错了。
    # 按列求和把整张图的亮度投到 x 轴上，才是"亮块中心在哪一列"。
    centers = [out_ok[i, 0].sum(dim=0).argmax().item() for i in range(6)]
    print(f"    每样本不同 tx → 亮点列号 {centers}")
    assert len(set(centers)) == 6, f"6 个样本出现了相同的位移 {centers}，不是各自独立"

    # ── 坑 1 的真实形态：形状合法、不报错、每样本拿到同一个参数 ──────
    # 比如在 random_affine 里图省事，只抽一个标量再复制 N 份。
    tx_same = torch.full((6,), 2.0)
    out_same = warp(x, make_theta(tx_same, torch.zeros(6), torch.zeros(6), 28))
    centers_same = [out_same[i, 0].sum(dim=0).argmax().item() for i in range(6)]
    print(f"    抽一个值复制 6 份 → 亮点列号 {centers_same}")
    print("    → 六个全一样。这才是坑 1 的真身：**形状完全合法、不报错**，")
    print("      但整批 128 张图被同一个变换处理，增强退化成「整批转一下」")
    assert len(set(centers_same)) == 1, "复制行为变了？"

    # ── 另一种错法：批大小不匹配，PyTorch 会直接拦住（这种是幸运的）──
    try:
        warp(x, make_theta(torch.tensor([2.0]), torch.zeros(1), torch.zeros(1), 28))
        print("    (1,2,3) 传 batch=6 居然跑通了？不应该")
        return False
    except RuntimeError:
        print("    批大小 (1,2,3) 对 batch=6 → RuntimeError 直接拦住")
        print("    → 同样是「参数传错了」，这一种 PyTorch 帮你挡，上一种只能靠自己")
    print("    ✓ 通过")
    return True


def gate4_shape_range() -> bool:
    """关 4：形状 / dtype 不变，值域不越界。"""
    print("\n[关 4] 形状、dtype、值域")
    x = _digits(8, seed=2)
    g = torch.Generator().manual_seed(0)
    out = random_affine(x, max_shift=3, max_angle=15, gen=g)
    print(f"    输入 {tuple(x.shape)} {x.dtype}  →  输出 {tuple(out.shape)} {out.dtype}")
    print(f"    值域 [{out.min():.3f}, {out.max():.3f}]（输入 [{x.min():.3f}, {x.max():.3f}]）")
    assert out.shape == x.shape, f"形状变了：{out.shape} vs {x.shape}"
    assert out.dtype == x.dtype, f"dtype 变了：{out.dtype}"
    # 增强后可能因为插值出现极小的负数（双线性插值不会，但保险起见查一下）
    assert out.min() >= -1e-6, f"出现负值 {out.min()}，插值或 padding 有问题"
    assert out.max() <= 1.0 + 1e-6, f"超过 1.0：{out.max()}"
    print("    ✓ 通过（值域没被撑破 —— 这很重要，后面接 Normalize 时不会偏移）")
    return True


def gate5_determinism() -> bool:
    """关 5：同 seed 可复现，不同 seed 不同；且生成器必须是持久的。"""
    print("\n[关 5] 复现性与「生成器必须持久」（坑 3）")
    x = _digits(8, seed=3)

    def run(seed: int) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(seed)
        return random_affine(x, 3, 15, g)

    a, b, c = run(0), run(0), run(1)
    d1 = (a - b).abs().max().item()
    d2 = (a - c).abs().max().item()
    print(f"    同 seed 两次   最大差 {d1:.2e}   （必须为 0）")
    print(f"    不同 seed 之间 最大差 {d2:.2e}   （必须 > 0）")
    assert d1 == 0.0, f"同 seed 结果不一致（{d1:.3e}），复现性断了"
    assert d2 > 0.0, "不同 seed 结果相同，说明种子根本没被用上"

    # ── 关键：同一个生成器连续抽两次，结果必须不同 ──────────────────
    # 这模拟的是"第 1 个 epoch 和第 2 个 epoch 拿到不同的变换"。
    g = torch.Generator()
    g.manual_seed(0)
    e1 = random_affine(x, 3, 15, g)
    e2 = random_affine(x, 3, 15, g)
    d3 = (e1 - e2).abs().max().item()
    print(f"    同一 gen 连抽两次 最大差 {d3:.2e}   （必须 > 0，否则每个 epoch 都一样）")
    assert d3 > 0.0, (
        "同一个生成器两次输出相同 —— 说明参数被缓存了，\n"
        "    那么每个 epoch 看到的增强图一样，enhancement 等于没做（坑 3）"
    )

    # Dataset 层再验一次：连续取同一个下标两次，应该拿到不同的图
    from torch.utils.data import TensorDataset
    ds = AugmentedDataset(TensorDataset(x, torch.zeros(8, dtype=torch.long)), 3, 15, seed=0)
    i1, _ = ds[0]
    i2, _ = ds[0]
    d4 = (i1 - i2).abs().max().item()
    print(f"    Dataset 连续取 [0] 两次  最大差 {d4:.2e}   （必须 > 0）")
    assert d4 > 0.0, "同一个下标两次拿到相同的增强图 → 每个 epoch 都一样"
    print("    ✓ 通过")
    return True


def gate6_rotation_geometry() -> bool:
    """关 6：旋转的几何正确性 —— 用亮点的位置来断言，而不是"看起来对"。

    这是我最想强调的一关。旋转矩阵的符号、坐标系 y 向下、affine_grid
    的 (x,y) 顺序 —— 三处都有符号自由度，靠肉眼"看起来转了"完全不够。
    唯一可靠的办法：放一个已知位置的亮点，断言它转到哪里。
    """
    print("\n[关 6] 旋转的几何正确性（亮点位置断言）")
    x = torch.zeros(1, 1, 28, 28)
    r0, c0 = 14, 20                    # 亮块中心偏右
    x[0, 0, r0 - 1 : r0 + 1, c0 - 1 : c0 + 1] = 1.0

    # 中心不动：把亮点放正中，旋转任何角度都应该还在正中
    xc = torch.zeros(1, 1, 28, 28)
    xc[0, 0, 13:15, 13:15] = 1.0
    outc = warp(xc, make_theta(torch.zeros(1), torch.zeros(1),
                               torch.tensor([37.0]), 28))
    rr = outc[0, 0].sum(dim=1).argmax().item()
    cc = outc[0, 0].sum(dim=0).argmax().item()
    print(f"    中心亮点旋转 37° → 落在 (row={rr}, col={cc})，应仍在 13/14 附近")
    assert abs(rr - 13.5) <= 1.5 and abs(cc - 13.5) <= 1.5, "中心点旋转后位移了"

    # 「右侧的点逆时针转到上方」
    out90 = warp(x, make_theta(torch.zeros(1), torch.zeros(1),
                               torch.tensor([90.0]), 28))
    r1 = out90[0, 0].sum(dim=1).argmax().item()
    c1 = out90[0, 0].sum(dim=0).argmax().item()
    print(f"    右侧亮点 (row={r0}, col={c0}) 转 +90° → (row={r1}, col={c1})")
    print("    预期：跑到**上方**靠近中列。因为 右→上 是逆时针，")
    print("          说明 theta 里的 -sin / +sin 位置是对的")
    assert r1 < r0 - 4, f"该往上方转（row 变小），实际 row={r1}（原 {r0}）"
    assert abs(c1 - 13.5) <= 3, f"应落到中列附近，实际 col={c1}"
    print("    ✓ 通过（符号、坐标系、坐标顺序三处都对得上）")
    return True


def gate7_real_data() -> bool:
    """关 7：真 MNIST 上跑一遍，看统计量被改动了多少 + 验"别增强验证集"。"""
    print("\n[关 7] 真 MNIST 上的统计量与「只增强训练集」")
    from torchvision import datasets, transforms

    from src.utils.config import Config
    from src.utils.paths import ROOT as PROJ_ROOT

    data_dir = PROJ_ROOT / "data"
    if not (data_dir / "MNIST").exists():
        print(f"    跳过：{data_dir / 'MNIST'} 不存在，先跑一次 scripts/train.py")
        return True

    base = datasets.MNIST(root=str(data_dir), train=True, download=False,
                          transform=transforms.ToTensor())
    idx = list(range(0, 2048))
    x = torch.stack([base[i][0] for i in idx])

    g = torch.Generator().manual_seed(0)
    aug = random_affine(x, max_shift=2, max_angle=10, gen=g)

    def stats(t):
        return (
            t.mean().item(),
            t.std().item(),
            (t > 0.5).float().mean().item(),
        )

    m0, s0, f0 = stats(x)
    m1, s1, f1 = stats(aug)
    print(f"    {'':<12}{'均值':>10}{'标准差':>10}{'亮像素占比':>12}")
    print(f"    {'原始':<12}{m0:>10.4f}{s0:>10.4f}{f0:>12.4f}")
    print(f"    {'增强后':<12}{m1:>10.4f}{s1:>10.4f}{f1:>12.4f}")
    print(f"    {'相对变化':<12}{(m1 - m0) / m0 * 100:>9.1f}%"
          f"{(s1 - s0) / s0 * 100:>9.1f}%{(f1 - f0) / f0 * 100:>11.1f}%")
    print("    → 均值几乎不动：MNIST 背景本来就是 0，旋转四角填 0 不改变均值")
    print("    → 标准差略降：双线性插值把笔画边缘磨平了一点，极值被拉低")
    print("      只要变化是个位数百分比，就说明增强没把数据分布改坏")
    print("      （如果标准差掉了一大半，那是增强过猛，图都快糊没了）")
    assert abs(m1 - m0) / m0 < 0.15, "均值变化过大，增强可能过猛"
    assert 0.7 < s1 / s0 < 1.4, f"标准差变化 {s1 / s0:.2f} 倍，超出合理范围"

    # ── 最重要的一条：验证集 / 测试集绝不能被增强 ────────────────────
    from src.data.mnist import build_loaders

    # ⚠️ Day 24 更新：判据从「train_set 是不是被 AugmentedDataset 包了」
    # 改成「train_loader 的 collate_fn 上有没有增强标记」。
    # 因为 Day 24 把增强从 Dataset 层挪到了 collate 层（数值完全等价，
    # 但每轮快 20 秒）。判据必须跟着实现走 —— 否则这一关会一直报
    # "增强没生效"，而增强其实好得很。**关卡本身也会过期。**
    from src.data.batch_aug import has_augment

    cfg = Config(data_dir=str(data_dir), augment_shift=2, augment_angle=10)
    tr, va, te = build_loaders(cfg)
    is_aug = [has_augment(d) for d in (tr, va, te)]
    print(f"    train/val/test 是否被增强：{is_aug}   （必须是 [True, False, False]）")
    assert is_aug == [True, False, False], (
        f"增强装错了地方：{is_aug}。\n"
        "      val/test 一旦被增强，val_loss 会人为变高且抖动变大，\n"
        "      早停会在错误的轮次触发 —— Day 15 已经吃过判据选错的亏。"
    )
    print("    ✓ 通过（只增强训练集，验证/测试保持干净）")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 23 手写数据增强自测")
    ap.add_argument("--train", action="store_true", help="跑全部七关")
    args = ap.parse_args()

    print("=" * 68)
    print("Day 23 · 手写数据增强（随机平移 + 随机旋转）自测")
    print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    print("=" * 68)

    gates = [
        ("关 1 恒等变换", gate1_identity),
        ("关 2 整数平移逐位对拍", gate2_integer_shift),
        ("关 3 每样本独立", gate3_per_sample),
        ("关 4 形状/值域", gate4_shape_range),
        ("关 5 复现性", gate5_determinism),
        ("关 6 旋转几何", gate6_rotation_geometry),
    ]
    if args.train:
        gates.append(("关 7 真数据统计", gate7_real_data))

    ok, passed = True, 0
    for name, fn in gates:
        try:
            if fn():
                passed += 1
        except AssertionError as e:
            print(f"    ✗ 未通过：{e}")
            ok = False
        except Exception as e:                       # noqa: BLE001
            print(f"    ✗ 意外错误：{type(e).__name__}: {e}")
            ok = False

    print("\n" + "=" * 68)
    print(f"{passed}/{len(gates)} 关通过")
    if ok:
        print("下一步：扫增强剂量")
        print('  & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\sweep_augment.py')
    else:
        print("有检查没过，先别扫剂量 —— 增强错了扫出来的曲线全是垃圾")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
