"""配置系统：yaml 文件 → Config 对象。

职责：把「一堆散在文件里的参数」变成一个「有字段、有类型、有默认值」的对象。
这是 Day 3 的交付物，也是 P1 之后每个实验的入口。

用法：
    from src.utils.config import Config, load_config, save_config

    cfg = load_config(CONFIGS / "default.yaml")
    print(cfg.seed)                 # 属性访问，不是 cfg["seed"]

    save_config(cfg, run_dir / "config.yaml")   # 存一份生效配置做备份

────────────────────────────────────────────────────────────────
────────────────────────
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

# 项目根：src/utils/config.py → utils → src → 项目根
ROOT = Path(__file__).resolve().parent.parent.parent

# 允许的取值。写死在这里，比散落在各处的 if 好维护
ALLOWED_DEVICES = ("cpu", "cuda")
ALLOWED_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

# Day 13 新增：训练相关枚举。同样写死在这里，别散落到训练脚本里
ALLOWED_OPTIMIZERS = ("sgd", "adam", "mysgd")   # mysgd = Day 12 手写的那个
ALLOWED_SCHEDULERS = ("none", "step")
ALLOWED_ACTS = ("relu", "tanh", "gelu")
# Day 16 新增：损失函数。myce = 手写 softmax+CE（src/utils/losses.py）
ALLOWED_LOSSES = ("ce", "myce")
# Day 17 新增：线性层。my = 手写 MyLinear（src/utils/layers.py）
ALLOWED_LINEARS = ("torch", "my")
# Day 18 新增：早停判据。两种判据在实测里给出**相反**的结论（Day 15），
# 所以要能选，而且选了哪个必须记进 config.yaml
ALLOWED_ES_METRICS = ("loss", "acc")
# Day 22 新增：卷积层实现。my = 手写 MyConv2d（src/utils/conv.py，七关对拍通过）
ALLOWED_CONVS = ("torch", "my")
# Day 25 新增：BatchNorm 实现。none = 不加，my = 手写 MyBatchNorm2d
ALLOWED_BNS = ("none", "torch", "my")
# Day 26 新增：混合精度。none = fp32，fp16 = autocast + GradScaler
# 为什么不是 "amp: true" 而是枚举：以后要加 bf16 时，多一个取值就行，
# 而 true/false 加第三态只能变成字符串，等于迟早要重构。
ALLOWED_AMPS = ("none", "fp16")
# Day 27 新增：数据后端。
#   pil = torchvision 原样（__getitem__ 里绕一圈 PIL）
#   u8  = 直接索引 uint8 张量，逐样本转（零额外内存，数值逐位相同）
#   f32 = 启动时一次性转 float32 常驻（+179MB，再快 2.5 倍）
ALLOWED_DATA_BACKENDS = ("pil", "u8", "f32")

# ⚠️ 注意这里**没有** ALLOWED_MODELS / ALLOWED_DATASETS。
# 那两张表在 src/engine/registry.py 里，由 builtin.py 注册填充。
# 如果在这里写死 ("mlp", "cnn", ...)，那么 P1 加 CNN 时就要改 config.py ——
# 而 config 是"所有实验都依赖"的文件，改它的风险最高。
# 现在这样：加模型只动 builtin.py 一个文件，config 完全不知道有哪些模型。
#
# 代价是「模型名拼错」不会在读配置那一刻报错，而是在 Trainer.build() 里报。
# 这个代价可以接受，因为：(1) 它仍在训练的第一秒内，不是第 1000 步；
# (2) registry 的报错信息带「已注册的有 [...]」，比这里能给的更有用。


# ═══════════════════════════════════════════════════════════════════
# 0b. 小工具：把 "16,32" 解析成 (16, 32)
# ═══════════════════════════════════════════════════════════════════
def parse_channels(value) -> tuple[int, ...]:
    """把配置里的 channels 解析成整数元组。

    接受四种写法（都是为了让人少踩坑）：
        "16,32"        yaml / --set 里的标准写法
        "16, 32"       带空格 —— yaml 里手写很容易带
        [16, 32]       yaml 里写成列表（safe_load 会读成 Python list）
        16             单个整数 → 解析成 (16,)

    为什么不在 yaml 里直接写列表就完事：
        `--set channels=8,16` 从命令行进来**永远是字符串**。
        如果 Config 里声明成 list、yaml 里写 list，
        那么 yaml 那条路是 list、--set 那条路是 str，两条路类型不一致。
        类型不一致的配置是最难查的 bug —— 它在某个分支里才会炸。
        统一用字符串 + 一个解析函数，两条路就都通了。
    """
    if isinstance(value, (list, tuple)):
        parts = [int(v) for v in value]
    else:
        # 顺手支持中文逗号：中文输入法下打 , 和 ，长得几乎一样
        text = str(value).replace("，", ",")
        raw = [p.strip() for p in text.split(",") if p.strip()]
        try:
            parts = [int(p) for p in raw]
        except ValueError:
            raise ValueError(
                f"channels 要写成逗号分隔的整数，比如 \"16,32\"，收到 {value!r}"
            ) from None

    if not parts:
        raise ValueError(f"channels 不能为空，收到 {value!r}")
    if any(p < 1 for p in parts):
        raise ValueError(f"channels 里每个通道数都必须 >= 1，收到 {value!r}")
    return tuple(parts)


# ═══════════════════════════════════════════════════════════════════
# 1. 配置的形状
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Config:
    """训练配置。

    这里的每个 `= 默认值` 都很重要：yaml 里没写的键就靠它补上。
    所以你**不必**在 yaml 里写全，只写跟默认值不一样的那几个就行。
    """

    seed: int = 0
    device: str = "cuda"
    output_dir: str = "outputs"
    log_level: str = "INFO"

    # ── Day 18 新增：选模型和数据集（通用模板的核心两行）──────────
    # 换成别的实现时，只改这两个字符串，**训练脚本一个字都不用动**。
    # 可选值见 `python scripts/train.py --list`（从注册表实时读，不会过时）
    dataset: str = "mnist"
    model: str = "mlp"
    # 要不要把输入摊平成一维。
    #   MLP 必须摊平：(B,1,28,28) → (B,784)
    #   CNN 必须不摊：卷积要的就是 28×28 的空间结构
    # 为什么不做成自动判断：框架无法知道"这个模型还要不要空间信息"。
    # 摊平后相邻像素的关系就永久丢了 —— 这也是 MLP 卡在 98% 的原因之一。
    flatten: bool = True

    # ── Day 13 新增：数据 ──────────────────────────────────────────
    data_dir: str = "data"          # MNIST 存在哪（相对项目根）
    batch_size: int = 128           # 每批多少条
    val_size: int = 6000            # 从 60000 里切多少条当验证集
    num_workers: int = 0            # 读数据的子进程数（Windows 上先留 0）

    # ── Day 27 新增：数据后端 ─────────────────────────────────────
    # 为什么默认用 u8 而不是原样 pil：
    # 实测 u8 与 pil **逐位相同**（差 0.00e+00，见 Day 27 关 1/关 3），
    # 额外内存 0，速度快 2.74 倍。**没有任何取舍的改动，就不该留成可选项的默认关。**
    #
    # 要不要升到 f32：再快 2.5 倍（端到端 6.9x），代价是 179MB 常驻。
    # 这个是有取舍的（内存 vs 时间），所以不设成默认，要的人自己开。
    #
    # 默认值跟着实测走 —— 和 amp 默认 none 是同一条原则的两个方向。
    data_backend: str = "u8"

    # ── Day 13 新增：模型 ──────────────────────────────────────────
    hidden: int = 256               # 隐藏层宽度
    n_layers: int = 2               # 总层数（含输出层）
    act: str = "relu"               # 激活函数：relu / tanh / gelu

    # ── Day 15 新增：正则化 ────────────────────────────────────────
    dropout: float = 0.0            # 丢弃概率（0 = 不加，Day 13 就是 0）
    early_stop_patience: int = 0    # 验证 loss 连续几轮不降就停（0 = 不停）
    # Day 18 从命令行挪进配置：以前它只存在于 argparse，
    # 导致"按 loss 早停"和"按 acc 早停"两次跑出来的 config.yaml
    # 一模一样 —— 半年后根本分不清哪个是哪个。
    # 凡是**会影响训练结果**的东西，都必须进 config.yaml。
    early_stop_metric: str = "loss"   # loss / acc

    # ── Day 16 新增：损失函数 ──────────────────────────────────────
    # ce   = nn.CrossEntropyLoss（PyTorch 官方实现）
    # myce = 手写 softmax + 交叉熵（src/utils/losses.py，七关对拍通过）
    # 两者数值上逐位一致，切它只是为了证明"你真的会自己写"
    loss: str = "ce"

    # ── Day 17 新增：线性层实现 ────────────────────────────────────
    # torch = nn.Linear
    # my    = 手写 MyLinear（src/utils/layers.py，七关对拍通过）
    # 换它比换 loss 更彻底：这个模型里所有全连接层都会被替掉
    linear: str = "torch"

    # ── Day 22 新增：卷积（CNN）────────────────────────────────────
    # 两个卷积层的输出通道数。写成 "16,32" 这种字符串而不是 list，
    # 是为了让 `--set channels=8,16` 也能用 —— 命令行传进来的永远是字符串，
    # 如果这里声明成 list，--set 就会塞一个字符串进去，类型悄悄不一致。
    # 解析统一走 parse_channels()，解析不了当场报错。
    channels: str = "16,32"
    in_channels: int = 1            # 输入通道数：MNIST 灰度图 = 1，RGB = 3
    # torch = nn.Conv2d
    # my    = 手写 MyConv2d（Day 22 七关对拍通过，30 步参数差 2.98e-08）
    conv: str = "torch"
    # ── Day 25 新增：BatchNorm ─────────────────────────────────────
    # none  = 不加
    # torch = nn.BatchNorm2d
    # my    = 手写 MyBatchNorm2d（Day 25 七关对拍通过）
    # 三个取值合成一个字段而不是「开关 + 实现」两个字段：
    # "不加 BN" 本身也是一种实现，拆成两个字段会出现
    # enable=False 却 bn_impl="my" 这种自相矛盾的组合。
    bn: str = "none"

    # ── Day 26 新增：混合精度 ─────────────────────────────────────
    # none = 全程 fp32；fp16 = autocast(fp16) + GradScaler
    # 默认 none：这台机器（GTX 1660 Ti，sm_75 的 TU116）实测 amp 比 fp32 **慢**
    # 1.5~4 倍，详见 docs/Day26-混合精度.md。默认值应该跟着实测走，
    # 不该跟着"amp 更快"这个流行说法走。
    amp: str = "none"

    # ── Day 26 新增：梯度裁剪 ──────────────────────────────────────
    # 0 = 关闭。为什么今天才加：amp 的 unscale_ 必须在 clip **之前**，
    # 没有 clip 就没法把这个坑讲清楚（Day 26 关 6）。
    # 另外 P2 的 PPO 会强制用到梯度裁剪，这个字段那时候直接能用。
    grad_clip: float = 0.0

    # ── Day 23 新增：数据增强（只作用于训练集）────────────────────
    # Day 22 的 CNN 在 ep7 之后 val_loss 回升 39%，是真过拟合。
    # Day 15 已经排除「加正则化」这条路（在 MNIST 上会让精度下降），
    # 剩下的就是「加数据约束」= 数据增强。
    #
    # 单位是**像素**和**度**，不用 torchvision 那种比例 ——
    # 比例写法在 28×28 上看不出到底动了多少，而且没法表达"就动 1 个像素"，
    # 那样就没法扫剂量（Day 15 的教训：不扫曲线就不用单点下结论）。
    #
    # ⚠️ 这两个值只对 **train_loader** 生效。
    #    val/test 永远不增强 —— 否则 val_loss 会被人为抬高、抖动变大，
    #    早停会在错误的轮次触发，整个 Day 23 的分析全部作废。
    augment_shift: float = 0.0      # 最大随机平移（像素），0 = 关闭
    augment_angle: float = 0.0      # 最大随机旋转（度），0 = 关闭

    # ── Day 13 新增：优化 ──────────────────────────────────────────
    optimizer: str = "sgd"          # sgd / adam / mysgd（自己手写的那个）
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0
    epochs: int = 10
    scheduler: str = "none"         # none / step
    step_size: int = 4              # 每几轮把 lr 乘一次 gamma
    gamma: float = 0.5

    def __post_init__(self) -> None:
        """字段赋值完成后自动调用 —— 校验的天然位置。

        为什么校验要放在这里，而不是等到训练时才发现：
        配错 device 如果拖到第 1000 步才报错，你已经浪费了半小时 GPU 时间。
        **配置的错，要在读配置的那一刻就炸。**
        """
        if self.device not in ALLOWED_DEVICES:
            raise ValueError(
                f"device 只能是 {ALLOWED_DEVICES} 之一，你给的是 {self.device!r}"
            )


        if self.log_level not in ALLOWED_LOG_LEVELS:
            raise ValueError(
                f"log_level 只能是 {ALLOWED_LOG_LEVELS} 之一，你给的是 {self.log_level!r}"
            )

        # ── Day 13 新增的校验 ──────────────────────────────────────
        # 配错一个字符串，如果拖到训练第 10 分钟才炸，那 10 分钟就白跑了。
        if self.optimizer not in ALLOWED_OPTIMIZERS:
            raise ValueError(
                f"optimizer 只能是 {ALLOWED_OPTIMIZERS} 之一，你给的是 {self.optimizer!r}"
            )

        if self.scheduler not in ALLOWED_SCHEDULERS:
            raise ValueError(
                f"scheduler 只能是 {ALLOWED_SCHEDULERS} 之一，你给的是 {self.scheduler!r}"
            )

        if self.act not in ALLOWED_ACTS:
            raise ValueError(f"act 只能是 {ALLOWED_ACTS} 之一，你给的是 {self.act!r}")

        # ── Day 16 新增的校验 ──────────────────────────────────────
        if self.loss not in ALLOWED_LOSSES:
            raise ValueError(
                f"loss 只能是 {ALLOWED_LOSSES} 之一（myce = 手写版），你给的是 {self.loss!r}"
            )

        # ── Day 17 新增的校验 ──────────────────────────────────────
        if self.linear not in ALLOWED_LINEARS:
            raise ValueError(
                f"linear 只能是 {ALLOWED_LINEARS} 之一（my = 手写版），你给的是 {self.linear!r}"
            )

        # ── Day 18 新增的校验 ──────────────────────────────────────
        # 这里只查「是不是非空字符串」，不查「是不是已注册的模型」——
        # 原因见文件顶部 ALLOWED_MODELS 那段注释（避免 config 依赖具体实现）。
        if not self.model or not isinstance(self.model, str):
            raise ValueError(f"model 必须是非空字符串，你给的是 {self.model!r}")
        if not self.dataset or not isinstance(self.dataset, str):
            raise ValueError(f"dataset 必须是非空字符串，你给的是 {self.dataset!r}")
        if not isinstance(self.flatten, bool):
            # 从 yaml 读进来的 "true"/"false" 会被 safe_load 解析成 bool，
            # 但如果有人写成字符串 "False"，这里就要拦住 ——
            # 非空字符串在 if 里恒为真，`flatten="False"` 会静默当成 True。
            raise ValueError(f"flatten 必须是 bool（true/false），你给的是 {self.flatten!r}")
        if self.early_stop_metric not in ALLOWED_ES_METRICS:
            raise ValueError(
                f"early_stop_metric 只能是 {ALLOWED_ES_METRICS} 之一"
                f"（推荐 loss，见 Day 15 的实测对比），你给的是 {self.early_stop_metric!r}"
            )

        # 范围检查：数值型的错（batch_size=0、lr 是负数）也会一路错到训练中期
        if self.batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1，你给的是 {self.batch_size}")
        if self.epochs < 1:
            raise ValueError(f"epochs 必须 >= 1，你给的是 {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr 必须 > 0，你给的是 {self.lr}")
        if not 0 <= self.val_size < 60000:
            raise ValueError(f"val_size 要在 [0, 60000) 内，你给的是 {self.val_size}")

        # ── Day 15 新增：正则化参数 ────────────────────────────────
        # dropout 写 1.0 → 神经元全被丢光，模型恒输出 0，训练完全不动，
        # 但**不报错**。这种"静默失效"必须靠在入口校验来挡。
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout 要在 [0, 1) 内，你给的是 {self.dropout}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay 不能为负，你给的是 {self.weight_decay}")
        if self.early_stop_patience < 0:
            raise ValueError(
                f"early_stop_patience 不能为负（0 表示关闭），你给的是 {self.early_stop_patience}"
            )

        # ── Day 22 新增：CNN 相关校验 ──────────────────────────────
        if self.conv not in ALLOWED_CONVS:
            raise ValueError(
                f"conv 只能是 {ALLOWED_CONVS} 之一（my = 手写版），你给的是 {self.conv!r}"
            )
        # ── Day 25 新增：BN 校验 ──────────────────────────────────
        if self.bn not in ALLOWED_BNS:
            raise ValueError(
                f"bn 只能是 {ALLOWED_BNS} 之一"
                f"（none = 不加，my = 手写版），你给的是 {self.bn!r}"
            )
        # BN 只对 CNN 有效。MLP 上加 BN2d 会在第一次 forward 报"需要 4 维输入"，
        # 但那个报错发生在第 0 秒之后、而且信息指向形状而不是配置 ——
        # 在这里拦一下，报错直接说"你给 mlp 配了 bn"。
        if self.bn != "none" and self.model != "cnn":
            raise ValueError(
                f"bn={self.bn!r} 只能配 model='cnn'（BatchNorm2d 需要 4 维输入），"
                f"当前 model={self.model!r}。\n"
                "  如果确实想在 MLP 上加归一化，那应该用 BatchNorm1d，"
                "  目前没有实现，不要用 bn 字段凑。"
            )
        if self.in_channels < 1:
            raise ValueError(f"in_channels 必须 >= 1，你给的是 {self.in_channels}")

        # ── Day 26 新增：amp 与梯度裁剪校验 ────────────────────────
        if self.amp not in ALLOWED_AMPS:
            raise ValueError(
                f"amp 只能是 {ALLOWED_AMPS} 之一（none = 全程 fp32），"
                f"你给的是 {self.amp!r}"
            )
        if self.grad_clip < 0:
            raise ValueError(
                f"grad_clip 不能为负（0 表示关闭），你给的是 {self.grad_clip}"
            )
        # amp 只在 cuda 上有意义。这里不拦，因为 Trainer 会自动降级并打印 ——
        # 但要在配 yaml 时就让人看见：配了 amp 又配 device=cpu 是白配。
        if self.amp != "none" and self.device == "cpu":
            raise ValueError(
                f"amp={self.amp!r} 需要 CUDA，但 device='cpu'。\n"
                "  fp16 的 autocast 在 CPU 上走的是另一套实现（甚至可能更慢），"
                "  不打算支持。要么把 device 改成 cuda，要么把 amp 改成 none。"
            )
        # 在入口就把 channels 解析一次。解析不了 = 配置写错，当场报。
        # 好处是后面 build_cnn 拿到的一定是合法的两个正整数。
        parse_channels(self.channels)

        # ── Day 23 新增：增强幅度校验 ──────────────────────────────
        # 这里只拦"明显是写错了"的情况（负数、角度 > 180）。
        # "平移 20 像素会不会把图推出画面"这种要结合图像边长才能判断，
        # 放到 build_loaders 里用 IMAGE_SIZE 校验 —— config 不该知道数据集多长。
        if self.augment_shift < 0:
            raise ValueError(
                f"augment_shift 不能为负（0 表示关闭），你给的是 {self.augment_shift}"
            )
        if not 0 <= self.augment_angle < 180:
            # 转 180° 和转 0° 对 MNIST 的数字来说几乎一样（6↔9 除外），
            # 再大就没有"几何扰动"的意义了，多半是写错了单位（弧度当度）。
            raise ValueError(
                f"augment_angle 要在 [0, 180) 度内，你给的是 {self.augment_angle}"
                "（单位是度，不是弧度）"
            )
        # ── Day 27 新增：数据后端校验 ──────────────────────────────
        if self.data_backend not in ALLOWED_DATA_BACKENDS:
            raise ValueError(
                f"data_backend 只能是 {ALLOWED_DATA_BACKENDS} 之一"
                f"（pil = 原样走 PIL，u8/f32 = 直索引张量），"
                f"你给的是 {self.data_backend!r}"
            )
        if self.num_workers < 0:
            raise ValueError(f"num_workers 不能为负，你给的是 {self.num_workers}")

        # ─────────────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════
# 2. 读 yaml
# ═══════════════════════════════════════════════════════════════════
def load_yaml(path: Path) -> dict:
    """读 yaml 文件，返回字典。空文件返回空字典。"""
    text = path.read_text(encoding="utf-8")

    # 用 safe_load，不用 load：load 能构造任意 Python 对象，
    # 等于「配置文件可以执行代码」，配置文件是不该有这种权力的。
    data = yaml.safe_load(text)

    # 空文件时 safe_load 返回 None，不是 {}。
    # 不兜底的话，后面 data.items() 会 AttributeError。
    return data or {}


# ═══════════════════════════════════════════════════════════════════
# 3. yaml → Config
# ═══════════════════════════════════════════════════════════════════
def load_config(path: Path, overrides: dict | None = None) -> Config:
    """读 yaml，返回 Config 对象。

    yaml 里没写的键，用 Config 里的默认值补上 —— 靠的是
    `Config(**{"seed": 1})` 这种只传部分键的写法，没传的字段自动取默认值。

    参数：
        overrides: 覆盖项，一般来自命令行。优先级高于 yaml。
                   传 None（默认）表示不覆盖。
                   注意：这里**只做合并**，合法性仍由 Config 的
                   __post_init__ 统一校验 —— 覆盖进来的错值一样会被拦。
    """
    raw = load_yaml(path)

    # 后到者胜：命令行盖在 yaml 上。
    # 因为调用方只传「用户真的传了」的键，所以没传的键在 yaml 里的值不受影响。
    if overrides:
        raw.update(overrides)

    known = {f.name for f in fields(Config)}
    unknown = sorted(set(raw) - known)

    if unknown:
        print(f"⚠️  配置里有不认识的键（多半是拼错了）: {unknown}", file=sys.stderr)


    cfg = Config(**{k: v for k, v in raw.items() if k in known})

    # ── 把「相对路径」按项目根解析成绝对路径 ──────────────────────
    # 【为什么必须在这里做】
    # yaml 里写的是 data_dir: "data"，这是个相对路径。相对路径是相对于
    # **当前工作目录（cwd）** 解析的，而 cwd 取决于你在哪个目录敲的命令：
    #
    #     在项目根跑   → <项目根>/data/          ✅
    #     在 scripts/ 跑 → <项目根>/scripts/data/  ❌ 又下了一份 64 MB
    #
    # 这个坑真的踩过：Day 13 跑完发现 scripts/data/MNIST 多出一份，
    # 磁盘上同一份数据存了两个地方。Day 2 写 paths.py 的目的就是
    # 「不依赖 cwd」，结果 config 这里又退回了相对路径 —— 现在补上。
    #
    # 一律转成 str，因为 dataclass 里声明的类型是 str，读代码的人
    # 不该遇到「有时是 str 有时是 Path」的惊喜。
    for name in ("data_dir", "output_dir"):
        p = Path(getattr(cfg, name))
        if not p.is_absolute():
            setattr(cfg, name, str(ROOT / p))

    return cfg


# ═══════════════════════════════════════════════════════════════════
# 4. Config → yaml（存一份生效配置做备份）
# ═══════════════════════════════════════════════════════════════════
def save_config(cfg: Config, path: Path) -> None:
    """把配置写回 yaml 文件。

    为什么需要它：一次实验跑完，你手里只有一堆日志。
    半年后想复现「当时那个好结果」，得知道**那一跑到底用了什么参数**。
    每个输出目录里存一份 config.yaml，就是为这件事。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    text = yaml.safe_dump(
        asdict(cfg),
        allow_unicode=True,   # 不写这个，中文会被转成 \u4e2d 这种转义
        sort_keys=False,      # 保持 dataclass 里定义的字段顺序
     )
    path.write_text(text, encoding="utf-8")
    # asdict 已经在文件顶部 import 了，直接用。
    # ─────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg_path = ROOT / "configs" / "default.yaml"

    print(f"[1] 读配置: {cfg_path}")
    cfg = load_config(cfg_path)
    print(f"    → {cfg}")
    print()

    print("[2] 属性访问（不是 cfg['seed']）")
    print(f"    cfg.seed       = {cfg.seed!r}")
    print(f"    cfg.device     = {cfg.device!r}")
    print(f"    cfg.output_dir = {cfg.output_dir!r}")
    print(f"    cfg.log_level  = {cfg.log_level!r}")
    print()

    print("[3] 转回字典")
    print(f"    {asdict(cfg)}")
    print()

    print("[4] 写回 yaml 备份")
    out = ROOT / "outputs" / "_config_selftest.yaml"
    save_config(cfg, out)
    print(f"    已写入 {out}")
    print("    内容：")
    for line in out.read_text(encoding="utf-8").splitlines():
        print(f"      {line}")
