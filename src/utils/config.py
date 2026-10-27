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
        if self.in_channels < 1:
            raise ValueError(f"in_channels 必须 >= 1，你给的是 {self.in_channels}")
        # 在入口就把 channels 解析一次。解析不了 = 配置写错，当场报。
        # 好处是后面 build_cnn 拿到的一定是合法的两个正整数。
        parse_channels(self.channels)
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
