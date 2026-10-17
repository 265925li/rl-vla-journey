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
