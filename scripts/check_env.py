"""环境自检：16 项，一次跑完，逐项告诉你哪里不对。

用法（在项目根目录或任意目录都行，脚本自己定位项目根）：

    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\check_env.py

期望末行：`结果：16/16 项通过`

【为什么要写这个脚本】
深度学习环境出问题的方式绝大多数是**静默的**：
  - PATH 里的 python 是另一个没装 torch 的版本 → 报 ModuleNotFoundError，但你可能
    以为是自己代码写错了
  - 装了 GPU 版 torch 但驱动太旧 → 训练能跑，只是悄悄地全在 CPU 上，慢 30 倍，
    或者报一个看不懂的 CUDA error
  - sm_75 上用了 bf16 → 不报错，但走的是软件模拟，比 fp32 还慢
  - matplotlib 缺中文字体 → 图上全是方块，不报错
这四类问题靠「打开 PyCharm 看看」是发现不了的。所以每天开工前跑一次，
把「我以为环境是对的」换成「我验证过环境是对的」。
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

# ── 让脚本在任意 cwd 下都能 import src.* ──────────────────────────────
# __file__ 是 scripts/check_env.py，.parent 是 scripts/，.parent.parent 是项目根
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows 控制台默认编码可能是 GBK，中文和特殊符号（✓✗）会炸。
# reconfigure 是 3.7+ 才有的，加 try 是为了兼容老解释器（本脚本本身
# 就是用来诊断"解释器对不对"的，所以不能要求解释器有多新）。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── 打分器：所有检查都往这里记结果 ────────────────────────────────────
RESULTS: list[tuple[bool, str, str]] = []   # (通过?, 项目名, 细节)


def check(ok: bool, name: str, detail: str = "", warn: bool = False) -> bool:
    """记一项检查。

    ok=False 且 warn=True → 记为「警告」，不计入失败（比如 bf16 是模拟的，
    这不是环境坏了，只是要记住别用）。
    """
    tag = "✓" if ok else ("!" if warn else "✗")
    RESULTS.append((ok or warn, name, detail))
    line = f"  {tag} {name:<34}"
    if detail:
        line += f"  {detail}"
    print(line)
    return ok


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 62 - len(title)))


def main() -> int:
    print("=" * 72)
    print("  rl-vla-journey 环境自检")
    print("=" * 72)

    # ══════════════════════════════════════════════════════════
    # 组 A：解释器（3 项）
    # 这组不过，后面的全部检查都没有意义 —— 你在检查一个错误的 Python
    # ══════════════════════════════════════════════════════════
    section("A. 解释器")

    exe = Path(sys.executable)
    # 判据：路径里要有 envs\pytorch。这台机器上 PATH 里的 python 指向
    # C:\Users\...\Programs\Python\Python312，那个没装 torch。
    ok_env = "envs" in exe.parts and "pytorch" in str(exe).lower()
    check(
        ok_env,
        "解释器在 conda 的 pytorch 环境内",
        f"{exe}",
    )

    ver = sys.version_info
    check(
        ver >= (3, 10) and platform.machine().endswith("64"),
        "Python >= 3.10（64 位）",
        f"{ver.major}.{ver.minor}.{ver.micro} · {platform.system()} · {platform.machine()}",
    )

    # ══════════════════════════════════════════════════════════
    # 组 B：核心依赖（4 项）
    # ══════════════════════════════════════════════════════════
    section("B. 依赖库")

    import torch

    torch_major = int(torch.__version__.split(".")[0])
    check(torch_major >= 2, "torch >= 2.0", torch.__version__)

    import numpy as np
    np_major, np_minor = (int(x) for x in np.__version__.split(".")[:2])
    check(
        (np_major, np_minor) >= (1, 24),
        "numpy >= 1.24",
        np.__version__,
    )

    import matplotlib
    check(True, "matplotlib 可用", matplotlib.__version__)

    import yaml
    check(True, "PyYAML 可用", yaml.__version__)

    # ══════════════════════════════════════════════════════════
    # 组 C：GPU（5 项）
    # 这组最容易「看起来对其实不对」，所以每一项都真的跑一次
    # ══════════════════════════════════════════════════════════
    section("C. GPU 与精度")

    # 编译进去的 CUDA 版本，和驱动支持的是两件事
    cuda_ver = torch.version.cuda
    check(cuda_ver is not None, "torch 是 CUDA 版（不是 cpu-only）", f"编译 CUDA {cuda_ver}")

    avail = torch.cuda.is_available()
    check(avail, "torch.cuda.is_available()", "True" if avail else "False（会静默退到 CPU）")

    if avail:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        sm = f"sm_{cap[0]}{cap[1]}"
        # 算力等级不是「过/不过」，是「决定你能用什么」：
        # sm_75 (Turing) 有 fp16 硬件加速，但没有 Tensor Core 的 bf16
        check(
            True,
            "显卡算力等级",
            f"{name} · {sm}"
            + ("  ← 支持 fp16，不支持原生 bf16" if cap == (7, 5) else ""),
        )

        check(
            torch.backends.cudnn.is_available(),
            "cuDNN 可用",
            str(torch.backends.cudnn.version()),
        )

        # 真跑一次半精度矩阵乘。只查 is_available 不够 ——
        # 有的环境 CUDA 能识别，跑到 fp16 算子时才炸
        try:
            a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
            b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
            c = (a @ b).float()
            ok_fp16 = torch.isfinite(c).all().item()
            check(ok_fp16, "fp16 矩阵乘实测通过", "64×64 无溢出")
        except Exception as e:
            check(False, "fp16 矩阵乘实测通过", f"{type(e).__name__}: {e}")

        # bf16：Turing 上 is_bf16_supported() 返回 True 是**软件模拟**，
        # 用了反而慢。这里不作判据，只提示，避免以后误用。
        try:
            bf16_supported = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_supported = False
        if cap == (7, 5):
            check(
                True,
                "bf16 可用性（仅提示）",
                f"is_bf16_supported()={bf16_supported} 但 sm_75 上是模拟实现，别用",
            )
    else:
        # 没 GPU 时把 4 项标成未通过，让总数仍然是 16
        for nm in ("显卡算力等级", "cuDNN 可用", "fp16 矩阵乘实测通过", "bf16 可用性（仅提示）"):
            check(False, nm, "跳过：CUDA 不可用", warn=True)

    # ══════════════════════════════════════════════════════════
    # 组 D：功能自检（2 项）
    # 到这一步环境已经「看起来」没问题了，但要确认它真的会算
    # ══════════════════════════════════════════════════════════
    section("D. 功能自测")

    # autograd 对拍：y = x²，dy/dx = 2x。手算和框架算必须一致。
    # 这一项能抓出「torch 装是装了但 autograd 后端坏了」这种怪情况
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    err = (x.grad - 2 * torch.tensor([1.0, 2.0, 3.0])).abs().max().item()
    check(err < 1e-6, "autograd 梯度对拍", f"max|2x - grad| = {err:.2e}")

    # 中文字体：缺字体时 plt 不报错，图上全是方块。先查字体表里有没有，
    # 再真正渲染一次 —— 有的环境字体表里有名字但文件是坏的
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    names = {f.name for f in font_manager.fontManager.ttflist}
    has_cn = bool({"SimHei", "Microsoft YaHei"} & names)
    if has_cn:
        font = "SimHei" if "SimHei" in names else "Microsoft YaHei"
        try:
            fig, ax = plt.subplots(figsize=(2, 1))
            ax.set_title("中文测试", fontname=font)
            fig.canvas.draw()
            plt.close(fig)
            check(True, "中文字体可渲染", f"{font}（图上不会出现方块）")
        except Exception as e:
            check(False, "中文字体可渲染", f"{type(e).__name__}: {e}")
    else:
        check(
            False,
            "中文字体可渲染",
            "字体表里找不到 SimHei / Microsoft YaHei，画图标题请用英文",
            warn=True,
        )

    # ══════════════════════════════════════════════════════════
    # 组 E：项目结构（2 项）
    # ══════════════════════════════════════════════════════════
    section("E. 项目结构")

    need_dirs = ["src/utils", "src/models", "src/data", "configs", "scripts"]
    missing = [d for d in need_dirs if not (ROOT / d).is_dir()]
    check(
        not missing,
        "关键目录齐全",
        "全部存在" if not missing else f"缺失 {missing}",
    )

    # 配置能读出来才算真的齐 —— 目录在、文件空，也是坏的
    try:
        from src.utils.config import load_config
        cfg = load_config(ROOT / "configs" / "mnist_mlp.yaml")
        # data_dir 必须是绝对路径（Day 15 修过的那个 bug：相对路径会跟着 cwd 跑）
        ok_abs = Path(cfg.data_dir).is_absolute()
        check(
            ok_abs,
            "配置文件可解析 + 路径已绝对化",
            f"hidden={cfg.hidden} epochs={cfg.epochs} loss={cfg.loss}"
            + ("" if ok_abs else f"  ← data_dir={cfg.data_dir} 是相对路径"),
        )
    except Exception as e:
        check(False, "配置文件可解析 + 路径已绝对化", f"{type(e).__name__}: {e}")

    # ══════════════════════════════════════════════════════════
    # 汇总
    # ══════════════════════════════════════════════════════════
    n_pass = sum(1 for ok, _, _ in RESULTS if ok)
    n_all = len(RESULTS)
    print("\n" + "=" * 72)
    if n_pass == n_all:
        print(f"  结果：{n_pass}/{n_all} 项通过 —— 环境就绪，可以开工。")
    else:
        print(f"  结果：{n_pass}/{n_all} 项通过 —— 下面这些没过：")
        for ok, name, detail in RESULTS:
            if not ok:
                print(f"    ✗ {name}  {detail}")
    print("=" * 72)

    # 顺手把解释器路径打出来：出问题时最容易搞错的就是「我到底在哪个 python 里」
    print(f"  本次使用解释器：{sys.executable}")
    print(f"  项目根目录    ：{ROOT}")
    return 0 if n_pass == n_all else 1


if __name__ == "__main__":
    sys.exit(main())
