"""Day 10 作业：自定义 MLP + 参数核对 + 三种初始化

用法（在项目根目录）：
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src/models/mlp.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src/models/mlp.py --depth 40 --width 256

今天有四关，按顺序报错。每填完一处就重跑一次，看下一个报什么。
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

from src.utils.logger import get_logger

LOG = ROOT / "outputs" / "day10_mlp.txt"

ACT_MAP = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "none": nn.Identity}


# ═══════════════════════════════════════════════════════════════════
# 1. 自定义 MLP
# ═══════════════════════════════════════════════════════════════════
class MLP(nn.Module):
    """n_layers 层的 MLP。最后一层是输出层，后面不接激活。

    in_dim -> hidden -> hidden -> ... -> out_dim
               └──── n_layers-1 个 hidden 层 ────┘  └─ 输出层 ─┘

    ── Day 15 新增 dropout 参数 ──────────────────────────────────────
    每个 hidden 单元变成  Linear → 激活 → Dropout

    dropout 是「丢弃概率」不是「保留概率」：
        dropout=0.0  不丢（等价于没加，Day 13 用的就是这个）
        dropout=0.2  每次前向随机把 20% 的神经元输出置 0
        dropout=0.5  丢一半（老论文里常用，对 MLP 已经很猛了）
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        n_layers: int,
        act: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError(f"n_layers 至少要 2（一个 hidden + 一个输出），收到 {n_layers}")
        if act not in ACT_MAP:
            raise ValueError(f"act 只能是 {list(ACT_MAP)}，收到 {act!r}")
        # Day 15 新增：概率必须落在 [0, 1) 里。
        # 写 1.0 会让所有神经元永远被丢光（模型输出恒为 0），这个错不报错、
        # 只是训练完全不动 —— 属于"静默失败"，所以在入口就拦掉。
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout 要在 [0, 1) 内，收到 {dropout}")

        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.act_name = act
        self.dropout_p = dropout

        act_cls = ACT_MAP[act]
        # 前 n_layers-1 层：Linear + 激活 (+ Dropout)；最后 1 层：Linear，不接激活
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(last, hidden))
            layers.append(act_cls())
            # ── 顺序：Linear → 激活 → Dropout ─────────────────────
            # 为什么不是 Linear → Dropout → 激活？
            # 两者数学上都成立，但惯例是「丢已经激活过的输出」。原因是
            # ReLU 之后的 0 本来就是死的，再丢它等于没丢（白丢），
            # 丢在 ReLU 前面才等价于「相当于没这个神经元」。
            # 这个顺序差异在实践中影响很小，但面试会问，值得记一下。
            if dropout > 0:
                # nn.Dropout 是**无参数**的层 —— 不占参数量，只改前向行为。
                # 所以 count_params_manual() 一个字都不用改。
                layers.append(nn.Dropout(dropout))
            last = hidden
        layers.append(nn.Linear(last, out_dim))

        # 关键：必须是 nn.ModuleList / nn.Sequential。
        # 普通 list 不会被 __setattr__ 注册，parameters() 会返回空 —— 不报错，但训练不动。
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ═══════════════════════════════════════════════════════════════════
# 2. 手算参数总数（过第一关的凭证）
# ═══════════════════════════════════════════════════════════════════
def count_params_manual(in_dim: int, hidden: int, out_dim: int, n_layers: int) -> int:
    """不算模型，直接按结构算参数个数（含 bias）。

    公式：
        第 1 层    : in_dim * hidden + hidden
        中间层     : hidden * hidden + hidden        （共 n_layers-2 层）
        输出层     : hidden * out_dim + out_dim
    """
    total = in_dim * hidden + hidden                  # 第一层
    for _ in range(n_layers - 2):                     # 中间层
        total += hidden * hidden + hidden
    total += hidden * out_dim + out_dim               # 输出层
    return total

# ═══════════════════════════════════════════════════════════════════
# 3. 三种初始化
# ═══════════════════════════════════════════════════════════════════
def init_weights(model: nn.Module, scheme: str) -> None:
    """按方案初始化所有 nn.Linear 的权重。

    scheme:
        "default"  保持 nn.Linear 自带初始化（什么都不做）
        "xavier"   nn.init.xavier_uniform_(m.weight)
        "kaiming"  nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    三种都把 bias 置零（让方差对比只反映 weight 的差异）。
    """
    if scheme not in ("default", "xavier", "kaiming"):
        raise ValueError(f"scheme 只能是 default/xavier/kaiming，收到 {scheme!r}")

    # 只处理 Linear：ReLU 这类模块没有 weight/bias
    for m in model.modules():
        if not isinstance(m, nn.Linear):
            continue
        if scheme == "xavier":
            nn.init.xavier_uniform_(m.weight)
        elif scheme == "kaiming":
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        # 不属于 if/elif 任何一方：三种 scheme 都要执行
        nn.init.zeros_(m.bias)


# ═══════════════════════════════════════════════════════════════════
# 4. 验证（已给，不用改）
# ═══════════════════════════════════════════════════════════════════
def check_structure(log) -> None:
    log.info("[1] 结构检查")
    model = MLP(in_dim=5, hidden=8, out_dim=3, n_layers=2)
    n_param = sum(p.numel() for p in model.parameters())

    assert n_param > 0, (
        "参数总数是 0 —— 层没有被注册。\n"
        "  最常见原因：用了普通 list 存层，或没有赋值给 self.layers。\n"
        f"  当前 model._modules = {list(model._modules.keys())}"
    )
    log.info(f"    参数总数 = {n_param}   （期望 75）")
    assert n_param == 75, f"MLP(5,8,3,2) 应该有 75 个参数，实际 {n_param}"

    names = [n for n, _ in model.named_parameters()]
    log.info(f"    参数名 = {names}")

    out = model(torch.randn(4, 5))
    assert out.shape == (4, 3), f"前向输出应为 (4,3)，实际 {tuple(out.shape)}"
    log.info(f"    前向 (4,5) -> {tuple(out.shape)}   OK")


def check_param_count(log) -> None:
    log.info("[2] 参数计数：手算 vs 程序")
    cases = [(5, 8, 3, 2), (5, 8, 3, 3), (784, 128, 10, 3), (20, 16, 10, 4)]
    for cfg in cases:
        by_hand = count_params_manual(*cfg)
        model = MLP(cfg[0], cfg[1], cfg[2], cfg[3])
        by_prog = sum(p.numel() for p in model.parameters())
        ok = by_hand == by_prog
        log.info(
            f"    in={cfg[0]:<4} h={cfg[1]:<4} out={cfg[2]:<3} L={cfg[3]}  "
            f"手算={by_hand:<7} 程序={by_prog:<7} {'OK' if ok else 'MISMATCH'}"
        )
        assert ok, (
            f"配置 {cfg} 手算 {by_hand} != 程序 {by_prog}。\n"
            "  逐项核对：第一层 in*h+h、中间层 h*h+h、输出层 h*out+out\n"
            "  最容易漏的是每层的 +h（bias）。"
        )


def check_init_variance(log, depth: int, width: int, batch: int) -> None:
    log.info(f"[3] 三种初始化在 {depth} 层网络里的方差（width={width}, batch={batch}）")
    log.info("    统计的是「每次过完 Linear+激活之后」的方差，所以建 depth+1 层")
    log.info("    （最后一层是输出层、不带激活，不参与统计）")
    head = f"    {'初始化':<10}{'第1层':>12}{'第5层':>12}{'第10层':>12}{'第20层':>12}"
    head += f"{'第' + str(depth) + '层':>12}"
    log.info(head)

    torch.manual_seed(0)
    x0 = torch.randn(batch, width)
    marks = sorted({d for d in (1, 5, 10, 20, depth) if 1 <= d <= depth})

    for scheme in ("default", "xavier", "kaiming"):
        torch.manual_seed(0)
        # depth 个 hidden 层（每层带激活）+ 1 个输出层
        model = MLP(width, width, width, depth + 1, act="relu")
        init_weights(model, scheme)
        model.eval()

        x = x0
        vals = []
        with torch.no_grad():
            idx = 0
            for m in model.layers:
                x = m(x)
                if isinstance(m, (nn.ReLU, nn.Tanh, nn.GELU, nn.Identity)):
                    idx += 1
                    if idx in marks:
                        vals.append(x.var().item())
        log.info(f"    {scheme:<10}" + "".join(f"{v:>12.3e}" for v in vals))

    log.info("    读法：default 每层缩到 ~1/3，到第 40 层接近 0；")
    log.info("          kaiming 把方差稳在 0.5 附近；xavier 配 ReLU 会衰减（假设错配）。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 10 nn.Module 作业")
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()

    log = get_logger("day10", LOG)

    log.info("=" * 68)
    log.info("Day 10 · nn.Module 与参数管理")
    log.info(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    log.info("=" * 68)

    check_structure(log)
    check_param_count(log)
    check_init_variance(log, args.depth, args.width, args.batch)

    log.info("=" * 68)
    log.info("全部通过。三关都过了，说明：")
    log.info("  1. 层被正确注册（parameters() 非空）")
    log.info("  2. 你的参数公式和 PyTorch 的实际结构一致")
    log.info("  3. 三种初始化都真的生效了（否则方差表三行会一样）")
    log.info("=" * 68)
    log.info(f"报告已写入 {LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
