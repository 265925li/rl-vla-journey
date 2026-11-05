"""probe_film.py — FiLM 条件化的五个可验证事实。

配套讲义：docs/VLA-FiLM与EfficientNet.md

回答的问题：
  A 参数量到底是多少？FiLM 加了多少？（不信"很轻量"这种形容词，算出来）
  B "逐通道广播"具体是什么意思？拿 std 验证。
  C identity 初始化真的能做到「插入前后输出逐位相同」吗？
  D 零初始化会不会像 ReLU 死亡那样把梯度卡死？
  E 四种条件注入方式的参数账，以及 FiLM vs cross-attention。

约定：只用算术和一次 backward，不训练。每个数字都打印出来，讲义里照抄。
运行：
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\probe_film.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\probe_film.py --only C
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3

D = 512          # 指令嵌入维度（RT-1 用 Universal Sentence Encoder，512 维）
DEV = "cpu"      # 参数量对账不需要 GPU


# ---------------------------------------------------------------- FiLM 本体
class FiLM(nn.Module):
    """逐通道仿射调制：y_c = (1 + γ_c) · x_c + β_c

    γ、β 由条件向量 z 经一个 Linear 算出来 —— 所以它们是**动态的**，
    这一点和 BN 的 γ、β（训练完就固定）本质不同。
    """

    def __init__(self, channels: int, cond_dim: int = D, identity_init: bool = False):
        super().__init__()
        self.channels = channels
        self.to_gamma = nn.Linear(cond_dim, channels)
        self.to_beta = nn.Linear(cond_dim, channels)
        if identity_init:
            # γ ≡ 0、β ≡ 0  ⟹  y = (1+0)·x + 0 = x
            for layer in (self.to_gamma, self.to_beta):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def coefficients(self, z):
        return self.to_gamma(z), self.to_beta(z)

    def forward(self, x, z):
        gamma, beta = self.coefficients(z)
        # (B, C) → (B, C, 1, 1)  ⟹ 沿 H、W 广播，同一通道所有位置乘同一个数
        return (1.0 + gamma)[:, :, None, None] * x + beta[:, :, None, None]


# ---------------------------------------------------------------- 实验 A
def run_a() -> dict:
    """参数量对账。分类头不算 —— RT-1 不用它。"""
    m = efficientnet_b3(weights=None)
    backbone = sum(p.numel() for p in m.features.parameters())

    # 逐块统计 MBConv 块的输入通道，用来算 FiLM 增量
    block_cin: list[int] = []
    for stage in m.features:
        if isinstance(stage, nn.Sequential) and len(stage) > 0 and hasattr(stage[0], "block"):
            for blk in stage:
                block_cin.append(blk.block[0][0].in_channels)

    film_params = sum(2 * (c * D + c) for c in block_cin)   # γ 和 β 各一套
    proj_params = 1536 * 512 + 512                           # 末层 1×1 卷积压到 512 维
    total = backbone + film_params + proj_params

    return {
        "n_blocks": len(block_cin),
        "sum_cin": sum(block_cin),
        "backbone": backbone,
        "film": film_params,
        "proj": proj_params,
        "total": total,
        "film_share": film_params / backbone,
    }


# ---------------------------------------------------------------- 实验 B
def run_b() -> dict:
    """「逐通道」到底逐的是什么：验证每个通道被整体缩放，空间上完全一致。"""
    torch.manual_seed(0)
    b, c, h, w = 2, 8, 5, 5
    x = torch.randn(b, c, h, w)
    film = FiLM(c, D)
    z = torch.randn(b, D)
    gamma, beta = film.coefficients(z)
    y = film(x, z)

    # 若真是「每个通道乘一个数」，那么
    #   (y − β)/x 在每个通道内应当处处相等，且等于 1+γ
    ratio = (y - beta[:, :, None, None]) / x
    per_channel_spread = (ratio - ratio.mean(dim=(2, 3), keepdim=True)).abs().max().item()
    expect = (1.0 + gamma)
    ratio_err = (ratio.mean(dim=(2, 3)) - expect).abs().max().item()

    # std 也被同一因子缩放：std(y_c) = |1+γ_c| · std(x_c)
    std_ratio = (y.std(dim=(2, 3)) / x.std(dim=(2, 3)))
    std_err = (std_ratio - expect.abs()).abs().max().item()

    return {
        "gamma_sample": gamma[0, :4].tolist(),
        "per_channel_spread": per_channel_spread,
        "ratio_err": ratio_err,
        "std_err": std_err,
        "shape": tuple(y.shape),
        "coef_shape": tuple(gamma.shape),
    }


# ---------------------------------------------------------------- C + D 共用
def _block_channels(m) -> list[int]:
    """按执行顺序列出每个 MBConv 块的**输出**通道（FiLM 加在块输出上）。"""
    chans = []
    for stage in m.features:
        if isinstance(stage, nn.Sequential) and len(stage) > 0 and hasattr(stage[0], "block"):
            for blk in stage:
                chans.append(blk.block[-1][0].out_channels if hasattr(blk.block[-1][0], "out_channels")
                             else blk.out_channels)
    return chans


def _make_films(m, identity_init: bool) -> nn.ModuleList:
    return nn.ModuleList([FiLM(c, D, identity_init=identity_init) for c in _block_channels(m)])


def _forward(m, films: nn.ModuleList | None, z: torch.Tensor | None, x: torch.Tensor):
    """跑一遍 features，可选地在每个 MBConv 块之后插 FiLM。"""
    k = 0
    for stage in m.features:
        if isinstance(stage, nn.Sequential) and len(stage) > 0 and hasattr(stage[0], "block"):
            for blk in stage:
                x = blk(x)
                if films is not None:
                    x = films[k](x, z)
                    k += 1
        else:
            x = stage(x)
    return x


# ---------------------------------------------------------------- 实验 C
def _prep(m):
    """train 模式（让 BN 用批统计，激活才不被逐层衰减），但关掉随机深度。

    ⚠️ torchvision 0.28 的 StochasticDepth 用 `p`（丢弃概率）不是 `survival_prob`，
       只改后者等于没改 —— 两次前向会差 5.93，逐位对比直接失效。两个都设。
    """
    m = m.train()
    for mod in m.modules():
        if mod.__class__.__name__ == "StochasticDepth":
            if hasattr(mod, "p"):
                mod.p = 0.0
            if hasattr(mod, "survival_prob"):
                mod.survival_prob = 1.0
    return m


def run_c() -> dict:
    """identity 初始化：插入 FiLM 后输出是否逐位不变？不初始化会坏多少？

    ⚠️ 三次前向必须用**同一个**骨干实例，否则比的是两个随机网络，结论无意义。
    ⚠️ 必须在 train 模式：eval 模式下的 BN 用的是 running stats（还是初始值 1），
       等于什么都不做，随机初始化的网络激活会逐阶段衰减（实测 std 1.0 → 4.2e-14）。
       那时候所有数都在 1e-14 量级，偏差当然也「很小」，测不出任何东西。
    """
    torch.manual_seed(0)
    x = torch.randn(1, 3, 300, 300, device=DEV)
    z = torch.randn(1, D, device=DEV)
    m = _prep(efficientnet_b3(weights=None).to(DEV))

    with torch.no_grad():
        base = _forward(m, None, None, x)
        # 关卡：前向必须是确定的，否则下面所有「逐位相同」都在噪声里
        base2 = _forward(m, None, None, x)
        if not torch.equal(base, base2):
            raise AssertionError("前向不确定，先关随机深度/随机性再比")

        films_id = _make_films(m, identity_init=True).to(DEV)
        out_id = _forward(m, films_id, z, x)

        torch.manual_seed(1)                      # 普通初始化（PyTorch 默认）
        films_rand = _make_films(m, identity_init=False).to(DEV)
        out_rand = _forward(m, films_rand, z, x)

    # 顺便量一下 eval 模式的衰减，说明为什么不能用它
    m_eval = efficientnet_b3(weights=None).to(DEV).eval()
    with torch.no_grad():
        e, stds = x, []
        for stage in m_eval.features:
            e = stage(e)
            stds.append(e.std().item())

    scale = base.abs().max().item()
    return {
        "ident_max_abs": (out_id - base).abs().max().item(),
        "ident_bitwise": bool(torch.equal(out_id, base)),
        "rand_max_abs": (out_rand - base).abs().max().item(),
        "rand_rel": (out_rand - base).abs().max().item() / scale,
        "act_scale": scale,
        "n_film": len(films_id),
        "eval_decay": (stds[0], stds[-1]),
    }


# ---------------------------------------------------------------- 实验 D
def run_d() -> dict:
    """零初始化会不会把梯度卡死？（答：不会，但原因值得记）"""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 300, 300, device=DEV)
    z = torch.randn(1, D, device=DEV)

    m = _prep(efficientnet_b3(weights=None).to(DEV))
    films = _make_films(m, identity_init=True).to(DEV)

    out = _forward(m, films, z, x)
    target = torch.randn_like(out)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    # 只看最后一层的 FiLM：深度上离 loss 最近，最能说明「零初始化是否卡住」
    last = films[-1]
    g_gamma = last.to_gamma.weight.grad
    g_beta = last.to_beta.weight.grad
    # 对照：骨干最后一个卷积的梯度（确认骨干也没被冻住）
    last_conv = None
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            last_conv = mod
    g_conv = last_conv.weight.grad
    return {
        "loss": loss.item(),
        "gamma_grad_norm": g_gamma.norm().item() if g_gamma is not None else None,
        "beta_grad_norm": g_beta.norm().item() if g_beta is not None else None,
        "gamma_grad_frac_zero": (g_gamma == 0).float().mean().item() if g_gamma is not None else None,
        "conv_grad_norm": g_conv.norm().item() if g_conv is not None else None,
    }


# ---------------------------------------------------------------- 实验 E
def run_e() -> dict:
    """四种条件注入方式的参数账（C=512、d=512、3×3 卷积、单指令 token）。"""
    c = d = 512
    rows = {
        "输入拼接（通道 concat）": d * c * 3 * 3,
        "加法偏置  β": d * c + c,
        "乘性缩放  γ": d * c + c,
        "仿射 FiLM  γ,β": 2 * (d * c + c),
        "cross-attention": 4 * d * d,
    }
    # 计算量：FiLM 每像素每通道 1 乘 1 加；attention 是 O(tokens²·d)
    tokens = 81
    flops = {
        "仿射 FiLM  γ,β": 2 * c * tokens,
        "cross-attention": 4 * d * d * tokens,
    }
    return {"rows": rows, "flops": flops, "tokens": tokens}


# ---------------------------------------------------------------- 输出
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="ABCDE")
    args = ap.parse_args()
    only = args.only.upper()

    if "A" in only:
        r = run_a()
        print("=" * 68)
        print("A · 参数量对账（EfficientNet-B3 + FiLM，d=512）")
        print("=" * 68)
        print(f"  MBConv 块数            : {r['n_blocks']}（与 RT-1 论文的 26 一致）")
        print(f"  各块输出通道之和       : {r['sum_cin']}")
        print(f"  骨干 features          : {r['backbone']:>10,}  ({r['backbone']/1e6:.2f}M)")
        print(f"  FiLM (γ,β) 增量        : {r['film']:>10,}  ({r['film']/1e6:.2f}M)")
        print(f"  末层 1×1 投影到 512    : {r['proj']:>10,}  ({r['proj']/1e6:.2f}M)")
        print(f"  合计                   : {r['total']:>10,}  ({r['total']/1e6:.2f}M)")
        print(f"  FiLM / 骨干            : {r['film_share']*100:.1f}%")
        print("  读法：FiLM 占了骨干三成多的参数，但几乎不增加卷积计算量 ——")
        print("        「轻量」说的是 FLOPs，不是参数量。这两件事别混。")

    if "B" in only:
        r = run_b()
        print("\n" + "=" * 68)
        print("B · 逐通道广播：γ、β 是 C 维向量，怎么作用到 C×H×W 上")
        print("=" * 68)
        print(f"  x 形状 {r['shape']}   γ 形状 {r['coef_shape']}  → γ[:, :, None, None] 后广播")
        print(f"  γ 前 4 个通道 = {['%.4f' % v for v in r['gamma_sample']]}")
        print(f"  (y−β)/x 在同一通道内的极差 : {r['per_channel_spread']:.3e}")
        print(f"  该比值与 (1+γ) 的最大偏差   : {r['ratio_err']:.3e}")
        print(f"  std(y)/std(x) 与 |1+γ| 偏差 : {r['std_err']:.3e}")
        print("  读法：同一通道的所有空间位置乘的是同一个数 ——")
        print("        这就是「只调找什么，不调在哪」。")

    if "C" in only:
        r = run_c()
        print("\n" + "=" * 68)
        print("C · identity 初始化：插入 FiLM 后输出变不变")
        print("=" * 68)
        print(f"  原始输出 |max|            : {r['act_scale']:.6f}")
        print(f"  identity 初始化 最大偏差  : {r['ident_max_abs']:.3e}"
              f"   逐位相同 = {r['ident_bitwise']}")
        print(f"  普通随机初始化 最大偏差   : {r['rand_max_abs']:.6f}"
              f"  （占激活尺度的 {r['rand_rel']*100:.1f}%）")
        print(f"  对照：eval 模式下激活 std {r['eval_decay'][0]:.3e} → {r['eval_decay'][1]:.3e}")
        print("  读法：零初始化不是「初始化得小一点」，是**精确的恒等映射**。")
        print("        预训练权重因此一个都没被破坏。")
        print("        注意 eval 那行 —— 随机初始化的网络在 eval 下信号会衰减到 1e-14，")
        print("        拿它做基准，什么偏差看起来都「很小」。基准错，结论全错。")

    if "D" in only:
        r = run_d()
        print("\n" + "=" * 68)
        print("D · 零初始化会不会把梯度卡死")
        print("=" * 68)
        print(f"  loss                     : {r['loss']:.6f}")
        print(f"  ∂L/∂W(γ) 范数            : {r['gamma_grad_norm']:.6e}")
        print(f"  ∂L/∂W(β) 范数            : {r['beta_grad_norm']:.6e}")
        print(f"  其中恰好为 0 的比例      : {r['gamma_grad_frac_zero']*100:.2f}%")
        print("  读法：γ 的权重为 0，但 ∂L/∂W = (∂L/∂γ)·xᵀ ——")
        print("        梯度里带着输入 x，所以不为 0。和 ReLU 死亡不是一回事。")

    if "E" in only:
        r = run_e()
        print("\n" + "=" * 68)
        print("E · 四种条件注入方式的账（C=d=512）")
        print("=" * 68)
        print(f"  {'方式':<24}{'参数量':>12}{'相对 FiLM':>12}")
        base = r["rows"]["仿射 FiLM  γ,β"]
        for k, v in r["rows"].items():
            print(f"  {k:<24}{v:>12,}{v/base:>11.2f}×")
        print(f"\n  计算量（{r['tokens']} 个 token）:")
        for k, v in r["flops"].items():
            print(f"  {k:<24}{v:>12,}")
        print("  读法：拼接把 512 维噪声塞进第一层，预训练权重全废；")
        print("        纯加法/纯乘性各缺一半自由度；attention 参数最贵且随 token 数平方涨。")


if __name__ == "__main__":
    main()
