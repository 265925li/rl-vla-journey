"""把具体实现注册进注册表 —— 这是唯一知道「有哪些模型/数据集」的文件。

想加一个模型，就在这个文件里加一个 @register_model；
想加一个数据集，就加一个 @register_dataset。
**trainer.py 和 cli.py 一个字都不用改。**

这个文件必须在程序启动早期被 import（哪怕只是为了副作用）。
cli.py 顶部会 `import src.engine.builtin  # noqa: F401` 触发注册 ——
那行 import 看着"没用到任何东西"，但它就是注册的动作本身。
"""
from __future__ import annotations

import torch.nn as nn

from src.data.mnist import IN_DIM, NUM_CLASSES, build_loaders
from src.engine.registry import register_dataset, register_model
from src.models.cnn import CNN, expected_params_cnn
from src.models.mlp import MLP
from src.utils.config import parse_channels
from src.utils.layers import MyLinear


# ═══════════════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════════════
@register_dataset("mnist")
def build_mnist(cfg):
    """MNIST：返回 (train_loader, val_loader, test_loader)。

    契约（新数据集必须遵守，否则 trainer 跑不了）：
        · 返回三个 loader，顺序固定是 train / val / test
        · 每个 loader 迭代出 (xb, yb)，xb 是 (B, ...)、yb 是 (B,) 的 long
        · **val 和 test 必须是两个不同的集合**（见下面那段警告）
    """
    return build_loaders(cfg)


# ═══════════════════════════════════════════════════════════════════
# 一个能被手算验证的检查（定义在用到它的地方之前）
# ═══════════════════════════════════════════════════════════════════
def expected_params(n_in: int, hidden: int, n_out: int, n_layers: int) -> int:
    """按公式手算参数量，用来和 model.parameters() 对账。

    为什么值得写这个：它是**唯一一处不依赖 PyTorch 的独立校验**。
    如果 build_mlp 里少加了一层、或者 hidden 传错了，
    参数量立刻对不上，而且报错信息会告诉你差多少。
    只看 loss 是发现不了的 —— 少一层的模型照样能训，只是效果差一点。

    公式（n_layers=2 时）：
        n_in*hidden + hidden        ← 第一层 + 它的 bias
        + hidden*n_out + n_out      ← 输出层 + 它的 bias
    实测 784-256-10 → 203530，和 model.parameters() 一致。
    """
    total = 0
    last = n_in
    for _ in range(n_layers - 1):
        total += last * hidden + hidden
        last = hidden
    total += last * n_out + n_out
    return total


# ═══════════════════════════════════════════════════════════════════
# 模型
# ═══════════════════════════════════════════════════════════════════
@register_model("mlp")
def build_mlp(cfg) -> nn.Module:
    """784 -> hidden -> ... -> 10 的全连接网络（Day 10 自己写的那一个）。

    Day 15：多传一个 dropout。cfg.dropout=0 时结构和 Day 13 完全相同。
    Day 17：cfg.linear="my" 时把所有全连接层换成手写的 MyLinear。

    这一步之所以能一行搞定，是因为 MyLinear 的构造签名和参数属性
    跟 nn.Linear 完全一致 —— 这叫「接口对齐」，是能替换的前提。
    """
    linear_cls = MyLinear if cfg.linear == "my" else nn.Linear
    model = MLP(
        in_dim=IN_DIM,
        hidden=cfg.hidden,
        out_dim=NUM_CLASSES,
        n_layers=cfg.n_layers,
        act=cfg.act,
        dropout=cfg.dropout,
        linear_cls=linear_cls,
    )

    # ── 参数量对账（模型自己负责自己的自检）─────────────────────
    # 为什么这个检查放在这里而不是 Trainer 里：
    # 公式因模型而异 —— CNN 的参数量算法和 MLP 完全不同。
    # 让每个 build_xxx 检查自己，Trainer 就能保持"不认识任何具体模型"，
    # 以后加 CNN 也不用回头改 Trainer。
    #
    # 这个检查抓什么：少加一层、hidden 传错、忘了 bias、误加了参数层。
    # 这些错误**都不会让训练崩**，只会让效果莫名其妙地差一点。
    actual = sum(p.numel() for p in model.parameters())
    manual = expected_params(IN_DIM, cfg.hidden, NUM_CLASSES, cfg.n_layers)
    assert actual == manual, (
        f"MLP 参数量对不上：运行时 {actual} vs 手算 {manual}（差 {actual - manual}）"
    )
    return model


@register_model("cnn")
def build_cnn(cfg) -> nn.Module:
    """两层卷积 + 两层全连接的小 CNN（Day 22）。

    ── Day 18 那次重构在这一刻兑现了 ────────────────────────────────
    加这个模型，一共只改了**两个文件**：

        src/models/cnn.py            新建
        src/engine/builtin.py        就是下面这十几行

    trainer.py / cli.py / loops.py / registry.py **一个字都没动**。
    这不是巧合，是 Day 18 把「谁认识具体模型」限制在一个文件里的结果。
    验证方法：`grep -n cnn src/engine/trainer.py src/engine/cli.py`
    在 Day 22 实测是 **0 行**。

    ── 三个参数怎么从 cfg 传进来 ───────────────────────────────────
    cfg.channels   "16,32" 字符串 → parse_channels() → (16, 32)
                   （解析在校验阶段已经跑过一次了，能走到这行就说明格式合法）
    cfg.in_channels  1（MNIST 灰度）。换成 RGB 数据集时这里要改
    cfg.conv       torch / my —— 选 nn.Conv2d 还是手写 MyConv2d
    """
    # 延迟 import：只有真正用 cnn 模型时才需要 conv.py。
    # 不用 cnn 的实验（比如纯 MLP 对照）完全不会碰到它 ——
    # 万一 conv.py 写坏了，也不会影响别的模型。
    conv_cls = nn.Conv2d
    if cfg.conv == "my":
        from src.utils.conv import MyConv2d
        conv_cls = MyConv2d

    channels = parse_channels(cfg.channels)
    model = CNN(
        in_channels=cfg.in_channels,
        channels=channels,
        hidden=cfg.hidden,
        out_dim=NUM_CLASSES,
        dropout=cfg.dropout,
        act=cfg.act,
        conv_cls=conv_cls,
    )

    # ── 参数量对账（和 MLP 同一个套路，公式换成卷积的）────────────
    # CNN 的参数量算法和 MLP 完全不同（卷积层是 C_out·(C_in·kh·kw+1)），
    # 所以这段检查**必须写在 build_cnn 里**，不能提到 Trainer 去。
    # 这正是 Day 18 定的规矩：模型自己负责自己的自检。
    actual = sum(p.numel() for p in model.parameters())
    manual = expected_params_cnn(
        in_channels=cfg.in_channels,
        channels=channels,
        hidden=cfg.hidden,
        out_dim=NUM_CLASSES,
    )
    assert actual == manual, (
        f"CNN 参数量对不上：运行时 {actual} vs 手算 {manual}（差 {actual - manual}）\n"
        f"  配置：in_channels={cfg.in_channels} channels={channels} hidden={cfg.hidden}\n"
        "  常见原因：池化层数改了但公式没跟着改、channels 解析成了别的值"
    )

    # ── 一个只有 CNN 才会犯的配置错误，在这里拦掉 ─────────────────
    # flatten=true 会让 Trainer 先把 (B,1,28,28) 摊成 (B,784)，
    # 然后 conv1 收到 2 维输入才报错。虽然会报错，但报错点在卷积层，
    # 而错因在 yaml —— 隔了一个文件，不好查。
    # 在这里提前拦，报错信息就能直接说清楚「去改 yaml」。
    if cfg.flatten:
        raise ValueError(
            "model=cnn 但 flatten=true。CNN 需要 4 维输入 (B,C,H,W)，"
            "必须在配置里写 flatten: false。\n"
            "  为什么必须手动关：框架无法知道哪个模型还要空间结构，"
            "所以它是显式开关（见 config.py 里 flatten 那段注释）。"
        )
    return model
