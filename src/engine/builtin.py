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
from src.models.mlp import MLP
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
