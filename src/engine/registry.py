"""注册表：把「名字字符串」映射到「构建函数」。

════════════════════════════════════════════════════════════════════
为什么需要这东西 —— 这是「通用模板」和「换个参数名」的分界线
════════════════════════════════════════════════════════════════════

一个训练脚本要算「通用」，关键在于：**换数据集、换模型，主程序要不要改？**

    反例（看起来通用，其实不通用）：
        model = MLP(in_dim=784, hidden=cfg.hidden, out_dim=10, ...)
        loader = build_loaders(cfg)          # 写死 MNIST
        # 想换 CNN？→ 改主程序；想换 CIFAR？→ 改主程序

    正例（真的通用）：
        model = build_model(cfg.model, cfg)  # cfg.model = "mlp" / "cnn" / ...
        loader = build_dataset(cfg.dataset, cfg)   # cfg.dataset = "mnist" / ...
        # 想换 CNN？→ 在 builtin.py 里加一行注册，主程序一个字不动

所以这张表的价值不在于现在能支持几个模型，
而在于**以后加模型的时候，改动面是一行而不是一片**。

════════════════════════════════════════════════════════════════════
分层，以及为什么这么分（防止循环 import）
════════════════════════════════════════════════════════════════════

    registry.py   纯机制，不 import 任何项目内的东西
         ↑
    builtin.py    import registry + models + data，把具体实现注册进去
         ↑
    trainer.py    只 import registry，**不知道有哪些具体模型**

好处：trainer 不认识 MLP、不认识 CNN。P1 加 CNN 时，
trainer.py 和 cli.py 都不用重新读一遍就能确定它们没被影响。

如果把「注册」写在 registry.py 里，registry 就要 import models，
而 models 里的 mlp.py 又想用 registry 的话，就成环了。
"""
from __future__ import annotations

from typing import Callable

# 两张表。键是配置里写的字符串，值是「吃 cfg、返回对象」的函数。
#
# 为什么 value 存函数而不是类：
#   类需要统一的构造签名（都叫 in_dim? 都叫 hidden?），CNN 和 MLP 的
#   超参根本不一样，硬统一只会造出一个塞满 None 的假接口。
#   存函数就没有这个问题 —— 每个 build_xxx(cfg) 自己从 cfg 里取自己要的。
DATASETS: dict[str, Callable] = {}
MODELS: dict[str, Callable] = {}


def register_dataset(name: str) -> Callable:
    """装饰器：把一个 build_loaders(cfg) 函数登记为名为 name 的数据集。

    用法：
        @register_dataset("mnist")
        def build_mnist(cfg):
            return train_loader, val_loader, test_loader
    """

    def deco(fn: Callable) -> Callable:
        if name in DATASETS:
            # 重复注册必须报错，不能静默覆盖。
            # 否则两个实验想用同一个名字时，后注册的悄悄赢，
            # 你会以为自己在跑 A，其实跑的是 B —— 而且没有任何提示。
            raise ValueError(f"数据集 {name!r} 已经被注册过了（{DATASETS[name].__name__}）")
        DATASETS[name] = fn
        return fn

    return deco


def register_model(name: str) -> Callable:
    """装饰器：把一个 build_model(cfg) 函数登记为名为 name 的模型。

    用法：
        @register_model("mlp")
        def build_mlp(cfg):
            return MLP(...)
    """

    def deco(fn: Callable) -> Callable:
        if name in MODELS:
            raise ValueError(f"模型 {name!r} 已经被注册过了（{MODELS[name].__name__}）")
        MODELS[name] = fn
        return fn

    return deco


def build_dataset(name: str, cfg):
    """按名字取数据集。名字不存在时，把已注册的名字列出来 ——

    这一点很重要：报错信息里带「有哪些可选」，你就不用回头翻代码。
    只写 "KeyError: 'cnn'"，你还得自己去 grep 才知道正确拼法。
    """
    if name not in DATASETS:
        raise KeyError(
            f"未知数据集 {name!r}。已注册的有：{sorted(DATASETS)}。"
            f"（新数据集请在 src/engine/builtin.py 里用 @register_dataset 注册）"
        )
    return DATASETS[name](cfg)


def build_model(name: str, cfg):
    """按名字取模型。"""
    if name not in MODELS:
        raise KeyError(
            f"未知模型 {name!r}。已注册的有：{sorted(MODELS)}。"
            f"（新模型请在 src/engine/builtin.py 里用 @register_model 注册）"
        )
    return MODELS[name](cfg)
