"""造零件：优化器、损失函数、学习率调度器，以及几个观测指标。

这里是「换实现」的唯一入口。Day 12 换了优化器、Day 16 换了损失函数、
Day 17 换了线性层 —— 三次都只改了这张表里的一个 if，调用方一行没动。

这个性质有个名字：**接口一致就能替换**。
它成立的前提是手写实现和官方实现对上了签名，而这正是前面几天
七关/五关对拍要做的事。没有对拍就替换，等于蒙着眼睛换零件。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.losses import MyCrossEntropy
from src.utils.optim import MySGD, MyStepLR


def build_optimizer(cfg, model) -> torch.optim.Optimizer:
    """按配置造优化器。三个选项里有一个是你自己写的（mysgd）。

    mysgd 和 sgd 的接口故意做成一样，所以你只要改一个 yaml 字段就能切换，
    训练脚本一个字都不用动。
    """
    if cfg.optimizer == "mysgd":
        # Day 12 手写的那个。参数名与 torch.optim.SGD 对齐
        return MySGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "adam":
        # Adam 没有 momentum 这个参数（它有 betas），所以这里只传 lr
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )


def build_criterion(cfg) -> nn.Module:
    """按配置造损失函数。两个选项的接口完全一样：crit(logits, target) -> 标量。

    ⚠️ 一个必须记住的前提：**模型的最后一层不要接 softmax。**
    交叉熵内部已经做了 log_softmax。多套一层 softmax 不会报错，
    只会让 loss 卡在 -ln(e/(e+9)) ≈ 1.4612 附近再也降不下去
    （Day 16 在真实模型上实测：正确 0.0307 → 多加一层 1.4837，差 48 倍）。
    """
    if cfg.loss == "myce":
        return MyCrossEntropy()
    return nn.CrossEntropyLoss()


def build_scheduler(cfg, opt):
    """学习率调度器。none 就返回 None，训练循环里会判一下。"""
    if cfg.scheduler == "none":
        return None
    if isinstance(opt, MySGD):
        # 手写优化器的 lr 挂在 opt.lr 上，用我们自己的 StepLR
        return MyStepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    # torch 优化器的 lr 藏在 param_groups 里，用它自带的
    return torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)


def get_lr(opt) -> float:
    """取当前学习率。两种优化器存 lr 的位置不一样，这里统一一下。

    不统一的话，日志里打印 lr 时要写两套分支，
    哪天加了第三种优化器就又要改一遍。
    """
    if isinstance(opt, MySGD):
        return opt.lr
    return opt.param_groups[0]["lr"]


def weight_norm(model) -> float:
    """所有参数拼起来的 L2 范数 sqrt(Σ p²)。

    Day 15 用它观察 weight decay 到底干了什么：
    wd 的作用是**把权重往 0 拉**，所以开了 wd 之后这个数应该比不开时小。
    实测 wd=5e-4 时 28.851 → 16.845（降 41.6%）。
    **如果开了 wd 这个数没变小，说明 wd 根本没生效** —— 比如传错了优化器，
    或者 momentum 把 λw 放大成了别的东西。
    """
    total = 0.0
    for p in model.parameters():
        total += p.detach().pow(2).sum().item()
    return total ** 0.5


def count_params(model) -> int:
    """参数总量。和手算值对一下，能抓出「模型结构和你想的不一样」。"""
    return sum(p.numel() for p in model.parameters())
