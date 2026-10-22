"""随机种子：让「同一条命令跑两次」得到完全一样的结果。

为什么需要它（真实场景，不是学术洁癖）：
    你调了三天，今晚终于训到 98.2%。明天想再确认一次，跑出来 97.1%。
    这时候你无法判断是「昨晚那次运气好」还是「今天改坏了什么」。
    没有复现性，所有调参结论都是噪音。

一个容易被忽略的事实：
    随机性不止一个来源。Python 的 random、numpy、PyTorch 的 CPU 和 CUDA
    各有一套随机数生成器，**它们互不相干**。只设 torch.manual_seed 而不管
    numpy，某天引入一个用 numpy 抽样的数据增强，复现性就悄悄断了。

用法：
    from src.utils.seed import set_seed
    set_seed(cfg.seed)
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """把四套随机数生成器全部固定到同一个种子。

    参数：
        seed          : 种子。同一个 seed 必须得到同一串随机数
        deterministic : 是否强制 cuDNN 走确定性算法

    注意：这个函数**只在程序开头调用一次**。
        如果你在训练中途再调一次，等于把随机流倒回起点 —— 后面每一步的
        数据顺序都会和「正常跑」不一致，看起来就像复现性坏了。
    """
    # ① Python 自带 random（比如 random.shuffle、random.random）
    random.seed(seed)

    # ② numpy（比如 np.random.rand、np.random.permutation）
    np.random.seed(seed)

    # ③ PyTorch。这一行同时管 CPU 和**所有** GPU 的默认生成器
    torch.manual_seed(seed)

    # ④ 多卡时每张卡还有各自的生成器，manual_seed 不一定覆盖到，补一刀
    #    （我们只有一张卡，写上是为了以后换机器时不用回来改）
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN 默认会「为了快而挑算法」：同一份输入，第一次跑用算法 A、
        # 第二次可能用算法 B，两者的浮点加法顺序不同，末位数字就不一样。
        # deterministic=True 强制「每次都挑同一个」，代价是可能慢一点。
        torch.backends.cudnn.deterministic = True

        # benchmark=True 会「先试几种算法，挑最快的」——这个过程本身
        # 就带随机性，和上一行冲突，必须关掉
        torch.backends.cudnn.benchmark = False

    # ⑤ Python 的字符串哈希也带随机化（每次启动解释器不同），
    #    影响 set / dict 的遍历顺序。必须在新进程启动前设置才生效，
    #    所以这一行在本进程内其实**已经晚了** —— 但写在这里是为了提醒：
    #    如果你哪天发现「换个终端跑结果就变了」，去查这个环境变量。
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_generator(seed: int) -> torch.Generator:
    """造一个独立的随机数生成器，专供 DataLoader 使用。

    为什么要单独造一个，而不是让 DataLoader 用全局生成器：
        全局生成器是「一条流水线」，模型初始化、dropout、数据打乱都从这条
        流水线上取数。你哪天在模型初始化前多加了一个随机操作，
        后面数据打乱的顺序就整体错位 —— 复现性以一种很难查的方式断掉。

        给 DataLoader 一个**自己的**生成器，它就从自己这条流水线取数，
        和模型初始化互不干扰。**这也是为什么「设了种子还是不 reproducible」
        的绝大多数原因就在 DataLoader 上。**
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
