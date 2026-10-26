"""MNIST 数据加载：把「磁盘上的 idx 文件」变成「能喂给模型的 batch」。

这层代码只做三件事，每件都值得记住：
    1. 定义 transform —— 决定一张图变成什么张量
    2. 切出验证集   —— 从训练集里分，绝不能拿测试集当验证集
    3. 造 DataLoader —— 决定 batch 怎么切、要不要打乱、用哪条随机流

用法：
    from src.data.mnist import build_loaders
    train_loader, val_loader, test_loader = build_loaders(cfg)
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.utils.seed import make_generator

# ── 数据集的固定形状 ─────────────────────────────────────────────────
# 放在数据集模块里，而不是散落在训练脚本里。
# 理由：这些数字是**数据本身的属性**，不是某个模型的超参。
# 换模型（MLP → CNN）时它们不变；换数据集（MNIST → CIFAR）时它们才变。
IMAGE_SIZE = 28
NUM_CLASSES = 10
IN_DIM = IMAGE_SIZE * IMAGE_SIZE      # 784，摊平之后的长度

# MNIST 全集的均值 / 标准差（官方给的数字，不是现算的）。
# 归一化后数据大致变成「均值 0、标准差 1」，模型更好训。
# 我们这版 MLP 只用 ToTensor（把 0~255 压到 0~1），已经够用；
# 数字留着，等你以后换 CNN 时打开 W3 里的归一化选项对比。
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def build_transform(normalize: bool = False) -> transforms.Compose:
    """定义「一张图 → 一个张量」的变换链。

    ToTensor() 做了两件事，两件都必须知道：
        · PIL 图像 (28, 28) uint8  →  张量 (1, 28, 28) float32
        · 数值从 0~255 **除以 255**  →  变成 0~1

    为什么多出来第 0 维：PyTorch 的图像张量约定是 (C, H, W)，
    灰度图 C=1。等你以后用 DataLoader 拼成 batch，就是 (B, 1, 28, 28)。
    """
    ops = [transforms.ToTensor()]
    if normalize:
        # Normalize 做的是 (x - mean) / std，按通道来
        ops.append(transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    return transforms.Compose(ops)


def build_datasets(cfg) -> tuple[Dataset, Dataset, Dataset]:
    """返回 (train, val, test) 三个数据集。

    ⚠️ 这里有一个初学者最常犯的错误，值得单独说：

        很多人直接拿 **test** 当验证集（"反正都是数据"）。
        这么做的后果不是报错，而是**你自己的判断失效**：
        你会按测试集的表现去调超参数，调完再报"测试准确率 98%"——
        这个数字已经被你偷看过很多次，它不再代表"没见过的新数据"。
        等到真正上线/交作业，表现会明显低于你的预期。

    正确做法：从 60000 条训练集里切一块（默认 6000 条）当验证集，
    测试集只在最后**看一次**。
    """
    tf = build_transform(normalize=False)

    # download=True：本地已有就直接读，没有才下载（我们刚才已经下好了，在 data/）
    train_full = datasets.MNIST(
        root=cfg.data_dir, train=True, download=True, transform=tf
    )
    test_set = datasets.MNIST(
        root=cfg.data_dir, train=False, download=True, transform=tf
    )

    # ── 切训练 / 验证 ────────────────────────────────────────────
    # 用「打乱后的下标」来切，而不是直接取前 6000 条。
    # 原因：MNIST 训练集是按标签大致排过序的，直接切前 6000 条会拿到
    # 一堆 0 和 1，验证集根本代表不了整体。
    # generator 用固定种子 → 切法固定，每次跑都切出同一批样本。
    g = make_generator(cfg.seed)
    n = len(train_full)
    perm = torch.randperm(n, generator=g)

    val_idx = perm[: cfg.val_size].tolist()
    train_idx = perm[cfg.val_size :].tolist()

    # Subset 就是「从这个数据集里挑我指定的那些下标」，不会复制数据
    return Subset(train_full, train_idx), Subset(train_full, val_idx), test_set


def build_loaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """返回 (train_loader, val_loader, test_loader)。

    三个 DataLoader 的唯一区别是 shuffle：
        train  打乱  —— 每个 epoch 的顺序不同，避免模型学到"顺序"
        val/test 不打乱 —— 反正只求平均，顺序不影响结果（还更快）

    还有一个容易漏的点：**只有 train_loader 需要 generator**。
    因为只有它在用随机数（打乱顺序）。给不 shuffle 的 loader 传 generator
    没有坏处，但会造成一种误解：以为"传了 generator 就 reproducible 了"。
    """
    train_set, val_set, test_set = build_datasets(cfg)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,                        # 每个 epoch 重新打乱
        generator=make_generator(cfg.seed),  # 打乱用哪条随机流
        num_workers=cfg.num_workers,         # Windows 上先用 0，多进程容易出怪问题
        drop_last=False,                     # 最后一批不满也保留（Day 11 算过这个）
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return train_loader, val_loader, test_loader
