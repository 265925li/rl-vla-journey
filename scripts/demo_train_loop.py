"""Day 11 作业：数据集与训练循环

用法（PowerShell，在项目根目录跑）：
    cd "E:\\work Buddy\\强化学习\\rl-vla-journey"
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\demo_train_loop.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\demo_train_loop.py --epochs 30 --lr 0.1
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\demo_train_loop.py --compare

今天有四关，按顺序报错。每填完一处就重跑一次，看下一个报什么。

    TODO 1  BlobsDataset   —— Dataset 接口只需要两个方法
    TODO 2  build_loader   —— TensorDataset + DataLoader
    TODO 3  train_step     —— 五步训练动作（今天最值钱的 5 行，也是盲写材料）
    TODO 4  evaluate       —— eval + no_grad + 按样本数加权

模型用的是 Day 10 的 MLP（src/models/mlp.py），不重写。
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mlp import MLP, init_weights  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.paths import OUTPUTS  # noqa: E402

LOG = OUTPUTS / "day11_train_loop.txt"


# ═══════════════════════════════════════════════════════════════════
# 0. 造数据（已给，不用改）
# ═══════════════════════════════════════════════════════════════════
def make_blobs(
    n_per_class: int = 1000,
    n_classes: int = 3,
    radius: float = 2.0,
    std: float = 0.7,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在圆周上均匀摆 n_classes 个类中心，每类撒 n_per_class 个点。

    为什么不用 MNIST：今天练的是**循环结构**，不是准确率。
    合成数据 3000 条、2 维，0.5 秒跑完 20 个 epoch，改一个参数立刻能看到结果。
    MNIST 要下载、要读图、要等——那些是 Day 13 的事。

    std=0.7 是调出来的：太大（>1.0）三团糊在一起，准确率封顶在 85%，曲线像卡住了；
    太小（<0.4）第一个 epoch 就 100%，曲线没形状，batch_size 实验也看不出差异。

    返回：
        X: (N, 2) float32    y: (N,) int64
    """
    g = torch.Generator().manual_seed(seed)
    angles = torch.arange(n_classes, dtype=torch.float32) * (2 * math.pi / n_classes)
    centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * radius

    n = n_classes * n_per_class
    X = centers.repeat_interleave(n_per_class, dim=0) + torch.randn(n, 2, generator=g) * std
    y = torch.arange(n_classes).repeat_interleave(n_per_class)
    return X, y


def split(
    X: torch.Tensor, y: torch.Tensor, val_ratio: float = 0.2, seed: int = 1234
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """按行打乱后切分 train / val（已给）。"""
    n = X.shape[0]
    n_val = int(n * val_ratio)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    vi, ti = perm[:n_val], perm[n_val:]
    return X[ti], y[ti], X[vi], y[vi]


# ═══════════════════════════════════════════════════════════════════
# TODO 1 · 手写一个 Dataset
# ═══════════════════════════════════════════════════════════════════
class BlobsDataset(Dataset):
    """把「两个张量」包装成 Dataset。

    Dataset 的全部要求就两条：
        __len__(self)       -> int                 一共多少条
        __getitem__(self, i)-> (张量, 张量)         第 i 条长什么样
    有了这两条，DataLoader 就能自己完成「按索引取一批、拼成一个 batch」。

    torch.utils.data.TensorDataset 干的就是同一件事，源码约等于：
        class TensorDataset(Dataset):
            def __getitem__(self, i):
                return tuple(t[i] for t in self.tensors)
    今天先自己写一遍（知道它有多简单），再用官方的（它带类型标注和边界检查）。
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X 有 {X.shape[0]} 行，y 有 {y.shape[0]} 行，对不上")
        self.X = X
        self.y = y

    def __len__(self) -> int:
        raise AssertionError("TODO 1a 未完成：__len__ 要返回样本总数（看 X 的第 0 维）")

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError(
            "TODO 1b 未完成：__getitem__ 要返回第 i 条样本 (X[i], y[i])\n"
            "  注意 X[i] 的形状是 (2,)，y[i] 是 0 维标量 —— 不是 (1,2) / (1,)"
        )


# ═══════════════════════════════════════════════════════════════════
# TODO 2 · 建 DataLoader
# ═══════════════════════════════════════════════════════════════════
def build_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader:
    """Dataset → DataLoader。

    DataLoader 只做三件事：
        1. 按 batch_size 把索引切片：[0..63], [64..127], ...
        2. shuffle=True 时每 epoch 重新打乱索引（训练集要打乱，验证集不要）
        3. 把取出来的样本沿第 0 维拼起来：64 个 (2,) → 一个 (64, 2)
    第 3 步叫 collate，是「Dataset 返回单条、模型要吃整批」之间那道桥。

    参数 seed 同时喂给 DataLoader 的 generator —— 打了三重种子
    （torch / 数据生成 / 这里），同一个 seed 才能每次都一样的打乱顺序。

    返回的形状：每次迭代给出 (xb, yb)，xb 是 (B, 2)，yb 是 (B,)。
    B 通常等于 batch_size，但**最后一个 batch 可能更小**（除非 drop_last=True）。
    """
    ds = TensorDataset(X, y)  # 已给：Dataset 这一层有了
    raise AssertionError(
        "TODO 2 未完成：把 ds 交给 DataLoader，参数是 batch_size / shuffle / generator\n"
        "  generator 用 torch.Generator().manual_seed(seed) —— 不传的话每 epoch 打乱顺序都不同"
    )


# ═══════════════════════════════════════════════════════════════════
# TODO 3 · 一个 batch 的训练（五步）
# ═══════════════════════════════════════════════════════════════════
def train_step(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, int]:
    """对一个 batch 做一次参数更新，返回 (本 batch 平均 loss, 预测对的条数)。

    五个动作，顺序不能变：
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    返回值是 (float, int) 而不是 (tensor, int) —— 记住 loss 要 .item()，
    否则调用方把它累加起来时，整张计算图会被一直拎着不放（内存只涨不降）。
    """
    raise AssertionError(
        "TODO 3 未完成：把上面那五个动作按顺序写出来，然后返回 (loss 的 float, 猜对的条数)\n"
        "  猜对的条数 = (logits.argmax(dim=1) == yb).sum().item()\n"
        "  别忘了把 xb / yb 搬到 device 上"
    )


# ═══════════════════════════════════════════════════════════════════
# TODO 4 · 评估
# ═══════════════════════════════════════════════════════════════════
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """在 loader 上跑一遍，返回 (平均 loss, 准确率)。

    跟 train_step 有三处不同，每一处都有原因：
        1. model.eval()       关掉 Dropout / 冻结 BN 的统计量
        2. torch.no_grad()    不建计算图（不做反向，建图纯属浪费）
        3. 不做 optimizer 的三件套：不 zero_grad、不 backward、不 step

    还有一处**相同**但其实很容易写错的：loss 要按样本数加权，
    因为最后一个 batch 可能不满（2400 条 / bs=64 → 最后一个只有 32 条）。
    直接对 batch loss 求平均，等于给了小 batch 过高的权重。
    """
    raise AssertionError(
        "TODO 4 未完成：eval 模式下遍历 loader，累计 loss 和猜对的条数\n"
        "  累计 loss 要乘上本 batch 的实际条数（xb.shape[0]），最后除以总条数\n"
        "  返回 (loss / n, 猜对的总数 / n)"
    )


# ═══════════════════════════════════════════════════════════════════
# 5. 一个 epoch 的循环（已给）
# ═══════════════════════════════════════════════════════════════════
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """遍历 loader，每个 batch 调一次 train_step，返回 (平均 loss, 准确率)。"""
    model.train()  # 告诉模型「在训练」—— 有 Dropout/BN 时才看得出区别
    total_loss, total_correct, n = 0.0, 0, 0

    for xb, yb in loader:
        loss, correct = train_step(model, xb, yb, optimizer, criterion, device)
        bs = xb.shape[0]           # ← 本 batch 的真实条数，不是 batch_size
        total_loss += loss * bs    # ← 加权：小 batch 不能和满 batch 等权
        total_correct += correct
        n += bs

    return total_loss / n, total_correct / n


def trace_shapes(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    log,
    n_batches: int = 3,
) -> None:
    """把一个 batch 从输入到 loss 的张量形状逐站打印一遍（已给）。

    这是训练循环的「X 光片」。以后任何一个模型接不上，先跑一遍这个，
    看一眼形状在哪一站对不上，比读十遍代码快。
    """
    model.eval()
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            if i >= n_batches:
                break
            xb_d, yb_d = xb.to(device), yb.to(device)
            logits = model(xb_d)
            loss = criterion(logits, yb_d)
            log.info(
                f"    [形状] batch {i}: xb={tuple(xb_d.shape)}  yb={tuple(yb_d.shape)}  "
                f"logits={tuple(logits.shape)}  loss={tuple(loss.shape)}"
            )
    log.info("    [形状] 读法：xb 第 0 维是条数 B，yb 是 (B,) 的类别下标，")
    log.info("    [形状]      logits 是 (B, 类别数)，loss 是 () —— 0 维标量，不是 (1,)")


# ═══════════════════════════════════════════════════════════════════
# 6. 三个验证（已给）
# ═══════════════════════════════════════════════════════════════════
def check_dataset_interface(X: torch.Tensor, y: torch.Tensor, log) -> None:
    """[关 1] 自定义 Dataset 和 TensorDataset 逐元素一致。"""
    log.info("[关 1] Dataset 接口")
    custom, builtin = BlobsDataset(X, y), TensorDataset(X, y)

    assert len(custom) == X.shape[0], f"len(ds) 应为 {X.shape[0]}，实际 {len(custom)}"
    log.info(f"    len(ds) = {len(custom)}   （= X 的行数）")

    for i in (0, 7, -1):
        xa, ya = custom[i]
        xb, yb = builtin[i]
        log.info(
            f"    ds[{i:>2}]: X{tuple(xa.shape)} y{tuple(ya.shape)} 0 维={ya.dim()}"
            f"  与官方的 TensorDataset 一致={torch.equal(xa, xb) and torch.equal(ya, yb)}"
        )
        assert xa.shape == (X.shape[1],), f"X[{i}] 应为 ({X.shape[1]},)，实际 {tuple(xa.shape)}"
        assert ya.dim() == 0, (
            f"y[{i}] 应该是 0 维标量，实际 {ya.dim()} 维、形状 {tuple(ya.shape)}。\n"
            "  写成 y[i:i+1] 就会变成 (1,) —— 那样 collate 之后 yb 会是 (B,1)，"
            "CrossEntropyLoss 会报 expected scalar type / 形状不匹配"
        )
        assert torch.equal(xa, xb) and torch.equal(ya, yb), "和 TensorDataset 取出的样本不一致"


def check_batching(X: torch.Tensor, y: torch.Tensor, log) -> None:
    """[关 2] 每个 batch 的形状对，且所有 batch 条数加起来 = 总数。"""
    log.info("[关 2] DataLoader 切批")
    n = X.shape[0]
    head = f"    {'batch_size':>10}{'每 epoch 步数':>14}{'最后一个 batch':>16}{'条数合计':>10}"
    log.info(head)

    for bs in (8, 64, 256, n):
        loader = build_loader(X, y, bs, shuffle=False)
        sizes = [xb.shape[0] for xb, _ in loader]
        ok = sum(sizes) == n
        log.info(
            f"    {bs:>10}{len(loader):>14}{sizes[-1]:>16}{sum(sizes):>10}   "
            f"{'OK' if ok else 'MISMATCH'}"
        )
        assert ok, f"batch_size={bs}：各 batch 条数合计 {sum(sizes)} != 总数 {n}"

        xb, yb = next(iter(loader))
        assert xb.shape == (bs, X.shape[1]), f"xb 形状应为 ({bs},{X.shape[1]})，实际 {tuple(xb.shape)}"
        assert yb.shape == (bs,), f"yb 形状应为 ({bs},)，实际 {tuple(yb.shape)}"
        assert xb.dtype == torch.float32 and yb.dtype == torch.int64, (
            f"dtype 不对：xb={xb.dtype}（要 float32）, yb={yb.dtype}（要 int64）"
        )

    log.info("    注意 bs=64 那一行：2400/64=37.5，所以 38 步，最后一步只有 32 条。")
    log.info("    这 32 条如果和 64 条的 batch 等权平均，epoch loss 就偏了 —— 所以累加要乘 bs。")


def check_loss_weighting(loader: DataLoader, model, criterion, device, log) -> None:
    """[关 3] 「按 bs 加权」和「直接平均」差多少（量化一下这个静默错误）。"""
    log.info("[关 3] epoch loss 的加权方式")
    model.eval()
    with torch.no_grad():
        losses = [criterion(model(xb.to(device)), yb.to(device)).item() for xb, yb in loader]
        sizes = [xb.shape[0] for xb, _ in loader]

    weighted = sum(l * s for l, s in zip(losses, sizes)) / sum(sizes)
    plain = sum(losses) / len(losses)
    log.info(f"    按 bs 加权 = {weighted:.6f}   直接平均 = {plain:.6f}")
    log.info(f"    相对偏差   = {abs(plain - weighted) / weighted * 100:.2f}%")
    log.info("    偏差不大，但它不报错、也不会自己消失 —— 属于「你永远不会发现」的那一类。")


# ═══════════════════════════════════════════════════════════════════
# 7. batch_size 实验（已给）
# ═══════════════════════════════════════════════════════════════════
def compare_batch_sizes(
    Xtr, ytr, Xva, yva, args, criterion, device, log
) -> None:
    """同 seed / 同 epoch 数 / 同 lr，只改 batch_size。"""
    n = Xtr.shape[0]
    log.info("=" * 68)
    log.info(f"batch_size 实验：epochs={args.epochs} lr={args.lr} 只改 batch_size")
    log.info(f"    {'bs':>6}{'步/epoch':>10}{'总更新次数':>12}{'首个 epoch':>12}{'末个 epoch':>12}{'val acc':>10}")

    for bs in (8, 64, n):
        tr = build_loader(Xtr, ytr, bs, shuffle=True, seed=args.seed)
        va = build_loader(Xva, yva, 512, shuffle=False)

        torch.manual_seed(args.seed)
        model = MLP(2, args.hidden, 3, args.n_layers, "relu")
        init_weights(model, "kaiming")
        model.to(device)
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)

        hist = []
        for _ in range(args.epochs):
            hist.append(train_one_epoch(model, tr, opt, criterion, device)[0])
        _, va_acc = evaluate(model, va, criterion, device)

        log.info(
            f"    {bs:>6}{len(tr):>10}{len(tr) * args.epochs:>12}"
            f"{hist[0]:>12.3f}{hist[-1]:>12.3f}{va_acc:>10.4f}"
        )
        log.info("           loss 轨迹: " + " ".join(f"{v:.3f}" for v in hist))

    log.info("    读法：bs=8 每个 epoch 走 300 步，20 个 epoch 就是 6000 次更新；")
    log.info("          全量 batch 每个 epoch 只走 1 步，20 个 epoch 只有 20 次更新。")
    log.info("          **epoch 不是统一的进度单位，更新次数才是。**")


# ═══════════════════════════════════════════════════════════════════
# 8. 主流程
# ═══════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Day 11 数据集与训练循环")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compare", action="store_true", help="跑 8 / 64 / 全量 三种 batch_size")
    args = parser.parse_args()

    log = get_logger("day11", LOG)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    log.info("=" * 68)
    log.info("Day 11 · 数据集与训练循环")
    log.info(f"torch {torch.__version__} | device={device}")
    log.info("=" * 68)

    # ── 数据
    X, y = make_blobs(seed=args.seed)
    Xtr, ytr, Xva, yva = split(X, y)
    log.info("[0] 数据")
    log.info(f"    全量 X{tuple(X.shape)} y{tuple(y.shape)}  每类 {torch.bincount(y).tolist()} 条")
    log.info(f"    train X{tuple(Xtr.shape)}   val X{tuple(Xva.shape)}")

    check_dataset_interface(Xtr, ytr, log)
    check_batching(Xtr, ytr, log)

    tr_loader = build_loader(Xtr, ytr, args.batch_size, shuffle=True, seed=args.seed)
    va_loader = build_loader(Xva, yva, 512, shuffle=False)

    # ── 模型（Day 10 的 MLP）
    torch.manual_seed(args.seed)
    model = MLP(2, args.hidden, 3, args.n_layers, "relu")
    init_weights(model, "kaiming")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    log.info("[1] 模型")
    log.info(f"    MLP(2, {args.hidden}, 3, n_layers={args.n_layers})  参数量={sum(p.numel() for p in model.parameters())}")
    log.info(f"    优化器 SGD(lr={args.lr})   batch_size={args.batch_size}   epochs={args.epochs}")

    log.info("[2] 张量形状全程追踪")
    trace_shapes(model, tr_loader, criterion, device, log)

    check_loss_weighting(tr_loader, model, criterion, device, log)

    # ── 训练
    log.info("[3] 训练")
    first = None
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, va_loader, criterion, device)
        if first is None:
            first = (tr_loss, va_loss)
        if epoch <= 5 or epoch % 5 == 0:
            log.info(
                f"    epoch {epoch:>2}/{args.epochs}  "
                f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  |  "
                f"val loss={va_loss:.4f} acc={va_acc:.4f}"
            )

    log.info("=" * 68)
    log.info(f"[4] 结果：train loss {first[0]:.4f} -> {tr_loss:.4f}   "
             f"val loss {first[1]:.4f} -> {va_loss:.4f}   val acc {va_acc:.4f}")
    assert tr_loss < first[0] * 0.5, (
        f"训练 {args.epochs} 个 epoch 后 train loss 只从 {first[0]:.4f} 降到 {tr_loss:.4f}，"
        "降幅不到一半。三个常见原因，按概率排：\n"
        "  ① 漏了 optimizer.step()          —— loss 会一动不动\n"
        "  ② 漏了 optimizer.zero_grad()     —— 梯度逐 batch 累加，等效 lr 越滚越大，loss 会先降后炸\n"
        "  ③ criterion 前面多加了 softmax   —— CrossEntropyLoss 自带 softmax，加两次损失信息，"
        "loss 卡在 0.56 附近下不去"
    )
    assert abs(tr_acc - va_acc) < 0.3, (
        f"train acc={tr_acc:.4f} 和 val acc={va_acc:.4f} 差了 {abs(tr_acc - va_acc):.2f}。\n"
        "  正常情况两者应该接近（差 0.01~0.05）。差得离谱通常是**统计写错了**而不是训练坏了：\n"
        "  ① 算准确率时 logits.argmax() 漏了 dim=1 —— 对整批求一次全局最大值，"
        "和每个标签比必然全错，train acc 会恒等于 0\n"
        "  ② 累计分母错了（除以 batch 数而不是样本数）"
    )
    assert va_acc > 0.95, (
        f"val acc 只有 {va_acc:.4f}。逐项查：\n"
        "  ① train_step 里是否真的 backward + step 了\n"
        "  ② 模型是否 .to(device)，数据是否也 .to(device)\n"
        "  ③ criterion 收到的是 logits 还是 softmax 之后的值（这里应该直接给 logits）"
    )
    log.info("    两关都过：loss 降幅过半 + val acc > 0.95")

    if args.compare:
        compare_batch_sizes(Xtr, ytr, Xva, yva, args, criterion, device, log)

    log.info("=" * 68)
    log.info(f"报告已写入 {LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
