"""Day 13 · MNIST + MLP 第一版：第一个真数据集

跑法（PowerShell，项目根目录）：
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\train_mnist.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\train_mnist.py --epochs 2 --tag repro
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\train_mnist.py --optimizer mysgd

这一版只追求「全流程跑通」，不追求最好成绩（那是 Day 15 的事）。
四个交付物都落在 outputs/<时间戳>_mnist_<tag>/ 里：
    config.yaml   这次跑到底用了什么参数
    train.log     完整日志
    history.json  每个 epoch 的 loss / 准确率（复现性就是对比这个文件）
    curve.png     两条曲线
    best.pt       验证集最好的那一版权重

════════════════════════════════════════════════════════════════════
整个脚本的执行顺序（顺序本身有讲究，别随手调换）

    读配置 → 定设备 → 建目录 → 设种子 → 建数据 → 建模型 → 建优化器 → 训练

  前三步在"造假"之前：目录建不出来要在花 10 分钟训练**之前**就知道。
  设种子必须在建模型、建数据**之前**：模型初始化要抽随机数，种子设晚了
  就等于让 Python 用上一个默认种子初始化的权重。
════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# 让 `python scripts/train_mnist.py` 也能 import 到 src（否则 ModuleNotFoundError）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 画图必须放在 pyplot 之前：Agg 是"只画到文件、不弹窗"的后端。
# 不加这一行，某些环境下 plt.show() 会卡住或报没有显示设备。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.mnist import build_loaders
from src.models.mlp import MLP, count_params_manual
from src.utils.config import load_config, save_config
from src.utils.logger import get_logger
from src.utils.optim import MySGD, MyStepLR
from src.utils.paths import CONFIGS, run_dir
from src.utils.seed import set_seed

# MNIST 的固定形状，写成常量，避免 784 这个数字散落在各处
IMG_SIZE = 28
IN_DIM = IMG_SIZE * IMG_SIZE   # 784
NUM_CLASSES = 10


# ═══════════════════════════════════════════════════════════════════
# 1. 造模型 / 优化器 / 调度器
# ═══════════════════════════════════════════════════════════════════
def build_model(cfg) -> nn.Module:
    """784 -> 256 -> 10 的两层 MLP（Day 10 自己写的那一个）。

    为什么输入是 784：一张 28×28 的灰度图摊平就是 784 个数。
    MLP 不认识"图像"，它只认识"一行数字"，所以图像必须先摊平。
    （摊平这件事放在训练循环里做，见 train_one_epoch 的第一行。）

    Day 15：多传一个 dropout。cfg.dropout=0 时结构和 Day 13 完全相同。
    """
    return MLP(
        in_dim=IN_DIM,
        hidden=cfg.hidden,
        out_dim=NUM_CLASSES,
        n_layers=cfg.n_layers,
        act=cfg.act,
        dropout=cfg.dropout,
    )


def weight_norm(model) -> float:
    """所有参数拼起来的 L2 范数 sqrt(Σ p²)。

    Day 15 用它观察 weight decay 到底干了什么：
    wd 的作用是**把权重往 0 拉**，所以开了 wd 之后这个数应该比不开时小。
    如果开了 wd 这个数没变小，说明 wd 根本没生效（比如传错了优化器）。
    """
    total = 0.0
    for p in model.parameters():
        total += p.detach().pow(2).sum().item()
    return total ** 0.5


def build_optimizer(cfg, model) -> torch.optim.Optimizer:
    """按配置造优化器。三个选项里有一个是你自己写的（mysgd）。

    mysgd 和 sgd 的接口故意做成一样，所以你只要改一个 yaml 字段就能切换，
    训练脚本一个字都不用动 —— 这就是"接口一致"的好处。
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


def build_scheduler(cfg, opt):
    """学习率调度器。none 就返回 None，训练循环里会判一下。"""
    if cfg.scheduler == "none":
        return None
    if isinstance(opt, MySGD):
        # 手写优化器的 lr 挂在 opt.lr 上，用我们自己的 StepLR
        return MyStepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    # torch 优化器的 lr 藏在 param_groups 里，用它自带的
    return torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)


def current_lr(opt) -> float:
    """取当前学习率。两种优化器存 lr 的位置不一样，这里统一一下。"""
    if isinstance(opt, MySGD):
        return opt.lr
    return opt.param_groups[0]["lr"]


# ═══════════════════════════════════════════════════════════════════
# 2. 训练 / 评估
# ═══════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, opt, crit, device, log=None) -> tuple[float, float, int]:
    """跑完一个 epoch，返回 (平均 loss, 准确率, 样本数)。

    循环体就是 Day 11 那五行，一行都没变：
        zero_grad → forward → loss → backward → step
    这里多的只是"累加统计量"和"第一次打印形状"。
    """
    # train() 打开训练模式。对纯 MLP 没有影响，但一旦加了 Dropout/BatchNorm
    # 就必须有，否则行为是错的（而且不报错）。写在这里是为了养成习惯。
    model.train()

    total_loss, total_correct, n = 0.0, 0, 0

    for step, (xb, yb) in enumerate(loader):
        # ── 只在第一个 batch 打印张量形状 ─────────────────────────
        # 这个动作看起来傻，但它能一眼看出"哪一步的形状不对"。
        # 养成习惯比省这几行输出重要。
        if step == 0 and log is not None:
            log.info(f"    形状检查  xb={tuple(xb.shape)}  yb={tuple(yb.shape)} "
                     f"({yb.dtype})")

        # 图像 (B, 1, 28, 28) → 摊平 (B, 784)。第 0 维是"几条"，不动它；
        # -1 让 PyTorch 自己算（1×28×28 = 784）。
        xb = xb.view(xb.size(0), -1)

        # 数据搬到和模型同一个设备上。漏了这行 → 报 "Expected all tensors
        # to be on the same device"（Day 11 实测过这个报错）
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad()              # ① 擦掉上一步的梯度
        logits = model(xb)           # ② 前向：(B,784) → (B,10)
        loss = crit(logits, yb)      # ③ 打分：一个标量 ()
        loss.backward()              # ④ 追责：算出每个参数的梯度
        opt.step()                   # ⑤ 微调：按梯度挪一小步

        # ── 累加统计（按样本数加权，Day 11 算过不加权会错多少）──
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        n += bs

    return total_loss / n, total_correct / n, n


@torch.no_grad()  # 评估不需要梯度，关掉能省显存、也能防止手滑改了参数
def evaluate(model, loader, crit, device) -> tuple[float, float]:
    """在验证/测试集上算 (平均 loss, 准确率)。"""
    model.eval()   # 评估模式：Dropout 关掉、BatchNorm 用running统计量

    total_loss, total_correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.view(xb.size(0), -1)
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = crit(logits, yb)

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        n += bs

    return total_loss / n, total_correct / n


# ═══════════════════════════════════════════════════════════════════
# 3. 画曲线
# ═══════════════════════════════════════════════════════════════════
def plot_curves(history: list[dict], out_path: Path) -> None:
    """把每个 epoch 的 loss / 准确率 / 两者差距画成三张图。

    train 和 val 一定画在同一张里 —— 单看 train 曲线你以为自己很厉害，
    两条放一起才能看出"是学会了，还是背下来了"（过拟合）。

    Day 15 加了第三张 gap = train_acc - val_acc：
    这张图是过拟合的"体温计"，单调往上爬就是过拟合在加重。
    看 loss 那两张也能判断，但 gap 这种"把差距单独拎出来"的画法
    更不容易被"两边都在涨"的表象骗过去。
    """
    epochs = [h["epoch"] for h in history]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.6))

    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], marker="s", label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["train_acc"] for h in history], marker="o", label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], marker="s", label="val")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 老格式的 history 里没有 gap 字段，用算式兜一下（不然画图会 KeyError）
    gaps = [h.get("gap", h["train_acc"] - h["val_acc"]) for h in history]
    ax3.plot(epochs, gaps, marker="^", color="crimson", label="train - val")
    ax3.axhline(0, color="gray", lw=1, ls="--")   # 0 线：两条曲线在这里交叉
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("acc gap")
    ax3.set_title("overfitting gap")
    ax3.legend()
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)   # 关掉，否则连着跑多次会积累一堆图对象


# ═══════════════════════════════════════════════════════════════════
# 4. 命令行
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(description="MNIST + MLP 训练")
    ap.add_argument("--config", type=Path, default=CONFIGS / "mnist_mlp.yaml")
    # 下面这些都设成 None：None 表示"用户没传"，就不去覆盖 yaml
    # 如果写成 default=10，就永远分不清"用户想要 10"和"用户没说话"
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--optimizer", type=str, default=None, choices=["sgd", "adam", "mysgd"])
    ap.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=None)
    # ── Day 15 新增：三个正则化旋钮，都能从命令行直接试 ──────────────
    ap.add_argument("--dropout", type=float, default=None, help="丢弃概率，0 表示不加")
    ap.add_argument("--weight-decay", type=float, default=None, help="L2 惩罚系数")
    ap.add_argument(
        "--patience", type=int, default=None,
        help="早停耐心值：验证指标连续这么多轮不改善就停（0 = 不早停）",
    )
    ap.add_argument(
        "--es-metric", type=str, default="loss", choices=["loss", "acc"],
        help="早停看哪个指标。默认 loss（连续、灵敏）；选 acc 会看到相反的结论",
    )
    ap.add_argument("--tag", type=str, default="run", help="输出目录后缀，方便区分多次实验")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # ── 命令行覆盖 yaml ───────────────────────────────────────────
    # 只把用户真的传了的键放进去（None 的丢掉）
    overrides = {
        k: v
        for k, v in {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "device": args.device,
            "seed": args.seed,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "early_stop_patience": args.patience,
        }.items()
        if v is not None
    }
    cfg = load_config(args.config, overrides)

    # ── 设备：配置要 cuda 但机器没有，就降级到 cpu 而不是崩掉 ──────
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[警告] 配置要 cuda，但这台机器没有可用 GPU，降级到 cpu")
        cfg.device = "cpu"
    device = torch.device(cfg.device)

    # ── 输出目录 + 日志 ──────────────────────────────────────────
    out = run_dir(f"mnist_{args.tag}")
    log = get_logger("train_mnist", log_file=out / "train.log")
    log.info("=" * 70)
    log.info(f"输出目录 {out}")
    log.info("=" * 70)

    # 存一份"这次到底用了什么参数"。半年后想复现，就靠这个文件
    save_config(cfg, out / "config.yaml")
    log.info(f"配置 {cfg}")

    # ── 设种子：必须在建模型、建数据之前 ─────────────────────────
    set_seed(cfg.seed)
    log.info(f"种子已固定 seed={cfg.seed}")

    # ── 数据 ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    train_loader, val_loader, test_loader = build_loaders(cfg)
    log.info(
        f"数据 train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"test={len(test_loader.dataset)}  "
        f"batch={cfg.batch_size}  步/epoch={len(train_loader)}  "
        f"({time.perf_counter() - t0:.2f}s)"
    )

    # ── 模型 ────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    actual = sum(p.numel() for p in model.parameters())
    manual = count_params_manual(IN_DIM, cfg.hidden, NUM_CLASSES, cfg.n_layers)
    log.info(f"模型 {cfg.n_layers} 层 hidden={cfg.hidden}  参数量={actual}")
    # 两种算法算同一个数，对不上说明模型结构和你想的不一样
    assert actual == manual, f"参数量对不上：运行时 {actual} vs 手算 {manual}"

    crit = nn.CrossEntropyLoss()
    opt = build_optimizer(cfg, model)
    sched = build_scheduler(cfg, opt)
    log.info(f"优化器 {cfg.optimizer}  lr={cfg.lr}  "
             f"momentum={cfg.momentum}  scheduler={cfg.scheduler}  "
             f"weight_decay={cfg.weight_decay}")
    log.info(f"正则化 dropout={cfg.dropout}  "
             f"early_stop_patience={cfg.early_stop_patience}  判据={args.es_metric}")

    # ── 训练 ────────────────────────────────────────────────────
    history: list[dict] = []
    timings: list[float] = []
    best_acc = -1.0
    best_path = out / "best.pt"

    # ── Day 15：早停所需的两个计数器 ───────────────────────────────
    # 【为什么要有两组计数】
    # 同一份训练记录，用 val_acc 当判据和用 val_loss 当判据，
    # 会得出**完全相反**的结论。原因：val_acc 是离散的。
    # 验证集 6000 条，正确 1 条 = 0.0167%，所以 acc 的"分辨率"就这么粗；
    # 而 val_loss 是连续值，模型每一点微小改善都能体现出来。
    # 两个计数器都留着，日志里都打出来，让你自己看见这个矛盾。
    best_val_loss = float("inf")
    bad_by_acc = 0     # 连续几轮 val_acc 没创新高
    bad_by_loss = 0    # 连续几轮 val_loss 没创新低
    stopped_at: int | None = None   # None = 没触发早停

    for epoch in range(1, cfg.epochs + 1):
        t_ep = time.perf_counter()

        tr_loss, tr_acc, n_tr = train_one_epoch(
            model, train_loader, opt, crit, device, log=log if epoch == 1 else None
        )
        va_loss, va_acc = evaluate(model, val_loader, crit, device)

        # 调度器按 epoch 走（不是按 batch —— 按 batch 会衰减快 469 倍）
        if sched is not None:
            sched.step()

        rec = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 6),
            "train_acc": round(tr_acc, 6),
            "val_loss": round(va_loss, 6),
            "val_acc": round(va_acc, 6),
            "gap": round(tr_acc - va_acc, 6),   # Day 15：过拟合的直接读数
            "wnorm": round(weight_norm(model), 6),  # Day 15：看 wd 有没有生效
            "lr": current_lr(opt),
        }
        history.append(rec)
        # 耗时**不进 history**：它是墙钟时间，每次跑都不一样，
        # 混进去之后 "diff 两个 history.json" 就永远差那么几行，
        # 你就没法用这个文件证明复现性了。计时单独放 timings。
        timings.append(round(time.perf_counter() - t_ep, 2))
        log.info(
            f"epoch {epoch:>2}/{cfg.epochs}  "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  |  "
            f"val loss={va_loss:.4f} acc={va_acc:.4f}  |  "
            f"gap={rec['gap']:+.4f}  |  "
            f"({timings[-1]}s)"
        )

        # 只在验证集创新高时存盘。存"最好那一版"而不是"最后一版"，
        # 因为训练到后期可能开始过拟合，最后一版反而不如中间那版
        #
        # ⚠️ improved_by_acc 必须在更新 best_acc **之前**算，
        #    否则 best_acc 已经变成新值了，"va_acc > best_acc" 恒为 False，
        #    早停计数会一路涨上去 —— 第 2 轮就停。这个 bug 不报错，只是行为错。
        improved_by_acc = va_acc > best_acc
        if improved_by_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)

        # 注意：checkpoint 里存的是 state_dict()（Day 10 讲过为什么不是 pickle 整个 model）

        # ── Day 15：早停判断 ──────────────────────────────────────
        # 两个计数器一起更新（就算只用一个判据，另一个也要记账，
        # 因为日志里要能对比"另一个判据会怎么判"）
        # 1e-6 是"最小可感知改善"。不加这个容差，浮点噪声级的抖动
        # （比如 0.0789 → 0.07891）也会被当成"有进步"，早停永远不触发。
        improved_by_loss = va_loss < best_val_loss - 1e-6

        bad_by_acc = 0 if improved_by_acc else bad_by_acc + 1
        bad_by_loss = 0 if improved_by_loss else bad_by_loss + 1
        if improved_by_loss:
            best_val_loss = va_loss

        if cfg.early_stop_patience > 0:
            # 按用户选的判据决定是否停
            bad = bad_by_acc if args.es_metric == "acc" else bad_by_loss
            if bad >= cfg.early_stop_patience:
                stopped_at = epoch
                log.info(
                    f"  ⏹ 早停：按 {args.es_metric} 判据已连续 {bad} 轮没改善"
                    f"（patience={cfg.early_stop_patience}）"
                )
                log.info(
                    f"     另一个判据此时是："
                    f"val_acc 连续 {bad_by_acc} 轮没新高、"
                    f"val_loss 连续 {bad_by_loss} 轮没新低"
                )
                break

    train_time = sum(timings)
    if stopped_at is not None:
        log.info(
            f"[早停] 实际跑了 {stopped_at}/{cfg.epochs} 轮，"
            f"省下 {cfg.epochs - stopped_at} 轮 ≈ {train_time / stopped_at * (cfg.epochs - stopped_at):.1f}s"
        )
    else:
        log.info(f"[早停] 未触发（patience={cfg.early_stop_patience}），跑满 {len(history)} 轮")

    # ── 曲线 + 记录 ─────────────────────────────────────────────
    plot_curves(history, out / "curve.png")
    log.info(f"曲线已保存 {out / 'curve.png'}")

    # ── 最后才看测试集：载入"最好那一版"，只测一次 ───────────────
    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_acc = evaluate(model, test_loader, crit, device)
    log.info("-" * 70)
    log.info(f"best val acc = {best_acc:.4f}")
    log.info(f"test loss={te_loss:.4f}  test acc={te_acc:.4f}   （只在最后看这一次）")
    log.info(f"总训练时间 {train_time:.1f}s")

    # history.json 就是"复现性"要对比的那个文件：
    # 同一个 seed 跑两次，这两个文件必须逐字节相同。
    #
    # 所以它里面**只能放"确定性"的内容**：参数、每轮指标、最终测试结果。
    # 耗时绝对不能写进来 —— 它是墙钟时间，每次跑都不一样，
    # 一旦写进去，`diff 两个 history.json` 就永远有差异，
    # 你就再也没法用这个文件证明"两次跑得一模一样"了。
    # （每轮耗时打在 train.log 里，想看随时能看。）
    record = {
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "history": history,
        "test": {"loss": round(te_loss, 6), "acc": round(te_acc, 6)},
        "best_val_acc": round(best_acc, 6),
        # Day 15：早停结算。都是确定性值，不影响复现性对比
        "early_stop": {
            "metric": args.es_metric,
            "patience": cfg.early_stop_patience,
            "stopped_at": stopped_at,          # None 表示没触发
            "epochs_run": len(history),
        },
        "final_wnorm": round(weight_norm(model), 6),
    }
    (out / "history.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(f"记录已保存 {out / 'history.json'}")


if __name__ == "__main__":
    main()
