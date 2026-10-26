"""Trainer：把整条训练流水线装进一个类。

════════════════════════════════════════════════════════════════════
执行顺序（顺序本身有讲究，别随手调换）
════════════════════════════════════════════════════════════════════

    定设备 → 建目录 → 设种子 → 建数据 → 建模型 → 建优化器 → 训练

· 前三步在"造假"之前：目录建不出来要在花 10 分钟训练**之前**就知道。
· **设种子必须在建模型、建数据之前**：模型初始化要抽随机数，
  种子设晚了就等于让 PyTorch 用上一个默认种子初始化的权重 ——
  而且它不报错，你只会觉得"怎么复现不了"。

════════════════════════════════════════════════════════════════════
为什么做成类，而不是继续写一个 500 行的 main()
════════════════════════════════════════════════════════════════════

不是为了"面向对象好看"，是因为有三样东西**训练中途要一直改写**：
    best_acc / best_val_loss / 早停计数器 / history / model
写成局部变量的话，要么层层传参（7 个参数），要么用 global。
写成实例属性，`self.best_acc` 在 fit() 里任何位置都能读写，
而且**跑完之后还能从外面读出来**（`trainer.best_acc`）——
这一点对 P1 之后要做的「跑完再分析」很有用。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

# 这行 import 看着"没用到任何东西"，但它就是**注册动作本身**：
# import builtin 模块 ⇒ 模块体里的 @register_model / @register_dataset 被执行
# ⇒ "mlp" 和 "mnist" 进了注册表。删掉这行，build_model("mlp") 会报「未知模型」。
# （这种"为了副作用而 import"的写法在 Python 里常见，加 noqa 注释是为了
#   告诉 linter 和读代码的人：这是故意的，不是忘了删。）
import src.engine.builtin  # noqa: F401
from src.engine.builders import (
    build_criterion,
    build_optimizer,
    build_scheduler,
    count_params,
    get_lr,
    weight_norm,
)
from src.engine.loops import evaluate, train_one_epoch
from src.engine.registry import build_dataset, build_model
from src.utils.config import save_config
from src.utils.logger import get_logger
from src.utils.paths import run_dir
from src.utils.plotting import plot_curves
from src.utils.seed import set_seed


class Trainer:
    """一次完整的训练 = 一个 Trainer 实例。

    用法：
        trainer = Trainer(cfg, tag="baseline")
        record = trainer.fit()
        print(record["test"]["acc"])
    """

    def __init__(self, cfg, tag: str = "run") -> None:
        self.cfg = cfg
        self.tag = tag

        # ── 设备：配置要 cuda 但机器没有，就降级到 cpu 而不是崩掉 ────
        # 为什么不直接 raise：因为「在笔记本上写完、拿到服务器上跑」
        # 是常态。降级 + 一句警告，比一个 Traceback 有用得多。
        # 但一定要**打印出来** —— 静默降级会让你以为自己在用 GPU，
        # 然后纳闷为什么这一跑慢了 20 倍。
        if cfg.device == "cuda" and not torch.cuda.is_available():
            print("[警告] 配置要 cuda，但这台机器没有可用 GPU，降级到 cpu")
            cfg.device = "cpu"
        self.device = torch.device(cfg.device)

        # ── 输出目录 + 日志 ──────────────────────────────────────────
        # 目录名带时间戳，所以同一个 tag 跑多次不会互相覆盖。
        # 前缀用 cfg.dataset（不是写死 "mnist"）—— 换数据集时目录名跟着变。
        self.out: Path = run_dir(f"{cfg.dataset}_{tag}")
        self.log = get_logger("train", log_file=self.out / "train.log")
        self.log.info("=" * 70)
        self.log.info(f"输出目录 {self.out}")
        self.log.info("=" * 70)

        # 存一份"这次到底用了什么参数"。半年后想复现，就靠这个文件
        save_config(cfg, self.out / "config.yaml")
        self.log.info(f"配置 {cfg}")

        # ── 待 fit() 填充的状态 ────────────────────────────────────
        self.model = None
        self.crit = None
        self.opt = None
        self.sched = None
        self.history: list[dict] = []
        self.timings: list[float] = []
        self.best_acc = -1.0
        self.best_val_loss = float("inf")
        self.stopped_at: int | None = None

    # ═══════════════════════════════════════════════════════════════
    # 组装：数据 → 模型 → 优化器
    # ═══════════════════════════════════════════════════════════════
    def build(self) -> None:
        """建数据、模型、损失、优化器、调度器。

        单独拆成一个方法（而不是塞进 fit）是为了**能在训练前单独调**：
            t = Trainer(cfg)
            t.build()
            print(sum(p.numel() for p in t.model.parameters()))   # 只想看参数量
        不需要为此白跑 10 个 epoch。
        """
        cfg, log = self.cfg, self.log

        # ── 设种子：必须在建模型、建数据之前 ─────────────────────
        set_seed(cfg.seed)
        log.info(f"种子已固定 seed={cfg.seed}")

        # ── 数据 ────────────────────────────────────────────────
        t0 = time.perf_counter()
        self.train_loader, self.val_loader, self.test_loader = build_dataset(
            cfg.dataset, cfg
        )
        log.info(
            f"数据 [{cfg.dataset}] train={len(self.train_loader.dataset)} "
            f"val={len(self.val_loader.dataset)} test={len(self.test_loader.dataset)}  "
            f"batch={cfg.batch_size}  步/epoch={len(self.train_loader)}  "
            f"({time.perf_counter() - t0:.2f}s)"
        )

        # ── 模型 ────────────────────────────────────────────────
        self.model = build_model(cfg.model, cfg).to(self.device)
        actual = count_params(self.model)
        log.info(f"模型 [{cfg.model}] {cfg.n_layers} 层 hidden={cfg.hidden}  参数量={actual}")
        # 参数量对账不在这里做 —— 它依赖「这个模型的公式」，
        # 而 trainer 不该认识任何具体模型。MLP 自己的公式在 build_mlp 里对。

        # ── 损失 / 优化器 / 调度器 ───────────────────────────────
        self.crit = build_criterion(cfg)
        self.opt = build_optimizer(cfg, self.model)
        self.sched = build_scheduler(cfg, self.opt)

        log.info(f"优化器 {cfg.optimizer}  lr={cfg.lr}  "
                 f"momentum={cfg.momentum}  scheduler={cfg.scheduler}  "
                 f"weight_decay={cfg.weight_decay}")
        log.info(f"损失函数 {cfg.loss}  →  {type(self.crit).__name__}")
        # 顺手打印第一层的类名，一眼确认到底用的是哪个 Linear。
        # 别信「我传了参数所以肯定生效了」—— 参数名拼错会被 argparse 拦下，
        # 但如果哪天 build_model 里的分支写反了，只有打印出来才看得见。
        first_linear = next((m for m in self.model.modules() if hasattr(m, "in_features")), None)
        if first_linear is not None:
            log.info(f"线性层 {cfg.linear}  →  {type(first_linear).__name__}")
        log.info(f"正则化 dropout={cfg.dropout}  flatten={cfg.flatten}  "
                 f"early_stop_patience={cfg.early_stop_patience}  "
                 f"判据={cfg.early_stop_metric}")

    # ═══════════════════════════════════════════════════════════════
    # 训练主循环
    # ═══════════════════════════════════════════════════════════════
    def fit(self) -> dict:
        """跑满（或早停），返回可以写进 history.json 的 record。"""
        cfg, log = self.cfg, self.log

        if self.model is None:
            self.build()

        best_path = self.out / "best.pt"

        # ── 早停所需的两个计数器 ─────────────────────────────────────
        # 【为什么要有两组计数】
        # 同一份训练记录，用 val_acc 当判据和用 val_loss 当判据，
        # 会得出**完全相反**的结论。原因：val_acc 是离散的。
        # 验证集 6000 条，正确 1 条 = 0.0167%，所以 acc 的"分辨率"就这么粗；
        # 而 val_loss 是连续值，模型每一点微小改善都能体现出来。
        # 两个计数器都留着，日志里都打出来，让你自己看见这个矛盾。
        bad_by_acc = 0     # 连续几轮 val_acc 没创新高
        bad_by_loss = 0    # 连续几轮 val_loss 没创新低

        for epoch in range(1, cfg.epochs + 1):
            t_ep = time.perf_counter()

            tr_loss, tr_acc, _ = train_one_epoch(
                self.model, self.train_loader, self.opt, self.crit,
                self.device, flatten=cfg.flatten,
                log=log if epoch == 1 else None,
            )
            va_loss, va_acc = evaluate(
                self.model, self.val_loader, self.crit, self.device, flatten=cfg.flatten
            )

            # 调度器按 epoch 走（不是按 batch —— 按 batch 会衰减快 469 倍）
            if self.sched is not None:
                self.sched.step()

            rec = {
                "epoch": epoch,
                "train_loss": round(tr_loss, 6),
                "train_acc": round(tr_acc, 6),
                "val_loss": round(va_loss, 6),
                "val_acc": round(va_acc, 6),
                "gap": round(tr_acc - va_acc, 6),        # 过拟合的直接读数
                "wnorm": round(weight_norm(self.model), 6),  # 看 wd 有没有生效
                "lr": get_lr(self.opt),
            }
            self.history.append(rec)
            # 耗时**不进 history**：它是墙钟时间，每次跑都不一样，
            # 混进去之后 "diff 两个 history.json" 就永远差那么几行，
            # 你就没法用这个文件证明复现性了。计时单独放 timings。
            self.timings.append(round(time.perf_counter() - t_ep, 2))
            log.info(
                f"epoch {epoch:>2}/{cfg.epochs}  "
                f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  |  "
                f"val loss={va_loss:.4f} acc={va_acc:.4f}  |  "
                f"gap={rec['gap']:+.4f}  |  "
                f"({self.timings[-1]}s)"
            )

            # 只在验证集创新高时存盘。存"最好那一版"而不是"最后一版"，
            # 因为训练到后期可能开始过拟合，最后一版反而不如中间那版。
            #
            # ⚠️ improved_by_acc 必须在更新 best_acc **之前**算，
            #    否则 best_acc 已经变成新值了，"va_acc > best_acc" 恒为 False，
            #    早停计数会一路涨上去 —— 第 2 轮就停。这个 bug 不报错，只是行为错。
            improved_by_acc = va_acc > self.best_acc
            if improved_by_acc:
                self.best_acc = va_acc
                # checkpoint 里存的是 state_dict()（不是 pickle 整个 model）
                torch.save(self.model.state_dict(), best_path)

            # ── 早停判断 ────────────────────────────────────────
            # 两个计数器一起更新（就算只用一个判据，另一个也要记账，
            # 因为日志里要能对比"另一个判据会怎么判"）。
            #
            # 1e-6 是"最小可感知改善"。不加这个容差，浮点噪声级的抖动
            # （比如 0.0789 → 0.07891）也会被当成"有进步"，早停永远不触发。
            improved_by_loss = va_loss < self.best_val_loss - 1e-6

            bad_by_acc = 0 if improved_by_acc else bad_by_acc + 1
            bad_by_loss = 0 if improved_by_loss else bad_by_loss + 1
            if improved_by_loss:
                self.best_val_loss = va_loss

            if cfg.early_stop_patience > 0:
                bad = bad_by_acc if cfg.early_stop_metric == "acc" else bad_by_loss
                if bad >= cfg.early_stop_patience:
                    self.stopped_at = epoch
                    log.info(
                        f"  [早停] 按 {cfg.early_stop_metric} 判据已连续 {bad} 轮没改善"
                        f"（patience={cfg.early_stop_patience}）"
                    )
                    log.info(
                        f"     另一个判据此时是："
                        f"val_acc 连续 {bad_by_acc} 轮没新高、"
                        f"val_loss 连续 {bad_by_loss} 轮没新低"
                    )
                    break

        # ── 早停结算 ────────────────────────────────────────────────
        train_time = sum(self.timings)
        if self.stopped_at is not None:
            saved = cfg.epochs - self.stopped_at
            log.info(
                f"[早停] 实际跑了 {self.stopped_at}/{cfg.epochs} 轮，"
                f"省下 {saved} 轮 ≈ {train_time / self.stopped_at * saved:.1f}s"
            )
        else:
            log.info(f"[早停] 未触发（patience={cfg.early_stop_patience}），"
                     f"跑满 {len(self.history)} 轮")

        # ── 曲线 ────────────────────────────────────────────────────
        plot_curves(self.history, self.out / "curve.png")
        log.info(f"曲线已保存 {self.out / 'curve.png'}")

        # ── 最后才看测试集：载入"最好那一版"，只测一次 ──────────────
        # 为什么不是用最后一版：见上面 best.pt 那段。
        # 为什么"只测一次"：每看一次测试集，你对它的判断就多一分污染。
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        te_loss, te_acc = evaluate(
            self.model, self.test_loader, self.crit, self.device, flatten=cfg.flatten
        )
        log.info("-" * 70)
        log.info(f"best val acc = {self.best_acc:.4f}")
        log.info(f"test loss={te_loss:.4f}  test acc={te_acc:.4f}   （只在最后看这一次）")
        log.info(f"总训练时间 {train_time:.1f}s")

        # ── 落盘 ────────────────────────────────────────────────────
        record = {
            # config 里放**全部**生效字段，包括 early_stop_metric。
            # 以前 es_metric 只存在于命令行，config.yaml 里没有它 ——
            # 那就意味着"按 loss 早停"和"按 acc 早停"两次跑出来的
            # config.yaml 一模一样，半年后根本分不清哪个是哪个。
            "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
            "history": self.history,
            "test": {"loss": round(te_loss, 6), "acc": round(te_acc, 6)},
            "best_val_acc": round(self.best_acc, 6),
            "early_stop": {
                "metric": cfg.early_stop_metric,
                "patience": cfg.early_stop_patience,
                "stopped_at": self.stopped_at,      # None 表示没触发
                "epochs_run": len(self.history),
            },
            "final_wnorm": round(weight_norm(self.model), 6),
        }
        # history.json 就是"复现性"要对比的那个文件：
        # 同一个 seed 跑两次，这两个文件必须逐字节相同。
        # 所以它里面**只能放"确定性"的内容**。耗时绝对不能写进来。
        (self.out / "history.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info(f"记录已保存 {self.out / 'history.json'}")
        return record
