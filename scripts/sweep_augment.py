"""Day 23 · 增强剂量扫描 + 早停离线分析

════════════════════════════════════════════════════════════════════
这个脚本回答两个问题，而它们是同一个问题的两面

  Q1  增强到底有没有用？用多少量？
  Q2  早停应该设几轮？用哪个判据？

── 为什么必须"扫"而不是"测一个点"──────────────────────────────
  Day 15 的血泪教训：当时只测了 weight_decay=5e-4 一个点，
  得出"正则化没用"的结论；后来补扫剂量，发现曲线是**倒 U 型**，
  最优在 1e-4，结论反了一半。

  数据增强一样有剂量问题：
      太小  → 模型看到的还是几乎一样的图，约束没增加，过拟合照旧
      太大  → 数字被转成别的数字（6 转成 9、7 转成 1），标签变成噪声，
              模型学不到东西，精度反而掉
  所以中间必然有个最优区间。只测一个点，你永远不知道自己在哪一侧。

── 为什么早停分析不用再跑一遍训练 ────────────────────────────
  早停是**只看验证曲线**做的决策：给定一条已经记录下来的
  val_loss / val_acc 序列，"patience=p 时会在第几轮停"是一个
  **确定性的、可以离线算出来的**函数。

  所以一次训练（每条曲线 15 轮）可以反推 10 种早停策略
  （2 个判据 × 5 个 patience）。这比"每种策略各训一遍"省 10 倍算力，
  而且保证 10 种策略比较的是**同一条曲线** —— 各训一遍的话，
  随机性会让它们根本不可比。

── 一个容易忽略的点：存 best checkpoint 时，早停不改变最终模型 ──
  Trainer 存的是「验证集上 best_acc 那一轮」的权重，不是最后一轮。
  所以在训练结束后 load best.pt 去测 test 的设定下，
  **早停唯一的作用是省时间，不改变结果** —— 除非它停得太早，
  把后面可能出现的更高点给掐掉了。

  这才是早停的真实风险，所以下面的分析会明确算出
  「停在 E 轮 vs 全局最优点 Q 轮」的差距。

用法（项目根目录）：
    # 全量 5 组 × 15 轮，约 14 分钟
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\sweep_augment.py

    # 先试水：3 轮、只跑前两组
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\sweep_augment.py --epochs 3 --only 0,1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # scripts/x.py → scripts → 项目根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.trainer import Trainer          # noqa: E402
from src.utils.config import load_config        # noqa: E402

CONFIG = ROOT / "configs" / "cnn_mnist.yaml"
OUTPUTS = ROOT / "outputs"


# ═══════════════════════════════════════════════════════════════════
# 1. 剂量表
# ═══════════════════════════════════════════════════════════════════
# 为什么是这个梯度（而不是 0/5/10 这种）：
#   MNIST 的数字线条宽度大约 2~3 像素，笔画间距也就 2~4 像素。
#   平移 1 像素 = 轻微抖动；2 像素 = 明显但可辨认；
#   3 像素以上 + 旋转 15°，已经能把 1 推成 7 的形状了。
#   所以 0/1/2/3 这四个点刚好覆盖「关 / 轻 / 中 / 过猛」四档。
#
# 旋转只从 10° 起步：手写数字本身有倾斜习惯（比如欧美人写 1 常常斜），
# 转太多会让 6 和 9、2 和 7 互相混淆 —— 那是真的在制造错标签。
DOSES: list[tuple[float, float, str, str]] = [
    (0.0, 0.0, "aug0", "baseline：无增强（应与 Day 22 的 cnn 逐位一致）"),
    (1.0, 0.0, "aug1", "轻：平移 ±1 像素"),
    (2.0, 0.0, "aug2", "中：平移 ±2 像素"),
    (2.0, 10.0, "aug3", "中上：平移 ±2 + 旋转 ±10°"),
    (3.0, 15.0, "aug4", "重：平移 ±3 + 旋转 ±15°（预期进入过猛区）"),
]


# ═══════════════════════════════════════════════════════════════════
# 2. 跑一组
# ═══════════════════════════════════════════════════════════════════
def run_dose(shift: float, angle: float, tag: str, epochs: int) -> dict:
    """用给定增强剂量训一次，返回 history.json 的内容。"""
    cfg = load_config(
        CONFIG,
        {
            "augment_shift": shift,
            "augment_angle": angle,
            "epochs": epochs,
            # 关掉早停：我们要拿到**完整 15 轮曲线**才能做离线早停分析。
            # 让 Trainer 自己早停的话，曲线会被截断，后面的分析就没数据了。
            "early_stop_patience": 0,
        },
    )
    trainer = Trainer(cfg, tag=tag)
    record = trainer.fit()
    print(f"\n    → {tag}: test acc = {record['test']['acc']:.4f}\n")
    return record


# ═══════════════════════════════════════════════════════════════════
# 3. 单条曲线的体检
# ═══════════════════════════════════════════════════════════════════
def curve_stats(record: dict) -> dict:
    """从一条 15 轮曲线里提取「过拟合程度」的读数。

    ── 为什么这么算 ────────────────────────────────────────────────
    Day 22 定的判据是三条同时出现才算真过拟合：
        val_loss 回升 + train_loss 还降 + gap 扩大
    这里把第一条量化成两个数：
        best_ep       val_loss 最低的那一轮（"该停在哪"的自然答案）
        rebound       最后一轮比最低点涨了百分之几
    """
    h = record["history"]
    losses = [r["val_loss"] for r in h]
    best_i = min(range(len(losses)), key=lambda i: losses[i])
    best_loss = losses[best_i]
    last_loss = losses[-1]
    return {
        "epochs": len(h),
        "best_ep": h[best_i]["epoch"],
        "best_val_loss": best_loss,
        "best_val_acc": max(r["val_acc"] for r in h),
        "last_val_loss": last_loss,
        "rebound_pct": (last_loss - best_loss) / best_loss * 100.0,
        "last_gap_pt": h[-1]["gap"] * 100.0,
        "best_gap_pt": h[best_i]["gap"] * 100.0,
        "test_acc": record["test"]["acc"],
        "train_acc_last": h[-1]["train_acc"],
    }


# ═══════════════════════════════════════════════════════════════════
# 4. 早停离线模拟
# ═══════════════════════════════════════════════════════════════════
def simulate_early_stop(history: list[dict], metric: str, patience: int) -> dict:
    """在一条已有曲线上模拟早停，返回「停在哪、那时什么水平」。

    逻辑必须和 Trainer.fit() 里那段**逐条对齐**，否则模拟出来的结论
    和你真跑一遍的结果对不上 —— 而这种不一致最难发现，因为两边都"看起来对"：
        · loss 判据：新低才算改善，容差 1e-6（和 Trainer 一致）
        · acc  判据：严格大于才算改善（Trainer 里是 va_acc > best_acc）
        · patience=0 → 不早停，跑满
    """
    if patience <= 0:
        last = history[-1]
        return {
            "stopped": False,
            "stop_ep": last["epoch"],
            "reason": "关闭",
        }

    best = float("inf") if metric == "loss" else -1.0
    bad = 0
    for r in history:
        v = r["val_loss"] if metric == "loss" else r["val_acc"]
        improved = (v < best - 1e-6) if metric == "loss" else (v > best)
        if improved:
            best = v
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            return {
                "stopped": True,
                "stop_ep": r["epoch"],
                "stop_val_loss": r["val_loss"],
                "stop_val_acc": r["val_acc"],
            }
    last = history[-1]
    return {
        "stopped": False,
        "stop_ep": last["epoch"],
        "stop_val_loss": last["val_loss"],
        "stop_val_acc": last["val_acc"],
    }


# ═══════════════════════════════════════════════════════════════════
# 5. 汇总打印
# ═══════════════════════════════════════════════════════════════════
def print_summary(results: list[tuple[str, str, dict]]) -> None:
    print("\n" + "=" * 96)
    print("一、剂量扫描（全部跑满，无早停）")
    print("=" * 96)
    print(f"{'剂量':<22}{'test acc':>10}{'best val':>10}{'最低val_loss':>14}"
          f"{'@ep':>5}{'末轮val_loss':>13}{'回升':>8}{'末gap(pt)':>11}")
    print("-" * 96)
    for tag, desc, st in results:
        print(f"{desc:<22}{st['test_acc']:>10.4f}{st['best_val_acc']:>10.4f}"
              f"{st['best_val_loss']:>14.4f}{st['best_ep']:>5}"
              f"{st['last_val_loss']:>13.4f}{st['rebound_pct']:>+7.1f}%"
              f"{st['last_gap_pt']:>+11.2f}")

    print("\n  读法：")
    print("    · 「回升」为正是过拟合的直接读数；接近 0 说明 15 轮内还没开始过拟合")
    print("    · 「@ep」是 val_loss 最低那一轮 —— 这天然就是「最该在哪停」的答案")
    print("    · test acc 是载入 best.pt（best val acc 那轮）后测的，全程只测一次")


def print_early_stop(results: list[tuple[str, str, dict]], records: dict) -> None:
    print("\n" + "=" * 96)
    print("二、早停离线分析（每条曲线反推 10 种策略，不需要额外训练）")
    print("=" * 96)
    print("  注意：因为存的是 best checkpoint，「停在哪」不改变最终模型，")
    print("        只改变训练耗时。真正的风险是**停太早、错过后面的更高点**。")
    print()
    header = f"{'剂量':<22}{'判据':<6}{'p=1':>8}{'p=2':>8}{'p=3':>8}{'p=4':>8}{'p=5':>8}"
    print(header)
    print("-" * 96)

    for tag, desc, st in results:
        h = records[tag]["history"]
        for metric in ("loss", "acc"):
            cells = []
            for p in (1, 2, 3, 4, 5):
                sim = simulate_early_stop(h, metric, p)
                mark = "" if sim["stopped"] else "*"      # 星号 = 没触发，跑满
                cells.append(f"{sim['stop_ep']:>7}{mark}")
            print(f"{desc:<22}{metric:<6}" + "".join(cells))
    print("-" * 96)
    print("  数字 = 会在第几轮停；带 * = 没触发，跑满了")
    print()
    print("  怎么用这张表：")
    print("    · 同一个剂量，loss 列和 acc 列停的轮数差得远 → 说明 acc 判据")
    print("      被离散化噪声带偏了（Day 15 已经量过：6000 条验证集里")
    print("      正确 1 条 = 0.0167%，判据分辨率就这么粗）")
    print("    · 和上表「@ep」对比：停的轮数应该 ≈ @ep + patience")
    print("      差太多就说明这个 patience 在该判据下不稳")


# ═══════════════════════════════════════════════════════════════════
# 6. 复现性检查：augment=0 是否真的没改变任何东西
# ═══════════════════════════════════════════════════════════════════
def check_no_regression(base_record: dict) -> bool:
    """augment=0 时，前 10 轮必须和 Day 22 的 cnn 记录逐位一致。

    这是 Day 18「重构不改变行为」验收方法的延续，也是今天最硬的一条自检：
    我们改动了 data 层（包了一层 AugmentedDataset），理论上增强=0 时
    __getitem__ 直接短路返回原图，DataLoader 的打乱流也没动 ——
    所以前 10 轮的每一个 loss/acc 都应该**一位不差**。

    如果这里对不上，说明要么包装引入了额外行为（比如随机流被消耗了），
    要么我的"短路"没生效。那今天所有剂量结论都建立在一个被动的管线上。
    """
    print("\n" + "=" * 96)
    print("三、回归检查：augment=0 是否真的等于 Day 22 的原始管线")
    print("=" * 96)

    cands = sorted(OUTPUTS.glob("*_mnist_cnn/history.json"))
    if not cands:
        print("  跳过：找不到 Day 22 的 outputs/*_mnist_cnn/history.json")
        return True

    ref = json.loads(cands[-1].read_text(encoding="utf-8"))
    ref_h, got_h = ref["history"], base_record["history"]

    n = min(len(ref_h), len(got_h))
    fields = ("train_loss", "train_acc", "val_loss", "val_acc", "gap", "wnorm", "lr")
    bad = []
    for i in range(n):
        for f in fields:
            if ref_h[i][f] != got_h[i][f]:
                bad.append((i + 1, f, ref_h[i][f], got_h[i][f]))

    print(f"  参照 {cands[-1].parent.name}/history.json  比较前 {n} 轮 × {len(fields)} 个字段")
    if bad:
        print(f"  ✗ {len(bad)} 处不一致，前 5 处：")
        for ep, f, a, b in bad[:5]:
            print(f"      ep{ep} {f}: Day22={a}  今天={b}")
        print("  → 增强=0 时数据管线不该有任何变化，去查 AugmentedDataset 的短路分支")
        return False

    print(f"  ✓ 前 {n} 轮 × {len(fields)} 个字段全部逐位一致")
    print("    → 包装 AugmentedDataset 没有改变原始管线，基线是干净的")
    return True


# ═══════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="Day 23 增强剂量扫描")
    ap.add_argument("--epochs", type=int, default=15, help="每组跑几轮（默认 15）")
    ap.add_argument("--only", type=str, default="", help="只跑指定下标，如 0,1（默认全跑）")
    args = ap.parse_args()

    doses = DOSES
    if args.only:
        idx = [int(x) for x in args.only.split(",")]
        doses = [DOSES[i] for i in idx]

    print("=" * 96)
    print(f"Day 23 · 增强剂量扫描   共 {len(doses)} 组 × {args.epochs} 轮")
    print("=" * 96)

    results: list[tuple[str, str, dict]] = []
    records: dict[str, dict] = {}
    for i, (shift, angle, tag, desc) in enumerate(doses, 1):
        print(f"\n{'─' * 96}")
        print(f"[{i}/{len(doses)}] {desc}   (shift={shift}, angle={angle})")
        print(f"{'─' * 96}")
        rec = run_dose(shift, angle, tag, args.epochs)
        records[tag] = rec
        results.append((tag, desc, curve_stats(rec)))

    print_summary(results)
    print_early_stop(results, records)

    ok = True
    if "aug0" in records:
        ok = check_no_regression(records["aug0"])

    # ── 落盘一份汇总，方便事后写讲义时引用 ──────────────────────
    summary = {
        "epochs": args.epochs,
        "doses": [
            {
                "tag": tag,
                "desc": desc,
                "shift": shift,
                "angle": angle,
                "stats": st,
            }
            for (shift, angle, tag, desc), (_t, _d, st) in zip(doses, results)
        ],
        "regression_ok": ok,
    }
    out = OUTPUTS / "sweep_augment_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总已保存 {out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
