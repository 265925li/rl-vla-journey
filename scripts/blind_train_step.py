"""Day 28 · 首次盲写：train_step

════════════════════════════════════════════════════════════════════
这是什么

  P0 到 Day 27，教学模式一直是「我写完整实现 + 逐行注释，你读懂跑通」。
  这个模式能验证「你读懂了」，**验证不了「你能自己写」**。

  这个文件就是用来验那件事的。

════════════════════════════════════════════════════════════════════
盲写规则（必须遵守，否则验不出东西）

  | 规则 | 理由 |
  |---|---|
  | 只能看函数签名和 docstring，不能看已有实现 | 看实现就变成抄写 |
  | 只能看 3 分钟 | 超时就说明「会」是假的 |
  | 写完必须跑 --check | 自己觉得对不算 |
  | 卡住时先记录卡在哪，再去看参考 | 卡点本身是最有价值的信息 |

  「已有实现」指这些地方，盲写期间不要打开：
      src/engine/loops.py                        （train_one_epoch 里就是这五行）
      docs/Day28-W4收口与盲写.md 的附录           （参考实现）
      docs/Day11-训练循环.md

  3 分钟计时：现在开始。

════════════════════════════════════════════════════════════════════
写完怎么验

    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\blind_train_step.py --check

四关全过才算会。最后再跑一次真实训练：

    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\train.py --epochs 1

和 Day 27 的记录一致 → 盲写通过。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = str(ROOT / "data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════
#  只有这一件事要做
# ═══════════════════════════════════════════════════════════════════

def train_step(model, x, y, loss_fn, optimizer):
    """跑一步训练，返回这一步的 loss（**标量张量**）。

    契约（五条，缺一条就过不了关）：

      1. 模型必须处于**训练态**（在函数内部保证，不依赖调用方先调 model.train()）
      2. 必须清掉上一步留下的梯度（不依赖调用方先 zero_grad）
      3. 前向 → 算 loss → 反向 → 更新参数，顺序固定
      4. 返回值必须是**标量张量**，而且**仍带计算图**
         （不是 .item() 出来的 float，也不是形状 (1,) 的张量）
         —— 调用方可能还要对它做加权求和
      5. 传入的 x / y 已经在正确设备上，函数内**不需要**再 .to(device)

    参数：
        model      nn.Module
        x          一个批的输入，形状 (B, ...)
        y          一个批的标签，形状 (B,)
        loss_fn    损失函数，签名 loss_fn(logits, y) -> 标量张量
        optimizer  优化器，已经绑定好 model.parameters()

    返回：
        标量张量，这一步的 loss
    """
    todo = False
    assert todo, (
        "TODO：这里要写 5 行。\n"
        "  顺序是：清梯度 → 前向 → 算 loss → 反向 → 更新。\n"
        "  卡住了先不要看参考 —— 把「卡在哪一步」写下来，那是今天最有价值的产物。"
    )


# ═══════════════════════════════════════════════════════════════════
#  以下是关卡，不需要你改
# ═══════════════════════════════════════════════════════════════════

class _Cfg:
    """关卡用的最小配置：只给 build_mlp / build_datasets 需要的字段。"""

    data_dir = DATA_DIR
    val_size = 6000
    seed = 0
    model = "mlp"
    hidden = 256
    n_layers = 2
    act = "relu"
    dropout = 0.0
    linear = "torch"
    flatten = True


def _build():
    from src.data.mnist import build_datasets
    from src.engine.builtin import build_mlp

    cfg = _Cfg()
    train_set, _, _ = build_datasets(cfg)
    torch.manual_seed(0)
    return build_mlp(cfg).to(DEV), train_set


def _batches(train_set, n_batches=2, bs=128):
    """从真实 MNIST 里取前 n 批，摊平并搬到设备上。"""
    out = []
    for b in range(n_batches):
        idx = range(b * bs, (b + 1) * bs)
        xs = torch.stack([train_set[i][0] for i in idx])
        ys = torch.tensor([train_set[i][1] for i in idx], dtype=torch.long)
        out.append((xs.view(bs, -1).to(DEV), ys.to(DEV)))
    return out


def gate1_return_value() -> bool:
    """关 1：返回的是「标量张量」，而且带计算图。"""
    print("\n[关 1] 返回值：标量张量 + 带计算图")
    model, train_set = _build()
    (x, y), = _batches(train_set, 1)
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    out = train_step(model, x, y, crit, opt)
    dtype = getattr(out, "dtype", None)
    shape = tuple(out.shape) if hasattr(out, "shape") else "N/A"
    print(f"    类型 {type(out).__name__}   dtype={dtype}   shape={shape}")
    print(f"    值 {out.item() if hasattr(out, 'item') else out:.6f}"
          if isinstance(out, torch.Tensor) else f"    值 {out}")

    assert isinstance(out, torch.Tensor), (
        f"返回值类型是 {type(out).__name__}，必须是 torch.Tensor。\n"
        "  常见错法：return loss.item() —— 那是 Python float。"
    )
    assert out.dim() == 0, (
        f"返回值 shape 是 {tuple(out.shape)}，必须是 0 维标量。\n"
        "  常见错法：loss 用了 reduction='none' 没取 mean；或手动 reshape 成 (1,)。"
    )
    assert out.grad_fn is not None, (
        "返回值不带计算图（grad_fn=None）。\n"
        "  说明它经过了 detach() / .item() / .numpy() —— 这样调用方就没法再对它求导。"
    )
    print("    ✓ 通过")
    return True


def gate2_params_moved() -> bool:
    """关 2：调用之后参数真的变了（说明 optimizer.step() 走到了）。"""
    print("\n[关 2] 调用之后参数变了")
    model, train_set = _build()
    (x, y), = _batches(train_set, 1)
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    before = [p.detach().clone() for p in model.parameters()]
    train_step(model, x, y, crit, opt)
    diffs = [(p.detach() - b).abs().max().item()
             for p, b in zip(model.parameters(), before)]
    moved = sum(1 for d in diffs if d > 0)
    print(f"    {moved}/{len(diffs)} 个参数张量变了    最大变化 {max(diffs):.3e}")
    assert moved == len(diffs), (
        f"只有 {moved}/{len(diffs)} 个参数变了。\n"
        "  最常见原因：忘了 optimizer.step()，或者顺序写成 step 在 backward 之前。"
    )
    print("    ✓ 通过")
    return True


def gate3_grad_not_accumulated() -> bool:
    """关 3：连续两次调用不能累加梯度（说明 zero_grad 真的在函数开头）。"""
    print("\n[关 3] 连续两次调用不累加梯度")

    def run_once_pair():
        model, train_set = _build()
        (x, y), = _batches(train_set, 1)
        crit = torch.nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        return model, opt, crit, x, y

    model1, opt1, crit1, x1, y1 = run_once_pair()
    train_step(model1, x1, y1, crit1, opt1)
    g1 = [p.grad.detach().clone() if p.grad is not None else None
          for p in model1.parameters()]

    model2, opt2, crit2, x2, y2 = run_once_pair()
    train_step(model2, x2, y2, crit2, opt2)
    g2a = [p.grad.detach().clone() if p.grad is not None else None
           for p in model2.parameters()]
    train_step(model2, x2, y2, crit2, opt2)
    g2b = [p.grad.detach().clone() if p.grad is not None else None
           for p in model2.parameters()]

    assert all(g is not None for g in g1), (
        "调用之后梯度全是 None —— 说明 backward 没执行，"
        "或者 zero_grad 被放在了 backward 之后（把刚算的梯度擦掉了）。"
    )
    ratios = []
    for a, b in zip(g2a, g2b):
        if a is not None and a.abs().max() > 0:
            ratios.append((b.abs().max() / a.abs().max()).item())
    ratio = max(ratios) if ratios else 1.0
    print(f"    第二次梯度 / 第一次梯度 的最大比值 {ratio:.3f}")
    print("    （接近 1 = 没累加；接近 2 = 累加了；差很多 = 参数已经被更新过）")
    assert ratio < 1.5, (
        f"第二次调用时梯度明显偏大（比值 {ratio:.2f}）—— 说明梯度在累加。\n"
        "  原因：zero_grad 不在函数开头。"
    )
    print("    ✓ 通过")
    return True


def gate4_loss_sequence() -> bool:
    """关 4（最硬）：你的 train_step 跑 100 步，和现有管线逐位相同。

    判据是「行为等价」，不是「代码长得一样」。
    用别的写法（比如 zero_grad(set_to_none=True)）一样能过。
    """
    print("\n[关 4] 100 步 loss 序列 vs 现有管线（train_one_epoch）")
    from src.engine.builtin import build_mlp
    from src.engine.loops import train_one_epoch
    from src.data.mnist import build_datasets

    cfg = _Cfg()
    train_set, _, _ = build_datasets(cfg)
    subset = torch.utils.data.Subset(train_set, list(range(128 * 100)))
    loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False)

    def fresh():
        torch.manual_seed(0)
        return build_mlp(cfg).to(DEV)

    # ── 路径 A：现有管线。criterion 里记一笔，把每步 loss 抄下来 ──
    a_losses = []

    class Recorder(torch.nn.Module):
        def forward(self, logits, target):
            loss = F.cross_entropy(logits, target)
            a_losses.append(loss.item())
            return loss

    model_a = fresh()
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.05, momentum=0.9)
    train_one_epoch(model_a, loader, opt_a, Recorder(), DEV, flatten=True)

    # ── 路径 B：你的 train_step ──
    b_losses = []
    model_b = fresh()
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.05, momentum=0.9)
    crit_b = torch.nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.view(xb.size(0), -1).to(DEV)
        yb = yb.to(DEV)
        b_losses.append(train_step(model_b, xb, yb, crit_b, opt_b).item())

    assert len(a_losses) == len(b_losses), f"步数不同：{len(a_losses)} vs {len(b_losses)}"
    worst = max(abs(a - b) for a, b in zip(a_losses, b_losses))
    print(f"    步数 {len(a_losses)}   最大差 {worst:.2e}")
    print(f"    前 3 步  A={[round(v, 6) for v in a_losses[:3]]}")
    print(f"             B={[round(v, 6) for v in b_losses[:3]]}")
    print(f"    末 3 步  A={[round(v, 6) for v in a_losses[-3:]]}")
    print(f"             B={[round(v, 6) for v in b_losses[-3:]]}")
    assert worst < 1e-6, (
        f"loss 序列对不上（最大差 {worst:.3e}）。按这个顺序查：\n"
        "  1. 漏了 zero_grad → 梯度从第 2 步开始累加，偏差会越来越大\n"
        "  2. 顺序：前向 / loss / backward / step\n"
        "  3. 模型有没有 model.train()（MLP 无 dropout 时不影响数值，但契约要求）\n"
        "  4. 有没有对 loss 做多余的重算（比如算两次 loss 会导致两次反向）"
    )
    print("    ✓ 通过（与现有管线逐位等价）")
    return True


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Day 28 盲写 train_step 的自检")
    ap.add_argument("--check", action="store_true", help="跑全部四关")
    args = ap.parse_args()

    if not args.check:
        print(__doc__)
        print("写完之后跑：python scripts/blind_train_step.py --check")
        return 0

    print("=" * 70)
    print("Day 28 · train_step 盲写验收")
    print(f"torch {torch.__version__}  device={DEV}")
    print("=" * 70)

    gates = [
        ("关 1 返回值是标量张量 + 带计算图", gate1_return_value),
        ("关 2 参数真的变了", gate2_params_moved),
        ("关 3 梯度不累加", gate3_grad_not_accumulated),
        ("关 4 100 步 loss 序列等价", gate4_loss_sequence),
    ]
    results, ok = [], True
    for name, fn in gates:
        try:
            good = bool(fn())
        except AssertionError as e:
            print(f"    ✗ 未通过：{e}")
            good = False
        except Exception as e:                        # noqa: BLE001
            print(f"    ✗ 意外错误：{type(e).__name__}: {e}")
            good = False
        results.append((name, good))
        ok = ok and good

    passed = sum(1 for _, g in results if g)
    print("\n" + "=" * 70)
    print(f"总计：{passed}/{len(results)} 关通过")
    for name, good in results:
        print(f"  {'✓' if good else '✗'}  {name}")
    print("=" * 70)
    if ok:
        print("四关全过。最后再跑一次真实训练，和 Day 27 的记录对一下：")
        print('  & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\train.py --epochs 1')
        print("")
        print("然后回 docs/Day28-W4收口与盲写.md 的附录对写法，")
        print("看有没有「你的版本更啰嗦」或「更简洁」的地方 —— 那是最值得记的。")
    else:
        print("没过就是没过。先记录卡在哪，再去看参考。")
        print("卡点比正确答案有价值：它决定 P2 的教学方式该怎么调。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
