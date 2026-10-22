"""Day 12 作业：手写 SGD / Momentum / Adam + 学习率调度

用法（PowerShell，在项目根目录跑）：
    cd "E:\\work Buddy\\强化学习\\rl-vla-journey"
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\utils\\optim.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\utils\\optim.py --train

────────────────────────────────────────────────────────────────
填空规则（只有两条）

  1. 找到带 TODO 的那一行，把占位内容换成真正的表达式
  2. 如果它下面紧跟一个 assert，删掉那行 assert（它只是报警器）

占位有三种长相：
      g = None                 需要赋值的 → 用 None 占位
      assert g is not None     紧跟其后，填完就删
      raise AssertionError(…)  纯语句位置（没有赋值）→ 整段 raise 删掉，换成你的代码
────────────────────────────────────────────────────────────────
报错顺序不是按编号排的，这很正常

  第一次跑 → ⑤    关 1 用默认参数，①②③④ 那些 if 分支根本不执行
  填完 ⑤   → ②    关 2 打开 momentum，首步要 clone
  填完 ②   → ③    第 2 步起走缓冲累加
  填完 ③   → ①    关 2 第 3 组打开 weight_decay
  填完 ①   → ④    关 2 第 4 组打开 nesterov
  填完 ④   → 关 2 全过，进入 2-①

  所以「一上来报 ⑤ 而不是 ①」不是你的错，是分支没被执行。
────────────────────────────────────────────────────────────────
四个原地操作（PyTorch 特有，数学里没有对应符号）

  x.add_(y)                  等价于  x = x + y          ← 下划线表示「改自己」
  x.mul_(k)                  等价于  x = x * k
  x.add_(y, alpha=k)         等价于  x = x + k * y
  x.addcmul_(a, b, value=k)  等价于  x = x + k * a * b

  为什么必须用带下划线的？因为 `p.data -= …` 改的是模型里那个张量本身；
  写成 `p = p - …` 只是让局部变量 p 指向一个新张量，模型参数根本没动（讲义 §7）。

对拍判据：max |手写 - torch| < 1e-6 就算一致。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# 让 `python src/utils/optim.py` 也能 import 到 src 下的模块
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════
# TODO 1 · 手写 SGD
# ═══════════════════════════════════════════════════════════════════
class MySGD:
    """纯 Python 的 SGD，语义与 torch.optim.SGD 完全一致。

    四个可选项，处理顺序不能变：
        1. weight_decay  先加到梯度上（L2 正则）
        2. momentum      梯度进一个缓冲 buf，更新用 buf 而不是梯度
        3. dampening     进缓冲时按 (1 - dampening) 缩一次
        4. nesterov      更新方向改成 g + momentum * buf

    ⚠️ 两个必须注意的点（都写在提示里，别跳过）：
        · 首步 buf 要**直接拷贝梯度**，不是从 0 累加 —— 默认参数下两者结果相同，
          只有 dampening > 0 时才差 4%，所以很容易漏
        · 拷贝要用 g.clone()，直接写 buf = g 会让 buf 和梯度是**同一个对象**，
          后面 buf.mul_() 会连梯度一起改掉
    """

    def __init__(
        self,
        params,
        lr: float = 0.1,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        self.params = [p for p in params]
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.buf: list[torch.Tensor | None] = [None for _ in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # ① L2 正则：先把 weight_decay * p 加到梯度上
            #    数学：g = p.grad + weight_decay × p
            g = p.grad
            if self.weight_decay:
                g = None
                assert g is not None, (
                    "TODO 1-① 未完成（weight_decay）\n"
                    "  一行：g = g + self.weight_decay * p"
                )

            # ② 有 momentum 时，方向不再等于梯度，而是一个会累积的缓冲 buf
            if self.momentum:
                if self.buf[i] is None:
                    # 首步：把这次梯度直接拷进去（不是从 0 累加）
                    self.buf[i] = None
                    assert self.buf[i] is not None, (
                        "TODO 1-② 未完成（momentum 首步）\n"
                        "  self.buf[i] = g.clone()\n"
                        "  ⚠ clone 不能省：不加的话 buf 和 grad 是同一个张量，后面 mul_ 会把梯度也改掉"
                    )
                else:
                    # ③ 缓冲滚动：buf = momentum × buf + (1 - dampening) × g
                    raise AssertionError(
                        "TODO 1-③ 未完成（缓冲滚动）\n"
                        "  把这两步合成一行：\n"
                        "    self.buf[i].mul_(self.momentum)                     # buf 先乘 momentum\n"
                        "    self.buf[i].add_(g, alpha=1 - self.dampening)       # 再加 (1-dampening)×g"
                    )
                direction = self.buf[i]

                # ④ nesterov：方向改成「当前梯度 + momentum × 缓冲」
                if self.nesterov:
                    direction = None
                    assert direction is not None, (
                        "TODO 1-④ 未完成（nesterov）\n"
                        "  direction = g + self.momentum * self.buf[i]"
                    )
            else:
                direction = g

            # ⑤ 更新参数：p = p - lr × 方向
            raise AssertionError(
                "TODO 1-⑤ 未完成（更新参数）\n"
                "  p.data -= self.lr * direction\n"
                "  ⚠ 必须是 p.data -=，写成 p = p - ... 模型里的参数不会变"
            )


# ═══════════════════════════════════════════════════════════════════
# TODO 2 · 手写 Adam
# ═══════════════════════════════════════════════════════════════════
class MyAdam:
    """纯 Python 的 Adam。m / v 两个缓冲已经在 __init__ 里建好。

    核心是「除一下」：m_hat / sqrt(v_hat) 把梯度的**大小**约掉，只留方向。
    所以 Adam 每一步走的距离恒约等于 lr，与梯度是 1e-3 还是 1e3 无关。

    ⚠️ 偏差校正用的 t 是**全局步数**（所有参数共用），不是每个参数各自计数。
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params = [p for p in params]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        # 全局步数，偏差校正要用。漏了这行 → 1 - 0.9**0 = 0 → 除零 → 参数变 nan
        self.t += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad

            # ① 一阶矩：m 是梯度的滑动平均，就是"惯性"
            #    数学：m = b1 × m + (1 - b1) × g
            raise AssertionError(
                "TODO 2-① 未完成（一阶矩 m）\n"
                "  合成一行：self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)\n"
                "  拆开看：   m 先乘 b1  →  再加 (1-b1)×g"
            )

            # ② 二阶矩：v 是「梯度平方」的滑动平均，用来衡量梯度有多大
            #    数学：v = b2 × v + (1 - b2) × g²
            #    torch 的 addcmul_(a, b, value=k) = 加 k×a×b，所以传两个 g 就是 g*g
            raise AssertionError(
                "TODO 2-② 未完成（二阶矩 v）\n"
                "  self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)\n"
                "  为什么传两次 g：addcmul_ 算 a*b，传两次就是 g×g"
            )

            # ③④ 偏差校正：m 和 v 都从 0 起步，前几步天然偏小，要除掉
            #     数学：m_hat = m / (1 - b1^t)，v_hat = v / (1 - b2^t)
            m_hat = None
            assert m_hat is not None, (
                "TODO 2-③ 未完成（m 的偏差校正）\n"
                "  m_hat = self.m[i] / (1 - self.b1 ** self.t)"
            )
            v_hat = None
            assert v_hat is not None, (
                "TODO 2-④ 未完成（v 的偏差校正）\n"
                "  v_hat = self.v[i] / (1 - self.b2 ** self.t)"
            )

            # ⑤ 更新：p = p - lr × m_hat / (sqrt(v_hat) + eps)
            #    除 sqrt(v_hat) 就是 Adam 的核心：把梯度的大小约掉，只留方向
            raise AssertionError(
                "TODO 2-⑤ 未完成（更新参数）\n"
                "  p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)"
            )


# ═══════════════════════════════════════════════════════════════════
# TODO 3 · 手写 StepLR
# ═══════════════════════════════════════════════════════════════════
class MyStepLR:
    """每走 step_size 步，把优化器的 lr 乘一次 gamma。

    与 torch.optim.lr_scheduler.StepLR 的序列一致：
        lr=0.1, step_size=5, gamma=0.5
        → 0.1 ×5步 → 0.05 ×5步 → 0.025 ...
    """

    def __init__(self, opt, step_size: int = 5, gamma: float = 0.5) -> None:
        self.opt = opt
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self) -> None:
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            raise AssertionError(
                "TODO 3 未完成（lr 衰减）\n"
                "  self.opt.lr *= self.gamma\n"
                "  第 5、10、15… 步各乘一次 → 0.1, 0.05, 0.025 …"
            )


# ═══════════════════════════════════════════════════════════════════
# 以下是检查代码，不需要改
# ═══════════════════════════════════════════════════════════════════
def _grad_at(step: int, n: int = 3) -> torch.Tensor:
    """固定的梯度序列（有正有负、幅度在变），保证每次跑都一样。"""
    idx = torch.arange(n, dtype=torch.float32)
    return torch.cos(idx * 0.7 + step * 0.3) * (1.0 + 0.1 * step)


def _new_param() -> nn.Parameter:
    return nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))


def _run_mine(cls, kw: dict, lr: float, steps: int) -> list[torch.Tensor]:
    p = _new_param()
    opt = cls([p], lr=lr, **kw)
    trail = [p.detach().clone()]
    for s in range(steps):
        p.grad = _grad_at(s)
        assert p.grad is not None
        opt.step()
        trail.append(p.detach().clone())
    return trail


def _run_torch(kind: str, kw: dict, lr: float, steps: int) -> list[torch.Tensor]:
    p = _new_param()
    opt = torch.optim.SGD([p], lr=lr, **kw) if kind == "sgd" else torch.optim.Adam([p], lr=lr, **kw)
    trail = [p.detach().clone()]
    for s in range(steps):
        p.grad = _grad_at(s)
        opt.step()
        trail.append(p.detach().clone())
    return trail


def _compare(log, name: str, mine, ref, tol: float = 1e-6) -> None:
    assert len(mine) == len(ref), f"{name}: 轨迹长度 {len(mine)} != {len(ref)}"
    # 先查 inf/nan —— 否则 max() 会把 nan 当"不算最大"悄悄跳过，检查就形同虚设
    for tag, trail in (("手写", mine), ("torch", ref)):
        bad = [i for i, t in enumerate(trail) if not torch.isfinite(t).all()]
        assert not bad, (
            f"{name}: {tag} 在第 {bad[:3]} 步出现 inf/nan —— 通常是除零或 sqrt(负数)\n"
            f"  检查 Adam 的 1 - beta**t 是否可能为 0（漏了 self.t += 1？）"
        )
    diffs = [(a - b).abs().max().item() for a, b in zip(mine, ref)]
    worst = max(diffs)
    log(f"    {name:<34} max diff = {worst:.3e}   {'OK' if worst < tol else 'MISMATCH'}")
    assert worst < tol, (
        f"{name}: 与 torch 的最大偏差 {worst:.3e} 超过 {tol:.0e}\n"
        f"  第 {diffs.index(worst)} 步就偏了，单步 diff = {diffs[:6]}"
    )


def check_plain_sgd(log) -> None:
    log("[关 1] plain SGD（p -= lr * g）")
    _compare(log, "无选项", _run_mine(MySGD, {}, 0.1, 10), _run_torch("sgd", {}, 0.1, 10))


def check_momentum(log) -> None:
    log("[关 2] Momentum（默认参数 + dampening=0.1）")
    _compare(log, "momentum=0.9",
             _run_mine(MySGD, {"momentum": 0.9}, 0.1, 10),
             _run_torch("sgd", {"momentum": 0.9}, 0.1, 10))
    _compare(log, "momentum=0.9 dampening=0.1",
             _run_mine(MySGD, {"momentum": 0.9, "dampening": 0.1}, 0.1, 10),
             _run_torch("sgd", {"momentum": 0.9, "dampening": 0.1}, 0.1, 10))
    _compare(log, "momentum=0.9 weight_decay=1e-3",
             _run_mine(MySGD, {"momentum": 0.9, "weight_decay": 1e-3}, 0.1, 10),
             _run_torch("sgd", {"momentum": 0.9, "weight_decay": 1e-3}, 0.1, 10))
    _compare(log, "momentum=0.9 nesterov=True",
             _run_mine(MySGD, {"momentum": 0.9, "nesterov": True}, 0.1, 10),
             _run_torch("sgd", {"momentum": 0.9, "nesterov": True}, 0.1, 10))


def check_adam(log) -> None:
    log("[关 3] Adam（两个矩 + 偏差校正）")
    _compare(log, "Adam lr=0.01",
             _run_mine(MyAdam, {}, 0.01, 10),
             _run_torch("adam", {}, 0.01, 10))
    _compare(log, "Adam lr=0.1",
             _run_mine(MyAdam, {}, 0.1, 10),
             _run_torch("adam", {}, 0.1, 10))


def check_scheduler(log) -> None:
    log("[关 4] StepLR 的 lr 序列")
    mine_opt = MySGD([_new_param()], lr=0.1)
    mine_sch = MyStepLR(mine_opt, step_size=5, gamma=0.5)
    mine = [mine_opt.lr]
    for _ in range(20):
        mine_opt.step()
        mine_sch.step()
        mine.append(mine_opt.lr)

    p = _new_param()
    ref_opt = torch.optim.SGD([p], lr=0.1)
    ref_sch = torch.optim.lr_scheduler.StepLR(ref_opt, step_size=5, gamma=0.5)
    ref = [ref_opt.param_groups[0]["lr"]]
    for _ in range(20):
        ref_opt.step()
        ref_sch.step()
        ref.append(ref_opt.param_groups[0]["lr"])

    same = all(abs(a - b) < 1e-12 for a, b in zip(mine, ref))
    log(f"    手写 = {[round(v, 6) for v in mine[:12]]}")
    log(f"    torch= {[round(v, 6) for v in ref[:12]]}")
    assert same, f"lr 序列对不上，第 {[i for i,(a,b) in enumerate(zip(mine,ref)) if abs(a-b)>1e-12][:3]} 步开始不同"
    log("    20 步序列逐位一致   OK")


def check_train_trajectory(log) -> None:
    """用自己写的 SGD 训练一个两层网络，loss 轨迹与 torch 一致。"""
    log("[关 5] 训练轨迹：自己的 MySGD vs torch.optim.SGD（50 步）")

    def build() -> nn.Module:
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))

    # 三个类中心按 120° 摆开，标签有规律可学（随机标签的话 loss 学不动，看不出差别）
    gen = torch.Generator().manual_seed(7)
    centers = torch.tensor([[1.0, 0.0], [-0.5, 0.866], [-0.5, -0.866]])
    y = torch.arange(3).repeat_interleave(80)
    X = centers[y] + torch.randn(240, 2, generator=gen) * 0.5
    crit = nn.CrossEntropyLoss()

    def run(use_mine: bool) -> list[float]:
        model = build()
        opt = MySGD(model.parameters(), lr=0.05) if use_mine else torch.optim.SGD(model.parameters(), lr=0.05)
        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    mine, ref = run(True), run(False)
    worst = max(abs(a - b) for a, b in zip(mine, ref))
    log(f"    loss[0]={mine[0]:.6f}  loss[49]={mine[-1]:.6f}   （torch 同）")
    log(f"    50 步最大偏差 = {worst:.3e}   {'OK' if worst < 1e-5 else 'MISMATCH'}")
    assert worst < 1e-5, (
        f"训练轨迹对不上，最大 loss 偏差 {worst:.3e}\n"
        f"  手写前 5 步 = {[round(v, 6) for v in mine[:5]]}\n"
        f"  torch 前 5 步 = {[round(v, 6) for v in ref[:5]]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="额外跑第 5 关（训练轨迹对拍）")
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    log("=" * 66)
    log("Day 12 · 手写优化器对拍")
    log("=" * 66)
    check_plain_sgd(log)
    check_momentum(log)
    check_adam(log)
    check_scheduler(log)

    log("")
    if args.train:
        check_train_trajectory(log)
    else:
        log("（加 --train 可跑第 5 关：用 MySGD 训练两层网络，对比 loss 轨迹）")

    log("")
    log("=" * 66)
    log("全部通过：你手写的更新规则与 torch 逐位一致")
    log("=" * 66)


if __name__ == "__main__":
    main()
