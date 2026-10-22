"""Day 12 · 手写 SGD / Momentum / Adam + 学习率调度（已完成的完整版）

用法（PowerShell，在项目根目录跑）：
    cd "E:\\work Buddy\\强化学习\\rl-vla-journey"
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\utils\\optim.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" src\\utils\\optim.py --train

════════════════════════════════════════════════════════════════════
这个文件怎么读（建议按这个顺序）

  1) 先看 MySGD.step()      —— 最短、最典型，只有 5 步
  2) 再看 MyStepLR.step()   —— 两行，看完就懂"lr 怎么衰减"
  3) 再看 MyAdam.step()     —— 结构同 SGD，只是多了一个 m 和一个 v
  4) 最后看 MySGD.__init__  —— 参数都存在哪几个属性里

  每个 step() 里的 ①②③④⑤ 标号，和讲义里的公式编号是一一对应的。
  注释里写「数学：xxx」的那一行，就是那个公式翻译成的代码。
════════════════════════════════════════════════════════════════════
读懂代码的前提：四个 PyTorch 原地操作

  x.add_(y)                  等价于  x = x + y           ← 下划线 = 改自己
  x.mul_(k)                  等价于  x = x * k
  x.add_(y, alpha=k)         等价于  x = x + k * y
  x.addcmul_(a, b, value=k)  等价于  x = x + k * a * b

  为什么要带下划线？
    x = x + y  让变量 x 指向一个**新的**张量，原来那个张量没变。
    x.add_(y)  直接在原来那块内存上改，所有指向它的地方都看得到变化。
    参数更新必须用后者，否则模型里的参数一点没动（而且不会报错）。

  @torch.no_grad() 又是什么？
    正常的张量运算会顺手记下"这个结果是怎么算出来的"（为了 backward）。
    优化器更新只是一次纯数值修改，不需要被记录，所以整个 step() 用
    @torch.no_grad() 关掉这套记账机制。torch 自己的优化器也是这么写的。
════════════════════════════════════════════════════════════════════
对拍判据：max |手写 - torch| < 1e-6 就算一致。
════════════════════════════════════════════════════════════════════
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
# ① 手写 SGD（含 Momentum / Dampening / Weight Decay / Nesterov）
# ═══════════════════════════════════════════════════════════════════
class MySGD:
    """纯 Python 的 SGD，语义与 torch.optim.SGD 完全一致。

    四个可选项，处理顺序不能变（顺序就是信息）：
        1. weight_decay  先加到梯度上（L2 正则）
        2. momentum      梯度进一个缓冲 buf，更新用 buf 而不是梯度
        3. dampening     进缓冲时按 (1 - dampening) 缩一次
        4. nesterov      更新方向改成 g + momentum * buf

    为什么顺序不能变：
        weight_decay 影响的是"梯度本身"，所以必须在"梯度进缓冲"之前加；
        不然缓冲里存的是没正则过的旧梯度，行为就和 torch 不一样了。
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
        # params 可能是个生成器（model.parameters() 就是），必须立刻存成列表，
        # 否则第一次遍历完就空了，后面 step() 时什么都拿不到
        self.params = [p for p in params]

        # 把几个超参数原样存下来，step() 里读
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # 动量缓冲：每个参数配一个 buf，初始为 None（表示"还没走过第一步"）
        # 长度和 params 一样，用下标 i 一一对应
        self.buf: list[torch.Tensor | None] = [None for _ in self.params]

    def zero_grad(self) -> None:
        """清空梯度。不清的话，这一步的梯度会和之前所有步的叠加在一起。"""
        for p in self.params:
            p.grad = None

    @torch.no_grad()  # 参数更新是纯数值操作，不需要记录计算图
    def step(self) -> None:
        # 逐个参数处理。一个模型通常有很多参数，每个参数有自己的 buf
        # enumerate 同时给出「下标 i」和「参数 p」，i 用来查 self.buf[i]
        for i, p in enumerate(self.params):
            # 有些参数可能没有梯度（没参与前向、或被冻结），直接跳过
            if p.grad is None:
                continue

            # ── 第 1 步：L2 正则（weight_decay）──────────────────────
            # 数学：g ← p.grad + weight_decay × p
            # 直觉：在梯度上额外加一份"把参数往 0 拉"的力，参数越大拉得越狠
            # 默认 weight_decay=0 时这一行不执行，g 就是原始梯度
            g = p.grad
            if self.weight_decay:
                g = p.grad + self.weight_decay * p

            # ── 第 2 步：决定"更新方向" direction ────────────────────
            # plain SGD 里 direction = g；有动量时 direction = buf
            if self.momentum:
                if self.buf[i] is None:
                    # 首步：缓冲里还是空的，直接把这次梯度拷进去。
                    # 注意是「直接拷贝」，不是「从 0 累加」（torch 的行为）。
                    # 必须 .clone()：写 buf = g 会让两者指向同一块内存，
                    # 下一句 buf.mul_() 会把 p.grad 一起改掉。
                    self.buf[i] = g.clone()
                else:
                    # 第 3 步：缓冲滚动  buf ← momentum × buf + (1 − dampening) × g
                    # 拆成两句，是为了看清顺序：
                    #   先 mul_：旧缓冲乘上 momentum（历史信息按比例衰减）
                    #   再 add_：加上本次梯度，(1-dampening) 就是加多少比例
                    # 默认 dampening=0，所以等于 buf = momentum*buf + g
                    self.buf[i].mul_(self.momentum)
                    self.buf[i].add_(g, alpha=1 - self.dampening)

                # 默认用缓冲作为方向
                direction = self.buf[i]

                # 第 4 步：Nesterov 动量
                # 普通动量：直接沿 buf 走
                # Nesterov ：先用 buf 往前"瞄一眼"，再在瞄到的位置取当前梯度
                #            方向 = g + momentum × buf
                if self.nesterov:
                    direction = g + self.momentum * self.buf[i]
            else:
                # 没有动量：方向就是梯度本身
                direction = g

            # ── 第 5 步：真正更新参数 ────────────────────────────────
            # 数学：p ← p − lr × direction
            # 用 p.data -= 而不是 p = p - ...：
            #   后者只是让局部变量 p 指向一个新张量，模型里那个参数没变，
            #   结果就是 loss 一动不动，而且**不会报任何错**。
            # p.data -= 是在原来那块内存上原地减，模型里立刻生效。
            p.data -= self.lr * direction


# ═══════════════════════════════════════════════════════════════════
# ② 手写 Adam
# ═══════════════════════════════════════════════════════════════════
class MyAdam:
    """纯 Python 的 Adam。m / v 两个缓冲已经在 __init__ 里建好。

    核心是「除一下」：m_hat / sqrt(v_hat) 把梯度的**大小**约掉，只留方向。
    所以 Adam 每一步走的距离恒约等于 lr，与梯度是 1e-3 还是 1e3 无关。

    m 和 v 的分工：
        m = 梯度的滑动平均       → 往哪个方向走（一阶矩，均值）
        v = 梯度平方的滑动平均   → 这个方向有多可靠（二阶矩，方差）
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

        # betas 是 (b1, b2)：b1 管 m 的衰减速度，b2 管 v 的衰减速度
        # b2 更接近 1，意思是"梯度的量级"要比"梯度的方向"更平稳地估计
        self.b1, self.b2 = betas
        self.eps = eps

        # 两个缓冲，形状和参数完全一样，全从 0 开始
        # zeros_like(p) 造一个"同形状、同 dtype、同 device"的全 0 张量
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

        # 全局步数，偏差校正要用。所有参数共用这一个计数
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        # 全局步数 +1。漏了这行 → 1 - 0.9**0 = 0 → 除零 → 参数变 nan
        self.t += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad

            # ── 第 1 步：一阶矩（梯度的滑动平均，就是"惯性"）─────────
            # 数学：m ← b1 × m + (1 − b1) × g
            # 代码：先原地乘 b1，再加上 g 的 (1-b1) 倍
            # b1=0.9 时，m 里约等于最近 1/(1-0.9)=10 步梯度的加权平均
            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)

            # ── 第 2 步：二阶矩（梯度平方的滑动平均，衡量梯度有多大）──
            # 数学：v ← b2 × v + (1 − b2) × g²
            # addcmul_(a, b, value=k) 的意思是"加上 k × a × b"
            # 所以传两个 g，就等于加上 (1-b2) × g × g = (1-b2) × g²
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)

            # ── 第 3、4 步：偏差校正 ─────────────────────────────────
            # 为什么需要校正：m 和 v 都从 0 起步，第一步时
            #   m = (1-b1) × g = 0.1 × g，只有真实梯度的 1/10，天然偏小。
            # 除掉 (1 - b1^t) 就是把这部分缩小补回来：
            #   t=1 时除以 0.1 → 恢复成 g 的量级
            #   t 很大时 b1^t → 0，除以约 1，几乎不再影响
            # 数学：m_hat = m / (1 − b1^t)，v_hat = v / (1 − b2^t)
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)

            # ── 第 5 步：更新参数 ───────────────────────────────────
            # 数学：p ← p − lr × m_hat / (sqrt(v_hat) + eps)
            # 除以 sqrt(v_hat) 是 Adam 的核心：梯度大就多除一点，梯度小就少除一点，
            # 一除之下大小被约掉，只剩下方向 —— 所以每步的位移基本恒定 ≈ lr。
            # eps 只是防止 v_hat 为 0 时除零（比如某步梯度恰好全 0）
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


# ═══════════════════════════════════════════════════════════════════
# ③ 手写 StepLR（学习率阶梯衰减）
# ═══════════════════════════════════════════════════════════════════
class MyStepLR:
    """每走 step_size 步，把优化器的 lr 乘一次 gamma。

    与 torch.optim.lr_scheduler.StepLR 的序列一致：
        lr=0.1, step_size=5, gamma=0.5
        → 0.1 ×5步 → 0.05 ×5步 → 0.025 ...
    """

    def __init__(self, opt, step_size: int = 5, gamma: float = 0.5) -> None:
        self.opt = opt            # 持有一个优化器，改的是它的 lr 属性
        self.step_size = step_size  # 每多少步衰减一次
        self.gamma = gamma          # 每次乘多少（0.5 = 减半）
        self.last_epoch = 0         # 已经走了多少步

    def step(self) -> None:
        # 记一步
        self.last_epoch += 1

        # 走到第 5、10、15… 步时触发衰减
        # % 是取余：10 % 5 == 0，所以第 10 步也会触发
        if self.last_epoch % self.step_size == 0:
            # 直接改优化器对象的 lr 属性，下一轮 train_step 就会读到新值
            # *= 是原地乘，等价于 self.opt.lr = self.opt.lr * self.gamma
            self.opt.lr *= self.gamma


# ═══════════════════════════════════════════════════════════════════
# 以下是检查代码，不需要改
# ═══════════════════════════════════════════════════════════════════
def _grad_at(step: int, n: int = 3) -> torch.Tensor:
    """固定的梯度序列（有正有负、幅度在变），保证每次跑都一样。"""
    idx = torch.arange(n, dtype=torch.float32)
    return torch.cos(idx * 0.7 + step * 0.3) * (1.0 + 0.1 * step)


def _new_param() -> nn.Parameter:
    """被测参数：一个三元素的向量，故意有正有负。"""
    return nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))


def _run_mine(cls, kw: dict, lr: float, steps: int) -> list[torch.Tensor]:
    """用手写的优化器跑 steps 步，每步记下参数值（轨迹）。"""
    p = _new_param()
    opt = cls([p], lr=lr, **kw)
    trail = [p.detach().clone()]  # 起点
    for s in range(steps):
        p.grad = _grad_at(s)      # 手动塞梯度，不用真的 backward
        assert p.grad is not None
        opt.step()
        trail.append(p.detach().clone())  # 每步之后的位置
    return trail


def _run_torch(kind: str, kw: dict, lr: float, steps: int) -> list[torch.Tensor]:
    """同样的配置，用 torch 官方优化器跑一遍，作为标准答案。"""
    p = _new_param()
    opt = torch.optim.SGD([p], lr=lr, **kw) if kind == "sgd" else torch.optim.Adam([p], lr=lr, **kw)
    trail = [p.detach().clone()]
    for s in range(steps):
        p.grad = _grad_at(s)
        opt.step()
        trail.append(p.detach().clone())
    return trail


def _compare(log, name: str, mine, ref, tol: float = 1e-6) -> None:
    """逐步比较两条轨迹，最大偏差小于 tol 就算一致。"""
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
    # 四组参数，一组比一组多开一个开关，这样能定位到"哪个开关写错了"
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
    # 两个 lr：Adam 号称"不挑 lr"，两个量级都该对上
    _compare(log, "Adam lr=0.01",
             _run_mine(MyAdam, {}, 0.01, 10),
             _run_torch("adam", {}, 0.01, 10))
    _compare(log, "Adam lr=0.1",
             _run_mine(MyAdam, {}, 0.1, 10),
             _run_torch("adam", {}, 0.1, 10))


def check_scheduler(log) -> None:
    log("[关 4] StepLR 的 lr 序列")
    # 手写版：每走一步同时调 opt.step() 和 sch.step()（和真实训练一致）
    mine_opt = MySGD([_new_param()], lr=0.1)
    mine_sch = MyStepLR(mine_opt, step_size=5, gamma=0.5)
    mine = [mine_opt.lr]
    for _ in range(20):
        mine_opt.step()
        mine_sch.step()
        mine.append(mine_opt.lr)

    # torch 版：同样的配置
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
        # 每次重建都重置随机种子，保证两个优化器拿到完全相同的初始权重
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
            opt.zero_grad()     # 擦掉上一步的梯度
            loss = crit(model(X), y)
            loss.backward()     # 算出每个参数的梯度
            opt.step()          # 按梯度挪一步
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
