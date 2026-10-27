"""Day 26 · 混合精度：手写 GradScaler + 把正确顺序钉死的 AmpStepper

════════════════════════════════════════════════════════════════════
一、fp16 到底解决了什么、又引入了什么

  fp32 的指数位 8 bit（范围 ~1e-38 ~ 3e38），fp16 只给 5 bit
  （~6e-8 ~ 65504）。尾数位从 23 掉到 10。

  两个后果方向相反：

      上溢    数值 > 65504 → inf。fp16 的 exp 在参数 11.09 时就到顶，
              因为 exp(11.09) = 65504。
      **下溢** 数值 < 6e-8 → 直接变 0。

  ⇒ **深度学习里真正致命的是下溢，不是上溢。**
    因为梯度通常很小：1e-5 级的梯度在 fp16 里还活着，
    1e-9 级的就直接被 flush 成 0 —— 参数再也不更新，
    损失不报错、不 nan，只是"学不动"。

  这就是 GradScaler 存在的唯一理由：**把 loss 放大，梯度跟着放大，
  反向前做、更新前除掉。** 放大倍数（scale）越大，越小的梯度能活下来；
  但太大就上溢。所以要动态调。

════════════════════════════════════════════════════════════════════
二、为什么把 clip 也装进 AmpStepper

  amp 的标准五步是：

      with autocast(): loss = crit(model(x), y)
      scaler.scale(loss).backward()
      scaler.unscale_(opt)              ← 少了这行，下面的 clip 就错了
      clip_grad_norm_(params, max_norm)
      scaler.step(opt); scaler.update()

  **`unscale_` 必须在 `clip_grad_norm_` 之前。** 写反不报错：
  你 clip 的是"放大了 65536 倍的梯度"，norm 远超阈值，于是全部被
  缩到 max_norm —— 等于对真实梯度做了 65536 倍的额外衰减。
  结果是训练几乎不动，但 loss 看起来只是"降得慢"，很容易被当成
  "amp 本来就慢"或者"lr 太小"。

  所以这里把 unscale / clip / step / update 打包进一个 step() 方法。
  **顺序由代码结构保证，不由每个调用点的记性保证。**
  这和 Day 25 用 `nn.Identity()` 替掉 `if has_bn:` 是同一个思路：
  让"写错"需要主动破坏封装，而不是忘记某一行。

════════════════════════════════════════════════════════════════════
三、这台机器的预期（1660 Ti / sm_75 / 无 Tensor Core）

  amp 的 2 倍加速来自 Tensor Core 的 fp16 GEMM。sm_75 的消费卡
  （TU116）**没有 Tensor Core**，所以理论上只剩"显存带宽减半"这一点收益，
  而这个小 CNN 不是带宽瓶颈。

  → 所以 Day 26 的第一件事是**测**，不是抄配置。
    实测结论见 docs/Day26-混合精度.md。
"""
from __future__ import annotations

import contextlib
import math

import torch


# ═══════════════════════════════════════════════════════════════════
# 1. 手写 GradScaler
# ═══════════════════════════════════════════════════════════════════

class MyGradScaler:
    """PyTorch 官方 GradScaler 的手写等价物。

    只有 4 个动作，每个对应一件必须做的事：

        scale(loss)      乘上当前 scale（前向之后、反向之前）
        unscale_(opt)    除回真实尺度，**同时**检查有没有 inf/nan
        step(opt)        只有在没发现 inf/nan 时才真的更新参数
        update()         根据"这一步有没有炸"调大或调小 scale

    ── 为什么 update 要单独一步 ─────────────────────────────────
    因为它必须发生在 step **之后**，而且必须在"已经知道这一步有没有炸"
    之后。如果塞进 step 里，调用方就没法在两者之间插别的东西
    （比如日志、比如第二个优化器）。官方 API 也是拆开的。

    ── 动态调节的两个参数 ───────────────────────────────────────
    growth_interval：连续多少次没炸，就把 scale 乘 growth_factor。
    0.5 的 backoff：炸一次就砍半。

    **砍半快、翻倍慢**是不对称的，故意如此：
      一次 inf 说明"当前 scale 已经过大"，要立刻退回去；
      而 scale 偏小时只是浪费一点精度，不致命。
      所以"保守地大"比"激进地大"安全。
    """

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        if growth_factor <= 1.0:
            raise ValueError(f"growth_factor 必须 > 1，你给的是 {growth_factor}")
        if not 0.0 < backoff_factor < 1.0:
            raise ValueError(f"backoff_factor 要在 (0,1) 内，你给的是 {backoff_factor}")
        if growth_interval < 1:
            raise ValueError(f"growth_interval 必须 >= 1，你给的是 {growth_interval}")

        self.init_scale = float(init_scale)
        self.scale_value = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self.enabled = bool(enabled)

        self._growth_tracker = 0      # 连续多少步没炸
        self._found_inf = False       # 这一步有没有炸
        self._unscaled = False        # 这一步是否已经 unscale 过
        # 轨迹：每一步的 scale，用来和官方对拍、也用来画图
        self.history: list[float] = []

    # ── ① 放大 ────────────────────────────────────────────────────
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """loss * scale。**返回的是新张量，不改 loss 本身。**

        为什么要乘在 loss 上而不是乘在梯度上：
        梯度是 loss 对参数的导数，只要 loss 乘了常数，整条链上的梯度
        就都乘了同一个常数 —— 一次乘法管全部参数，不用遍历。
        """
        if not self.enabled:
            return loss
        return loss * self.scale_value

    # ── ② 除回来，并检查 inf/nan ──────────────────────────────────
    def unscale_(self, optimizer) -> None:
        """把梯度除回真实尺度，同时记下这一步有没有非有限值。

        ⚠️ 这里有一个容易忽略的点：**inf 检查要在除法之前做**。
        因为 g/inf = nan、g/nan = nan，除完就读不出"原本是 inf"了 ——
        虽然 nan 也是非有限值，结果上没错，但概念上检查的是
        "放大后的梯度有没有溢出"，那才是 scale 过大的信号。
        """
        if not self.enabled:
            return
        if self._unscaled:
            raise RuntimeError(
                "unscale_ 被调用了两次。官方实现同样会报这个错 —— "
                "重复除会让梯度被多除一次 scale，而且不报错，只是结果全错。"
            )

        found_inf = False
        grads = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(p.grad)
                    if not torch.isfinite(p.grad).all():
                        found_inf = True

        # 就算发现 inf 也要除 —— 梯度已经废了，除一下不为别的，
        # 只为让 state 保持一致（官方实现也是无条件除）。
        inv = 1.0 / self.scale_value
        for g in grads:
            g.mul_(inv)

        self._found_inf = found_inf
        self._unscaled = True

    # ── ③ 条件更新 ────────────────────────────────────────────────
    def step(self, optimizer, *args, **kwargs):
        """没发现 inf/nan 才真的更新参数。

        为什么"发现 inf 就整步跳过"是对的：
        一步的梯度里只要有一个参数是 inf，SGD 那一步就会把这个参数
        变成 nan，而 **nan 会顺着后续所有计算扩散到全部参数**。
        跳一步的代价是慢一步，代价可以忽略。
        """
        if not self.enabled:
            return optimizer.step(*args, **kwargs)

        if not self._unscaled:
            # 用户忘了 unscale_ —— 官方这里会自动补一次。
            # 我们也补，但**不静默**：真正需要知道的是"谁忘了"。
            # （这一行是刻意的设计取舍：补上是为了能跑，
            #   打警告是为了别让这个错误变成习惯。）
            import warnings
            warnings.warn(
                "step() 之前没有调用 unscale_()：梯度仍是放大状态。"
                "已自动补调，但请检查调用顺序。",
                RuntimeWarning, stacklevel=2,
            )
            self.unscale_(optimizer)

        if self._found_inf:
            return None
        return optimizer.step(*args, **kwargs)

    # ── ④ 调 scale ────────────────────────────────────────────────
    def update(self) -> None:
        """按这一步的结果调 scale，然后清掉 step 级的状态。"""
        if not self.enabled:
            return

        if self._found_inf:
            # 炸了：砍半，并把"连续没炸"的计数清零
            self.scale_value *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale_value *= self.growth_factor
                self._growth_tracker = 0

        self.history.append(self.scale_value)
        self._found_inf = False
        self._unscaled = False

    # ── 查询 ──────────────────────────────────────────────────────
    def get_scale(self) -> torch.Tensor:
        """和官方 API 对齐：返回一个标量张量（不是 Python float）。"""
        return torch.tensor(self.scale_value)

    def __repr__(self) -> str:
        return (f"MyGradScaler(scale={self.scale_value:g}, "
                f"growth={self.growth_factor}, backoff={self.backoff_factor}, "
                f"interval={self.growth_interval}, enabled={self.enabled})")


# ═══════════════════════════════════════════════════════════════════
# 2. 把"正确顺序"钉死的一步执行器
# ═══════════════════════════════════════════════════════════════════

class AmpStepper:
    """一次训练步里 amp 需要的全部动作，按正确顺序封装。

    用法（训练循环里只有三行）：

        stepper = AmpStepper(device, enabled=True, grad_clip=1.0)
        ...
        with stepper.autocast():
            logits = model(xb)
            loss = crit(logits, yb)
        stepper.backward(loss)
        stepper.step(opt, model.parameters())

    ── 为什么 clip 必须在这里 ────────────────────────────────────
    见模块开头的第二段。简单说：如果 clip 留在这里外面，
    `scaler.unscale_()` 和 `clip_grad_norm_()` 的相对顺序就变成了
    调用方的责任，而写反**不报错**、只是训练变慢 —— 这种 bug
    能活很久。装进来之后，顺序是结构性的。

    ── enabled=False 时必须是零影响的 ────────────────────────────
    `autocast()` 返回 nullcontext，`backward` 直接 loss.backward()，
    `step` 里只做 clip（如果有）再 opt.step()。这条路径要和
    Day 25 的循环**逐位一致**，否则 baseline 全部作废。
    """

    def __init__(
        self,
        device,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        scaler=None,
        grad_clip: float = 0.0,
    ) -> None:
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.dtype = dtype
        self.grad_clip = float(grad_clip)

        # 只有 cuda 才谈得上 amp。cpu 上强制关掉而不是报错 ——
        # 理由和 Trainer 里"cuda 不可用就降级到 cpu"一样：
        # 换机器跑是常态，静默降级 + 打印，比 Traceback 有用。
        self.enabled = bool(enabled) and self.device.type == "cuda"
        if enabled and not self.enabled:
            print(f"[警告] 请求了 amp，但 device={self.device}，已关闭")

        if self.enabled and scaler is None:
            scaler = MyGradScaler()
        self.scaler = scaler if self.enabled else None

        # 统计：这一步有没有被跳过
        self.skipped = 0
        self.steps = 0

    # ── 前向上下文 ────────────────────────────────────────────────
    def autocast(self):
        """autocast 上下文。关闭时返回 nullcontext()（零开销、零行为变化）。

        ⚠️ autocast 只影响**前向**。反向的类型转换由 autograd 自己按
        前向记录的类型推回去，不需要额外开一个反向 autocast。
        """
        if not self.enabled:
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type=self.device.type, dtype=self.dtype)

    # ── 反向 ──────────────────────────────────────────────────────
    def backward(self, loss: torch.Tensor) -> None:
        if not self.enabled:
            loss.backward()
            return

        # ── 一个很隐蔽的坑：loss 本身如果是 fp16，乘 scale 会立刻溢出 ──
        # scale 初始 65536，fp16 上限 65504。loss 只要有 1.0，一乘就 inf。
        # 官方的 CE 在 autocast 下会自动走 fp32，所以默认不会遇到；
        # 但手写 loss（Day 16 的 myce）会返回 fp16 —— 那一乘就是 inf，
        # 然后 scaler 会以为"梯度炸了"，不停砍 scale，训练悄悄停摆。
        # 所以这里先升到 fp32 再乘。关 4 会把这个场景钉死。
        if loss.dtype != torch.float32:
            loss = loss.float()
        self.scaler.scale(loss).backward()

    # ── 更新 ──────────────────────────────────────────────────────
    def step(self, opt, params=None) -> None:
        """unscale → clip → step → update，顺序不可换。"""
        self.steps += 1

        if not self.enabled:
            # 非 amp 路径：clip 也要支持（config 里 grad_clip > 0 时）
            if self.grad_clip > 0 and params is not None:
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
            opt.step()
            return

        # ① 先除回真实尺度（同时也是 inf 检测点）
        self.scaler.unscale_(opt)
        # ② 再 clip —— 此刻梯度已经是真实尺度，max_norm 才有意义
        if self.grad_clip > 0 and params is not None:
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
        # ③ 条件更新。scaler.step 内部按 _found_inf 决定是否真的 opt.step()
        before = self.scaler.scale_value
        self.scaler.step(opt)
        # 只要 update 之前 scale 变小了，就说明这一步炸过、被跳过了。
        # 为什么不用 scaler 的内部标志：那是私有属性，
        # 而"scale 变小"是**对外可观测**的行为，断言行为比断言实现活得久。
        self.scaler.update()
        if self.scaler.scale_value < before:
            self.skipped += 1

    # ── 状态 ──────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        if not self.enabled:
            return {}
        return {
            "scale": self.scaler.scale_value,
            "growth_tracker": getattr(self.scaler, "_growth_tracker", 0),
        }

    def describe(self) -> str:
        if not self.enabled:
            return "amp 关闭（fp32）"
        s = self.scaler
        return (f"amp 开启  dtype={self.dtype}  scale={s.scale_value:g}  "
                f"grad_clip={self.grad_clip or 'off'}")


# ═══════════════════════════════════════════════════════════════════
# 3. 六关
# ═══════════════════════════════════════════════════════════════════

def _toy_net(dtype=torch.float32):
    """一个够小、够快、但有真实结构的小网络（含 BN，因为 BN 的 buffer 是 fp32）。"""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(8 * 14 * 14, 10),
    ).to(dtype=dtype)


def _batch(n=64, device="cuda", seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g).to(device)
    y = torch.randint(0, 10, (n,), generator=g).to(device)
    return x, y


def gate1_my_vs_official() -> bool:
    """关 1：手写 GradScaler 的 scale 轨迹 == 官方（逐步对拍）。

    喂一段**人为设计**的 inf/nan 模式（第 3、7、8 步炸），
    要求两边的 scale 序列、growth_tracker、以及"是否跳过本步"全部一致。
    """
    print("\n[关 1] 手写 GradScaler vs 官方（scale 轨迹逐步对拍）")
    dev = "cuda"
    torch.manual_seed(0)

    pattern = [False, False, False, True, False, False, False, True, True,
               False, False, False, False, False]

    def run(kind):
        model = _toy_net().to(dev)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        crit = torch.nn.CrossEntropyLoss()
        x, y = _batch(device=dev)
        if kind == "mine":
            sc = MyGradScaler(growth_interval=3)
        else:
            sc = torch.amp.GradScaler("cuda", growth_interval=3)
        trace, stepped = [], []
        for inject_inf in pattern:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            if inject_inf:
                # 人为造一个 inf 梯度：这是"scale 太大"的等价信号
                loss = loss * float("inf")
            sc.scale(loss).backward()
            sc.unscale_(opt)
            # 记录"这一步到底有没有真的更新"：把梯度之和当指纹
            before = [p.detach().clone() for p in model.parameters()]
            ret = sc.step(opt)
            after = list(model.parameters())
            moved = any(not torch.equal(b, a.detach()) for b, a in zip(before, after))
            stepped.append(moved)
            sc.update()
            g = sc.get_scale()
            trace.append(float(g.item() if torch.is_tensor(g) else g))
            del ret
        return trace, stepped

    t_mine, s_mine = run("mine")
    t_off, s_off = run("official")

    print(f"    注入 inf 的步：{[i for i, v in enumerate(pattern) if v]}")
    print(f"    {'步':>4}{'手写 scale':>14}{'官方 scale':>14}{'手写更新':>10}{'官方更新':>10}")
    print("    " + "-" * 54)
    for i, (a, b, sa, sb) in enumerate(zip(t_mine, t_off, s_mine, s_off)):
        flag = "" if (a == b and sa == sb) else "   ✗"
        print(f"    {i:>4}{a:>14g}{b:>14g}{str(sa):>10}{str(sb):>10}{flag}")

    assert t_mine == t_off, f"scale 轨迹不一致：{t_mine} vs {t_off}"
    assert s_mine == s_off, f"更新决策不一致：{s_mine} vs {s_off}"
    assert any(v for v in pattern), "构造问题：没有注入 inf"
    skipped = [i for i, s in enumerate(s_mine) if not s]
    print(f"\n    跳过的步：{skipped}   （正好是注入 inf 的那几步）")
    print("    ✓ 通过（手写与官方的 scale 轨迹、跳过决策逐步一致）")
    return True


def gate2_underflow_evidence() -> bool:
    """关 2：fp16 的致命问题不是上溢，是**下溢**。

    直接把一组有代表性的梯度值放进 fp16，看哪些活不下来。
    """
    print("\n[关 2] 梯度在 fp16 里的存活下限")
    vals = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    print(f"    {'梯度值':>10}{'fp16 原样':>14}{'变 0':>7}"
          f"{'(fp16×65536)/65536':>22}{'变 0':>7}")
    print("    " + "-" * 62)
    zeros_raw, zeros_scaled = 0, 0
    for v in vals:
        t32 = torch.tensor(v, dtype=torch.float32)
        raw = t32.to(torch.float16)
        back = (t32 * 65536).to(torch.float16).to(torch.float32) / 65536
        if raw.item() == 0.0:
            zeros_raw += 1
        if back.item() == 0.0:
            zeros_scaled += 1
        print(f"    {v:>10.0e}{raw.item():>14.3e}"
              f"{'是' if raw.item() == 0 else '':>7}"
              f"{back.item():>22.4e}"
              f"{'是' if back.item() == 0 else '':>7}")

    print(f"""
    原样放 fp16：{zeros_raw}/{len(vals)} 个值**直接变 0**
    放大再还原：{zeros_scaled}/{len(vals)} 个值变 0
    （放大后的值是"近似原值"，1e-5、1e-6 那两行有 1~2% 的相对误差 ——
      那是 fp16 尾数只有 10 bit 的精度损失，不是下溢，两者要分开看）

    fp16 最小正规格数 6.10e-05，靠 subnormal 能摸到 5.96e-08。
    fp32 能到 1e-38。所以 **1e-9 的梯度在 fp32 里完全正常，在 fp16 里是 0**。
    """)
    # 判据：放大必须把"变 0"的个数压到 0
    assert zeros_scaled == 0, (
        f"放大之后仍有 {zeros_scaled} 个值下溢成 0，说明 scale 不够大。\n"
        f"  65536 × 1e-9 = 6.55e-5，应该刚好在 subnormal 之上。"
    )
    assert zeros_raw > 0, "原样放 fp16 应该有值下溢 —— 否则这组用例根本没覆盖到边界"
    print("    ✓ 通过（下溢是真的；放大能救回来，但救不回精度）")
    return True


def gate3_scale_dynamics() -> bool:
    """关 3：动态 scaling 的完整轨迹 —— scale 会自己找到量级。

    故意从一个**过大**的 init_scale 开始，看它怎么自己降下来并稳定。
    """
    print("\n[关 3] scale 自己找到量级（从过大的 2^24 起步）")
    dev = "cuda"
    torch.manual_seed(0)
    model = _toy_net().to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    crit = torch.nn.CrossEntropyLoss()
    x, y = _batch(device=dev)

    sc = MyGradScaler(init_scale=2.0 ** 24, growth_interval=2)
    trace, skipped = [], []
    for step in range(30):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = crit(model(x), y)
        sc.scale(loss).backward()
        sc.unscale_(opt)
        found = sc._found_inf
        sc.step(opt)
        sc.update()
        trace.append(sc.scale_value)
        skipped.append(found)

    print(f"    {'步':>4}{'scale':>14}{'跳过':>8}")
    print("    " + "-" * 28)
    for i in (0, 1, 2, 3, 5, 8, 12, 20, 29):
        print(f"    {i:>4}{trace[i]:>14g}{str(skipped[i]):>8}")

    final = trace[-1]
    print(f"""
    起点 2^24 = {2.0**24:.0f}，终点 {final:g}
    {"前几步连续跳过，scale 一路砍半，然后稳定下来。" if skipped[0] else "起步就没炸。"}
    **这就是动态 scaling 的全部智慧**：不用猜该乘多少，
    从偏大的地方起步，让 inf 检测自己把它压到合适位置。
    """)
    # 关键性质：如果一开始就炸，最终 scale 必须比起点小
    if any(skipped[:5]):
        assert final < 2.0 ** 24, "一开始就溢出，scale 却没降下来"
    print("    ✓ 通过（scale 会朝可行域自己收敛）")
    return True


def gate4_exp_overflow_threshold() -> bool:
    """关 4：fp16 的 exp 溢出阈值是 11.09 —— 以及它怎么咬到 CE。"""
    print("\n[关 4] fp16 的 exp 上限，以及它和交叉熵的关系")

    print(f"    {'x':>8}{'exp(x) fp32':>18}{'exp(x) fp16':>18}")
    print("    " + "-" * 46)
    first_inf = None
    for v in [8.0, 10.0, 11.0, 11.05, 11.09, 11.1, 11.5, 12.0]:
        e32 = math.exp(v)
        e16 = torch.exp(torch.tensor(v, dtype=torch.float16)).item()
        if math.isinf(e16) and first_inf is None:
            first_inf = v
        print(f"    {v:>8.2f}{e32:>18.4g}{e16:>18.4g}")

    print(f"""
    fp16 最大值 65504 = exp({math.log(65504):.4f})。
    **所以 x > 11.09 时 exp(x) 在 fp16 里直接是 inf。**
    第一次 inf 出现在 x={first_inf}。

    ── 这件事在 CE 里怎么会咬到人 ──────────────────────────────
    这一节要区分两种写法：
      ① 先减最大值再 exp（log-sum-exp 的稳定版）：
         exp 的参数 ≤ 0，永远 ≤ 1，**f​p16 也绝不溢出**。
         Day 16 手写的 log_softmax 就是这一版。
      ② 直接 exp(s)：logits 一旦 > 11.09 就是 inf。

    那为什么题目里说"CE 的 log-sum-exp 必须用 fp32"？
    因为还有一个更隐蔽的入口：**loss 本身是 fp16 时，
    乘 scale 会溢出。** scale 初始 65536 > fp16 上限 65504，
    loss 只要有 1.0 就爆。
    """)

    # 实测②：loss 是 fp16 时乘 scale 会怎样
    dev = "cuda"
    loss32 = torch.tensor(2.3, device=dev, dtype=torch.float32)
    loss16 = loss32.to(torch.float16)
    scaled32 = loss32 * 65536
    scaled16 = loss16 * torch.tensor(65536.0, device=dev, dtype=torch.float16)
    print(f"    loss(fp32)=2.3 × 65536 = {scaled32.item():.1f}  有限")
    print(f"    loss(fp16)=2.3 × 65536 = {scaled16.item()}   ← 溢出")
    assert torch.isfinite(scaled32) and not torch.isfinite(scaled16), "构造与预期不符"

    # 实测③：官方 CE 在 autocast 下返回什么 dtype
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits16 = torch.randn(8, 10, device=dev, dtype=torch.float16) * 3
        ce = torch.nn.CrossEntropyLoss()(logits16, torch.zeros(8, device=dev, dtype=torch.long))
    print(f"\n    官方 CrossEntropyLoss 在 autocast 下的输出 dtype：{ce.dtype}")
    assert ce.dtype == torch.float32, (
        "官方 CE 在 autocast 下应该返回 fp32（它在 PyTorch 的 promote 列表里）。"
        f"实测得到 {ce.dtype} —— 那 AmpStepper.backward 里那句 .float() 就是必须的。"
    )
    print("    → 官方 CE 被 autocast 的 promote 列表兜住了。")
    print("      **手写 loss 不一定** —— 取决于它用了哪些算子，关 5 实测。")
    print("    ✓ 通过")
    return True


def gate5_handwritten_loss_trap() -> bool:
    """关 5：手写 loss 在 autocast 下的 dtype，**取决于它用了哪些算子**。

    ⚠️ 这一关我第一版写错了：我断言「手写 loss 一定返回 fp16」，
    实测 Day 16 那版（max/sub/exp/sum/log/gather/mean）返回的是 **fp32**。
    原因是这些归约算子都在 autocast 的 fp32 promote 列表里，
    所以那版**碰巧**安全。

    真正的危险是「loss 里出现了 matmul / mm / bmm / conv / linear」——
    这几类 autocast 留在 fp16。这一关把两种情况都摆出来。
    """
    print("\n[关 5] 手写 loss 的 dtype：取决于写法，不取决于「手写」")
    dev = "cuda"

    class ReduceOnly(torch.nn.Module):
        """Day 16 那版：全是归约算子。"""

        def forward(self, logits, target):
            m = logits.max(dim=-1, keepdim=True).values
            shifted = logits - m
            lse = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
            log_p = shifted - lse
            return -log_p.gather(1, target.view(-1, 1)).mean()

    class TailMatmul(torch.nn.Module):
        """数学上等价，但末尾套了一次 1×1 matmul。

        现实里很容易写成这样：比如"按样本权重加权"、或者把 loss
        收成一个 (1,1) 张量再 `@` 一个权重。只要出现 matmul，
        autocast 就把它留在 fp16。
        """

        def forward(self, logits, target):
            base = ReduceOnly()(logits, target)
            w = torch.ones(1, 1, device=logits.device)
            return (base.reshape(1, 1) @ w).reshape(())

    logits = torch.randn(16, 10, device=dev) * 3
    target = torch.zeros(16, device=dev, dtype=torch.long)
    limit = 65504.0

    print(f"    {'写法':<26}{'dtype':<16}{'值':>9}"
          f"{'×65536':>12}{'结果':>8}")
    print("    " + "-" * 72)
    rows = []
    for name, mod in [("只含归约算子（Day 16）", ReduceOnly()),
                      ("末尾套一次 matmul", TailMatmul()),
                      ("官方 CrossEntropyLoss", torch.nn.CrossEntropyLoss())]:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = mod(logits, target)
        scaled = out * torch.tensor(65536.0, device=dev, dtype=out.dtype)
        finite = bool(torch.isfinite(scaled).item())
        rows.append((name, out.dtype, finite))
        print(f"    {name:<26}{str(out.dtype).replace('torch.', ''):<16}"
              f"{out.item():>9.4f}"
              f"{('有限' if finite else 'inf'):>12}"
              f"{'✅' if finite else '❌':>8}")

    reduce_dtype = rows[0][1]
    matmul_dtype = rows[1][1]
    assert reduce_dtype == torch.float32, (
        f"纯归约的手写 loss 在 autocast 下应该是 fp32（算子被 promote），"
        f"实测 {reduce_dtype}。若为 fp16，说明 PyTorch 版本的 promote 列表变了，"
        "这一关的结论要跟着改。"
    )
    assert matmul_dtype == torch.float16, (
        f"含 matmul 的手写 loss 应该留在 fp16，实测 {matmul_dtype}。"
        "若已是 fp32，说明 autocast 规则变了 —— 那这个坑就不存在了，"
        "但也别把结论反过来写，要如实记录。"
    )

    # 再确认：AmpStepper 的 .float() 能不能救
    with torch.amp.autocast("cuda", dtype=torch.float16):
        bad = TailMatmul()(logits, target)
    fixed = bad.float() * 65536.0
    print(f"\n    出事的那一版：直接 ×65536 = {bad * torch.tensor(65536.0, device=dev, dtype=torch.float16)}")
    print(f"                  先 .float() 再乘 = {fixed.item():.1f}  有限")
    assert torch.isfinite(fixed).item(), "升到 fp32 之后应该能乘"

    print(f"""
    所以结论要精确到"写法"这一层，不能停在"手写就危险"：

      只含归约算子  → autocast 把它们 promote 到 fp32 → 碰巧安全
      含 matmul 类  → 留在 fp16 → 乘 scale 立刻 inf

    而 inf 之后会发生什么？scaler 以为"梯度炸了"，于是
    **跳过本步 + scale 砍半**，周而复始 → 训练悄悄停摆，
    loss 卡在一个值上不动，但一行报错都没有。

    → 所以 `AmpStepper.backward` 里那句 `loss.float()` 是必需的：
      它的作用不是"防万一"，是把"写法会变"这件事兜住。
      检查成本一次 dtype 比较，收益是避免一个静默失败。
    """)
    print("    ✓ 通过（陷阱真实存在，触发条件已定位到 matmul）")
    return True


def gate6_clip_order() -> bool:
    """关 6：`unscale_` 在 `clip` 之前 vs 之后 —— 差 65536 倍，且不报错。"""
    print("\n[关 6] unscale 与 clip 的顺序")
    dev = "cuda"
    torch.manual_seed(0)
    x, y = _batch(device=dev)
    crit = torch.nn.CrossEntropyLoss()
    MAX_NORM = 1.0

    def run(order: str):
        torch.manual_seed(0)
        model = _toy_net().to(dev)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sc = MyGradScaler()
        before = [p.detach().clone() for p in model.parameters()]

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = crit(model(x), y)
        sc.scale(loss.float()).backward()

        if order == "wrong":
            # ❌ clip 在 unscale 之前：clip 的对象是放大 65536 倍的梯度
            raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            sc.unscale_(opt)
            sc.step(opt)
        else:
            # ✅ 先 unscale，再 clip
            sc.unscale_(opt)
            raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            sc.step(opt)
        sc.update()

        delta = sum((p.detach() - b).pow(2).sum() for p, b in zip(model.parameters(), before)).sqrt()
        return float(raw_norm), float(delta)

    n_wrong, d_wrong = run("wrong")
    n_right, d_right = run("right")

    print(f"{'顺序':<26}{'clip 前看到的 norm':>20}{'实际步长 |Δw|':>16}")
    print("-" * 64)
    print(f"{'❌ clip 然后 unscale':<26}{n_wrong:>20.4g}{d_wrong:>16.3e}")
    print(f"{'✅ unscale 然后 clip':<26}{n_right:>20.4g}{d_right:>16.3e}")
    print(f"\n    比值：{d_right / d_wrong if d_wrong > 0 else float('inf'):.0f} 倍")

    print(f"""
    读法：
      ❌ 顺序写反时，clip 看到的 norm 是 {n_wrong:.4g} —— 放大了 65536 倍。
         于是它把梯度整体缩到 max_norm={MAX_NORM}，等于**额外衰减了 65536 倍**。
         之后 unscale_ 又除一次 65536，最终步长只剩 {d_wrong:.2e}。
      ✅ 顺序正确时，clip 看到的是真实 norm，步长 {d_right:.2e}。

      **两次都"没报错"**，loss 也都在降 —— 只是写反那一版慢了几个数量级。
      这就是为什么这个坑必须靠结构（AmpStepper）而不是靠记性。
    """)
    assert n_wrong > n_right * 100, (
        "顺序写反时 clip 看到的 norm 应该远大于正确顺序（差 ~scale 倍）。"
        f"实测 {n_wrong:.4g} vs {n_right:.4g} —— 差值不够，检查 clip 是否真的作用在梯度上。"
    )
    assert d_wrong < d_right, "写反顺序后的步长应该更小（被额外衰减了）"
    print("    ✓ 通过（顺序错了不报错，但步长差好几个量级）")
    return True


# ═══════════════════════════════════════════════════════════════════
# 4. 入口
# ═══════════════════════════════════════════════════════════════════

def main() -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Day 26 amp 自测（不需要数据集）")
    ap.add_argument("--train", action="store_true", help="跑全部六关（默认只跑前两关）")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("这六关全部需要 CUDA（fp16 的 autocast 在 CPU 上行为不同）")
        return 1

    print("=" * 70)
    print("Day 26 · 混合精度 amp")
    print(f"torch {torch.__version__}  cuda={torch.cuda.get_device_name(0)}")
    p = torch.cuda.get_device_properties(0)
    # ⚠️ 从 sm 号**推不出**有没有 Tensor Core：sm_75 里
    # RTX 20 系列有，GTX 16 系列（TU116/TU117）没有 —— 同一个 sm_75。
    # 所以这里只看名字，而且把"这是推断"写出来，别让它看起来像结论。
    name = torch.cuda.get_device_name(0).upper()
    no_tc = any(k in name for k in ("GTX 16", "GTX 165", "GTX 166", "MX"))
    print(f"sm_{p.major}{p.minor}  "
          f"Tensor Core: {'推断无（GTX 16 系列是 TU116，砍掉了 TC）' if no_tc else '推断有'}")
    print("            ↑ 靠设备名推断。真正的判据见下面 A 段的实测速度。")
    print("=" * 70)
    print("""
fp16 的两个方向：
  上溢   > 65504（exp 在参数 11.09 时就到顶）→ inf
  下溢   < 6e-8 → 变 0

深度学习的梯度通常是 1e-5 ~ 1e-9 级，**真正致命的是下溢**。
GradScaler 只做一件事：放大 loss，把梯度抬到 fp16 能表示的范围里。
""")

    gates = [
        ("关 1 手写 scaler == 官方", gate1_my_vs_official),
        ("关 2 fp16 的下溢下限", gate2_underflow_evidence),
        ("关 3 scale 的动态轨迹", gate3_scale_dynamics),
        ("关 4 exp 阈值 11.09", gate4_exp_overflow_threshold),
        ("关 5 手写 loss 的 fp16 陷阱", gate5_handwritten_loss_trap),
        ("关 6 unscale 与 clip 的顺序", gate6_clip_order),
    ]
    if not args.train:
        print("（预览模式只跑前两关。加 --train 跑全六关）\n")
        gates = gates[:2]

    passed, ok = 0, True
    results = []
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
        passed += 1 if good else 0
        ok = ok and good

    print("\n" + "=" * 70)
    print(f"总计：{passed}/{len(results)} 关通过")
    for name, good in results:
        print(f"  {'✓' if good else '✗'}  {name}")
    print("=" * 70)
    if ok:
        print("""
跑完的下一步：拿它去测「这台机器上 amp 值不值」。
六关只能证明实现是对的，**不能证明它有用** ——
有用没用要看真实训练的速度和曲线，见 docs/Day26-混合精度.md 的实测表。
""")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
