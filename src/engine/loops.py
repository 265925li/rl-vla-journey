"""训练循环与评估循环。

整个深度学习最核心的五个动作就在这里，从 Day 11 到现在一行都没变：

    zero_grad → forward → loss → backward → step

后面所有花哨的东西（正则化、手写算子、通用模板）都是围在这五行外面的壳。

【分层上的一个选择】
这两个函数**只收 flatten 这一个布尔值，不收整个 cfg**。
理由：核心循环不应该认识 Config —— 一旦它开始读 cfg.xxx，
以后每加一个配置字段，你都得怀疑「这个字段会不会影响训练循环」。
现在它只依赖三个东西：模型、loader、优化器/损失，接口面最小。

Day 26 加了 amp，这条原则被检验了一次：循环需要知道"要不要 autocast、
要不要 scaler"。有两条路：

    ❌ 传 cfg 进来，循环里 `if cfg.amp == "fp16":`
       → 循环从此认识配置系统，以后每个 amp 相关字段都要在这里加分支
    ✅ 传一个已经装好的 AmpStepper 进来，循环只调它的三个方法
       → 循环仍然不认识 Config，只认识一个"一步怎么走"的小对象

选了后者。代价是 loops.py 多了一个可选参数（默认 None = 纯 fp32，
行为与 Day 25 完全相同），收益是 amp 的全部复杂度被关在
src/utils/amp_scaler.py 一个文件里。
"""
from __future__ import annotations

import contextlib

import torch


def _to_device(xb, device, flatten: bool):
    """把一批图像整成「模型能吃的形状 + 在正确设备上」。

    ── 摊平 ───────────────────────────────────────────────────────
    为什么要摊平：MLP 不认识"图像"，它只认识"一行数字"。
    (B, 1, 28, 28) 的 784 个像素要排成 (B, 784) 才能喂进第一层 Linear。

    为什么用 flatten 开关而不是让框架自动判断：
    因为 **CNN 恰恰不能摊平** —— 卷积要的就是 28×28 的空间结构，
    摊平之后"相邻像素相关"这个信息就永久丢了（这也是 MNIST+MLP
    卡在 98% 而上不去的原因之一）。
    框架无法替你决定"这个模型要不要空间结构"，只有你知道，所以要显式配置。

    ⚠️ 用 `view(xb.size(0), -1)` 而不是 `xb.flatten(1)`：结果相同，
    但 view 在内存连续的张量上零拷贝，只是换个形状标签。
    更实际的理由是：保持和重构前完全一致的调用，
    这样"重构没改变数值行为"才验得干净。

    ── 搬设备 ─────────────────────────────────────────────────────
    漏 `.to(device)` → 报 "Expected all tensors to be on the same device"
    （Day 11 实测过）。注意这个错**只在有 GPU 的机器上出现**，
    纯 CPU 环境跑是看不出来的 —— 所以「我本地能跑」不等于没 bug。
    """
    if flatten:
        xb = xb.view(xb.size(0), -1)
    return xb.to(device)


def train_one_epoch(model, loader, opt, crit, device, flatten: bool = True,
                    log=None, stepper=None) -> tuple[float, float, int]:
    """跑完一个 epoch，返回 (平均 loss, 准确率, 样本数)。

    循环体就是那五行，多的只是「累加统计量」和「第一次打印形状」。

    参数 stepper：None 表示纯 fp32（Day 25 的行为，**逐位不变**）。
                  传 AmpStepper 时，前向包 autocast、反向往 loss 上乘 scale、
                  更新走 unscale→clip→step→update 这条固定顺序。

    ⚠️ stepper=None 这条路径必须和 Day 25 逐位一致。所以这里**不写**
    `with nullcontext():` 包住整个循环体 —— 而是直接在分支里选两条路径。
    少一层缩进变化，也让"没传 stepper 时行为完全不变"在源码上一眼可见。
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

        xb = _to_device(xb, device, flatten)
        yb = yb.to(device)

        opt.zero_grad()              # ① 擦掉上一步的梯度
        if stepper is None:
            logits = model(xb)           # ② 前向
            loss = crit(logits, yb)      # ③ 打分：一个标量 ()
            loss.backward()              # ④ 追责：算出每个参数的梯度
            opt.step()                   # ⑤ 微调：按梯度挪一小步
        else:
            # Day 26：还是那五步，但前向/反向/更新各自多了 amp 的动作。
            # 顺序由 AmpStepper 保证 —— 尤其 unscale_ 必须在 clip 之前（关 6）。
            with stepper.autocast():
                logits = model(xb)
                loss = crit(logits, yb)
            stepper.backward(loss)
            stepper.step(opt, model.parameters())

        # ── 累加统计（按样本数加权，Day 11 算过不加权会错多少）──
        # 末批通常不满（60000 % 128 ≠ 0），简单平均会让这一批的权重过大。
        # 实测极端情况（val 600 条、bs=512）简单平均偏差 64%。
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        n += bs

    # n 一定是正数，除非 loader 是空的。空 loader 会让 0/0 抛
    # "division by zero" —— 报错信息完全不提「你的数据集是空的」，
    # 所以这里主动给一句人话。
    if n == 0:
        raise ValueError("训练集是空的：loader 一个 batch 都没产出")
    return total_loss / n, total_correct / n, n


@torch.no_grad()  # 评估不需要梯度，关掉能省显存、也能防止手滑改了参数
def evaluate(model, loader, crit, device, flatten: bool = True) -> tuple[float, float]:
    """在验证/测试集上算 (平均 loss, 准确率)。"""
    model.eval()   # 评估模式：Dropout 关掉、BatchNorm 用 running 统计量

    total_loss, total_correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb = _to_device(xb, device, flatten)
        yb = yb.to(device)
        logits = model(xb)
        loss = crit(logits, yb)

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        n += bs

    if n == 0:
        raise ValueError("评估集是空的：loader 一个 batch 都没产出")
    return total_loss / n, total_correct / n
