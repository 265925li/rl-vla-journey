"""通用训练入口。

跑法（PowerShell，项目根目录）：
    $PY = "E:\\anaconda\\envs\\pytorch\\python.exe"

    & $PY scripts\\train.py                                    # 用默认配置
    & $PY scripts\\train.py --config configs/mnist_mlp.yaml
    & $PY scripts\\train.py --list                             # 有哪些模型/数据集
    & $PY scripts\\train.py --dry-run                          # 只看最终配置，不训练
    & $PY scripts\\train.py --epochs 3 --tag quick              # 覆盖单个字段
    & $PY scripts\\train.py --set dropout=0.2 --set hidden=512   # 覆盖任意字段
    & $PY scripts\\train.py --loss myce --linear my             # 全换手写实现

════════════════════════════════════════════════════════════════════
实现去哪了（Day 18 的模块化重构）
════════════════════════════════════════════════════════════════════
这个文件只有十几行，因为真正的实现搬进了 src/engine/：

    src/engine/registry.py   名字 → 构建函数的映射表
    src/engine/builtin.py    注册 mnist 数据集和 mlp 模型
    src/engine/builders.py   优化器 / 损失 / 调度器
    src/engine/loops.py      训练循环、评估循环
    src/engine/trainer.py    Trainer，串起整个流程
    src/engine/cli.py        参数解析（本文件调的就是它）
    src/utils/plotting.py    画曲线

为什么入口要这么薄：
**入口越薄，越不会有两份实现。** scripts/train.py 和 scripts/train_mnist.py
共用同一个 main()，就不可能出现"用这个脚本跑出来的结果和那个不一样"。

回头看 Day 13 的时候，train_mnist.py 有 546 行，什么都往里塞：
参数表、造模型、训练循环、画图、写 json。当时那样写是对的 ——
**先跑通，再整理**。但一旦要跑第二个数据集、第二个模型，
那种结构就必须拆，否则每加一个实验就是复制一份 546 行再改。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让 `python scripts/train.py` 也能 import 到 src（否则 ModuleNotFoundError）。
# 为什么不用 `pip install -e .`：这样零安装、换台机器也能直接跑。
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.cli import main


if __name__ == "__main__":
    main()
