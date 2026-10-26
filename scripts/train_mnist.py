"""MNIST 训练入口 —— 现在只是一个兼容薄壳。

════════════════════════════════════════════════════════════════════
为什么这个文件还留着
════════════════════════════════════════════════════════════════════
Day 18 把原来的 546 行实现拆进了 src/engine/，逻辑全部搬走了。
但这个文件名被三处引用，删掉就会断：

    1. scripts/run_ablation.py    用 subprocess 调它跑四组对照实验
    2. README.md                  命令示例里写着它
    3. 你自己的肌肉记忆            前面十几天一直在敲这个名字

所以保留成薄壳：**命令完全不变，行为完全不变，只是实现搬了个家。**
（`python -c "import src.engine.cli; main()"` 这种"删了再想怎么补"的做法
 才是糟糕的重构 —— 重构的第一原则是调用方无感。）

新代码请直接用 scripts/train.py，它支持 --config / --list / --set 等通用能力。

跑法（和以前完全一样）：
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\train_mnist.py
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\train_mnist.py --epochs 2 --tag repro
    & "E:\\anaconda\\envs\\pytorch\\python.exe" scripts\\train_mnist.py --optimizer mysgd
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 和 scripts/train.py 调的是**同一个函数**。
# 共用实现是「不会有第二份真相」的唯一保证 ——
# 要是这里再复制一遍参数表，两边迟早会漂移。
from src.engine.cli import main


if __name__ == "__main__":
    main()
