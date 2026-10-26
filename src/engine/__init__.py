"""engine 包：可复用的训练引擎。

    registry.py   注册表机制（名字 → 构建函数），零项目内依赖
    builtin.py    把 mnist / mlp 注册进去 —— 只有这个文件知道具体实现
    builders.py   优化器 / 损失 / 调度器 / 指标
    loops.py      训练循环、评估循环
    trainer.py    Trainer 类，把上面这些串成一次完整训练
    cli.py        命令行入口

为什么这里**刻意不 import 任何东西**：
一旦 `__init__.py` 里写 `from .trainer import Trainer`，
那么"import 本包"就会连锁 import trainer → builtin → registry。
这套依赖本身没环，但会带来一个隐蔽的副作用 ——
**只要有人 import 任何一个子模块，注册动作就发生了**，
于是"注册表里有什么"变得依赖 import 顺序，排查问题时非常难受。

现在这样，注册动作只在明确 `import src.engine.builtin` 时发生，
责任清晰：谁需要注册，谁显式去 import。

用法：
    from src.engine.trainer import Trainer
    from src.engine.cli import main
"""
