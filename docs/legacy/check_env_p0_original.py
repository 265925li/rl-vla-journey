"""
P0 环境自检脚本
================
用途：一次性验证 PyTorch + CUDA + Gymnasium + MuJoCo + Box2D + Stable-Baselines3 全链路可用。

运行：
    E:\\anaconda\\envs\\pytorch\\python.exe check_env.py

预期：全部显示 [PASS]，最后打印 "全部通过"。
任何一项 [FAIL] 都会告诉你具体原因。
"""

import os
import sys
import time
import traceback

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib

matplotlib.use("Agg")

results = []


def check(name, fn):
    try:
        info = fn()
        results.append((name, True, info))
        print(f"  [PASS] {name}" + (f"  ->  {info}" if info else ""))
        return True
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        results.append((name, False, msg))
        print(f"  [FAIL] {name}  ->  {msg}")
        return False


print("=" * 68)
print(" P0 环境自检")
print("=" * 68)

# ---------------------------------------------------------------- 1. 基础
print("\n[1/6] 基础环境")
print(f"  解释器 : {sys.executable}")
print(f"  Python : {sys.version.split()[0]}")
print(f"  工作目录: {os.getcwd()}")


def _torch():
    import torch

    return f"torch {torch.__version__}"


def _cuda():
    import torch

    assert torch.cuda.is_available(), "CUDA 不可用"
    p = torch.cuda.get_device_properties(0)
    x = torch.randn(512, 512, device="cuda")
    y = (x @ x).sum().item()
    del x
    torch.cuda.empty_cache()
    return f"{p.name} / {p.total_memory / 1024**3:.1f}GB / sm_{p.major}{p.minor} / 矩阵乘验证 ok({y:.0f})"


check("import torch", _torch)
check("CUDA 可用且能算", _cuda)

# ---------------------------------------------------------------- 2. 科学计算
print("\n[2/6] 科学计算栈")


def _np():
    import numpy as np

    return f"numpy {np.__version__}"


def _scipy():
    import scipy

    return f"scipy {scipy.__version__}"


def _pd():
    import pandas as pd

    return f"pandas {pd.__version__}"


def _mpl():
    import matplotlib

    return f"matplotlib {matplotlib.__version__}"


check("import numpy", _np)
check("import scipy", _scipy)
check("import pandas", _pd)
check("import matplotlib", _mpl)

# ---------------------------------------------------------------- 3. 环境库
print("\n[3/6] RL 环境库")


def _gym():
    import gymnasium as gym

    return f"gymnasium {gym.__version__}"


def _mujoco():
    import mujoco

    return f"mujoco {mujoco.__version__}"


check("import gymnasium", _gym)
check("import mujoco", _mujoco)

# ---------------------------------------------------------------- 4. 环境实例
print("\n[4/6] 环境实例化与交互")


def make_env(candidates):
    """按候选 ID 依次尝试，返回第一个可用环境（不渲染）。"""
    import gymnasium as gym

    last = None
    for eid in candidates:
        try:
            env = gym.make(eid, render_mode=None)
            return eid, env
        except Exception as e:
            last = e
    raise last


def _cartpole():
    eid, env = make_env(["CartPole-v1"])
    env.reset(seed=0)
    total = 0.0
    for _ in range(50):
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            env.reset()
    env.close()
    return f"{eid} | obs{env.observation_space.shape} act{env.action_space.n} | 50 步随机累计奖励 {total:.0f}"


def _lunar():
    eid, env = make_env(["LunarLander-v3", "LunarLander-v2"])
    env.reset(seed=0)
    total = 0.0
    for _ in range(50):
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            env.reset()
    env.close()
    return f"{eid} | obs{env.observation_space.shape} act{env.action_space.n} | Box2D 可用"


def _mujoco_env():
    eid, env = make_env(["HalfCheetah-v5", "HalfCheetah-v4"])
    env.reset(seed=0)
    total = 0.0
    for _ in range(50):
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            env.reset()
    env.close()
    return f"{eid} | obs{env.observation_space.shape} act{env.action_space.shape} | 连续控制可用"


check("CartPole（经典控制）", _cartpole)
check("LunarLander（Box2D）", _lunar)
check("HalfCheetah（MuJoCo）", _mujoco_env)

# ---------------------------------------------------------------- 5. 算法库
print("\n[5/6] 算法库与工具")


def _sb3():
    import stable_baselines3 as sb3

    return f"stable-baselines3 {sb3.__version__}"


def _sb3_gpu():
    import torch

    return "SB3 将使用 " + ("GPU (cuda)" if torch.cuda.is_available() else "CPU")


def _wandb():
    import wandb

    return f"wandb {wandb.__version__}"


def _tb():
    import tensorboard

    return f"tensorboard {tensorboard.__version__}"


check("import stable_baselines3", _sb3)
check("SB3 计算设备", _sb3_gpu)
check("import wandb", _wandb)
check("import tensorboard", _tb)

# ---------------------------------------------------------------- 6. 真训练
print("\n[6/6] 端到端训练验证（DQN 打 CartPole，约 20-60 秒）")


def _train():
    from stable_baselines3 import DQN
    from stable_baselines3.common.evaluation import evaluate_policy

    env = make_env(["CartPole-v1"])[1]
    t0 = time.time()
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=500,
        batch_size=64,
        gamma=0.99,
        verbose=0,
        seed=0,
        device="auto",
    )
    model.learn(total_timesteps=8000, progress_bar=False)
    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    env.close()
    dt = time.time() - t0
    return f"8000 步训练 {dt:.0f}s | 10 局评估均分 {mean_r:.1f} ± {std_r:.1f}"


ok_train = check("DQN 完整训练 + 评估", _train)

# ---------------------------------------------------------------- 汇总
print("\n" + "=" * 68)
failed = [r for r in results if not r[1]]
print(f" 结果：{len(results) - len(failed)}/{len(results)} 项通过")

if failed:
    print("\n 未通过项：")
    for name, _, msg in failed:
        print(f"   - {name}: {msg}")
    print("\n 环境尚未就绪，请把上面的 FAIL 信息发给助手。")
    sys.exit(1)

print("\n 全部通过 —— 可以开始 P0 阶段了。")
if ok_train:
    print(" 提示：DQN 只训练 8000 步，均分偏低是正常的，CartPole 需要约 5 万步才收敛。")
print("=" * 68)
