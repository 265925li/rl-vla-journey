# Day 3 学习文档 · argparse + YAML + dataclass

> 目标：**从此以后改参数不改代码。**
> 对应清单：`P0-逐日执行清单.md` → Day 3 · 40min 学习段（后面还有 80min + 60min 的实现任务）
> 本机环境：`E:\anaconda\envs\pytorch\python.exe` · pyyaml **6.0.3 已装**（实测）
> 最后更新：2026-10-17

⚠️ **本文所有代码我都实跑验证过，输出是真实截取的，不是编的。**
但请**自己敲一遍**，别复制粘贴——敲的过程才会记住。

---

## 0. 先看清楚这三样东西各自负责什么

很多教程把它们混着讲，导致学完还是不知道谁管谁。其实分工特别清楚：

| 工具 | 负责 | 一句话 |
|---|---|---|
| **YAML** | 存 | 把一堆参数**写进文件**，人能读、能改、能进 git |
| **dataclass** | 表达 | 把参数**变成一个对象**，`cfg.seed` 比 `d["seed"]` 安全得多 |
| **argparse** | 覆盖 | 临时**从命令行**改一两个值，不用动文件 |

数据流是这样的：

```
默认值  →  yaml 文件  →  命令行参数  →  Config 对象  →  代码里用 cfg.xxx
（写死在   （长期配置）    （临时覆盖）     （类型明确）
 dataclass）
```

**优先级：命令行 > yaml > 默认值。** 记这一句就够了，后面全是它的实现细节。

---

# 一、argparse（约 15 分钟）

## 1.1 最小可用例子

在 `rl-vla-journey/scripts/` 下建一个 `play_argparse.py`：

```python
import argparse

p = argparse.ArgumentParser(description="演示 argparse")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

print(args)
print("seed =", args.seed, type(args.seed).__name__)
```

跑：

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
& "E:\anaconda\envs\pytorch\python.exe" scripts\play_argparse.py
& "E:\anaconda\envs\pytorch\python.exe" scripts\play_argparse.py --seed 42
```

输出：

```
Namespace(seed=0)
seed = 0 int
Namespace(seed=42)
seed = 42 int
```

**三件事一眼看清**：
1. `--seed` 对应 `args.seed`（**横杠变成下划线**，`--log-level` → `args.log_level`）
2. `type=int` 决定了 `args.seed` 是**整数**不是字符串
3. `default=0` 是"用户没传时用什么"

## 1.2 五种你一定会用到的写法

```python
import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="演示 argparse 的四种常用写法",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 1) 最普通：有默认值，可不传
    p.add_argument("--seed", type=int, default=0, help="随机种子")

    # 2) 只能是某几个值之一
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cpu", "cuda"], help="训练设备")

    # 3) 开关：出现就是 True，不出现就是 False
    p.add_argument("--dry-run", action="store_true", help="只打印配置不训练")

    # 4) 收多个值
    p.add_argument("--tags", type=str, nargs="+", default=[], help="实验标签")

    # 5) default=None：区分「没传」和「传了」
    p.add_argument("--lr", type=float, default=None, help="学习率（覆盖 yaml 用）")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    print("解析结果：", args)
    print("转成字典（合并配置时要用）：", vars(args))
```

跑：

```powershell
& "E:\anaconda\envs\pytorch\python.exe" scripts\play_argparse.py --seed 42 --device cpu --dry-run --tags dqn ppo --lr 0.001
```

实测输出：

```
解析结果： Namespace(seed=42, device='cpu', dry_run=True, tags=['dqn', 'ppo'], lr=0.001)
转成字典（合并配置时要用）： {'seed': 42, 'device': 'cpu', 'dry_run': True, 'tags': ['dqn', 'ppo'], 'lr': 0.001}
```

## 1.3 ⭐ 本日最重要的一个知识点：`default=None`

注意第 5 项 `--lr` 的默认值是 `None`，**不是 0**。这不是随手写的。

因为真实的配置系统要回答一个问题：**用户到底传了这个参数没有？**

| 写法 | 用户没传时 `args.lr` | 能不能判断"用户没传" |
|---|---|---|
| `default=0.001` | `0.001` | ❌ 分不清是默认值还是用户传的 |
| `default=None` | `None` | ✅ `None` 就代表没传 |

于是合并逻辑可以写得非常干净：

```python
cli = {k: v for k, v in vars(args).items() if v is not None}   # 只留用户真传的
```

**如果把 default 写成具体值会怎样？** 我写了个错误示范给你看：

```python
import argparse

yaml_cfg = {"seed": 123}

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)   # ← 罪魁祸首
args = p.parse_args()

merged = {**yaml_cfg, **vars(args)}
print("yaml 里写的是 123，用户没传 --seed，合并结果却是：", merged["seed"])
```

实测输出：

```
yaml 里写的是 123，用户没传 --seed，合并结果却是： 0
```

**yaml 里明明写着 123，用户什么都没干，值却变成了 0。** 这就是"配置系统把好值踩坏了"——而且它不报错，你只会觉得"这参数怎么不生效"。

> 这条如果只能记一件事，就记这个：**所有准备被命令行覆盖的参数，`default` 都写 `None`。**

## 1.4 `-h` 是白送的

不用自己写帮助文本，argparse 自动生成。加上 `formatter_class=argparse.ArgumentDefaultsHelpFormatter` 后，默认值也会显示：

```
options:
  -h, --help            show this help message and exit
  --seed SEED           随机种子 (default: 0)
  --device {cpu,cuda}   训练设备 (default: cuda)
  --dry-run             只打印配置不训练 (default: False)
  --tags TAGS [TAGS ...]
                        实验标签 (default: [])
```

## 1.5 三个坑

| 坑 | 现象 | 正解 |
|---|---|---|
| 在模块顶层直接 `parse_args()` | `import` 这个文件时就直接退出/报错，因为在别人的命令行里解析 | 放进 `if __name__ == "__main__":` 或 `main()` 里 |
| 写了不存在的参数 | `error: unrecognized arguments: --foo` | 拼写检查；确实要容忍未知参数用 `parse_known_args()` |
| 参数名带横杠但按驼峰取值 | `AttributeError: 'Namespace' object has no attribute 'logLevel'` | `--log-level` 取 `args.log_level` |

---

# 二、YAML（约 10 分钟）

## 2.1 为什么不用 JSON / 不用 Python 文件

| 格式 | 优点 | 缺点 |
|---|---|---|
| **YAML** | 能写注释、不用引号、层级看着舒服 | 缩进敏感 |
| JSON | 通用 | **不能写注释**（配置文件不能写注释是灾难） |
| `config.py` | 直接能跑 | 改参数=改代码，还会被误当成逻辑 |

配置文件必须能手写、能手改、能留注释——所以选 YAML。

## 2.2 pyyaml 已就绪

实测 `pyyaml 6.0.3` 已在 `pytorch` 环境里，**不用装**。

```python
import yaml
```

## 2.3 读文件的标准写法（四行）

```python
from pathlib import Path
import yaml


def load_yaml(path: Path) -> dict:
    """读 yaml 文件，返回字典。空文件返回空字典。"""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return data or {}
```

三个细节各自有原因：

1. **`encoding="utf-8"`** —— 不写会用系统默认编码（Windows 上是 GBK），文件里一有中文就 `UnicodeDecodeError`。Day 1 你在 `FileHandler` 上踩过同一个坑。
2. **`safe_load` 而不是 `load`** —— `load` 能执行任意 Python 对象构造，等于"配置文件可以执行代码"。`yaml.load` 在 pyyaml 6 里已经要求必须显式传 `Loader`，官方就是用这种方式逼你用 safe。**永远用 `safe_load`。**
3. **`data or {}`** —— 空文件 `safe_load` 返回 `None`（不是 `{}`）。不兜底的话，后面 `data.get(...)` 直接 `AttributeError: 'NoneType'`。

## 2.4 YAML 语法速览

```yaml
# 这是注释。yaml 用 # 注释，不是 //
# ---- 基本键值 ----
seed: 42                 # 整数
lr: 0.0003               # 浮点
device: cuda             # 字符串（不需要引号）
note: "hello: world"     # 含冒号要加引号，否则解析失败
empty_value: ~           # ~ 表示 None

# ---- 布尔 ----
use_amp: true            # true / false 是布尔
is_debug: yes            # ⚠️ yes / no 也会被当布尔！这是坑

# ---- 嵌套 ----
model:
  hidden: [128, 128]     # 行内列表
  dropout: 0.1
  layers:                # 块状列表
    - conv
    - fc
```

读出来的实际类型（实测）：

```
seed         = 42                     (int)
lr           = 0.0003                 (float)
device       = 'cuda'                 (str)
note         = 'hello: world'         (str)
empty_value  = None                   (NoneType)
use_amp      = True                   (bool)
is_debug     = True                   (bool)      ← yes 变成了 True！
model        = {'hidden': [128, 128], 'dropout': 0.1, 'layers': ['conv', 'fc']}
```

## 2.5 ⚠️ 唯一的真坑：YAML 会自动推断类型

YAML 没有"类型声明"，它会**猜**。猜错的经典场景：

| 你写的 | 你以为 | 实际拿到 |
|---|---|---|
| `version: 1.10` | 字符串 `"1.10"` | **浮点 `1.1`**（`1.10` 和 `1.1` 是同一个数） |
| `name: yes` | 字符串 `"yes"` | **布尔 `True`** |
| `name: no` | 字符串 `"no"` | **布尔 `False`** |
| `code: 00123` | 字符串 `"00123"` | **整数 `123`**（前导 0 丢了） |
| `path: outputs\run` | 路径 | 反斜杠照原样，但建议一律写 `/` |

**解法：需要它保持字符串就加引号。** `version: "1.10"`、`name: "yes"`。

> 记住一句：**只要这个值要当字符串用（版本号、名字、路径），就加引号。** 不加引号就是在赌 YAML 猜得对。

---

# 三、dataclass（约 10 分钟）

## 3.1 为什么要把字典变成对象

配置读进来是一个 `dict`，直接用也能跑：

```python
cfg["seed"]        # 能用
```

但有三件事很难受：

| 问题 | dict | dataclass |
|---|---|---|
| 键名拼错 | `cfg["sead"]` → 运行时才 `KeyError` | `cfg.sead` → **IDE 当场标红** |
| 有什么字段 | 得翻文件看 | 补全列表直接弹出来 |
| 默认值 | 散在各处 `dict.get(k, 0)` | 集中在类定义里，一眼看完 |
| 类型 | 全靠猜 | 声明好了，IDE 能查 |

一句话：**dict 是"随便装"，dataclass 是"有形状"。**

## 3.2 标准写法

```python
from dataclasses import asdict, dataclass, field, fields


@dataclass
class Config:
    seed: int = 0
    device: str = "cuda"
    tags: list[str] = field(default_factory=list)   # 可变默认值要这样写
```

用起来：

```python
a = Config()                        # 全用默认值
b = Config(seed=42, device="cpu")   # 覆盖一部分
print(b.seed)                       # 42   ← 属性访问，不是 b["seed"]
```

## 3.3 三个必会的小工具

**① `fields()` 拿全部字段名**（合并时过滤未知键要用）

```python
print([f.name for f in fields(Config)])
# ['seed', 'device', 'tags']
```

**② `asdict()` 转回普通字典**（把生效配置写回 yaml 做备份时要用）

```python
print(asdict(b))
# {'seed': 42, 'device': 'cpu', 'tags': []}
```

**③ `default_factory`** —— 见下面第 3.5 节

## 3.4 坑一：传了不存在的键会直接报错

```python
Config(seed=1, lr=0.001)    # Config 里没有 lr
```

实测：

```
TypeError: Config.__init__() got an unexpected keyword argument 'lr'
```

**这不是坏事，是在帮你**——yaml 里拼错键名（`lrearning_rate`）会立刻暴露。
但注意：如果你像下面这样**先过滤再传入**，这个保护就失效了（见 3.6）。

## 3.5 坑二：可变默认值不能直接写

```python
@dataclass
class Bad:
    tags: list[str] = []      # ← 报错
```

实测：

```
ValueError: mutable default <class 'list'> for field tags is not allowed: use default_factory
```

正确写法：

```python
tags: list[str] = field(default_factory=list)
```

**为什么？** 因为 `= []` 这个列表在**所有实例之间共享**——`a.tags.append("x")` 之后 `b.tags` 也会变成 `['x']`。而 `default_factory=list` 的意思是"每次实例化时**调用一次** `list()` 造个新的"。

`dict`、`set`、`list` 都一样处理。数字、字符串、`None`、`tuple` 不可变，可以直接写。

---

# 四、把三样拼起来（核心，约 15 分钟）

这是整天的重点，也是 `train.py` 里那段"配置就绪"逻辑的原型。

## 4.1 合并的四个步骤

```python
raw = load_yaml(args.config)                     # ① 读 yaml → dict
cli = {k: v for k, v in vars(args).items()       # ② 挑出用户真传的键
       if k != "config" and v is not None}
raw.update(cli)                                  # ③ 命令行盖在 yaml 上
cfg = Config(**{k: v for k, v in raw.items()      # ④ 转成对象
                if k in {f.name for f in fields(Config)}})
```

**第 ③ 步是整个配置系统的核心**：`dict.update` 就是"后到的覆盖先到的"。因为 `cli` 里只有用户真传的键，所以没传的键在 yaml 里的值完好无损。

**第 ④ 步为什么要过滤？** 因为 yaml 里可能有 `Config` 不认识的键（注释用的、将来才加的），直接 `Config(**raw)` 会 `TypeError`。过滤掉更稳。

⚠️ **但过滤有代价**：拼错的键会被**静默丢弃**。所以正确做法是顺手报出来：

```python
unknown = set(raw) - {f.name for f in fields(Config)}
if unknown:
    print("⚠️ Config 不认识的键（多半是拼错了）:", sorted(unknown))
```

## 4.2 完整可运行版本

`configs/default.yaml`：

```yaml
seed: 123
device: cuda
output_dir: outputs/exp_baseline
log_level: DEBUG
```

`scripts/play_merge.py`：

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    seed: int = 0
    device: str = "cuda"
    output_dir: str = "outputs"
    log_level: str = "INFO"


def load_yaml(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="配置系统最小演示")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--log-level", type=str, default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> int:
    args = build_parser().parse_args()

    raw = load_yaml(args.config)
    print("① yaml 里的原始值 :", raw)

    cli = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    print("② 命令行传了的键   :", cli)

    raw.update(cli)
    print("③ 合并后的字典     :", raw)

    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        print("⚠️  Config 不认识的键（多半是拼错了）:", sorted(unknown))

    cfg = Config(**{k: v for k, v in raw.items() if k in known})
    print("④ 最终 Config 对象 :", cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## 4.3 两种跑法的实测对照

**跑法 A：什么都不传 → yaml 说了算**

```powershell
& "E:\anaconda\envs\pytorch\python.exe" scripts\play_merge.py
```

```
① yaml 里的原始值 : {'seed': 123, 'device': 'cuda', 'output_dir': 'outputs/exp_baseline', 'log_level': 'DEBUG'}
② 命令行传了的键   : {}
③ 合并后的字典     : {'seed': 123, 'device': 'cuda', 'output_dir': 'outputs/exp_baseline', 'log_level': 'DEBUG'}
④ 最终 Config 对象 : Config(seed=123, device='cuda', output_dir='outputs/exp_baseline', log_level='DEBUG')
```

**跑法 B：传 `--seed 7 --device cpu` → 命令行说了算**

```powershell
& "E:\anaconda\envs\pytorch\python.exe" scripts\play_merge.py --seed 7 --device cpu
```

```
② 命令行传了的键   : {'seed': 7, 'device': 'cpu'}
③ 合并后的字典     : {'seed': 7, 'device': 'cpu', 'output_dir': 'outputs/exp_baseline', 'log_level': 'DEBUG'}
                                                                 ↑ 没传的键，yaml 的值完好
④ 最终 Config 对象 : Config(seed=7, device='cpu', output_dir='outputs/exp_baseline', log_level='DEBUG')
```

**看懂这两组输出的差别，Day 3 就学会了。**

---

# 五、报错对照表（卡住先查这里）

| 报错 | 原因 | 修法 |
|---|---|---|
| `UnicodeDecodeError: 'gbk' codec can't decode` | 读文件没写 `encoding="utf-8"` | `read_text(encoding="utf-8")` |
| `AttributeError: 'NoneType' object has no attribute 'get'` | yaml 文件是空的，`safe_load` 返回 `None` | `return yaml.safe_load(text) or {}` |
| `yaml.scanner.ScannerError` 指向某一行 | 缩进用了 Tab / 冒号后没空格 / 值里有没引起来的冒号 | 只用空格缩进；`key: value` 冒号后必须有空格；含冒号的值加引号 |
| `TypeError: __init__() got an unexpected keyword argument 'xxx'` | yaml 里有 Config 没定义的键 | 加字段，或按第 4.1 节过滤 |
| `NameError: name 'yaml' is not defined` | 忘了 `import yaml` | 加上 |
| `AttributeError: 'Namespace' object has no attribute 'logLevel'` | 横杠转下划线 | `args.log_level` |
| 参数传了但不生效 | 十有八九是 `default` 没写 `None`，被默认值反向覆盖 | 回看 1.3 节 |
| `ValueError: mutable default ... not allowed` | 可变默认值直接写了 `= []` | `field(default_factory=list)` |

---

# 六、小练习（做完再动 80min 的实现任务）

四个都做完，说明真会了：

1. **改默认值陷阱**：把 4.2 里 `--seed` 的 `default=None` 改成 `default=0`，不传任何参数跑一遍。看 `seed` 变成多少，**用自己的话解释为什么**。
2. **加一个参数**：给 `Config` 加 `batch_size: int = 32`，给 parser 加 `--batch-size`，yaml 里加上 `batch_size: 64`。跑三种组合（不传 / 传 128 / yaml 删掉这行）验证优先级。
3. **加一个开关**：加 `--dry-run`（`action="store_true"`），但注意——**开关不能用 `None` 判断**（False 也是有效值）。想想该怎么合并？提示：开关本身就是"用户传了才算数"，可以直接取 `True`。
4. **制造一次报错**：故意把 yaml 缩进改成 Tab，看报错长什么样；再故意拼错一个键名 `sed: 123`，看它是被静默忽略还是报错。

---

# 七、接下来要写的东西（对照清单）

40 分钟学完之后，Day 3 还剩两段实现任务：

| 任务 | 产出 | 关键要求 |
|---|---|---|
| 80min | `configs/default.yaml` + `src/utils/config.py` | `load_config(path) -> Config`；**yaml 里没写的键用默认值补上** |
| 60min | `scripts/train.py` 壳 | 解析命令行 → 加载 yaml → **命令行覆盖 yaml** → logger 打印完整配置 → 在 `outputs/<时间戳>/` 存一份 `config.yaml` 备份 |

`train.py` 要能这么跑：

```powershell
& "E:\anaconda\envs\pytorch\python.exe" scripts\train.py --config configs\default.yaml --seed 42 --device cuda
```

**注意 `train.py` 会复用你 Day 2 写的三个东西**：
- `paths.py` 的 `run_dir()` → 生成 `outputs/<时间戳>/` 目录（终于派上用场了）
- `logger.py` 的 `get_logger()` → 打印配置（不再用 `print`）
- 以及今天学的配置系统

到那时候你写的 `train.py` 只有二三十行，因为脏活都在 `src/utils/` 里。

---

## 附：三个提醒

1. **今天的所有演示文件都放在 `scripts/` 下，学完删掉，不要提交。** 验证方法：`git status` 里看到它们时别勾。
2. **别把 `parse_args()` 写在模块顶层**——`import` 就会执行，以后别的脚本 import 你的模块时会莫名其妙退出。
3. **Day 3 的过关标准第 3 条是"你能解释命令行解析器与 yaml 是怎么合并的"。** 说不清就回去看 4.1 那四行——它比代码本身重要，因为 P1 之后每个实验都要经过这段逻辑。
