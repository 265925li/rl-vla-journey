# Day 2 填空指南 · Python 工程基本功

> 目标：把 Day 1 那个"什么都塞在一个文件里"的 `env_info.py`，拆成**可以反复使用**的模块。
> 骨架我已经搭好并实测通过，你只需要填 **6 处**。

---

## 一、今天的文件长什么样

```
rl-vla-journey/
├── src/
│   ├── __init__.py          ✅ 已写好（让 src 成为正规包）
│   └── utils/
│       ├── __init__.py      ✅ 已写好
│       ├── paths.py         ⬜ 你填 2 处
│       └── logger.py        ⬜ 你填 2 处
└── scripts/
    └── env_info.py          ⬜ 你填 2 处（另有 1 处改造提示）
```

### 你 Day 1 写的代码去哪了

没有浪费，全都**搬进了新模块**，这是今天最核心的一步：

| Day 1 里写在哪 | 现在搬到了哪 |
|---|---|
| `ROOT / OUTPUTS / REPORT` 三行路径推导 | `src/utils/paths.py` 顶部 |
| `get_logger()` 里的 StreamHandler、FileHandler | `src/utils/logger.py` |
| `collect_env()` 的 torch 部分 | 留在 `scripts/env_info.py` |
| TODO 6 的 `"不可用"` 占位分支 | 留在 `scripts/env_info.py` |

区别是：**现在这些代码不再只属于 `env_info.py`，而是属于整个项目。**
以后你写 `train.py`、`eval.py`、`play.py`，全都 `import` 它们，不用复制粘贴。

这就是"工程化"最朴素的定义：**同一个逻辑只写一次。**

---

## 二、TODO 长什么样，怎么改

每一处都长这样：

```python
todo = False
assert todo, "TODO n 未完成：..."
```

**改法两步**（和 Day 1 同一个套路）：

1. 把 `todo = False` 那一行，换成注释里让你写的代码
2. 把紧跟的 `assert` 整行**删掉**（不是注释掉）

不做第 2 步不会报错，但那是垃圾代码——你 Day 1 已经踩过一次了。

> 搜 `TODO` 可以定位全部位置（PyCharm 里按 `Ctrl + F`）。

---

## 三、别看全部，跟着报错走

**跑一次只会报一个，而且是按依赖顺序报的。**（这是我在临时副本上实测出来的顺序）

| 第几次跑 | 报什么 | 在哪个文件 |
|---|---|---|
| 1 | `TODO 1 未完成：用 p.mkdir() 创建目录` | `src/utils/paths.py` |
| 2 | `TODO 1 未完成：建一个输出到控制台的 handler` | `src/utils/logger.py` |
| 3 | `TODO 2 未完成：建一个写入 log_file 的 handler` | `src/utils/logger.py` |
| 4 | `TODO 1 未完成：把 sys.executable 加进报告` | `scripts/env_info.py` |
| 5 | **成功**（打印报告 + `报告已写入: ...`） | — |
| 6 | `TODO 2 未完成：补上「怎么修」的提示` | 只在没装 torch 的环境里触发，见第五节 |

**所以别一次改 6 处。** 改一处 → 跑一次 → 看它报下一个 → 再改。
如果一次改完 6 处然后报错，你就不知道是哪一个写错了。

---

## 四、6 处分别要什么（提示，不是答案）

### `src/utils/paths.py`

| TODO | 要做什么 | 提示 |
|---|---|---|
| 1 | 创建目录 | `p.mkdir(parents=True, exist_ok=True)`——两个参数各解决一个问题，注释里写了 |
| 2 | 调用 `ensure_dir` 创建刚拼出来的 `d` | `d = ensure_dir(d)`。这行是"复用"第一次真正发生的地方 |

✅ **可以自测**：这个文件底部有 `if __name__ == "__main__":` 自测段。用这条跑：

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
& "E:\anaconda\envs\pytorch\python.exe" -m src.utils.paths
```

应该看到项目根路径 + 一个新建的 `outputs/2026-10-16_xxxxxx_selftest/`。

---

### `src/utils/logger.py`

| TODO | 要做什么 | 提示 |
|---|---|---|
| 1 | 建控制台 handler | `console = logging.StreamHandler()` |
| 2 | 建文件 handler | `fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")` |

**这两处你在 Day 1 已经填对过一次**，今天是换个位置再写一遍。重点不在"会不会"，在于**把它们变成肌肉记忆**。

TODO 2 的两个参数想一下（你在 Day 1 因为 `mode='a'` 那个默认值踩过坑）：
- `mode="w"`——每次覆盖。默认是 `"a"`，不改的话跑两次文件里就有两份报告
- `encoding="utf-8"`——不指定中文会乱码

---

### `scripts/env_info.py`

| TODO | 要做什么 | 提示 |
|---|---|---|
| 1 | 把当前解释器路径加进报告 | `rows.append(("解释器", sys.executable))` |
| 2 | 给"没有 torch"的情况写建议 | 补 2 行 `logger.error(...)`，说明**怎么修**，见第五节 |
| 3 | 给异常加中文提示 | 见第六节，**这一处不用 assert 导航，需要主动触发** |

**TODO 1 为什么可能是整份报告最有价值的一行**：
你以后会遇到无数种"装了什么却不生效"的怪现象，99% 的原因是你敲的 python 和你想的不是同一个。把解释器路径打出来，一眼就能确认。

---

## 五、TODO 2 怎么触发（平时跑不出来）

TODO 2 在 `if not HAS_TORCH:` 分支里——**只有解释器里根本没装 torch 时才走得到**。

你机器上正好有三个真实可用的测试环境：

| 环境 | torch 情况 | 会走哪个分支 |
|---|---|---|
| `E:\anaconda\envs\pytorch` | 2.13.0+cu126 | 正常分支 |
| `E:\anaconda\envs\py311` | 2.3.1+cpu（**装了，只是没有 GPU**） | 正常分支（`import` 成功！） |
| `C:\Users\...\Programs\Python\Python312` | **没装 torch** | ← **这个才会触发 TODO 2** |

⚠️ 注意中间那个坑：`py311` 是 **CPU 版 torch**，`import torch` 是**成功**的，只是 `cuda.is_available()` 为 False。所以它不会触发 TODO 2——**"没有 GPU" 和 "没装 torch" 是两回事**。

触发命令（填完前 4 处之后跑）：

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
& "C:\Users\不会挽留\AppData\Local\Programs\Python\Python312\python.exe" scripts\env_info.py
```

现在会报 `TODO 2 未完成`。填完之后，应该看到你自己写的中文提示，退出码 1。

**这正好又用上了你 Day 1 学的那条**：错误处理路径要主动触发才能测。

---

## 六、TODO 3 怎么触发（平时跑不出来，也没占位断言）

TODO 3 在 `if __name__ == "__main__":` 的 `except` 里——**只有 `main()` 真的抛异常才会执行**。

所以它没有 `assert` 占位（有也没用，永远轮不到）。它的任务是：把默认的"英文 traceback 糊脸"改成"中文提示"。

**触发方法**：

1. 打开 `scripts/env_info.py`，找到第 49 行附近：

   ```python
   REPORT = OUTPUTS / "env_report.txt"
   ```

2. 临时改成：

   ```python
   REPORT = Path("Z:/no_such_disk/env_report.txt")
   ```

3. 跑一次：

   ```powershell
   & "E:\anaconda\envs\pytorch\python.exe" scripts\env_info.py
   ```

4. **改之前**你会看到：一大段英文 `FileNotFoundError` / `OSError` traceback
5. **改成 try/except + print 之后**：应该看到你自己写的中文提示，退出码 1

6. **验完把 `REPORT` 改回来**（不改回来的话后面所有输出都会失败）

---

## 七、怎么跑

### PyCharm 里（推荐）

打开 `scripts/env_info.py` → 按 `Ctrl + Shift + F10`。

前提：右下角解释器显示 `Python 3.10 (pytorch)`。如果显示的不是，先按 Day 1 那套改成 `E:\anaconda\envs\pytorch\python.exe`。

### 终端里

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
& "E:\anaconda\envs\pytorch\python.exe" scripts\env_info.py
```

⚠️ 再强调一次：**命令敲在终端里，不要粘进 .py 文件**（Day 1 踩过）。
判断方法：敲完回车，如果什么都没发生、只是换了一行，那你多半在编辑器里。

---

## 八、过关标准（清单里的三条）

- [ ] 两个模块里每个函数都有类型注解，没有任何裸 `print`
      —— 注解我已经写好了，你只要别删；`print` 只在 `paths.py` 的自测段里
- [ ] 故意写错一个路径，程序报出**可读的错误**而不是英文 traceback 糊脸
      —— 就是第六节那个 `Z:/` 测试
- [ ] 今天 ≥2 次 commit

### 提交

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
git add -A
git commit -m "refactor: extract logger and path utils into src/utils"
```

建议中间再提交一次（比如 paths + logger 填完时），凑够"≥2 次"：
第一次 `feat: add logger and path utils`，第二次 `refactor: env_info uses shared utils`。

---

## 九、卡住了怎么办

**某一处卡超过 10 分钟，直接把那几行 + 完整报错贴过来。** 不用自己耗。

如果 6 处里有 4 处以上填不出来，直接说"给完整版"，我把答案发你**逐行讲一遍**，然后我们再一起删掉、你重写一遍。这不是作弊——先看一遍正确的形状，再自己复现，是正常的学编程方式。

---

## 十、今天的验收为什么是"6 个小填空"而不是"从零写三个文件"

因为今天的**真正内容不是这 6 行代码**，是这三件事：

1. **模块化**：一个逻辑写一次，多个地方复用（`ensure_dir` 被 `run_dir` 复用，就是最小样本）
2. **依赖方向**：`logger` 和 `paths` 只依赖标准库，谁都不依赖谁——所以它们在任何环境里都能跑
3. **面向失败设计**：`HAS_TORCH`、`TODO 3` 的存在，都是因为"工具必须比它要诊断的东西活得更久"

这 6 行是外壳，那三件事才是你今天真正带走的东西。
