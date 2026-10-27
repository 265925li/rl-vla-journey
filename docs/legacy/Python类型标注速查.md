# Python 类型标注速查（配合 typing 官方文档使用）

> 文档地址：https://docs.python.org/zh-cn/3/library/typing.html
> 你当前环境：Python **3.10.20**（pytorch 环境），项目里每个文件第一行都有 `from __future__ import annotations`
> 最后更新：2026-10-16

---

## 一、先解决"这份文档是不是简要"这个问题

**不是。它是一本完整参考手册（Reference），准确说是"词典"。**

它在开头的自我定位原话是：

> "本模块提供了对类型提示的**运行时支持**。"
> "Python 运行时不强制要求函数与变量类型标注。它们可被类型检查器、IDE、语法检查器等第三方工具使用。"

翻成大白话：**这东西是给你"查"的，不是给你"读"的。** 它是按字母顺序排列的类型构造清单，就像字典不会教你怎么写作文。

所以：**不要从头读到尾**，那是无效努力。正确用法是遇到具体问题时 Ctrl+F。

### 页面上真正值得读的部分（不到全文 10%）

| 位置 | 内容 | 要不要读 |
|---|---|---|
| 开头引言 | 类型系统定位 | 读，2 分钟 |
| 类型别名 / NewType / 标注可调用对象 / 泛型 / 标注元组 / 标注生成器 | 散文讲解，有例子 | **只读你需要的那一节** |
| 「模块内容」之后 | 所有类型构造的逐条 API 说明 | 当词典查，别通读 |

### 页面自己推荐的"入门替代品"（比这页更适合从头学）

页面底部的"参见"里指向了三个更该先看的地方：

1. **类型系统速查卡（Cheat Sheet）** — `https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html`
   一页纸看完 90% 常用语法，**推荐你先看这个**。
2. **Python 静态类型社区文档** — `https://typing.python.org/en/latest/`
   不限特定检查器的系统讲解，有最佳实践。
3. **typing_extensions 包** — 新版类型特性对老 Python 的向下移植（你 3.10 以后会用到，见第四节）。

---

## 二、你现在真正需要用到的，只有这 4 个写法

**重要结论：你 Day 1 和 Day 2 写的代码，一行 `from typing import ...` 都不需要。**

因为用到的全是**内置类型**，Python 本来就认识，不用 import 任何东西。

| 写法 | 在项目里的真实例子 | 说明 |
|---|---|---|
| `变量/参数: 类型` | `ROOT: Path = ...`（paths.py:40）<br>`def ensure_dir(p: Path) -> Path:`（paths.py:53） | 冒号后面是"这个变量装什么类型"，箭头后面是"这个函数返回什么类型" |
| 内置类型直接写 | `prefix: str`、`level: int = logging.INFO`、`-> logging.Logger`、`-> Path` | `str` `int` `float` `bool` `list` `dict` `Path` `logging.Logger` 都是现成的 |
| `X \| None` | `base: Path \| None = None`（paths.py:73）<br>`log_file: Path \| None = None`（logger.py:45） | "要么是 Path，要么是 None"。**这是最常写的一种**：参数可以传 None 时就得这么标 |
| 返回类型 `->` | `def get_logger(...) -> logging.Logger:`（logger.py:47） | 函数要返回东西，就顺手写 `->`，代码读完不用猜 |

补充两条你的环境相关的：

- **`X | None` 在 Python 3.10 是原生支持的**（PEP 604，3.10 新增），不需要 import。
  老的写法是 `Optional[X]`，得 `from typing import Optional`。**你现在两种都行，但新代码统一用 `X | None` 更短。**
- **`from __future__ import annotations` 是干嘛的**：它让所有标注变成"字符串"，运行时根本不去求值。
  好处是：① 启动更快 ② 可以放心用新语法（哪怕将来降级到 3.8 也不会报错）③ 解决"类还没定义就要引用它"的循环问题。
  代价：**运行时拿不到真实类型对象**，要用 `typing.get_type_hints()` 才能求值出来。你现在不用管这个。

---

## 三、往后会陆续用到的（先认个脸，别背）

看到能想起来"哦有这么个东西"就够了，真用到再回来查。

| 构造 | 长什么样 | 你会第一次用到它的场景 |
|---|---|---|
| `Any` | `def f(x: Any) -> Any` | 接 gym 的返回值——`env.reset()` 返回啥类型它自己也说不清，只能标 Any |
| `Literal` | `Literal["dqn", "ppo"]` | 算法名字这种"只能取这几个固定值"的参数，写错拼写 IDE 当场报错 |
| `Callable` | `Callable[[Tensor], Tensor]` | 把函数当参数传时（比如传一个 loss 函数给训练循环） |
| `Sequence` | `Sequence[float]` | "list 和 tuple 我都收"时用它，比写 `list` 更宽容 |
| `TypeVar` / `Generic` | `class Buffer(Generic[T])` | 写经验回放池这种"装什么都行"的通用容器时 |
| `Protocol` | `class Env(Protocol): ...` | 给"环境"定一个接口：只要有 `reset()`/`step()` 就算，不要求继承 |
| `TypedDict` | `class Config(TypedDict): lr: float` | 描述 config 字典里到底有哪些字段时（替代散装的 `dict`） |
| `Optional[X]` | `Optional[Path]` | `X \| None` 的老写法，看到要认识 |
| `cast` | `cast(Tensor, x)` | 类型检查器误报时，手动告诉它"相信我，就是这个类型" |
| `Self` | `-> Self` | 返回自己的方法（**3.11 才有，你 3.10 要用得装 typing_extensions**） |

---

## 四、版本对照（专治"为什么这段代码在我这儿跑不了"）

你环境是 **3.10**，所以：

| 写法 | 你（3.10）能用吗 |
|---|---|
| `list[int]`、`dict[str, float]` 内置泛型 | 能（3.9 起） |
| `X \| Y`、`X \| None` | 能（3.10 起） |
| `match` 语句 | 能（3.10 起） |
| `Self`、`LiteralString`、`Never`、`TypeVarTuple` | **不能**（3.11 才有）→ 装 `typing_extensions` 后 `from typing_extensions import Self` |
| `class Foo[T]:` 新泛型语法、`type Alias = ...` | **不能**（3.12 才有，PEP 695） |
| `TypeIs`、`ReadOnly` | **不能**（3.13 才有） |

> 遇到"教程里的写法你跑不通"，先查版本，**大概率不是你的错**。
> 解决办法统一是：`pip install typing_extensions`，然后从 `typing_extensions` 里 import。

---

## 五、三条最容易踩的坑

1. **`typing.List` / `typing.Dict` 已经过时了。**
   旧教程里的 `List[int]` 现在写 `list[int]`，`Dict[str, int]` 写 `dict[str, int]`。文档里 `typing.List` 一节写着"自 3.9 起弃用"。

2. **类型标注不产生任何运行时效果。**
   `def f(x: int) -> str:` 你传个字符串进去，Python **一句话都不会说**，照样跑。
   只有装了类型检查器（mypy/pyright）或在 PyCharm 里，它才会给你画黄线。
   **所以：别指望标注能"防错"，它是给人和工具看的文档。**

3. **PyCharm 的黄色波浪线不是报错。**
   只是"建议"，标红才是真语法错误。**不要把波浪线当成"程序坏了"。**

---

## 六、给你的行动建议

- **现在（Day 2–Day 30）**：什么都不用看。继续写 `X | None` 这种，遇到不确定的查第二节那张表。
- **想系统学类型标注时**：读 mypy 的 Cheat Sheet（一页纸），不要读这份参考手册。
- **卡住时**：Ctrl+F 关键词 → 只读那一小节。

看文档的正确姿势不是"读完"，是"知道它在这儿、知道该翻哪一页"。
