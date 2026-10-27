# Day 4 学习文档 · Git 完整流程 + 上 GitHub

> 目标：把"每天提交"变成习惯，让作品第一次出现在公网上，并且**亲手解决一次冲突**。
> 对应清单：`P0-逐日执行清单.md` → Day 4（60min 学 / 50min 练习 / 60min 推 GitHub / 30min 补文件）
> 本机前置（已实测）：Git **2.55.0** ✓ · SSH **OpenSSH 10.3** ✓ · **无 `~/.ssh`**（要新建） · **无 `gh` CLI**（用网页建仓库） · 全局身份 `RL <265925li@gmail.com>` ✓
> 最后更新：2026-10-17

⚠️ **本文的 Git 输出全部来自真实操作**（在临时仓库里把整条冲突流程跑通后才写下来）。

---

# 〇、先立一个心智模型：三个区

**这是今天唯一必须记住的东西。** 90% 的 Git 困惑，都是因为不知道命令动的是哪个区。

```
   工作区                暂存区                仓库
  (你编辑的文件)      (准备提交的内容)      (已提交的历史)
       │                   │                   │
       │   git add ────────>                   │
       │                   │  git commit ──────>
       │                   │
       │ <──── git restore --staged            │
       │ <──────────────── git restore ────────┘(从 HEAD 取内容覆盖)
```

| 区 | 也叫什么 | 里面是什么 |
|---|---|---|
| **工作区** | working tree / 工作目录 | 你现在用 PyCharm 打开、正在改的文件 |
| **暂存区** | staging area / index | 你 `git add` 过、准备这次提交的内容 |
| **仓库** | repository / HEAD / 历史 | 已经 `git commit` 的记录，永久留着 |

**一句话串起来**：`git add` = 工作区 → 暂存区；`git commit` = 暂存区 → 仓库。

> 昨天那个事故现在就能解释了：`play_argparse.py` 状态是 `AM`，意思是"**已经 add 进暂存区**（A），之后**工作区又改了**（M）"。
> 而 Amend 是"**改掉仓库里上一次提交**"——它动的是第三个区，所以历史被重写了。

---

# 一、status：先看清楚自己在哪

```powershell
git status          # 详细版
git status -s       # 简短版（-s = short），日常用这个
```

`-s` 的输出是**两列**，这是最容易看错的地方：

```
 M scripts/train.py
M  src/utils/config.py
AM scripts/play_argparse.py
?? scripts/play_merge.py
 D src/utils/.gitkeep
```

**第一列 = 暂存区 vs 仓库；第二列 = 工作区 vs 暂存区。**

| 符号 | 含义 | 该怎么做 |
|---|---|---|
| `??` | 未跟踪（新文件，Git 不认识） | 要提交就 `git add` |
| ` M` | **工作区**改了，还没 add | `git add` 或 `git restore` 丢掉 |
| `M ` | 已 add 进**暂存区** | 直接 `commit` |
| `MM` | add 过之后又改了 | 再 `git add` 一次 |
| `AM` | 新文件 add 过，之后又改了 | 再 `git add` 一次（**昨天就是这个**） |
| ` D` | 工作区删了文件，未 add | `git add` 记录删除 |
| `D ` | 删除已进暂存区 | 直接 `commit` |

**记忆法：左列是"已装箱"，右列是"还没装箱"。**

---

# 二、diff：看"改了什么"

`git diff` 有三个变体，区别恰恰就是三个区：

| 命令 | 比较的是 | 用来看 |
|---|---|---|
| `git diff` | 工作区 ↔ 暂存区 | **还没 add** 的改动 |
| `git diff --staged` | 暂存区 ↔ 仓库 | **下次提交会带上的**改动（提交前最后检查用） |
| `git diff HEAD` | 工作区 ↔ 仓库 | 两者加起来，全部改动 |

```powershell
git diff                 # 改了什么
git diff --staged        # 我马上要提交什么 ← 提交前必看
git diff src/utils/config.py    # 只看某个文件
```

`git diff --staged` 是我最推荐养成习惯的一条 —— **提交前扫一眼，能挡住 90% 的"把调试代码提交上去了"**。

---

# 三、log：看历史

```powershell
git log --oneline                  # 一行一条，日常用
git log --oneline --graph --all    # 带分支图（今天会用到）
git show 04342bf                   # 某次提交改了什么
git log -p src/utils/config.py     # 某个文件的完整改动史
```

`--graph --all` 的好处：**能看见分支从哪儿分出去、又合回来**。今天练冲突时会用到。

---

# 四、restore：撤销（分两种情况，别搞混）

| 场景 | 命令 | 效果 |
|---|---|---|
| 改了文件，**还没 add**，想丢掉 | `git restore 文件` | 工作区回到暂存区的内容 ⚠️ **改动永久丢失** |
| 已经 add，想**退回未暂存** | `git restore --staged 文件` | 暂存区退回仓库的内容，**工作区改动保留** ✅ 安全 |
| 想恢复成历史上某个版本 | `git restore --source=HEAD~1 文件` | 从上次提交取内容 |

```powershell
git restore --staged scripts/play_argparse.py   # 昨天用的就是这条
git restore scripts/train.py                    # ⚠️ 危险：未提交的改动直接没了
```

**⚠️ 第一条是今天唯一会让你丢代码的命令。** 它没有确认提示，执行即生效。

丢了的救法：`git reflog` 找回提交（但**没提交过的**内容找不回来）。

---

# 五、branch：分支只是一个指针

**先破除一个误解**：分支**不是**"把代码复制一份"。它只是一个**指向某次提交的指针**，几十字节。

所以建分支是**瞬间完成的**，跟仓库大小无关。

```powershell
git branch                       # 看有哪些分支（* 是当前）
git switch -c exp/logger-format  # 新建并切过去（-c = create）
git switch main                  # 切回 main
git branch -d exp/logger-format  # 删分支（已合并才行）
```

> 老教程里的 `git checkout -b xxx` 也能用，但 `checkout` 一词在 Git 2.23 之后被拆成了 `switch`（切分支）和 `restore`（恢复文件）——**同一个命令干两件事太容易搞错**。用新命令。

**当前在哪个分支**：看 `git status` 第一行，或 `git branch` 的 `*`。**做任何操作前先确认这句**。

---

# 六、merge 与冲突（今天的重点）

## 6.1 merge 的两种情形

```powershell
git merge exp/logger-format      # 把 exp 分支合进当前分支
```

| 情形 | 什么时候发生 | 结果 |
|---|---|---|
| **快进（fast-forward）** | 当前分支**没有**新提交 | 指针直接往前挪，不产生新提交 |
| **三方合并** | 两边**都**有新提交 | 产生一个 merge 提交 |
| **冲突** | 两边改了**同一处** | 停下，让你手动决定 |

## 6.2 冲突是怎么产生的

**同一文件的同一区域，被两边都改了。** Git 不敢替你选，于是把两边内容都摆进文件，让你决定。

只改不同文件、或同一文件的不同位置，Git 能自动合并，不会冲突。

## 6.3 冲突现场长什么样（实测）

在临时仓库里制造冲突，`git merge` 的真实输出：

```
Auto-merging logger.py
CONFLICT (content): Merge conflict in logger.py
Automatic merge failed; fix conflicts and then commit the result.
```

打开文件，会看到 Git 塞进去的标记：

```python
<<<<<<< HEAD
CONSOLE_FORMAT = "[主线版] %(asctime)s | %(message)s"
=======
CONSOLE_FORMAT = "[分支版] %(levelname)-7s | %(message)s"
>>>>>>> exp/logger-format
```

**怎么读这三个标记：**

| 标记 | 含义 |
|---|---|
| `<<<<<<< HEAD` | 从这里开始，是**当前分支**（你切过去的那个）的内容 |
| `=======` | 分界线 |
| `>>>>>>> exp/logger-format` | 到这里结束，是**要合进来的分支**的内容 |

`git status` 这时会说：

```
On branch main
You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)

Unmerged paths:
  (use "git add <file>..." to mark resolution)
	both modified:   logger.py
```

## 6.4 解决冲突：三步，没有第四步

**① 手动编辑文件** —— 把三个标记删掉，留下你想要的内容。可以只留一边，也可以两边合并：

```python
CONSOLE_FORMAT = "[主线版+分支版] %(asctime)s | %(levelname)-7s | %(message)s"
```

**② `git add 文件`** —— 这是告诉 Git"这个文件我处理完了"。**不 add 它就一直在冲突状态。**

**③ `git commit`** —— 完成合并。Git 已经准备好了提交信息（形如 `Merge branch 'exp/logger-format'`），直接保存即可。

实测完成后的历史：

```
*   cf3973e merge: keep both timestamp and level
|\
| * 2740862 exp: shorten console format
* | 00bb36a refactor: console format
|/
* 6e1ce45 init
```

那个分叉再合回来的形状，就是 merge 的可视化。

## 6.5 逃生舱

**搞砸了想撤销整个合并**：

```powershell
git merge --abort
```

回到 merge 之前的状态，工作区也恢复。**冲突期间不敢乱来时，先 abort 退回安全状态。**

---

# 七、reset：会改写历史，慎用

```powershell
git reset --soft HEAD~1     # 撤销上次提交，改动留在【暂存区】
git reset HEAD~1            # 撤销上次提交，改动留在【工作区】（--mixed，默认）
git reset --hard HEAD~1     # 撤销上次提交，改动【全丢】
```

对照三个区就清楚了：

| 模式 | 仓库 | 暂存区 | 工作区 |
|---|---|---|---|
| `--soft` | 退 | 保留 | 保留 |
| `--mixed`（默认） | 退 | 清空 | 保留 |
| `--hard` | 退 | 清空 | **清空** |

**⚠️ `--hard` 会丢代码，不用确认、直接执行。**

**救命命令**：

```powershell
git reflog
```

它记录 HEAD 的每一次移动（包括 reset、commit、amend）。昨天 `180cfa6` 被 amend 掉之后能救回来，就是靠它。

**`--hard` 丢掉的**：如果改动从没提交过，reflog 也救不了，因为 reflog 只记提交。

---

# 八、上 GitHub

## 8.1 先建 SSH key（本机还没有 `~/.ssh`）

```powershell
ssh-keygen -t ed25519 -C "265925li@gmail.com"
```

三个提示怎么答：

1. `Enter file in which to save the key` → **直接回车**（用默认 `C:\Users\不会挽留\.ssh\id_ed25519`）
2. `Enter passphrase` → **直接回车**（不设密码，本机自用够安全；设了每次 push 都要输）
3. `Enter same passphrase again` → 再回车

然后**打印公钥**（公钥可以公开，私钥绝不能给任何人）：

```powershell
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

复制输出那一整行（以 `ssh-ed25519` 开头、以邮箱结尾）。

## 8.2 把公钥加到 GitHub

网页操作：

1. 右上角头像 → **Settings**
2. 左侧 **SSH and GPG keys** → **New SSH key**
3. Title 随便写（如 `HUAWEI-FRD-WX9`），Key 粘贴刚才那一整行
4. **Add SSH key**

## 8.3 测试连通

```powershell
ssh -T git@github.com
```

第一次会问 `Are you sure you want to continue connecting (yes/no)?` → 输 **`yes`** 回车（这是在确认服务器指纹，不是权限问题）。

成功的样子：

```
Hi <你的用户名>! You've successfully authenticated, but GitHub does not provide shell access.
```

看到 `successfully authenticated` 就成了。`does not provide shell access` 这句是正常的，不用管。

## 8.3.5 ⚠️ 本机专属坑：中文用户名 + Git 自带 ssh

**这台机器的用户名是中文（`不会挽留`），而 Git 自带的那个 `ssh.exe` 处理不了这种路径。** 实测现象：

```
$ ssh -T git@github.com
Host key verification failed.
```

加 `-v` 才看得出来真正原因：

```
debug1: load_hostkeys: fopen /c/Users/\262\273\273\341\315\354\301\364/.ssh/known_hosts: No such file or directory
```

**`\262\273...` 就是"不会挽留"三个字的 GBK 字节被当成乱码打印出来了。** 于是 ssh 去找一个根本不存在的路径，找不到 `known_hosts`，就报"主机密钥验证失败"。

**更糟的是它连私钥也读不到**（同样的原因），所以 `git push` 会一直 `Permission denied`。

### 解法：让 git 用 Windows 自带的 OpenSSH

运行一次即可，只影响 git，不改任何系统设置：

```powershell
git config --global core.sshCommand "C:/Windows/System32/OpenSSH/ssh.exe"
```

Windows 自带的 OpenSSH 走 Unicode API，中文路径没问题。设完验证：

```powershell
git config --global --get core.sshCommand
# 应输出 C:/Windows/System32/OpenSSH/ssh.exe
```

**注意**：以后凡是手动测试连通性，也要用这个程序，不要用裸 `ssh`：

```powershell
& "C:\Windows\System32\OpenSSH\ssh.exe" -T git@github.com
```

> 判定标准：
> - `Host key verification failed` → **本机 ssh 的问题**（就是这条）
> - `Permission denied (publickey)` → ssh 正常，只是**公钥还没加到 GitHub**
> - `Hi <用户名>! You've successfully authenticated` → 全通了

### 顺手做掉的一件事：host key 已预置

已经用 GitHub 官方公布的指纹核对过，并把 `github.com` 的主机密钥写进了 `C:\Users\不会挽留\.ssh\known_hosts`：

```
256 SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU github.com (ED25519)
```

和 GitHub 官网公布的一致。**所以你不会再遇到 `Are you sure you want to continue connecting (yes/no)?` 那个提示**，直接就能连。

> 为什么要核对：`ssh-keyscan` 是"从网络上取对方公钥"，理论上可能被中间人替换。核对官方指纹是唯一可靠的确认方式。这一步以后连陌生服务器时也该做。

## 8.4 建仓库并推送

网页建仓库：

1. 右上角 **+** → **New repository**
2. Repository name：`rl-vla-journey`
3. 选 **Public**
4. **⚠️ 千万不要勾 "Add a README file" / "Add .gitignore" / "Choose a license"**
   —— 勾了会生成一个初始提交，和你本地的历史冲突，推的时候要额外处理
5. Create repository

然后本地三条命令（**注意把 `<你的用户名>` 换成 GitHub 用户名**）：

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
git remote add origin git@github.com:<你的用户名>/rl-vla-journey.git
git push -u origin main
```

- `remote add origin` = 给这个远程地址起名叫 `origin`（约定俗成的名字）
- `push -u` = 推上去，并**记住**"本地 main 对应 origin/main"，以后直接 `git push` 就行

**验证**：`git remote -v` 能看到两行（fetch / push），浏览器打开 `https://github.com/<你的用户名>/rl-vla-journey` 能看到代码。

---

# 九、requirements.txt 怎么筛

清单要求"留下你能说清用途的"。别直接 `pip freeze > requirements.txt` 就完事——那会带出上百个间接依赖。

```powershell
& "E:\anaconda\envs\pytorch\python.exe" -m pip freeze > requirements_all.txt
```

然后**手写**一个精简版（新建 `requirements.txt`），只写你项目真正直接 import 的：

```
# 直接依赖 —— 每一条都能说清为什么需要它
torch>=2.0
numpy>=1.24
pyyaml>=6.0

# 日志与可视化
tensorboard>=2.14

# RL 环境（P1 起才真正用上）
gymnasium>=1.0
```

**为什么要精简**：`pip freeze` 会把 `torch` 依赖的 `sympy`、`networkx`、`filelock` 等几十个包全列出来。它们由 pip 自动解析，不需要你钉版本。**钉死了反而害自己**——将来某个包升级，你的文件就把版本锁死了。

**判断标准**：能回答"这个包是我 import 的，还是别人 import 的？"只留前者。

---

# 十、今天的实操清单（按顺序，别跳）

## 阶段 A：学习（60min）

读 `status / diff / log / restore / branch / merge / reset` 七节。**重点吃透第 〇 节的三区图**。

## 阶段 B：刻意练习冲突（50min）—— **今天最重要的一段**

真实项目里操作，别用临时仓库：

```powershell
cd "E:\work Buddy\强化学习\rl-vla-journey"
```

1. **确认当前在 main 且干净**
   ```powershell
   git status
   ```
   应该是 `working tree clean`。

2. **建分支并改 logger 格式**
   ```powershell
   git switch -c exp/logger-format
   ```
   打开 `src/utils/logger.py`，把 `CONSOLE_FORMAT` 这一行改成（比如）去掉日期：
   ```python
   CONSOLE_FORMAT = "[分支版] %(levelname)-7s | %(message)s"
   ```
   跑一次确认没坏，然后提交：
   ```powershell
   git add src/utils/logger.py
   git commit -m "exp: shorten console format"
   ```

3. **切回 main，改同一行**
   ```powershell
   git switch main
   ```
   把**同一行**改成另一种写法：
   ```python
   CONSOLE_FORMAT = "[主线版] %(asctime)s | %(message)s"
   ```
   ```powershell
   git add src/utils/logger.py
   git commit -m "refactor: console format"
   ```

4. **合并 → 制造冲突**
   ```powershell
   git merge exp/logger-format
   ```
   应该看到 `CONFLICT (content): Merge conflict in src/utils/logger.py`。

5. **解决冲突**：打开文件，删掉 `<<<<<<<` / `=======` / `>>>>>>>` 三行标记，留下你想要的版本。然后：
   ```powershell
   git add src/utils/logger.py
   git commit
   ```
   （`git commit` 不带 `-m`，Git 会弹出准备好的合并信息，保存即可。终端里若卡住，用 `git commit -m "merge: console format"`。）

6. **看合并结果**
   ```powershell
   git log --oneline --graph --all
   ```

7. **顺手练 `restore`**：随便改一行 `logger.py`，**不要 add**，然后
   ```powershell
   git restore src/utils/logger.py
   ```
   改动消失了 —— 亲眼确认它有多危险。

8. **跑一次确认程序没坏**
   ```powershell
   & "E:\anaconda\envs\pytorch\python.exe" scripts\train.py --dry-run
   ```

## 阶段 C：推 GitHub（60min）

按第八节走。**建仓库时千万别勾任何初始化选项。**

## 阶段 D：补三个文件（30min）

- `README.md` —— **已有 140 行**（Day 1 写的骨架说明），今天要补的是**实质内容**：这个项目在做什么、现在能跑什么、怎么跑
- `.gitignore` —— **已有 79 行**，今天刚加了练习脚本一节，不动
- `requirements.txt` —— 按第九节新建

## 阶段 E：提交收尾

```powershell
git add README.md requirements.txt
git commit -m "docs: add usage guide and dependency list"
git push
```

---

# 十一、过关前自问（第 3 条要求"能口述"）

不用背术语，能用自己的话说清就行：

1. **`git add` 和 `git commit` 各自把东西从哪挪到哪？**
2. **` M` 和 `M ` 有什么区别？** 为什么同一个文件会变成 `MM`？
3. **冲突的那三个标记，哪个是"我的"、哪个是"对面的"？**
4. **`git restore` 和 `git restore --staged` 差在哪？哪个会丢代码？**
5. **`git reset --soft` 和 `--hard` 差在哪？**
6. **如果昨天手滑 `--hard` 丢了提交，用什么命令能找回？**

答不上来就回去翻对应小节 —— 这几条比记住命令本身重要得多。

---

# 附：今天的红线（贴在心里）

| 危险操作 | 后果 | 防法 |
|---|---|---|
| `git restore 文件` | 未提交的改动**永久丢失** | 执行前 `git status` 确认 |
| `git reset --hard` | 未提交的改动 + 提交全丢 | 先 `git log` 看清，优先用 `--soft` |
| PyCharm 勾 **Amend** | **改写**上一条提交 | 每次提交前扫一眼左下角 |
| `git push --force` | 覆盖远程历史，协作者遭殃 | **今天用不到。看到了就别点。** |
| 建仓库时勾 README/.gitignore | 远程有初始提交，push 被拒 | 建空仓库 |
| 在错误分支上 commit | 提交跑到别的分支去了 | **操作前先 `git branch` 看 `*`** |
