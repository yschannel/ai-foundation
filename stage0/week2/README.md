# 第 2 周 · 向量化编程

目标：不写循环，用数组运算实现 softmax、交叉熵和距离矩阵，并用「循环版」验证「向量化版」。

每个函数都按这个顺序走：先手算一个 2×2 或 3 个元素的小例子，再写下每个中间变量的 shape，然后写循环版、写向量化版，最后用测试确认两者一致。上周自检暴露的问题是：规则单独问都会，**串在一起就容易错**，而且容易跳过手算，所以这周每次都从手算开始。

## 进度与安排

上周实际用时约为计划的两倍，所以 4 个任务拆成 5 次课。11-29 没空，第 1 次提前到 11-28 和第 1 周收尾同一天上。后面的日期是建议，有变化就在这里改。每次课结束后在「时长」一栏记下实际用时。

| 次 | 建议日期 | 内容 | 时长 | 状态 / 产出 |
| --- | --- | --- | --- | --- |
| 1 | 11-28 | 热身复习 + `exp`、softmax 概念、数值稳定性、手算，2 轮共 10 题 | 2 h | ✓ `2026-11-28-softmax-concept.md` |
| 2 | 11-30 | 一维 softmax 和按行 softmax，配循环版和测试 | – | – |
| 3 | 12-01 | 交叉熵：独热标签、`log`、按 batch 求平均 | – | – |
| 4 | 12-02 | 距离矩阵：广播版和展开公式版 | – | – |
| 5 | 12-03 | 用 `time` 比较速度 + 口述自检，第 2 周收尾 | – | – |
| – | 12-04 | 开始第 3 周：梯度与向量化求导 | – | – |

## 文件

| 文件 | 内容 | 谁写 |
| --- | --- | --- |
| `README.md` | 本文件 | Claude |
| `YYYY-MM-DD-<topic>.md` | 每次课的英文学习笔记 | Claude 整理 |
| `vec_funcs.py` | 向量化版和循环版（`*_loop`）函数 | 骨架由 Claude 写，函数体自己写 |
| `test_vec_funcs.py` | pytest 测试：先检查 shape，再检查值，最后和循环版对照 | Claude |
| `bench.py` | 计时脚本：比较循环版和向量化版的速度 | 骨架由 Claude 写，计时部分自己写 |

骨架在对应那次课开始时再建，不提前建。

## 环境（Windows PowerShell）

直接用第 1 周的虚拟环境，不用重新装。在 `week2` 目录运行：

```powershell
Set-Location -LiteralPath 'D:\AAA000AI\stage0\week2'
..\week1\.venv\Scripts\python.exe -m pytest -q
..\week1\.venv\Scripts\python.exe bench.py
```

要进入交互模式就执行 `..\week1\.venv\Scripts\python.exe`。在编辑器里选择解释器时，选 `D:\AAA000AI\stage0\week1\.venv\Scripts\python.exe`。

## 第 1 次 · softmax 概念与数值稳定性

1. 热身（5 分钟，不看笔记）：重答第 1 周自检笔记 §3 的陷阱题和 §5。
2. 阅读 softmax 的定义，以及减去最大值的 *shift invariance* 那一段。 [d2l: The Softmax](https://d2l.ai/chapter_linear-classification/softmax-regression.html#sec-softmax-operation)
3. 查 `np.exp` 输入和输出的 shape、dtype，再看看输入很大时会发生什么。 [numpy.exp](https://numpy.org/doc/stable/reference/generated/numpy.exp.html)
4. 手算 `z = [1, 2, 3]` 的 softmax。想一想 `z = [1000, 1001, 1002]` 直接算会出什么问题，减去最大值后为什么结果不变。

读的时候，遇到不懂的英文和概念随时贴；读完再做预测题和巩固题。

## 第 2 次 · 一维与按行 softmax

- 先写一维版 `softmax(z)`，`(k,)` → `(k,)`，再写按行版，`(n, k)` → `(n, k)`，每一行各自做 softmax。
- 按行版要对**每一行**减最大值、除以**每一行**的和，这正是上周「按行操作要 `keepdims=True`」的陷阱。写之前先把每个中间变量的 shape 写出来。 [numpy.max](https://numpy.org/doc/stable/reference/generated/numpy.max.html)
- 循环版：对每一行调用一维版。测试要用**至少 2 行、尺度不同**的输入，否则写成「全局最大值」也能通过（上周的单行测试教训）。
- 用 `np.allclose` 对照循环版和向量化版。 [numpy.allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)

## 第 3 次 · 交叉熵

- 读交叉熵损失一节，只看定义和公式，最大似然那部分留到第 4 周。 [d2l: The Cross-Entropy Loss](https://d2l.ai/chapter_linear-classification/softmax-regression.html#sec-softmax-regression-loss-func)
- 输入：预测概率 `p` `(n, k)`、独热标签 `y` `(n, k)`；输出：一个标量（batch 平均）。 [numpy.log](https://numpy.org/doc/stable/reference/generated/numpy.log.html)
- 手算 2 个样本、3 个类别的例子。想一想：独热标签乘进去以后，每行实际只剩哪一项？
- 处理 `log(0)`：给输入加一个极小值，或者用 `np.clip`。

## 第 4 次 · 距离矩阵

- 输入 `X` `(n, d)`，输出 `D` `(n, n)`，其中 `D[i, j]` 是第 i 个点和第 j 个点之间的欧氏距离。
- 写法 A（广播）：把 `X` 变成 `(n, 1, d)` 和 `(1, n, d)` 再相减。先把三步对齐规则手写一遍，并估算中间结果有多大。 [NumPy 广播规则](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules)
- 写法 B（矩阵乘法）：‖a − b‖² = ‖a‖² + ‖b‖² − 2a·b，其中 `X @ X.T` 的 shape 是什么？ [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
- 手算 3 个二维点，比如 `(0, 0)`、`(3, 4)`、`(1, 1)`。检查对角线是否为 0、矩阵是否对称。写法 B 可能因为浮点误差出现极小的负数，开方前要处理。

## 第 5 次 · 计时与自检

- 用 `time.perf_counter()` 给循环版和向量化版计时，每个跑几次取最快的一次，数据规模逐步加大（如 n = 100、1000）。 [time.perf_counter](https://docs.python.org/3/library/time.html#time.perf_counter)
- 把结果记成表，解释为什么向量化更快，以及写法 A 和写法 B 哪个更省内存。
- 口述自检（不看笔记）：
  - softmax 减去最大值为什么不改变结果？
  - 距离矩阵用了怎样的广播或矩阵乘法？各中间变量的 shape 是什么？
  - 按行 softmax 如果忘了 `keepdims`，会报错还是悄悄算错？

## 推送到 GitHub（每次学完）

在仓库根目录按「看 → 加 → 再看 → 提交 → 推送」操作：

```powershell
Set-Location -LiteralPath 'D:\AAA000AI'
git status
git add stage0/week2
git status                 # 确认没有 __pycache__、.pytest_cache 之类
git commit -m "week2: <what you did> (YYYY-MM-DD)"
git push
```

根目录的 `.gitignore` 已经忽略了 `.venv/`、`__pycache__/`、`.pytest_cache/`，`week2` 不需要再单独建一份。常见报错的处理见第 1 周 README 的最后一节。
