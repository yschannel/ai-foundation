# 第 1 周 · NumPy 基础

练习时先写下输入的 shape、表达式能否运行，以及结果的 shape 和题目要求的值。运行后，把有出入的地方记成「我的预测 → 实际结果 → 我用错的规则」。

## 进度与安排（2026-11-18 调整）

NumPy 基础为零、按逐句精读推进，本周实际用时约为计划的两倍（预计共 9 小时左右），所以第 1 周延长到 11-22，后续各周顺延约 3 天，占用第 6 周的缓冲时间。

| 日期 | 内容 | 状态 / 产出 |
| --- | --- | --- |
| 11-16 | 第一次 block 1–2：ndarray、shape、索引 | ✓ `2026-11-16-ndarray-shape.md` |
| 11-17 | 第一次 block 2：reshape、newaxis、expand_dims、view | ✓ `2026-11-17-reshape-newaxis.md` |
| 11-18 | 遗留 reshape 实验 + 第一次 block 3：布尔索引、`nonzero` | ✓ `2026-11-18-boolean-indexing.md` |
| 11-19 | 第一次 block 4：`sum` 的 `axis` / `keepdims` + 8 道总结题 | 第一次完成 |
| 11-20 | 第二次：广播规则，6 道给定题 + 自己出 4 道 | 广播笔记 |
| 11-21 | 第三次（上）：函数骨架与测试，先写简单的几个 | 函数与测试 |
| 11-22 | 第三次（下）：重点「每行标准化」，`pytest -q` 全部通过，口述自检 | 第 1 周收尾 |
| 11-23 | 开始第 2 周：向量化编程 | – |

## 环境（Windows PowerShell）

环境在 `D:\AAA000AI\阶段0\week1\.venv`，直接调用它的 Python 即可。写完第三次的小函数和测试后，在同一目录运行最后一行。

```powershell
Set-Location -LiteralPath 'D:\AAA000AI\阶段0\week1'
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import numpy; print(numpy.__version__)"
.\.venv\Scripts\python.exe -m pytest --version
.\.venv\Scripts\python.exe -m pytest -q
```

要验证表达式时，在同一目录执行 `.\.venv\Scripts\python.exe` 进入交互模式，在 `>>>` 后输入 Python 表达式，输完用 `exit()` 退出。用编辑器运行代码时，选择 `.venv\Scripts\python.exe` 作为解释器。

### 重建环境

在 `week1` 目录运行以下命令。第一条指定 `uv` 管理的 Python 安装位置，第二条在需要时下载 Python 3.12 并创建 `.venv`，第三条把依赖装进这个虚拟环境。 [uv 的 Python 版本管理](https://docs.astral.sh/uv/concepts/python-versions/) · [uv 虚拟环境指南](https://docs.astral.sh/uv/pip/environments/)

```powershell
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location).Path '.python'
uv venv --python 3.12 --cache-dir .uv-cache .venv
uv pip install --python .\.venv\Scripts\python.exe --cache-dir .uv-cache numpy pytest
```

## 第一次 · 读入门文档，做 8 道预测题

用约 30 分钟按这个顺序读，遇到别的章节可以先跳过。

1. 认识 `ndarray`，分清 `.shape`、`.ndim`、`.size`。 [数组是什么](https://numpy.org/doc/stable/user/absolute_beginners.html#what-is-an-array) · [数组的形状和大小](https://numpy.org/doc/stable/user/absolute_beginners.html#how-do-you-know-the-shape-and-size-of-an-array)
2. 看 `reshape` 如何重新安排维度，以及 `None`（`np.newaxis`）如何增加一维。 [重塑数组](https://numpy.org/doc/stable/user/absolute_beginners.html#can-you-reshape-an-array) · [给数组增加一维](https://numpy.org/doc/stable/user/absolute_beginners.html#how-to-convert-a-1d-array-into-a-2d-array-how-to-add-a-new-axis-to-an-array)
3. 对照整数索引、切片和布尔筛选的例子。 [索引和切片](https://numpy.org/doc/stable/user/absolute_beginners.html#indexing-and-slicing)
4. 在 `sum` 的参数表里查 `axis` 和 `keepdims`。如果第 6–8 题卡在两个 shape 怎么对齐，先读第二次的广播规则一节。 [NumPy `sum` 参数说明](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)

## 第二次 · 广播，6 道给定题 + 自己出 4 道

读广播规则和后面的 shape 示例。先做 Claude 给的 6 道预测题，再自己出 4 道。全部 10 道题合起来覆盖 `axis`、`keepdims`、切片、布尔索引和广播，广播题里各放一个能运行和会报错的例子。 [NumPy 广播规则](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules)

## 第三次 · 6 个小函数和 pytest

6 个函数：每行均值、每列最大值、每行标准化、一维向量变列向量、统计偶数个数、取出大于均值的元素。「变列向量」和「取大于均值的元素」用的就是已学的 `None` 和布尔索引，可以写快一些；把时间留给「每行标准化」，它同时用到 `keepdims` 和广播。骨架和测试由 Claude 写，核心逻辑自己实现。

- 查 `mean`、`max` 和 `std` 时，留意输入与输出的 shape。变列向量和条件筛选可以回查第一次的增维和索引小节。 [NumPy 常用数组运算](https://numpy.org/doc/stable/user/absolute_beginners.html#more-useful-array-operations)
- 看 pytest 怎样发现 `test_*.py` 文件和 `test_*` 函数。写测试时先确定期望的 shape，再检查值。浮点结果可用 `assert_allclose`，要求精确相等的数组可用 `assert_array_equal`。 [pytest 入门](https://docs.pytest.org/en/stable/getting-started.html#create-your-first-test) · [浮点近似比较](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html) · [数组精确比较](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_array_equal.html)

做完主线还有时间，可以在 Quickstart 里查同类例子，或在题库前 30 题里挑几道练习。 [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html) · [100 NumPy Exercises](https://github.com/rougier/numpy-100)

## 推送到 GitHub（每次学完）

```
你的文件夹  --add-->  暂存区  --commit-->  本地仓库  --push-->  GitHub
```

在仓库根目录按「看 → 加 → 再看 → 提交 → 推送」操作：

```powershell
Set-Location -LiteralPath 'D:\AAA000AI'
git status                 # 看改了什么
git add 阶段0/week1         # 换成这次的目录
git status                 # 确认只有该交的文件（不能有 .venv 之类）
git commit -m "weekN: <what you did> (YYYY-MM-DD)"
git push
```

提交说明用英文，例如 `week1: add reshape/newaxis study note (2026-11-17)`。

- 中文路径显示成 `"\351\230\266..."`：运行一次 `git config --global core.quotepath false`。
- `push` 被拒绝并提示 `fetch first`：先 `git pull --rebase`，再 `git push`。
- 还没 push 时发现说明写错：`git commit --amend -m "new message"`。
