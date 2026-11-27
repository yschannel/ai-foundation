# Small Functions: axis, None, Boolean Indexing in Code (Session 3, part 1 done)

- Date: 2026-11-27
- Stage: Stage 0 / Week 1 / Session 3, part 1 (skeleton + pytest, the five easy functions)
- Files: `small_funcs.py` (signatures + docstrings), `test_small_funcs.py` (15 tests, shape first, then values)
- Result: `pytest -q` -> 11 passed, 4 failed. The 4 failures are all `standardize_rows` (tomorrow).
- Today was the first time I turned the week 1 ideas into running code: write the body, save, run pytest, read the failure, fix.

## 1. Workflow that worked

1. Before the code, write input shape -> output shape in a comment (which number disappears?).
2. Write the body, **`Ctrl+S`**, then run one function at a time: `python -m pytest -q -k row_mean`.
3. Read the `E` lines of the failure. Shape errors show `assert (3,) == (4,)`; value errors show ACTUAL vs DESIRED.
4. `13 deselected` is not a failure: `-k` only picked the tests whose names match.

## 2. The five functions

| Function | Shape | My solution | Idea |
| --- | --- | --- | --- |
| `row_mean` | `(n, d)` -> `(n,)` | `x.mean(axis=1)` | one mean per row: the `d` disappears |
| `col_max` | `(n, d)` -> `(d,)` | `x.max(axis=0)` | one max per column: the `n` disappears |
| `to_column` | `(n,)` -> `(n, 1)` | `v[:, None]` | new axis goes **last** |
| `count_even` | any -> int | `int((x % 2 == 0).sum())` | bool array, `True` counts as 1 |
| `above_mean` | any -> `(k,)` | `x[x > x.mean()]` | mean of the WHOLE array, strict `>`, result is 1-D in row-major order |

- `axis=k` deletes the k-th number of the shape. `axis=0` removes the row count -> **one result per column**.
- `x.mean()` without `axis` averages everything, so `above_mean` is a global comparison, not per row.
- A boolean mask of the same shape as `x` picks elements and flattens them to 1-D.
- `.sum()` of a bool array returns `np.int64`, not Python `int`. The docstring says "plain int", so I wrapped it in `int(...)`.
- After `return ...` the line `raise NotImplementedError` is dead code. Delete it together with `# TODO`.

## 3. Mistakes and fixes

- **`col_max`: `axis` flipped again** (third time: 11-25, 11-26, today). Symptom: shape `(3,)` instead of `(4,)`, and `-5` instead of `-3` in the negative test (`-5` was a **row** max). Fix: `axis=0`. Check: the result shape must equal the input shape with the k-th number removed.
- **`count_even`: `.max` instead of `.sum()`.** Two bugs in one:
  - without `()` I got the **method object** (`<built-in method max ...>`) instead of a value;
  - even with `()`, `max` answers "what is the largest value", not "how many". Counting = sum of a bool array.
- **`above_mean`: `x(...)` instead of `x[...]`.** `TypeError: 'numpy.ndarray' object is not callable`. `()` calls a function, `[]` indexes.
- Rule of thumb: error text with `<built-in method ...>` -> a missing `()`; `not callable` -> a wrong bracket.

## 4. Vocabulary

| Term | Meaning |
| --- | --- |
| skeleton | 骨架 (signatures + docstrings, body left as TODO) |
| deselected | 未被选中的 (pytest skipped them because of `-k`) |
| callable | 可被调用的 (a function / method; an ndarray is not) |
| dead code | 死代码 (code that can never run) |
| strictly greater | 严格大于 (`>`, not `>=`) |
| row-major order | 行优先顺序 (read row by row) |
| plain `int` | 普通的 Python 整数 (not `np.int64`) |
| standardize | 标准化 (subtract mean, divide by std) |
| population std | 总体标准差 (`ddof=0`, NumPy's default) |

## 5. Self-check (explain out loud)

- [ ] Which number does `axis=0` remove from `(3, 4)`? What shape do I get?
- [ ] Why does `x.max` (no parentheses) not give a number?
- [ ] `x(mask)` vs `x[mask]`: what does each do?
- [ ] Why does `above_mean` return a 1-D array for a 2-D input?
- [ ] Why does `.sum()` of a bool array count the `True`s? Is the result a plain `int`?

## 6. Next steps

- Clean up: delete `raise NotImplementedError` and `# TODO` in the five finished functions.
- Commit: `week1: implement five small functions (2026-11-27)`
- 11-28, Session 3 (part 2): `standardize_rows` (`keepdims` + broadcasting, same idea as 11-26 P4a). Before coding, write down the shape of the row mean and the row std so they broadcast against `(n, d)`. Then `pytest -q` all pass, oral self-check -> week 1 done.
- 11-29: start week 2 (vectorized programming). The week 6 buffer is used up, so no more slipping.
