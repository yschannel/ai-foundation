# Small Functions: axis, None, Boolean Indexing, keepdims + Broadcasting in Code (Session 3 done)

- Date: 2026-11-27
- Stage: Stage 0 / Week 1 / Session 3 (skeleton + pytest, all six functions)
- Files: `small_funcs.py` (signatures + docstrings), `test_small_funcs.py` (15 tests, shape first, then values)
- Result: `pytest -q` -> **15 passed**. Finished `standardize_rows` a day early (planned for 11-28); only the oral self-check is left.
- Today was the first time I turned the week 1 ideas into running code: write the body, save, run pytest, read the failure, fix.

## 1. Workflow that worked

1. Before the code, write input shape -> output shape in a comment (which number disappears?).
2. Write the body, **`Ctrl+S`**, then run one function at a time: `python -m pytest -q -k row_mean`.
3. Read the `E` lines of the failure. Shape errors show `assert (3,) == (4,)`; value errors show ACTUAL vs DESIRED.
4. `13 deselected` is not a failure: `-k` only picked the tests whose names match.
5. A test that passes is not proof of correctness: a single-row input can pass by coincidence (see §4, `standardize_rows`).

## 2. The five easy functions

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

## 3. Standard deviation (what `std` computes)

For one row `x_1 ... x_d`:

1. mean: `mu = (1/d) * sum(x_j)`
2. deviations: `x_j - mu`
3. variance: average of the squared deviations, `(1/d) * sum((x_j - mu)^2)`
4. std: `sqrt(variance)`

- Example `[2, 4, 4, 4, 5, 5, 7, 9]`: mu = 5, squared deviations sum to 32, variance = 32 / 8 = 4, std = 2.
- Why square: deviations have both signs and sum to 0. Why sqrt: bring the unit back (m^2 -> m).
- Meaning: on average, how far the data is from the mean.
- `ddof=0` (NumPy default) divides by `d` (population std). `ddof=1` divides by `d - 1` (sample std, for later stats courses).
- `std` is a method like `mean`, `sum`, `max`: `x.std(axis=1, keepdims=True)` = `np.std(x, axis=1, keepdims=True)`. Same `axis` / `keepdims` rules.

## 4. `standardize_rows`

Goal: every row gets mean 0 and std 1, each row on its own (`(n, d)` -> `(n, d)`).

```python
y = x - x.mean(axis=1, keepdims=True)
return y / y.std(axis=1, keepdims=True)
```

- Row mean with `keepdims=True` is **`(n, 1)`**, a column. (I first guessed `(1, n)`: wrong, that is one row of n numbers. Worse, when `n == d` it still broadcasts, silently subtracting a per-column value.)
- Check with the 11-26 rule: `(n, d)` vs `(n, 1)` -> right-aligned: `d` vs `1` ok (stretch), `n` vs `n` ok -> `(n, d)`.
- `y.std(...)` equals `x.std(...)`: shifting by the row mean does not change the spread. Both work.
- `y` is a new array, so `x` is not modified (`x -= ...` would modify it).
- Hand check: `[1, 2, 3]` -> mean 2, std sqrt(2/3) ~ 0.8165 -> `[-1.2247, 0, 1.2247]`. `[100, 300, 200]` gives the same pattern.

## 5. Mistakes and fixes

- **`col_max`: `axis` flipped again** (third time: 11-25, 11-26, today). Symptom: shape `(3,)` instead of `(4,)`, and `-5` instead of `-3` in the negative test (`-5` was a **row** max). Fix: `axis=0`. Check: the result shape must equal the input shape with the k-th number removed.
- **`count_even`: `.max` instead of `.sum()`.** Two bugs in one:
  - without `()` I got the **method object** (`<built-in method max ...>`) instead of a value;
  - even with `()`, `max` answers "what is the largest value", not "how many". Counting = sum of a bool array.
- **`above_mean`: `x(...)` instead of `x[...]`.** `TypeError: 'numpy.ndarray' object is not callable`. `()` calls a function, `[]` indexes.
- **`standardize_rows`: `y.std()` without `axis`.** It computed one std for the whole matrix: the output shape was right but the per-row test failed (`-1.7320` instead of `-1.2247`, because I divided by the global std). Also, the first version returned a scalar (shape `()`) and `out[1]` raised `IndexError: invalid index to scalar variable`.
- **Coincidence check again:** `test_standardize_rows_hand_value` passed with the buggy `y.std()` because it has only one row (global std = row std). Only the multi-row test exposed it.
- **Row-mean shape guess:** `(1, n)` -> correct is `(n, 1)` (see §4).
- Rules of thumb: error text with `<built-in method ...>` -> a missing `()`; `not callable` -> a wrong bracket; a result shape `()` -> something collapsed every axis, check `axis`.

## 6. Vocabulary

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
| deviation | 偏差 (value minus mean) |
| variance | 方差 (mean of squared deviations) |
| standard deviation | 标准差 (sqrt of variance) |
| population std | 总体标准差 (`ddof=0`, NumPy's default) |
| scalar | 标量 (a single number, shape `()`) |

## 7. Self-check (explain out loud)

- [ ] Which number does `axis=0` remove from `(3, 4)`? What shape do I get?
- [ ] Why does `x.max` (no parentheses) not give a number?
- [ ] `x(mask)` vs `x[mask]`: what does each do?
- [ ] Why does `above_mean` return a 1-D array for a 2-D input?
- [ ] Why does `.sum()` of a bool array count the `True`s? Is the result a plain `int`?
- [ ] What is the shape of the row mean with `keepdims=True` for `x` of shape `(n, d)`? Why is `(1, n)` wrong even when it does not raise an error?
- [ ] Walk through the four steps of std on `[1, 2, 3]`. Why square? Why sqrt?
- [ ] Why can `test_standardize_rows_hand_value` pass while `y.std()` is wrong?

## 8. Next steps

- Commit: `week1: update small functions note (2026-11-27)`
- 11-28: oral self-check only (the questions in this note and the 11-26 note, without looking). Short session; keep the spare time as buffer, since the week 6 buffer is used up. Optional: re-implement `std` by hand from §3 and compare with `np.std` using `np.allclose`.
- 11-29: start week 2 (vectorized programming: softmax, cross-entropy, distance matrix).
