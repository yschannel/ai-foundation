# Week 1 Oral Self-Check: axis, keepdims, Broadcasting, View/Copy, std (Week 1 done)

- Date: 2026-11-28
- Stage: Stage 0 / Week 1 / Session 3 (part 2): oral self-check, no notes, no running code
- Format: 4 rounds x 5 questions, graded after each round
- Result: shape rules are solid **when asked one at a time**. Weak spots: chaining two rules together, view/copy vocabulary, Python call vs index semantics, and skipping the hand calculation.
- Week 1 closes today. Week 2 (vectorized programming) starts 2026-11-29.

## 1. Scoreboard

| Round | Topic | Right | Half | Wrong / blank |
| --- | --- | --- | --- | --- |
| 1 | axis / keepdims | 2 | 3 | 0 |
| 2 | broadcasting | 1 | 3 | 1 |
| 3 | hand calc, view/copy, std | 2 | 2 | 1 |
| 4 | small-function details | 1 | 1 | 3 |

## 2. axis and keepdims

- Rule: `axis=k` **deletes** the k-th number of the shape. `keepdims=True` replaces it with `1` instead.
- `(3, 4)`, max of each column: `axis=0` -> `(4,)`. NOT `(1, 4)`: that only happens with `keepdims=True`. (My mistake: mixed the two up.)
- "Max of each column" squeezes out the **row count** (3); each column becomes one number, so 4 numbers remain.
- `keepdims` changes **only the shape** (`(n,)` vs `(n, 1)`). Values, count and order are identical. It has nothing to do with views; the reason to keep the axis is to broadcast correctly against the original array.
- `m.mean(...)` on an int array gives `float64` (division may not be whole); `m.sum(...)` stays int.

## 3. Broadcasting

Three steps:

1. Right-align the two shapes; pad the shorter one with `1`s on the **left**.
2. Each position: equal, or one of them is `1` -> compatible; the `1` is stretched.
3. Any position with two different numbers, neither `1` -> error.

- `(8, 1, 6, 1)` with `(7, 1, 5)` -> `(8, 7, 6, 5)`. Stretched array per position: second, first, second, first. (I answered "first, second, first, second": that is who **supplies** the size, the opposite question.)
- A 1-D array `(n,)` broadcasts as a **row** `(1, n)`. Make it a column with `v[:, None]`.
- Broadcasting does not copy the inputs, but the **result** is full-size: `(n, 1)` with `(1, n)` -> `(n, n)`, which can be huge.

### The trap I fell into (round 2, Q1)

`x` is `(5, 5)`, `x - x.mean(axis=1)` (no keepdims): no error, but wrong.
`x.mean(axis=1)` is `(5,)` -> treated as a row -> `x[i, j]` minus the mean of **row j**, not row i.

Hand example, `x = [[1, 2], [10, 20]]`, row means `[1.5, 15]`:

| Expression | Result | Meaning |
| --- | --- | --- |
| `x - x.mean(axis=1)` | `[[-0.5, -13], [8.5, 5]]` | nonsense |
| `x - x.mean(axis=1, keepdims=True)` | `[[-0.5, 0.5], [-5, 5]]` | each row minus its own mean ✓ |

Per **column** needs no keepdims: `x = [[0, 4], [2, 6]]`, `x - x.mean(axis=0)` = `[[-1, -1], [1, 1]]` ✓, because the column means `(d,)` broadcast as a row, which is exactly what we want.

> Per-row ops need `keepdims=True`; per-column ops work either way.

Fixed habit for any expression: (1) write both shapes, (2) right-align, (3) ask "is the small one copied as rows or as columns?"

## 4. Indexing, view vs copy

- **buffer**: one contiguous block of raw numbers in memory.
- **header**: how to read the buffer: shape, strides, dtype, offset.
- View = new header + same buffer (slices, `reshape`). Copy = new buffer.
- `m.sum(axis=0)` is a copy: the sums are new numbers that don't exist in the old buffer, so no header can describe them.
- Fancy indexing (a list as index, `y[:, [2, 3]]`) **always returns a copy**, never a view.
  - `c = y[:, [2, 3]]; c += 1` -> `y` unchanged.
  - `y[:, [2, 3]] += 1` -> `y` changed, because Python expands it to `y[:, [2, 3]] = y[:, [2, 3]] + 1`: indexing on the **left** of `=` writes back into `y`.
- `x[x > 9]` -> the **values**. `np.nonzero(x > 9)` -> the **positions** (one index array per dimension).
- `above_mean` returns 1-D because the number of `True`s can differ from row to row, so the result can't be a rectangle. NumPy flattens it in row-major order.

## 5. Python: `()` vs `[]`, method vs call

- `x.max` (no parentheses) **is valid Python**. It is the method object itself (`<built-in method max ...>`), not a number. Python checks nothing at compile time the way C does. Closest C analogy: a function pointer vs a function call.
- `x.max()` calls it and returns a value.
- `x[mask]` = indexing (select elements). `x(mask)` = try to **call** `x` like a function -> `TypeError: 'numpy.ndarray' object is not callable`.

## 6. Standard deviation on `[1, 2, 3]`

1. mean: 2
2. deviations: -1, 0, 1
3. variance: (1 + 0 + 1) / 3 = 2/3
4. std: sqrt(2/3) ~ 0.8165

- Why square: deviations always sum to 0, so their plain average is useless.
- Why sqrt: bring the **unit** back (m^2 -> m) so std is comparable to the data. NOT "the number is too big".

## 7. Testing lesson

`test_standardize_rows_hand_value` passed with the buggy `y.std()` because it has **one row**: with one row, global std == row std, so the bug is invisible. Tests need inputs where the right and wrong versions give different answers (here: at least 2 rows with different spreads). A passing test is not a proof.

## 8. Mistakes to watch in week 2

- [ ] Knowing rules separately but not chaining them (1-D = row, applied inside a bigger expression).
- [ ] Mixing up `(d,)` and `(1, d)`: no keepdims = the axis is gone.
- [ ] Skipping hand calculations. Always compute a 2x2 example.
- [ ] "view" used loosely. Ask: same buffer or new buffer?
- [ ] Forgetting what I fixed yesterday (`x.max`, `x(...)`, the one-row test). Re-read §5 and §7 before week 2.

## 9. Vocabulary

| Term | Meaning |
| --- | --- |
| right-align | 右对齐 |
| pad (with 1s) | 补齐（补 1） |
| stretch | 拉伸 |
| fancy indexing | 花式索引（用列表/数组做索引，返回副本） |
| buffer | 数据缓冲区（存原始数字的连续内存） |
| header | 头信息（shape、strides、dtype、偏移量） |
| method object | 方法对象（没被调用的方法本身） |
| function pointer | 函数指针 |
| invisible bug | 不报错、难发现的 bug |

## 10. Next steps

- Delete the leftover `# TODO` and `raise NotImplementedError` lines in `small_funcs.py`, run `pytest -q` (still 15 passed).
- Commit: `week1: add self-check note, close week 1 (2026-11-28)`
- 11-29: start week 2. First 5 minutes: re-answer §3 trap and §5 without looking.
