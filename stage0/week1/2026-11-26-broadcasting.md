# NumPy Basics: Broadcasting (Session 2 done)

- Date: 2026-11-26
- Stage: Stage 0 / Week 1 / Session 2 (warm-up leftovers + broadcasting rules + 6 given questions + 4 reverse challenges)
- Source: NumPy docs, [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html): intro, General broadcasting rules, Broadcastable arrays (up to the outer-addition example; skipped *vector quantization*, back to it in week 2 for the distance matrix)
- Changed today: instead of writing 4 of my own questions (boring), did 4 "given input + target output, write one line" challenges (C1–C4)

## 0. Warm-up: leftovers from 11-25

| Code | Actual | Why |
| --- | --- | --- |
| `m.sum(axis=2)` on 2-D `m` | `AxisError: axis 2 is out of bounds for array of dimension 2` | valid axes: `0, 1` (or `-1, -2`) |
| `a.reshape(4, -1)`, `a` of size 6 | `ValueError: cannot reshape array of size 6 into shape (4,newaxis)` | `-1` = "you compute it", 6 / 4 is not whole; `(3, -1)` -> `(3, 2)` works |
| `a[1::2].flags['C_CONTIGUOUS']` | `False` | step 2 skips elements, but it is still a **view** (slice = view) |

## 1. Why broadcasting

- Broadcasting = how NumPy does **arithmetic operations** (`+ - * /`, element-wise) on arrays of **different shapes**.
- It **vectorizes**: the loop still happens, but in C, not in Python -> much faster. (C view: `c[i] = a[i] * b;` with `b` a single number.)
- The "stretching" is **only conceptual**: the smaller array is **not copied**.
- But the **result** has the full broadcast shape and is real memory. Example: `(n,1,d) - (1,n,d)` -> `(n,n,d)`; with n = 10000, d = 100 that is 10^10 float64 ≈ 80 GB. -> watch out in the week 2 distance matrix.

## 2. The rule

1. Line the shapes up on the **right** (start at the trailing dimension, move left).
2. Each pair of dimensions is **compatible** if they are **equal**, or **one of them is 1**.
3. A missing dimension (fewer axes) counts as 1.
4. Result: the **larger** size in each position. Any incompatible pair -> `ValueError: operands could not be broadcast together with shapes ...`

```
A      (4d):  8 x 1 x 6 x 1
B      (3d):      7 x 1 x 5
Result (4d):  8 x 7 x 6 x 5
```

- "One is a multiple of the other" (like 2 and 4) is **not** compatible.
- Incompatible examples from the docs: `3` vs `4` (last dim); `2 x 1` vs `8 x 4 x 3` (second-to-last: 2 vs 4).

## 3. A 1-D array acts like a ROW

`(6,)` -> treated as `(1, 6)` -> stretched down to `(5, 6)`. Every row gets the same `c`.

To get a **column**, add an axis in the **index**: `a[:, None]` -> `(n, 1)`.

**Outer operation** = column op row -> matrix, `r[i, j] = a[i] op b[j]`:

```
a[:, None] + b      (4,1) + (3,) -> (4,1) + (1,3) -> (4,3)
```

Link to linear algebra (vectors are columns by default):

| Form | Shapes | Result |
| --- | --- | --- |
| `uᵀ v` (inner / dot) | (1×n)(n×1) | one number; lengths must match |
| `u vᵀ` (outer) | (m×1)(1×n) | m×n matrix; lengths can differ |

NumPy 1-D arrays have no row/column; `None` plays the role of the transpose.

## 4. When do I need `keepdims`?

- Reduce the **last** axis (per row): `(3, 4)` -> `(3,)` does **not** line up with `(3, 4)` (3 vs 4) -> need `keepdims=True` -> `(3, 1)`.
- Reduce the **first** axis (per column): `(3, 4)` -> `(4,)` lines up with the last dim -> works without `keepdims` (acts as a row).
- Only needed when the result is **used again** with the original array. A final answer does not need it (C3).

## 5. Writing into an array

- `y[:, [2, 3]] += ...` **does** change `y`: `[]` on the left of `=` means "write into these positions", even with an int-array index.
- `c = y[:, [2, 3]]; c += 1` does **not** change `y`: `c` is a copy.
- `y[:, [2, 3]] + ...` alone only computes a new array.

## 6. Python list vs NumPy array

- `None` adds an axis **only inside an index**: `np.array([...])[:, None]`. Putting `None` in a list (`[[100, 200, 300], None]`) -> ragged list -> `ValueError: ... inhomogeneous shape`.
- `reshape` is an array method; a list has none -> `AttributeError: 'list' object has no attribute 'reshape'`. Use `np.array([...]).reshape(3, 1)`.
- A nested list `[[100], [200], [300]]` is converted to a `(3, 1)` array and broadcasts fine.

## 7. Experiment log

`x = np.arange(12).reshape(3, 4)`

| # | Code | My prediction | Actual | Note |
| --- | --- | --- | --- | --- |
| P1 | `x + np.array([10,20,30,40])` | `(3,4)`, row 0 `[10 21 32 43]` | same | ✓ |
| P2 | `x + np.array([10,20,30])` | error, last axis | `shapes (3,4) (3,)` | ✓ |
| P3 | `x + np.array([10,20,30])[:, None]` | `(3,4)`, col 0 `[10 24 38]` | same | ✓ |
| P4a | `x - x.mean(axis=1, keepdims=True)` | `(3,1)`, row 0 `[-1.5 -0.5 0.5 1.5]` | `(3,4)`, all rows the same | ½ gave the shape of the mean, not the result |
| P4b | `x - x.mean(axis=1)` | error | `shapes (3,4) (3,)` | ✓ |
| P5 | `(2,1,3)&(4,1)`, `(5,4)&(5,)`, `(6,1,4)&(3,4)`, `(2,3)&(3,2)` | `(2,4,3)`, error, `(6,3,4)`, error (both dims) | (not run) | ✓ |
| P6 | `a[:, None] > b`, `a=arange(3)`, `b=arange(4)` | `(3,4)`, lower-triangle, sum `3` | same | ✓ (wrote 0/1, it prints True/False) |
| C1 | divide by column max | `x / x[2]` | right values by luck | last row = col max only because `arange` increases; then `axis=1` (row max) ✗; final `x / x.max(axis=0)` ✓ |
| C2 | add `[100,200,300]` by row to last 2 cols of `y` | `y[:, [1,2]] + ...` | not written back, wrong cols | final `y[:, [2,3]] += [[100],[200],[300]]` ✓ |
| C3 | count elements > row mean | `(...).sum(axis=1, keepdims=True)` | `(3,1)` | final without 2nd `keepdims` -> `[2 2 2]` `(3,)` ✓ |
| C4 | table `a[i]*10 + b[j]` | (skipped prediction) | `a * 10 + b` -> `shapes (3,) (4,)` | `a[:, None] * 10 + b` ✓ |

## 8. Mistakes and fixes

- P4a: the result of broadcasting has the **stretched** shape (max per dim), not the shape of the smaller operand. 4 numbers in a row already means it can't be `(3,1)`.
- `y[:][1, 2]` = one element (`6`). Rows and columns go in **one** bracket: `y[rows, cols]`. Same trap as 11-25 R1 (`x[:][2]` = row, `x[:, 2]` = column).
- Columns are 0-indexed: the last two of 4 are `[2, 3]`; slicing `2:` / `-2:` avoids counting.
- `axis` direction again (C1): `axis=k` deletes the k-th number in the shape. Per column -> the row count disappears -> `axis=0`.
- Coincidence check: a result that "looks right" on `arange` data can be wrong in general. Test on shuffled data.
- `[[...], None]` and `list.reshape`: see §6.
- Write arrays with all brackets: `array([[...], [...]])`; boolean results print `True/False`.

## 9. Vocabulary

| Term | Meaning |
| --- | --- |
| arithmetic operations | 算术运算 (element-wise `+ - * /`) |
| subject to (certain constraints) | 在满足（某些限制）的前提下 |
| constraint | 限制、约束条件 |
| provide a means of doing X | 提供一种做 X 的手段 (means = method, same in plural) |
| vectorize | 向量化 (no explicit Python loop) |
| looping occurs in C | 循环在 C 层面执行 (occur = 发生) |
| lead to | 导致 |
| inefficient | 低效的 (opposite of efficient) |
| be stretched into | 被拉伸成 |
| analogy | 类比、比喻 |
| conceptual | 概念上的 (not what really happens) |
| as memory and computationally efficient as possible | 在内存和计算上都尽可能高效 |
| element-wise | 逐元素地; for shapes: dimension by dimension |
| trailing (dimension) | 最后的（那一维） |
| i.e. / e.g. | 也就是 / 例如 |
| rightmost | 最右边的 |
| work its way left | 一步步往左推进 |
| compatible / incompatible | 兼容的 / 不兼容的 |
| expand | 扩展 (cf. `np.expand_dims`) |
| broadcastable | 可以广播的 |
| valid | 有效的 |
| act like | 表现得像（其实不是） |
| therefore | 因此 |
| outer product | 外积 (u vᵀ) |
| index operator | 索引运算符 (what goes in `[]`) |
| yield | 产生、得出 (≈ produce) |
| inhomogeneous | 不规则的、长短不一的 |

## 10. Self-check (explain out loud)

- [ ] State the broadcasting rule in 3 steps. What does a missing dimension count as?
- [ ] `(8,1,6,1)` with `(7,1,5)`: result shape? Which array is stretched in each position?
- [ ] Why is `2` vs `4` not compatible?
- [ ] Does broadcasting copy the smaller array? Then how can it still waste memory?
- [ ] Is a 1-D array stretched as a row or a column? How do I make it a column?
- [ ] Per-row vs per-column: when is `keepdims=True` required, and why?
- [ ] `y[:, [2, 3]] += 1` vs `c = y[:, [2, 3]]; c += 1`: which one changes `y`?

## 11. Next steps

- Optional: run P5 with `np.ones(...)` to confirm the 4 shapes
- Commit: `week1: add broadcasting study note (2026-11-26)`
- 11-27, Session 3 (part 1): skeleton + pytest for the 6 small functions, write the easy ones first
- 11-28, Session 3 (part 2): "standardize each row" (`keepdims` + broadcasting, just like P4a), `pytest -q` all pass, oral self-check -> week 1 done
