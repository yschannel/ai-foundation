# NumPy Basics: `sum` with `axis` and `keepdims` (Session 1 done)

- Date: 2026-11-25
- Stage: Stage 0 / Week 1 / Session 1 (quick review of 11-16 to 11-18 + block 4 + 8 summary questions)
- Source: NumPy docs, [`numpy.sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html): parameters `a`, `axis`, `keepdims`, Returns, Examples
- Study time: about 1 hour
- Back after a break (11-19 to 11-24); schedule moved, see README

## 1. Review: two old mistakes came back

**Rule (view vs copy):** the **kind of index** decides, not whether the chosen elements sit next to each other.

| Index | Example | Result |
| --- | --- | --- |
| int, slice, `None` | `s = x[2, ::2]` | view: `s[0] = 0` changes `x` |
| boolean array, int array | `c = x[x > 9]` | copy: `c[:] = 0` leaves `x` alone |

To change `x` itself: `x[x > 9] = 0` (`[]` on the left of `=` means "write into").

> My mistake (again, same as 11-18 P4): predicted `x[2, 3]` becomes 0 after `c[:] = 0`. It stays 11.

**Rule (`nonzero`):** `x[mask]` gives **values**; `np.nonzero(mask)` gives **positions**, a tuple with one index array per axis.

```
x = np.arange(12).reshape(3, 4)
x[x > 9]            -> [10 11]                          values
np.nonzero(x > 9)   -> (array([2, 2]), array([2, 3]))   rows, cols -> (2,2) (2,3)
x[np.nonzero(x > 9)] == x[x > 9]
```

> My mistake: answered `array([10, 11])` for `np.nonzero(x > 9)`.

## 2. `axis`: which axes to sum over

**Rule:** the number at position k disappears from the shape. A tuple removes several at once.

| `axis` | Meaning | For shape `(2, 3, 4)` |
| --- | --- | --- |
| `None` (default) | sum all elements -> scalar | `()` |
| int `k` | sum along axis k | `axis=1` -> `(2, 4)` |
| negative | count from the last axis | `axis=-1` = `axis=2` -> `(2, 3)` |
| tuple of ints | sum along every listed axis | `axis=(0, 2)` -> `(3,)` |
| out of range | error | `m.sum(axis=2)` on a 2-D `m` -> `AxisError` |

## 3. `keepdims`: keep the reduced axis as length 1

**Rule:** without `keepdims` the number k is **deleted**; with `keepdims=True` it is **replaced by 1**. Same values, different shape.

```
m = [[0 1 2]
     [3 4 5]]                     shape (2, 3)

m.sum(axis=1)                 -> [ 3 12]        (2,)
m.sum(axis=1, keepdims=True)  -> [[ 3]
                                  [12]]         (2, 1)

x.sum(axis=(0, 2), keepdims=True).shape  -> (1, 3, 1)
```

Why it matters: a `(2, 1)` result lines up with the original `(2, 3)` array, so `m - row_sums` works row by row (broadcasting, next session). This is what "standardize each row" needs.

## 4. Returns

- An array with the summed axes removed (or set to 1 with `keepdims=True`)
- `axis=None` or a 0-d input -> a **scalar**, e.g. `np.int64(3)`
- The result is always **new data** (a copy): the sums cannot be read from the original buffer by changing the header, so `s = m.sum(axis=0); s[0] = 100` leaves `m` unchanged

## 5. dtype in reductions

- `dtype` = the data type of the elements (see 11-16, §1.10)
- `np.ones` is `float64` by default, so its sums print as `3.`, not `3`
- A boolean array sums `True` as 1 and `False` as 0, giving an int: `(m > 2).sum()` -> `3` = "how many elements are > 2"
- `mean` always gives a float, even for an int array: `m.mean(axis=1)` -> `[1. 4.]`
- `mean` takes the same `axis` / `keepdims` as `sum` (and so do `max`, `min`, `std`)

## 6. Experiment log

| # | Code | My prediction | Actual | Note |
| --- | --- | --- | --- | --- |
| R1 | `x[:][2]`, `x[:, 2]` | `[8 9 10 11]` `(4,)`, `[2 6 10]` `(3,)` | same | ✓ |
| R2 | `np.ones((2,3,4)).sum(axis=1)` | `(2, 4)`, all 3 | same | ✓ |
| R3 | `b = a.reshape(2,3); b[1,0] = -7`, then `a` | `(6,)`, `[0 1 2 -7 4 5]` | same | ✓ |
| R4 | `expand_dims(x, axis=1).shape`, `a[:, None].shape` | `(3,1,4)`, `(6,1)` | same | ✓ |
| R5 | `x[(x > 3) & (x % 2 == 0)]` | `[4 6 8 10]` `(4,)` | same | ✓ |
| R6 | `c = x[x > 9]; c[:] = 0; x[2,3]` / `s = x[2, ::2]; s[0] = 0; x[2,0]` | `0` / `0` | `11` / `0` | ✗ / ✓ §1 |
| R7 | `np.nonzero(x > 9)` | `array([10, 11])` | `(array([2,2]), array([2,3]))` | ✗ §1 |
| R8 | `a[1::2]`, contiguous? | `[1 3 5]`, no | `[1 3 5]` | ✓ (`reshape(4,-1)` and the flag: run later) |
| S1 | `m.sum(axis=0)`, `m.sum(axis=1, keepdims=True)` | `[3 5 7]` `(3,)`, `[[3] [12]]` `(2,1)` | same | ✓ |
| S2 | `x.sum(axis=-1).shape`, `x.sum(axis=(0,2)).shape` | `(2,3)`, `(3,)` | same | ✓ |
| S3 | same with `keepdims=True` | `(2,1,4)`, `(1,3,1)` | same | ✓ |
| S4 | `m.sum(axis=2)` | error, "out_of_range" | not run yet | expected `AxisError` |
| S5 | `np.ones((3,4)).sum(axis=0, keepdims=True)` | `[[3] [3] [3] [3]]`, `(1,4)` | `[[3. 3. 3. 3.]]` | ½ shape right; values written as a column; dtype float |
| S6 | `(m > 2).sum()`, `(m > 2).sum(axis=1)` | `3`, `[0 3]` | same | ✓ |
| S7 | `s = m.sum(axis=0); s[0] = 100`, `m` | unchanged, copy | unchanged | ✓ |
| S8 | `m.mean(axis=1, keepdims=True)` | `[[3] [12]]`, `(2,1)` | `[[1.] [4.]]` | ½ shape right; wrote the sums, not the means |

## 7. Mistakes and fixes

- Boolean index = copy, again (R6). Rule in §1.
- Mixed up "values" and "positions" for `nonzero` (R7).
- S5: wrote a `(1, 4)` result as a column `[[3] [3] [3] [3]]`, which is `(4, 1)`. Shape `(1, 4)` = one row with 4 numbers: `[[3. 3. 3. 3.]]`.
- S8: copied the sums instead of dividing: 3 / 3 = 1, 12 / 3 = 4, and the result is float.
- `NameError: name 'Ture' is not defined`: typo for `True`. Python's `Did you mean: 'True'?` hint helps.

## 8. Vocabulary

| Term | Meaning |
| --- | --- |
| along which a sum is performed | 沿着它进行求和 (perform = 执行) |
| axis / axes | one / several axes (轴；复数) |
| count from the last to the first | 从最后一个往前数 |
| specified | given explicitly (被指定的) |
| set A to B | 把 A 设成 B |
| reduce / reduction | combine the values along an axis into one (sum, mean, max) (归约) |
| be left in the result | 被保留在结果里 |
| broadcast against X | 和 X 一起做广播运算 |
| pass through to | 原样转交给 |
| sub-class | 子类 (inheritance, like C++/Java) |
| implement | 实现 |
| raise an exception | 抛出异常 (like `throw`) |
| with X removed | X 被去掉的状态 |
| scalar | a single number, not an array (标量) |
| a reference to | 对……的引用 (same object, no copy) |

## 9. Self-check (explain out loud)

- [ ] For shape `(2, 3, 4)`, what are the shapes after `sum(axis=-1)`, `sum(axis=(0, 2))`, and `sum(axis=(0, 2), keepdims=True)`?
- [ ] What exactly does `keepdims=True` change, and what does it not change?
- [ ] Why does "standardize each row" need `keepdims=True`?
- [ ] Is `m.sum(axis=0)` a view or a copy? Explain with "buffer + header".
- [ ] What does `(m > 2).sum()` count?
- [ ] Why is `m.mean(axis=1)` float even though `m` is int?
- [ ] `x[x > 9]` vs `np.nonzero(x > 9)`: values or positions?

## 10. Next steps

- Run what I skipped: `m.sum(axis=2)` (read the error name), `a.reshape(4, -1)`, `a[1::2].flags['C_CONTIGUOUS']`
- Session 1 is done
- 11-26, Session 2: broadcasting rules, 6 given questions + 4 of my own
