# NumPy Basics: slicing step, boolean indexing, nonzero and tuples

- Date: 2026-11-18
- Stage: Stage 0 / Week 1 / Session 1 (block 3 done; warm-up = leftovers from 11-17)
- Source: *NumPy: the absolute basics for beginners*, "Indexing and slicing"
- Study time: about 1.5 hours

## 1. Slicing with a step: `start:stop:step`

**Rule:** any part can be left out. Defaults are `start=0`, `stop=end`, `step=1`.

```
a = [0 1 2 3 4 5]
a[::2]   = a[0:6:2]  -> [0 2 4]
a[1::2]              -> [1 3 5]
a[::-1]              -> [5 4 3 2 1 0]
```

> My mistake: read `a[::2]` as a reshape. `::` is slicing; it picks elements and does not rearrange the shape.

## 2. Buffer + header, with strides

- **buffer**: the block of memory that holds the numbers.
- **header**: how to read the buffer: `shape`, `dtype`, **strides** (bytes to jump for one step along each axis).

```
memory:     [0][1][2][3][4][5]
a[::2]:      ^     ^     ^        gaps -> C_CONTIGUOUS False
.copy():    [0][2][4]             packed -> C_CONTIGUOUS True

a.strides       -> (8,)    one int64 = 8 bytes
a[::2].strides  -> (16,)   jump 2 elements
```

`a[::2]` is a view: same buffer, new header (stride 16).

## 3. `reshape` leftovers

| Code | Result | Why |
| --- | --- | --- |
| `a.reshape(7)` | Error | size must stay 6 |
| `a.reshape(-1, 2)` | `(3, 2)` | 6 / 2 = 3 |
| `a.reshape(4, -1)` | Error | 6 / 4 is not an integer (cannot be worked out) |
| `a.reshape(-1, -1)` | Error | two unknowns, many answers (not unique) |
| `a.reshape(-1)` | `(6,)` | flatten |

## 4. Boolean indexing

**Rule 1:** a comparison works element by element and keeps the shape.
**Rule 2:** a boolean array inside `[]` keeps only the `True` positions.
**Rule 3:** a mask with the **same shape** as the array gives a **1-D** result (True positions do not form a rectangle; read in C order).

```
x = [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]                        shape (3, 4)

(x % 3 == 0).shape   -> (3, 4)            Rule 1
x[x % 3 == 0]        -> [0 3 6 9]  (4,)   Rule 3
x[~(x > 5)]          -> [0 1 2 3 4 5]
```

- The mask is only a "tick list" (no values). `x[...]` says **which array** to take from.
- So the mask can come from another array of matching shape: `scores[names == 'Bob']`, later `X[y == 1]`.
- A mask of shape `(3,)` on a `(3, 4)` array selects **rows** (along axis 0): `x[[True, False, True]]` -> rows 0 and 2, shape `(2, 4)`.

## 5. Combining conditions: `&` `|` `~` and parentheses

**Rule:** wrap every condition in parentheses; use `&` `|` `~`, never `and` `or` `not`.

```
x > 2 & x < 9                       & binds tighter than > and <
= x > (2 & x) < 9                   chained comparison
= (x > (2 & x)) and ((2 & x) < 9)   Python's 'and' asks "is this whole array True?"
-> ValueError: The truth value of an array with more than one element is ambiguous
```

`&` / `|` only build a new boolean array; `[]` does the selecting.

> My mistake: predicted `x[x > 2 & x < 9]` runs.

## 6. View or copy: the refined rule

| Indexing with | Example | Result |
| --- | --- | --- |
| int, slice, `None` | `x[1]`, `x[::2]`, `x[:, None]` | always **view** |
| boolean array, int array | `x[x > 8]`, `x[b]` | always **copy** |

NumPy decides by the **kind of index**, not by whether the chosen elements happen to sit next to each other.

```
c = x[x > 8]; c[0] = -1    -> x unchanged (c is a copy)      x[2, 1] = 9
x[x > 8] = -1              -> x changed (assignment into x)  x[2] = [8 -1 -1 -1]
```

`[]` on the left of `=` means "write into", not "take out". Example: `x[x < 0] = 0` is ReLU.

> My mistake: predicted `c` is a view. 9, 10, 11 are contiguous, but boolean indexing always copies.

## 7. `np.nonzero`: positions instead of values

**Rule:** returns the indices of non-zero elements; `False` = 0, `True` = 1, so on a mask it gives the `True` positions. It returns **a tuple with one index array per axis**.

```
a = [[1 2 3 4] [5 6 7 8] [9 10 11 12]]
b = np.nonzero(a < 5)    -> (array([0, 0, 0, 0]), array([0, 1, 2, 3]))
                                    rows                  cols
read vertically: (0,0) (0,1) (0,2) (0,3)

a[b]                     -> [1 2 3 4]    same as a[a < 5]
np.nonzero(a > 10)       -> (array([2, 2]), array([2, 3]))
np.nonzero(a == 42)      -> (array([], dtype=int64), array([], dtype=int64))
np.nonzero(x[0] > 1)     -> (array([2, 3]),)    1-D: one-element tuple
```

`zip` pairs the arrays point by point (like a zipper):

```python
for coord in list(zip(b[0], b[1])):
    print(coord)          # (np.int64(0), np.int64(0)) ...
```

`np.int64(0)` is a NumPy integer shown inside a tuple; same value as `0`.

## 8. Tuple

**Rule:** an ordered group of values that **cannot be changed**. Commas make it; `(7,)` is a tuple, `(7)` is just `7`.

| | list `[ ]` | tuple `( )` |
| --- | --- | --- |
| ordered, indexable | yes | yes |
| can be changed | yes | no |

In NumPy, "one thing per axis" is a tuple: `shape`, `strides`, the index in `x[1, 2]`, the result of `nonzero`. So `len(x.shape) == x.ndim` always.

> My mistake: said `np.zeros((2, 3, 4)).shape` is `(3,)`. It is `(2, 3, 4)`; `len` = 3 = `ndim`.

## 9. Python REPL errors I hit

| Error | Cause | Fix |
| --- | --- | --- |
| `IndentationError: expected an indented block` | loop body after `...` not indented | 4 spaces, then an empty line to finish |
| `IndentationError: unexpected indent` | indented line at `>>>`, outside any block | no indent at `>>>` |
| `NameError: ... Did you mean` | typo `list_of_coodinates` | check spelling first |

## 10. Experiment log

| # | Code | My prediction | Actual | Note |
| --- | --- | --- | --- | --- |
| W1 | `a.reshape(7)` | error | error | ✓ |
| W2 | `a[::2]`, `C_CONTIGUOUS` | `[[0,1],[2,3],[4,5]]` | `[0 2 4]`, `False` | ✗ §1 |
| W3 | `a[::2].copy()` contiguous | forgot | `True` | §2 |
| W4–W7 | reshape with `-1` | all right | – | ✓ §3 |
| W8 | `a.shape`, `a[0]` after `b[0,0]=100` | `(6,)`, `100` | same | ✓ fixed yesterday's mistake |
| P1 | `(x%3==0).shape`, `x[x%3==0]` | `(2,4)`, `(3,4)` | `(3,4)`, `[0 3 6 9]` `(4,)` | ✗ §4 |
| P2 | `x[~(x > 5)]` | – | `[0 1 2 3 4 5]` | – |
| P3 | `x[x > 2 & x < 9]` | runs | `ValueError` | ✗ §5 |
| P4 | `c = x[x > 8]; c[0] = -1; x[2, 1]` | `-1`, view | `9`, copy | ✗ §6 |
| P5 | `x[x > 8] = -1; x[2]` | `[8 -1 -1 -1]` | same | ✓ |
| P6 | `np.nonzero(x[0] > 1)` | `(array([2, 3]),)` | same | ✓ |
| P7 | `x[np.array([True, False, True])]` | shape `(2, 4)` | rows 0 and 2 | ✓ challenge |
| – | `np.nonzero(a > 10)` | `(array([2,2]), array([2,3]))` | same | ✓ |

## 11. Vocabulary

| Term | Meaning |
| --- | --- |
| straightforward | simple, no tricks (简单直接) |
| fulfill / meet / satisfy a condition | 满足条件 |
| make use of | use (利用) |
| logical operators | `&` `|` `~` (逻辑运算符) |
| specify whether | show whether (表明是否); compare "specify an argument" (指定) |
| categorical values | values that name a group: names, colors, labels (类别值) |
| index / indices | position number(s) (下标；复数) |
| one for each dimension | one per axis (每个维度一个) |
| represent | stand for (表示) |
| coordinates | (row, col) positions (坐标) |
| iterate over | go through each item in a loop (遍历) |
| ambiguous | has more than one meaning (有歧义的) |
| truth value | True / False of something (真值) |

## 12. Self-check (explain out loud)

- [ ] What does `a[1::2]` give for `a = np.arange(6)`? Why is it not contiguous?
- [ ] Why is `x[x % 3 == 0]` 1-D even though `x` is 2-D?
- [ ] Why does `x[x > 2 & x < 9]` fail? Rewrite it correctly.
- [ ] Which kinds of index give a view and which give a copy?
- [ ] What is the difference between `c = x[m]; c[0] = -1` and `x[m] = -1`?
- [ ] What does `np.nonzero` return for a 2-D mask, and why is it a tuple?
- [ ] Is `(7)` a tuple? What is `len(x.shape)` always equal to?

## 13. Next steps

- Block 4: `sum` with `axis` and `keepdims`
- Then the 8 summary prediction questions -> Session 1 done
- Habit: write a prediction for **every** item before running (P2 was left blank)
