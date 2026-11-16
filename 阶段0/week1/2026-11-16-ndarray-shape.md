# NumPy Basics: ndarray, indexing, axis and creating arrays

- Date: 2026-11-16
- Stage: Stage 0 / Week 1 / NumPy basics
- Source: NumPy docs, *NumPy: the absolute basics for beginners* (from the opening up to "Adding, removing, and sorting elements")
- Study time: about 1.5 hours

## 1. What I learned

### 1.1 An ndarray is a grid in space

- 1D: a row of cells, e.g. `[1, 2, 3]`
- 2D: a table with rows and columns
- 3D: several tables stacked like pages in a book (`shape=(3, 4, 5)` reads as 3 pages, each with 4 rows and 5 columns)
- The same idea extends to any number of dimensions; the "n" in `ndarray` means the number of dimensions is not fixed

### 1.2 The shape must be rectangular, not jagged

- Every row must have the same length; every page must have the same number of rows and columns
- Unlike a C array of pointers (where each row can have a different length), an ndarray is a regular multi-dimensional block
- Only a regular grid can be described by `shape` and `axis`, and vectorization and broadcasting later depend on this

### 1.3 Outer layer vs inner layer (example: `[[1, 2, 3], [4, 5]]`)

- Outer layer: the outermost pair of brackets. The number of elements inside = the number of rows
- Inner layer: the brackets of each row. The number of elements inside = the number of columns
- Rule of thumb: the outer layer decides how many rows, the inner layer decides how many columns per row
- `np.array([[1, 2, 3], [4, 5, 0]]).shape` is `(2, 3)`: the 2 comes from the outer layer, the 3 from the inner layer

### 1.4 `ndim` vs `shape`

- `ndim` is how many indices you need to locate one value
- For `[1, 2, 3]`, `ndim` is 1 and `shape` is `(3,)`
- This is not the same as "a 3-dimensional vector" (a vector with 3 components)
- A shape is a tuple. `(3,)` needs the comma; `(3)` is just the integer 3

### 1.5 The array is mutable

- Mutable: can be changed in place (`list`, `dict`, `ndarray`). Immutable: cannot (`tuple`, `str`, `int`)
- `a[0] = 99` changes the array directly
- A `tuple` raises `TypeError: 'tuple' object does not support item assignment` for the same operation

### 1.6 Slicing: `a[start:stop:step]`

- `start` is included, `stop` is excluded (left-closed, right-open, like C++ `[begin, end)`)
- Length = `stop - start`, and `a[:k]` plus `a[k:]` split an array with no overlap and no gap
- Omitted `start` means from the beginning, omitted `stop` means to the end, omitted `step` means 1
- Negative index: `-1` is the last element
- Negative step walks backwards, so `a[::-1]` reverses the array

### 1.7 Slice = view, not copy

- A NumPy slice is a **view**: it shares memory with the original, so changing one changes the other
- A Python `list` slice is a copy, so this is a difference to remember
- `.copy()` gives an independent array
- `b.base is a` is `True` when `b` is a view of `a`; `np.shares_memory(a, b)` checks the same thing
- `is` asks "same object?" and `==` asks "same content?"
- Own data (a fresh array or a `.copy()`) has `.base` equal to `None` (to verify: `a.base is None`, `c.base is None`)

### 1.8 Axis

- In NumPy, a dimension is sometimes called an **axis** (plural: axes)
- `axis=0` is the first dimension, `axis=1` the second, and so on
- Rule: `sum(axis=k)` adds up the k-th index (it "runs through" that index) and keeps the other indices fixed, so the k-th number disappears from the shape
- Example, `m = [[1,2,3],[4,5,6]]`, shape `(2, 3)`:
  - `axis=0` gives `[5 7 9]` (each column added from top to bottom), shape `(3,)`
  - `axis=1` gives `[6 15]` (each row added from left to right), shape `(2,)`
- No `axis` argument: all elements are added into one number, shape `()`
- C-style picture: `axis=0` fixes the column `j` and loops over the row `i`; `axis=1` fixes the row `i` and loops over the column `j`

### 1.9 Indexing an element: one pair of brackets, commas between axes

- Nested list: `lol[1][2]` (two pairs of brackets). Array: `m[1, 2]` (one pair, one number per axis)
- The numbers follow the axes in order: `axis=0`, `axis=1`, `axis=2`, so a 3D array is `x[i, j, k]`

### 1.10 Array attributes (no parentheses)

| Attribute | Meaning | For `[[1,2,3],[4,5,6]]` |
| --- | --- | --- |
| `ndim` | number of axes | `2` |
| `shape` | length of each axis (a tuple) | `(2, 3)` |
| `size` | total number of elements = product of the shape | `6` |
| `dtype` | data type of the elements | `int64` |

- Attribute = stored information, read without `()`. Method = an action, called with `()`, like `m.sum()`
- All elements of an array share one dtype. Mixing ints and floats promotes everything to float: `np.array([1, 2, 3.5]).dtype` is `float64`
- Float arrays print with a dot (`1.`), integer arrays without

### 1.11 Creating arrays

| Function | Result |
| --- | --- |
| `np.zeros(3)` | `[0. 0. 0.]` (float by default) |
| `np.ones((2, 2))` | 2 x 2 array of `1.` |
| `np.empty(shape)` | memory allocated but **not initialized**, values are arbitrary, never read before assigning |
| `np.arange(2, 10, 3)` | `[2 5 8]` (stop is excluded) |
| `np.linspace(0, 1, 5)` | `[0.   0.25 0.5  0.75 1.  ]` (5 evenly spaced points, stop is included) |

- The shape argument is a tuple, so `np.ones((2, 2))` has two pairs of parentheses; a single int `3` means `(3,)`
- Printed format: elements are separated by spaces, not commas (a list uses commas); a 2D array prints one row per line; columns are aligned

### 1.12 Sorting and concatenating

- `np.sort(a)` returns a sorted **copy**; `a` is unchanged. `a.sort()` sorts in place and returns `None`, so never write `a = a.sort()`
- `np.concatenate((x, y))` takes a **tuple** of arrays and builds a new array; `(3,)` + `(3,)` gives `(6,)`
- Array size is fixed, so "adding" and "removing" really mean building a new array
- Related functions (just know they exist):
  - `argsort`: indirect sort, returns the indices that would sort the array; `a[np.argsort(a)]` is the sorted array
  - `lexsort`: indirect stable sort on several keys (the last key has the highest priority)
  - `searchsorted`: position where a value would be inserted into a sorted array (binary search)
  - `partition`: partial sort, only the k-th element is in its final place, smaller ones before and larger ones after, no order inside each side

## 2. Experiment log (predict first, then run)

| Code | My prediction | Actual result | Why |
| --- | --- | --- | --- |
| `np.array([[1,2,3],[4,5]])` | (to fill in) | `ValueError: ... inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.` | The outer layer has 2 rows, but the rows have different lengths (3 and 2), so they cannot form a rectangle |
| `a = np.array([1,2,3]); a[0] = 99` | not recorded | `[99  2  3]` | The array is mutable |
| `t = (1,2,3); t[0] = 99` | not recorded | `TypeError: 'tuple' object does not support item assignment` | A tuple is immutable |
| `a[1:4]`, `a[:2]`, `a[3:]` on `[1,2,3,4,5]` | not recorded | `[2 3 4]`, `[1 2]`, `[4 5]` | `stop` is excluded |
| `a[-1]`, `a[-3:]`, `a[::2]`, `a[::-1]` | not recorded | `5`, `[3 4 5]`, `[1 3 5]`, `[5 4 3 2 1]` | Negative index and step |
| `b = a[1:4]; b[0] = 99; print(a)` | `a` becomes `[1 99 3 4 5]` (correct) | `[ 1 99  3  4  5]` | `b` is a view, `b[0]` is `a[1]` |
| `c = a[1:4].copy(); c[0] = 99; print(a)` | not recorded | `c` is `[99 3 4]`, `a` unchanged | `.copy()` makes independent data |
| `b.base is a`, `np.shares_memory(a, b)` | not recorded | `True`, `True` | Same memory |
| `m.sum(axis=0)`, `m.sum(axis=1)` | not recorded | `[5 7 9]`, `[ 6 15]` | See 1.8 |
| `x = np.zeros((2,3,4))`, shape after `sum(axis=0/1/2)` | `(3,4)`, `(2,4)`, `(2,3)` (correct) | `(3, 4)`, `(2, 4)`, `(2, 3)` | The k-th number of the shape disappears |
| `x.sum()` | not recorded | `np.float64(0.0)` | All elements added; `np.zeros` is all 0 and float by default |
| `y = np.ones((2,3,4)); y.sum()` | 24 (correct) | `24.0` | 2 x 3 x 4 = 24 ones |
| `y.sum(axis=0)` | each position 24, shape `(3,4)` | each position `2.`, shape `(3, 4)` | Only the 2 values along axis 0 are added, so 1 + 1 = 2 |
| `m.ndim`, `m.shape`, `m.size`, `m.dtype` for `[[1,2,3],[4,5,6]]` | not recorded | `2`, `(2, 3)`, `6`, `int64` | `size` = 2 x 3; integers without a dot give an integer dtype |
| `np.array([1, 2, 3.5]).dtype` | not recorded | `float64` | Mixed ints and floats are promoted to float |
| `np.zeros(3)`, `np.ones((2, 2))`, `np.arange(2, 10, 3)`, `np.linspace(0, 1, 5)` | not recorded | `[0. 0. 0.]`, 2 x 2 of `1.`, `[2 5 8]`, `[0. 0.25 0.5 0.75 1.]` | `arange` excludes the stop, `linspace` includes it |
| `np.sort(a)` then `print(a)` for `a = [2,1,5,3,7,4,6,8]` | not recorded | sorted copy `[1 ... 8]`; `a` unchanged | `np.sort` returns a copy |
| `np.concatenate((x, y))` for `[1,2,3]` and `[4,5,6]` | not recorded | `[1 2 3 4 5 6]` | New array of shape `(6,)` |
| `np.argsort(a)` and `a[np.argsort(a)]` for `[30,10,20]` | not recorded | `[1 2 0]` and `[10 20 30]` | `argsort` returns indices, not values |

Reading the first error: "inhomogeneous" means not uniform. "detected shape was (2,)" means NumPy counted 2 elements in the outer layer. "+ inhomogeneous part" means the inner lengths do not match.

Fixes: pad the short row so the array is regular, or use `dtype=object` to force it (not a normal numeric array, ignore for now).

## 3. Mistakes and fixes

- `NameError: name 'b' is not defined`: I only predicted `b = a[1:4]` in my head and never ran it, so the name did not exist. Define a variable before using it.
- `print(x.sum)` printed `<built-in method sum of numpy.ndarray object ...>`: I forgot the parentheses. `x.sum` is the method itself, `x.sum()` calls it.
- I predicted 24 for every position of `y.sum(axis=0)`; the correct value is 2. 24 is the total with no `axis`.
- Wrote shapes with blanks like `(2, , 5)`: the removed dimension should simply be dropped, and 5 was a typo for 4.

## 4. Vocabulary

| Term | Meaning |
| --- | --- |
| rectangular | every row has the same length (矩形的) |
| jagged | rows of uneven length (锯齿状的) |
| stacked | placed one on top of another (堆叠的) |
| arbitrary | any, not fixed (任意的) |
| generalized to | extended to cover more cases (推广到) |
| inhomogeneous | not uniform (不一致的) |
| exploit (verb) | take advantage of (利用); as a noun in security it means a vulnerability exploit |
| mutable / immutable | can / cannot be changed after creation (可变 / 不可变) |
| tuple | an immutable sequence, written with parentheses (元组) |
| item assignment | `x[i] = value` (给元素赋值) |
| slice / slicing | taking a part with `a[start:stop]` (切片) |
| view / copy | shares memory / has its own memory (视图 / 拷贝) |
| axis / axes | one dimension of an array, seen as a direction (轴) |
| refer to A as B | call A by the name B (把 A 称作 B) |
| collapse | remove a dimension by combining its values (压掉) |
| dtype | the data type of the elements, e.g. `float64` |
| attribute | information stored on an object, read without `()` (属性) |
| element | one value in an array (元素) |
| a list of lists | a nested list (嵌套列表) |
| concatenate | join end to end (拼接) |
| indirect sort | returns the sorting indices, not the sorted values (间接排序) |
| stable sort | equal elements keep their original order (稳定排序) |
| key | the criterion used for sorting (排序依据) |
| type promotion | mixed types are converted to a wider common type (类型提升) |
| in addition to | besides (除了……之外) |
| covers | deals with, includes (涵盖、介绍) |

## 5. Question log

- Q: What are the outer and inner layers?
  A: The outer layer is the outermost brackets, which gives the number of rows. The inner layer is each row's own brackets, which gives the number of columns.
- Q: Is the Python `>>>` prompt NumPy?
  A: No. It is the Python interpreter. NumPy is a third-party library: run `import numpy as np`, and if it is missing, `pip install numpy`. (Confirmed installed.)
- Q: What is a tuple?
  A: An immutable sequence in Python, written `(1, 2, 3)`. NumPy shapes are tuples.
- Q: What does `b.base is a` mean?
  A: `.base` is the array that owns the data; `is` checks object identity. `True` means `b` is a view of `a`.
- Q: Did I mix up rows and columns in `sum(axis=...)`?
  A: Think "which index is added up": `axis=0` adds over the row index `i`, `axis=1` adds over the column index `j`.
- Q: What is `np.float64(0.0)`?
  A: A NumPy scalar of type 64-bit float. Newer NumPy (2.x) shows the type name in the repr.
- Q: What is the format of the output `[0. 0. 0.]`?
  A: It is NumPy's display format, not Python syntax. `0.` is `0.0` (float), elements are separated by spaces, and 2D arrays print one row per line.
- Q: Is it feasible to read the docs section by section? Will I miss the context?
  A: Read the whole *absolute basics* page in order (it builds step by step), then skim *Quickstart* and read *Broadcasting* as in the Week 1 plan. Skim the headings first for the big picture. Look up other docs only when needed.

## 6. Self-check (explain out loud)

- [ ] Why must an ndarray be rectangular?
- [ ] How do I read `shape=(3, 4, 5)`? Which dimension does `axis=0` collapse?
- [ ] What is the difference between `ndim` and `shape`?
- [ ] Without notes, give two ways to make `np.array([[1,2,3],[4,5]])` not raise an error
- [ ] Why does changing `b = a[1:4]` change `a`, and how do I avoid it?
- [ ] For `shape=(2, 3, 4)`, what is the shape after `sum(axis=1)`? And after `sum()`?
- [ ] What is the difference between `is` and `==`?
- [ ] What do the numbers in `m[1, 2]` mean, and how does it differ from `lol[1][2]`?
- [ ] What happens to the dtype when ints and floats are mixed in one array?
- [ ] Does `np.sort(a)` change `a`? What about `a.sort()`?
- [ ] Which one includes the stop value, `np.arange` or `np.linspace`?

## 7. Next steps

- Continue the *absolute basics* page from "How do you know the shape and size of an array?"
- After the page: skim *Quickstart*, read *Broadcasting* (Week 1 plan)
- Run the exercises I have not run yet: `m[0, 1]`, `m[1, -1]`, `m[1][2]`, `lol[1, 2]` (a plain list), `np.empty((2, 3))`, `a.base is None`, `c.base is None`
- Write my own shape-prediction questions: predict first, then run (Week 1, task 2)
- Fill in the missing prediction for the jagged-array row in section 2
