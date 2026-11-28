# Softmax: Definition, Properties, Numerical Stability

- Date: 2026-11-28
- Time spent: 2 h
- Stage: Stage 0 / Week 2 / Session 1: softmax concept, `np.exp`, overflow, hand calculations (no code yet)
- Reading: [d2l 3.1.1.3 The Softmax](https://d2l.ai/chapter_linear-classification/softmax-regression.html#sec-softmax-operation) · [numpy.exp](https://numpy.org/doc/stable/reference/generated/numpy.exp.html)
- Result: round 1 = 2 right, 1 half, 2 wrong; round 2 = 4 right, 1 half. Concepts are solid; weak spot is **math vs what the computer actually computes**.

## 1. Warm-up (week 1 review)

- `x` `(5, 5)`, `x - x.mean(axis=1)`: no error, but each `x[i, j]` minus the mean of **row j**. Fix: `keepdims=True`. ✓
- `x.max` = the method object (not called). `x.max()` = the max value. `max()` = Python built-in, needs an argument. `np.max(z)` = same as `z.max()`.
- `x(mask)` -> `TypeError: 'numpy.ndarray' object is not callable`. `()` calls, `[]` selects.

## 2. From linear layer to probabilities

```
x (4,) --linear layer: o = Wx + b--> o (3,) logits --softmax--> y_hat (3,) probabilities
         W (3, 4), b (3,)
```

- **Linear layer**: one matrix multiply plus a bias. Each row of `W` gives one class score.
- **Affine**: `Wx` alone is linear (f(0) = 0); `Wx + b` adds a shift, so strictly it is affine. In deep learning both names are used.
- **Logits / affine scores**: any real numbers, can be negative or > 1, do not sum to 1. So they are not probabilities.

## 3. Softmax

$$\hat y_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}$$

1. `exp` makes every term positive (and keeps the order, it is increasing).
2. Dividing by the sum makes the terms add up to 1.

Hand example, `o = [2.0, -1.0, 0.5]`: exp = `[7.389, 0.368, 1.649]`, sum 9.406, result `[0.786, 0.039, 0.175]`.

## 4. Three properties

All come from e^(a+b) = e^a · e^b ("exponentials turn addition into multiplication").

### 4.1 Shift invariance: only differences matter

softmax(o + c) = softmax(o): exp(c) appears in both numerator and denominator and cancels.

Equivalent form (holds for **any** number of classes):

$$o_i - o_j = \log\frac{\hat y_i}{\hat y_j}$$

Logits differing by 1 -> probability ratio e. Differing by ln 2 -> ratio 2, so `softmax([5, 5 + ln 2]) = [1/3, 2/3]`, NOT `[1/2, 1/2]` (my mistake: only a difference of 0 gives equal probabilities).

### 4.2 Two classes = logistic regression

$$\hat y_1 = \frac{1}{1 + e^{-(o_1 - o_2)}} = \sigma(o_1 - o_2)$$

- Only one number matters: t = o₁ − o₂. Sigmoid is softmax with q = 2.
- t is the **log-odds**: t = log(p / (1 − p)). This function is called the **logit**: origin of the word.
- logit and sigmoid are inverses: sigmoid = score -> probability, logit = probability -> score.
- Symmetry: σ(−t) = 1 − σ(t), logit(1 − p) = −logit(p). E.g. σ(2) ≈ 0.881, σ(−2) ≈ 0.119.
- Use **t** for the input of σ, not **p**: p means a probability, which lives on the other side.

| p | odds p/(1−p) | logit |
| --- | --- | --- |
| 0.5 | 1 | 0 |
| 0.881 | 7.39 | 2 |
| 0.119 | 0.135 | −2 |

### 4.3 Clean gradient (preview, derive in weeks 3–4)

With cross-entropy loss: ∂loss/∂o = ŷ − y, shape `(q,)`. Same pattern as σ(Xw) − y in logistic regression.

## 5. Inference: argmax

- Softmax keeps the order, so argmax ŷ = argmax o. At inference time we can skip softmax.
- `np.argmax` returns the **position**, not the value. `np.max` returns the value.
- `x` `(4, 3)` (4 samples, 3 classes): `np.argmax(x, axis=1)` -> `(4,)`, the predicted class of each sample.

## 6. Shift vs scale

- `softmax([1, 2, 3])` ≈ `[0.090, 0.245, 0.665]`
- `softmax([2, 4, 6])` ≈ `[0.016, 0.117, 0.867]`: differences doubled -> more concentrated.
- Adding a constant changes nothing; multiplying changes the differences, so it changes the result.

## 7. `np.exp`

- **element-wise**: `out[i, j] = exp(x[i, j])`. Output shape = input shape. No `axis` needed.
- Reductions (`sum`, `max`, `argmax`) are the opposite: they remove an axis. In softmax, only the sum step needs `axis` / `keepdims`.
- Integer input -> `float64` output (e¹ is not an integer). Same reason as `mean` on ints.
- 1-D input `(3,)` -> output `(3,)`, NOT `(1, 3)` (my mistake again: `(d,)` vs `(1, d)`).
- Scalar in -> scalar out. No `out` given -> a freshly-allocated array (a copy, input unchanged).
- `where=mask` computes only where True; use it together with `out`, otherwise False positions are uninitialized. Not needed this week.

## 8. Overflow and numerical stability

| Expression | Result | Why |
| --- | --- | --- |
| `np.exp(1000.0)` | `inf` + `RuntimeWarning: overflow` | > float64 max (~1.8e308); limit is about exp(710) |
| `np.exp(-1000.0)` | `0.0`, no warning | underflow |
| `z = [1000, 1001, 1002]`, direct formula | `[nan, nan, nan]` | exp -> inf, inf / inf -> nan |
| same `z`, subtract max first | `[0.090, 0.245, 0.665]` | logits become `[-2, -1, 0]`, same differences as `[1, 2, 3]` |

My mistake: I answered the **math** result for the direct formula. The question was what the **computer** gets.

Why subtract the **max**:

1. Result unchanged (shift invariance).
2. Every exponent ≤ 0, so every exp is in (0, 1]: no overflow.
3. The largest term becomes exactly exp(0) = 1.0, so the denominator ≥ 1: never divide by 0.

Subtracting the **min** does not work: `[0, 1000]` minus its min is still `[0, 1000]`.

Softmax output can be exactly 0 on a computer **without any inf**: `softmax([0, -1000])` -> `[1.0, 0.0]`, because exp(−1000) underflows. This matters for cross-entropy: log(0) = −inf (session 3).

### Warnings are not reliable

The second `exp` overflow did not print a warning: Python shows the same warning at the same location (`<stdin>:1`) only once. The two warnings map to the two steps of nan:

- `overflow encountered in exp` -> inf
- `invalid value encountered in divide` -> inf / inf = nan

In tests, check actively: `assert np.isfinite(out).all()`.

## 9. Mistakes to watch in session 2

- [ ] `(d,)` vs `(1, d)`: 1-D has one number in its shape. (Second time.)
- [ ] Math answer vs computer answer: always ask "does any intermediate value overflow or underflow?"
- [ ] Row-wise softmax: `(2, 3) / (2,)` errors, but `(3, 3) / (3,)` silently computes the wrong thing. Test with a **square** input and rows on different scales.
- [ ] Only difference 0 means equal probabilities.

## 10. Vocabulary

| Term | Meaning |
| --- | --- |
| logit | 原始分数；也指 log-odds 函数 log(p/(1−p)) |
| affine | 仿射的（线性变换 + 平移） |
| linear layer | 线性层，o = Wx + b |
| probabilities | 概率（复数） |
| nothing stops A from doing B | 没有什么能阻止 A 做 B，即 A 完全可能 B |
| turn negative | 变成负数 |
| exceed | 超过 |
| exponentiate | 取指数 |
| logarithm | 对数，缩写 log；ML 中默认以 e 为底 |
| endow A with B | 赋予 A 某种 B |
| foundational | 基础的、根本的 |
| shift invariance | 平移不变性 |
| matter (v.) | 要紧、有影响 |
| scalar constant | 标量常数 |
| entry | （向量/矩阵的）元素 |
| numerator / denominator | 分子 / 分母 |
| odds / log-odds | 比值 / 比值的对数 |
| safeguard | 保护措施 |
| floating-point overflow | 浮点数溢出 |
| underflow | 下溢（太小变成 0） |
| reduction to | 归结为、退化为 |
| solely | 仅仅 |
| thus | 因此 |
| identical to | 与……完全相同 |
| inference time | 推理阶段（参数固定，只做预测） |
| argmax | 最大值所在的位置 |
| clean (gradient) | 简洁的 |
| under (maximum likelihood) | 在……框架下 |
| element-wise | 逐元素的 |
| ufunc | universal function，逐元素计算的 NumPy 函数 |
| array_like | 能转成数组的对象 |
| freshly-allocated | 新分配（内存）的 |

## 11. Next steps

- Commit: `week2: add softmax concept note (2026-11-28)`
- Session 2 (11-30): first 5 minutes, re-answer §8 table and the `[5, 5 + ln 2]` question without looking. Then write 1-D and row-wise softmax with loop versions and tests.
