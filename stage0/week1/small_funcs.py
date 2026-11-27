"""Stage 0 · Week 1 · Session 3: six small NumPy functions.

Rules
- Replace each `raise NotImplementedError` with your own code.
- No Python loops needed. Each one is 1-2 lines with what you learned
  (axis / keepdims / None / boolean indexing / broadcasting).
- Before writing, note the input shape and the output shape in a comment.
"""

import numpy as np


def row_mean(x):
    """Mean of each row.

    x: 2-D array, shape (n, d)
    returns: 1-D array, shape (n,)

    >>> row_mean(np.array([[1, 2, 3], [4, 5, 6]]))
    array([2., 5.])
    """
    # TODO
    return x.mean(axis=1)
    raise NotImplementedError


def col_max(x):
    """Maximum of each column.

    x: 2-D array, shape (n, d)
    returns: 1-D array, shape (d,)

    >>> col_max(np.array([[1, 9, 3], [4, 5, 6]]))
    array([4, 9, 6])
    """
    # TODO
    return x.max(axis = 0)
    raise NotImplementedError


def to_column(v):
    """Turn a 1-D vector into a column vector.

    v: 1-D array, shape (n,)
    returns: 2-D array, shape (n, 1), same values in the same order

    >>> to_column(np.array([1, 2, 3])).shape
    (3, 1)
    """
    # TODO
    return v[:,None]
    raise NotImplementedError


def count_even(x):
    """Number of even elements in an integer array of any shape.

    x: integer array, any shape (negative numbers and 0 count normally)
    returns: a plain int

    >>> count_even(np.array([[1, 2], [3, 4], [0, -6]]))
    4
    """
    # TODO
    return int((x % 2 == 0).sum())
    raise NotImplementedError


def above_mean(x):
    """All elements strictly greater than the mean of the WHOLE array.

    x: array, any shape
    returns: 1-D array of those elements, in row-major order
             (the order you get reading row by row)

    >>> above_mean(np.array([[1, 5], [3, 7]]))   # mean = 4
    array([5, 7])
    """
    # TODO
    return x[x > x.mean()]
    raise NotImplementedError


def standardize_rows(x):
    """Standardize each row: subtract the row mean, divide by the row std.

    x: 2-D float array, shape (n, d); assume no row is constant (std > 0)
    returns: new array, shape (n, d); every row has mean 0 and std 1
             (population std, i.e. np.std's default ddof=0)
    Must not modify x.

    Hint: what shape do the row mean and row std need so that
    they broadcast against x? (Compare 11-26 P4a.)

    >>> standardize_rows(np.array([[1., 2., 3.]])).round(4)
    array([[-1.2247,  0.    ,  1.2247]])
    """
    # TODO
    raise NotImplementedError
