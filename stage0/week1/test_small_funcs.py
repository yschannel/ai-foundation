"""Tests for small_funcs.py. Run in week1:  .\\.venv\\Scripts\\python.exe -m pytest -q

Each test checks the SHAPE first, then the values.
Data is deliberately not arange-ordered, so a "looks right by coincidence"
answer still fails (11-26 lesson).
"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from small_funcs import (
    above_mean,
    col_max,
    count_even,
    row_mean,
    standardize_rows,
    to_column,
)

X = np.array([[3, 8, 1, 4],
              [9, 2, 7, 6],
              [5, 0, 10, 1]])          # shape (3, 4)


# ---------- row_mean ----------

def test_row_mean():
    out = row_mean(X)
    assert out.shape == (3,)
    assert_allclose(out, [4.0, 6.0, 4.0])


def test_row_mean_single_row():
    out = row_mean(np.array([[2, 4]]))
    assert out.shape == (1,)
    assert_allclose(out, [3.0])


# ---------- col_max ----------

def test_col_max():
    out = col_max(X)
    assert out.shape == (4,)
    assert_array_equal(out, [9, 8, 10, 6])


def test_col_max_negative():
    out = col_max(np.array([[-5, -1], [-3, -7]]))
    assert_array_equal(out, [-3, -1])


# ---------- to_column ----------

def test_to_column():
    v = np.array([7, 3, 5])
    out = to_column(v)
    assert out.shape == (3, 1)
    assert_array_equal(out, [[7], [3], [5]])


def test_to_column_broadcasts_like_a_column():
    # a (3,1) column plus a (4,) row must give a (3,4) table
    out = to_column(np.array([0, 10, 20])) + np.array([1, 2, 3, 4])
    assert out.shape == (3, 4)
    assert out[2, 0] == 21


# ---------- count_even ----------

def test_count_even():
    assert count_even(X) == 6          # 8, 4, 2, 6, 0, 10


def test_count_even_negative_and_zero():
    assert count_even(np.array([-4, -3, 0, 1])) == 2


def test_count_even_returns_int():
    assert isinstance(count_even(np.array([1, 2])), (int, np.integer))


# ---------- above_mean ----------

def test_above_mean():
    out = above_mean(X)                # mean of all 12 = 56 / 12 ~ 4.67
    assert out.ndim == 1
    assert_array_equal(out, [8, 9, 7, 6, 5, 10])


def test_above_mean_strict():
    # elements equal to the mean are NOT included
    out = above_mean(np.array([2, 4, 6]))   # mean = 4
    assert_array_equal(out, [6])


# ---------- standardize_rows ----------

def test_standardize_rows_hand_value():
    out = standardize_rows(np.array([[1.0, 2.0, 3.0]]))
    assert out.shape == (1, 3)
    assert_allclose(out, [[-1.224744871, 0.0, 1.224744871]], atol=1e-8)


def test_standardize_rows_properties():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=3.0, size=(4, 6))
    out = standardize_rows(x)
    assert out.shape == (4, 6)
    assert_allclose(out.mean(axis=1), np.zeros(4), atol=1e-12)
    assert_allclose(out.std(axis=1), np.ones(4), atol=1e-12)


def test_standardize_rows_is_per_row_not_global():
    # rows on very different scales: each must be standardized on its own
    x = np.array([[1.0, 2.0, 3.0],
                  [100.0, 300.0, 200.0]])
    out = standardize_rows(x)
    assert_allclose(out[1], [-1.224744871, 1.224744871, 0.0], atol=1e-8)


def test_standardize_rows_does_not_modify_input():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])
    before = x.copy()
    standardize_rows(x)
    assert_array_equal(x, before)
