"""Shared NumPy helpers for Part 1 modules."""

from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray
LinAlgError = np.linalg.LinAlgError
NAN = np.nan
INF = np.inf


def as_1d_float_array(values: Any, name: str) -> Array:
    """Convert a vector-like input to a 1D float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.ravel()
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array or a 2D column vector.")
    return array


def as_2d_float_array(values: Any, name: str) -> Array:
    """Convert a matrix-like input to a 2D float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    return array


def validate_regression_shapes(X: Array, y: Array) -> tuple[int, int]:
    """Validate common OLS input dimensions and return ``(n, p)``."""
    n_samples, n_features = X.shape
    if y.shape[0] != n_samples:
        raise ValueError("X and y must contain the same number of samples.")
    if n_samples <= n_features:
        raise ValueError("OLS requires n_samples > n_features for sigma2 inference.")
    return n_samples, n_features


def inverse(matrix: Array) -> Array:
    return np.linalg.inv(matrix)


def solve(matrix: Array, values: Array) -> Array:
    return np.linalg.solve(matrix, values)


def norm(values: Array) -> float:
    return float(np.linalg.norm(values))


def all_close(left: Array, right: Array, atol: float = 1e-8) -> bool:
    return bool(np.allclose(left, right, atol=atol))


def is_close(left: Any, right: Any, atol: float = 1e-8) -> bool:
    return bool(np.isclose(left, right, atol=atol))


def mean(values: Any, axis: int | None = None) -> Any:
    return np.mean(values, axis=axis)


def sum_values(values: Any, axis: int | None = None) -> Any:
    return np.sum(values, axis=axis)


def sqrt(values: Any) -> Any:
    return np.sqrt(values)


def absolute(values: Any) -> Any:
    return np.abs(values)


def diagonal(matrix: Array) -> Array:
    return np.diag(matrix)


def divide(numerator: Any, denominator: Any, out: Array, where: Any) -> Array:
    return np.divide(numerator, denominator, out=out, where=where)


def full_like(values: Array, fill_value: float, dtype: type = float) -> Array:
    return np.full_like(values, fill_value, dtype=dtype)


def zeros(length_or_shape: int | tuple[int, ...]) -> Array:
    return np.zeros(length_or_shape)


def zeros_like(values: Array, dtype: type = float) -> Array:
    return np.zeros_like(values, dtype=dtype)


def ones(shape: int | tuple[int, ...], dtype: type = float) -> Array:
    return np.ones(shape, dtype=dtype)


def eye(size: int) -> Array:
    return np.eye(size)


def array(values: Any) -> Array:
    return np.array(values)


def arange(stop: int) -> Array:
    return np.arange(stop)


def column_stack(values: list[Array]) -> Array:
    return np.column_stack(values)


def concatenate(values: list[Array], axis: int = 0) -> Array:
    return np.concatenate(values, axis=axis)


def array_split(values: Array, sections: int) -> list[Array]:
    return np.array_split(values, sections)


def swapaxes(values: Array, axis1: int, axis2: int) -> Array:
    return np.swapaxes(values, axis1, axis2)


def clip(values: Array, min_value: float, max_value: float | None) -> Array:
    return np.clip(values, min_value, max_value)


def where(condition: Any, x: Any, y: Any) -> Array:
    return np.where(condition, x, y)


def delete(values: Array, index: int, axis: int = 0) -> Array:
    return np.delete(values, index, axis=axis)


def random_seed(seed: int) -> None:
    np.random.seed(seed)


def random_shuffle(values: Array) -> None:
    np.random.shuffle(values)


def random_normal(
    loc: float = 0.0,
    scale: float = 1.0,
    size: int | tuple[int, ...] | None = None,
) -> Array:
    return np.random.normal(loc=loc, scale=scale, size=size)


def random_randn(*shape: int) -> Array:
    return np.random.randn(*shape)


def assert_allclose(actual: Any, desired: Any, atol: float, err_msg: str = "") -> None:
    np.testing.assert_allclose(actual, desired, atol=atol, err_msg=err_msg)


def assert_array_equal(actual: Any, desired: Any, err_msg: str = "") -> None:
    np.testing.assert_array_equal(actual, desired, err_msg=err_msg)
