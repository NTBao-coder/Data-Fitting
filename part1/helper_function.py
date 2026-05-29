"""Pure-Python math helpers for Part 1 modules."""

from __future__ import annotations

import math
import random
from typing import Any, Iterable

NAN = math.nan
INF = math.inf


class LinAlgError(Exception):
    """Raised when a linear algebra operation cannot be completed."""


class Vector(list):
    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def T(self) -> "Vector":
        return self

    def copy(self) -> "Vector":
        return Vector(self)

    def mean(self, axis: int | None = None) -> float:
        if axis is not None:
            raise ValueError("axis is not valid for a 1D vector.")
        return mean(self)

    def __getitem__(self, key: Any) -> Any:
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return Vector(result)
        return result

    def __matmul__(self, other: Any) -> Any:
        return matmul(self, other)

    def __add__(self, other: Any) -> "Vector":
        return add(self, other)

    def __radd__(self, other: Any) -> "Vector":
        return add(other, self)

    def __sub__(self, other: Any) -> "Vector":
        return subtract(self, other)

    def __rsub__(self, other: Any) -> "Vector":
        return subtract(other, self)

    def __mul__(self, other: Any) -> "Vector":
        return multiply(self, other)

    def __rmul__(self, other: Any) -> "Vector":
        return multiply(other, self)

    def __truediv__(self, other: Any) -> "Vector":
        return divide_values(self, other)

    def __pow__(self, power: float) -> "Vector":
        return Vector([value**power for value in self])

    def __neg__(self) -> "Vector":
        return Vector([-value for value in self])

    def __eq__(self, other: Any) -> bool | list[bool]:
        if isinstance(other, (int, float)):
            return [value == other for value in self]
        return list.__eq__(self, other)

    def __gt__(self, other: Any) -> list[bool]:
        return [value > other for value in self]

    def __ge__(self, other: Any) -> list[bool]:
        return [value >= other for value in self]

    def __lt__(self, other: Any) -> list[bool]:
        return [value < other for value in self]

    def __le__(self, other: Any) -> list[bool]:
        return [value <= other for value in self]


class Matrix(list):
    @property
    def shape(self) -> tuple[int, int]:
        return (len(self), len(self[0]) if self else 0)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def T(self) -> "Matrix":
        return transpose(self)

    def copy(self) -> "Matrix":
        return Matrix([Vector(row) for row in self])

    def mean(self, axis: int | None = None) -> Any:
        return mean(self, axis=axis)

    def squeeze(self, axis: int | None = None) -> Any:
        if axis is None:
            if self.shape[1] == 1:
                return Vector([row[0] for row in self])
            return self
        if axis == 1 and self.shape[1] == 1:
            return Vector([row[0] for row in self])
        raise ValueError("Only singleton column squeeze is supported.")

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            row_key, col_key = key
            rows = _select_rows(self, row_key)
            if isinstance(row_key, int):
                rows = Matrix([rows])
                single_row = True
            else:
                single_row = False

            selected = [_select_cols(row, col_key) for row in rows]
            single_col = isinstance(col_key, int)
            if single_row and single_col:
                return selected[0]
            if single_row:
                return Vector(selected[0])
            if single_col:
                return Vector(selected)
            return Matrix([Vector(row) for row in selected])

        result = super().__getitem__(key)
        if isinstance(key, slice):
            return Matrix([Vector(row) for row in result])
        if isinstance(key, list):
            return Matrix([Vector(self[i]) for i in key])
        return Vector(result)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, tuple):
            row, col = key
            self[row][col] = float(value)
            return
        super().__setitem__(key, value)

    def __matmul__(self, other: Any) -> Any:
        return matmul(self, other)

    def __add__(self, other: Any) -> "Matrix":
        return add(self, other)

    def __radd__(self, other: Any) -> "Matrix":
        return add(other, self)

    def __sub__(self, other: Any) -> "Matrix":
        return subtract(self, other)

    def __rsub__(self, other: Any) -> "Matrix":
        return subtract(other, self)

    def __mul__(self, other: Any) -> "Matrix":
        return multiply(self, other)

    def __rmul__(self, other: Any) -> "Matrix":
        return multiply(other, self)


Array = Vector | Matrix


def _to_python(values: Any) -> Any:
    if isinstance(values, (Vector, Matrix)):
        return values
    if hasattr(values, "tolist"):
        return values.tolist()
    if hasattr(values, "values") and hasattr(values.values, "tolist"):
        return values.values.tolist()
    return values


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))


def _is_matrix_like(values: Any) -> bool:
    values = _to_python(values)
    return bool(_is_sequence(values) and values and _is_sequence(values[0]))


def _flatten_column(values: list[Any]) -> list[Any]:
    if values and all(_is_sequence(row) and len(row) == 1 for row in values):
        return [row[0] for row in values]
    return values


def as_1d_float_array(values: Any, name: str) -> Vector:
    """Convert a vector-like input to a 1D float vector."""
    values = _to_python(values)
    if not _is_sequence(values):
        raise ValueError(f"{name} must be a 1D array or a 2D column vector.")
    values = list(values)
    if values and _is_sequence(values[0]):
        values = _flatten_column(values)
    if values and _is_sequence(values[0]):
        raise ValueError(f"{name} must be a 1D array or a 2D column vector.")
    return Vector([float(value) for value in values])


def as_2d_float_array(values: Any, name: str) -> Matrix:
    """Convert a matrix-like input to a 2D float matrix."""
    values = _to_python(values)
    if not _is_sequence(values) or not values:
        raise ValueError(f"{name} must be a 2D array.")
    rows = list(values)
    if not all(_is_sequence(row) for row in rows):
        raise ValueError(f"{name} must be a 2D array.")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be a rectangular 2D array.")
    return Matrix([Vector([float(value) for value in row]) for row in rows])


def shape(values: Any) -> tuple[int] | tuple[int, int]:
    values = _to_python(values)
    if _is_matrix_like(values):
        return (len(values), len(values[0]) if values else 0)
    return (len(values),)


def validate_regression_shapes(X: Matrix, y: Vector) -> tuple[int, int]:
    """Validate common OLS input dimensions and return ``(n, p)``."""
    n_samples, n_features = X.shape
    if len(y) != n_samples:
        raise ValueError("X and y must contain the same number of samples.")
    if n_samples <= n_features:
        raise ValueError("OLS requires n_samples > n_features for sigma2 inference.")
    return n_samples, n_features


def transpose(matrix: Any) -> Matrix:
    matrix = as_2d_float_array(matrix, "matrix")
    return Matrix([Vector(row) for row in zip(*matrix)])


def dot(left: Any, right: Any) -> float:
    left = as_1d_float_array(left, "left")
    right = as_1d_float_array(right, "right")
    if len(left) != len(right):
        raise ValueError("Vectors must have the same length.")
    return sum(a * b for a, b in zip(left, right))


def matmul(left: Any, right: Any) -> Any:
    left = _to_python(left)
    right = _to_python(right)
    left_is_matrix = _is_matrix_like(left)
    right_is_matrix = _is_matrix_like(right)

    if not left_is_matrix and not right_is_matrix:
        return dot(left, right)

    if left_is_matrix and not right_is_matrix:
        matrix = as_2d_float_array(left, "left")
        vector = as_1d_float_array(right, "right")
        if matrix.shape[1] != len(vector):
            raise ValueError("Matrix columns must match vector length.")
        return Vector([dot(row, vector) for row in matrix])

    if not left_is_matrix and right_is_matrix:
        vector = as_1d_float_array(left, "left")
        matrix = as_2d_float_array(right, "right")
        if len(vector) != matrix.shape[0]:
            raise ValueError("Vector length must match matrix rows.")
        return Vector([dot(vector, column(matrix, j)) for j in range(matrix.shape[1])])

    left_matrix = as_2d_float_array(left, "left")
    right_matrix = as_2d_float_array(right, "right")
    if left_matrix.shape[1] != right_matrix.shape[0]:
        raise ValueError("Left columns must match right rows.")
    right_t = transpose(right_matrix)
    return Matrix(
        [
            Vector([dot(left_row, right_col) for right_col in right_t])
            for left_row in left_matrix
        ]
    )


def solve(matrix: Any, values: Any) -> Any:
    matrix = as_2d_float_array(matrix, "matrix")
    rhs_is_matrix = _is_matrix_like(values)
    rhs = as_2d_float_array(values, "values") if rhs_is_matrix else as_1d_float_array(values, "values")
    n_rows, n_cols = matrix.shape
    if n_rows != n_cols:
        raise LinAlgError("Matrix must be square.")
    if rhs_is_matrix and rhs.shape[0] != n_rows:
        raise LinAlgError("Right-hand side has incompatible shape.")
    if not rhs_is_matrix and len(rhs) != n_rows:
        raise LinAlgError("Right-hand side has incompatible shape.")

    rhs_matrix = rhs if rhs_is_matrix else Matrix([Vector([value]) for value in rhs])
    augmented = Matrix(
        [Vector(list(matrix[i]) + list(rhs_matrix[i])) for i in range(n_rows)]
    )
    rhs_width = rhs_matrix.shape[1]

    for pivot_idx in range(n_rows):
        pivot_row = max(
            range(pivot_idx, n_rows), key=lambda row: abs(augmented[row][pivot_idx])
        )
        if is_close(augmented[pivot_row][pivot_idx], 0.0, atol=1e-12):
            raise LinAlgError("Matrix is singular.")
        if pivot_row != pivot_idx:
            augmented[pivot_idx], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_idx]

        pivot = augmented[pivot_idx][pivot_idx]
        augmented[pivot_idx] = Vector([value / pivot for value in augmented[pivot_idx]])

        for row_idx in range(n_rows):
            if row_idx == pivot_idx:
                continue
            factor = augmented[row_idx][pivot_idx]
            if is_close(factor, 0.0, atol=1e-15):
                continue
            augmented[row_idx] = Vector(
                [
                    value - factor * pivot_value
                    for value, pivot_value in zip(augmented[row_idx], augmented[pivot_idx])
                ]
            )

    solution = Matrix([Vector(row[-rhs_width:]) for row in augmented])
    if rhs_is_matrix:
        return solution
    return Vector([row[0] for row in solution])


def inverse(matrix: Any) -> Matrix:
    matrix = as_2d_float_array(matrix, "matrix")
    return solve(matrix, eye(matrix.shape[0]))


def norm(values: Any) -> float:
    vector = as_1d_float_array(values, "values")
    return math.sqrt(sum(value * value for value in vector))


def all_close(left: Any, right: Any, atol: float = 1e-8) -> bool:
    left = _to_python(left)
    right = _to_python(right)
    if _is_matrix_like(left) or _is_matrix_like(right):
        left_matrix = as_2d_float_array(left, "left")
        right_matrix = as_2d_float_array(right, "right")
        if left_matrix.shape != right_matrix.shape:
            return False
        return all(
            is_close(a, b, atol=atol)
            for left_row, right_row in zip(left_matrix, right_matrix)
            for a, b in zip(left_row, right_row)
        )
    left_vector = as_1d_float_array(left, "left")
    right_vector = as_1d_float_array(right, "right")
    if len(left_vector) != len(right_vector):
        return False
    return all(is_close(a, b, atol=atol) for a, b in zip(left_vector, right_vector))


def is_close(left: Any, right: Any, atol: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= atol


def mean(values: Any, axis: int | None = None) -> Any:
    values = _to_python(values)
    if axis is None:
        if _is_matrix_like(values):
            flat = [value for row in values for value in row]
            return sum(flat) / len(flat)
        values = list(values)
        return sum(values) / len(values)
    matrix = as_2d_float_array(values, "values")
    if axis == 0:
        return Vector([mean(column(matrix, j)) for j in range(matrix.shape[1])])
    if axis == 1:
        return Vector([mean(row) for row in matrix])
    raise ValueError("axis must be None, 0, or 1.")


def sum_values(values: Any, axis: int | None = None) -> Any:
    values = _to_python(values)
    if axis is None:
        if _is_matrix_like(values):
            return sum(value for row in values for value in row)
        return sum(values)
    matrix = as_2d_float_array(values, "values")
    if axis == 0:
        return Vector([sum(column(matrix, j)) for j in range(matrix.shape[1])])
    if axis == 1:
        return Vector([sum(row) for row in matrix])
    raise ValueError("axis must be None, 0, or 1.")


def _map_values(values: Any, func: Any) -> Any:
    values = _to_python(values)
    if _is_matrix_like(values):
        return Matrix([Vector([func(value) for value in row]) for row in values])
    if _is_sequence(values):
        return Vector([func(value) for value in values])
    return func(values)


def sqrt(values: Any) -> Any:
    return _map_values(values, math.sqrt)


def absolute(values: Any) -> Any:
    return _map_values(values, abs)


def diagonal(matrix: Any) -> Vector:
    matrix = as_2d_float_array(matrix, "matrix")
    return Vector([matrix[i][i] for i in range(min(matrix.shape))])


def divide(numerator: Any, denominator: Any, out: Vector, where: Any) -> Vector:
    numerator = as_1d_float_array(numerator, "numerator")
    denominator = as_1d_float_array(denominator, "denominator")
    where = list(where)
    result = out.copy()
    for idx, should_divide in enumerate(where):
        if should_divide:
            result[idx] = numerator[idx] / denominator[idx]
    return result


def divide_values(left: Any, right: Any) -> Any:
    return _binary_elementwise(left, right, lambda a, b: a / b)


def full_like(values: Any, fill_value: float, dtype: type = float) -> Vector | Matrix:
    values = _to_python(values)
    fill = dtype(fill_value)
    if _is_matrix_like(values):
        return Matrix([Vector([fill for _ in row]) for row in values])
    return Vector([fill for _ in values])


def zeros(length_or_shape: int | tuple[int, ...]) -> Vector | Matrix:
    if isinstance(length_or_shape, tuple):
        if len(length_or_shape) == 1:
            return Vector([0.0 for _ in range(length_or_shape[0])])
        rows, cols = length_or_shape[:2]
        return Matrix([Vector([0.0 for _ in range(cols)]) for _ in range(rows)])
    return Vector([0.0 for _ in range(length_or_shape)])


def zeros_like(values: Any, dtype: type = float) -> Vector | Matrix:
    return full_like(values, 0.0, dtype=dtype)


def ones(shape: int | tuple[int, ...], dtype: type = float) -> Vector | Matrix:
    if isinstance(shape, tuple):
        if len(shape) == 1:
            return Vector([dtype(1.0) for _ in range(shape[0])])
        rows, cols = shape[:2]
        return Matrix([Vector([dtype(1.0) for _ in range(cols)]) for _ in range(rows)])
    return Vector([dtype(1.0) for _ in range(shape)])


def eye(size: int) -> Matrix:
    return Matrix(
        [
            Vector([1.0 if row == col else 0.0 for col in range(size)])
            for row in range(size)
        ]
    )


def array(values: Any) -> Vector | Matrix:
    values = _to_python(values)
    if _is_matrix_like(values):
        return as_2d_float_array(values, "values")
    return as_1d_float_array(values, "values")


def arange(stop: int) -> Vector:
    return Vector([float(value) for value in range(stop)])


def column_stack(values: list[Any]) -> Matrix:
    columns = [as_1d_float_array(value, "value") for value in values]
    if not columns:
        return Matrix([])
    length = len(columns[0])
    if any(len(col) != length for col in columns):
        raise ValueError("All columns must have the same length.")
    return Matrix([Vector([col[row] for col in columns]) for row in range(length)])


def concatenate(values: list[Any], axis: int = 0) -> Any:
    if axis == 0:
        result = []
        for value in values:
            converted = _to_python(value)
            result.extend(converted)
        return array(result)
    if axis == 1:
        matrices = [as_2d_float_array(value, "value") for value in values]
        rows = matrices[0].shape[0]
        if any(matrix.shape[0] != rows for matrix in matrices):
            raise ValueError("All matrices must have the same number of rows.")
        return Matrix(
            [
                Vector([item for matrix in matrices for item in matrix[row_idx]])
                for row_idx in range(rows)
            ]
        )
    raise ValueError("Only axis 0 and 1 are supported.")


def array_split(values: Any, sections: int) -> list[Vector]:
    values = as_1d_float_array(values, "values")
    n = len(values)
    base, remainder = divmod(n, sections)
    result = []
    start = 0
    for idx in range(sections):
        stop = start + base + (1 if idx < remainder else 0)
        result.append(Vector(values[start:stop]))
        start = stop
    return result


def swapaxes(values: Any, axis1: int, axis2: int) -> Any:
    if {axis1, axis2} == {0, 1}:
        return transpose(values)
    raise ValueError("Only 2D axis swap is supported.")


def clip(values: Any, min_value: float, max_value: float | None) -> Any:
    def _clip(value: float) -> float:
        clipped = max(value, min_value)
        if max_value is not None:
            clipped = min(clipped, max_value)
        return clipped

    return _map_values(values, _clip)


def where(condition: Any, x: Any, y: Any) -> Any:
    condition = _to_python(condition)
    if isinstance(condition, bool):
        return x if condition else y
    x_is_scalar = not _is_sequence(_to_python(x))
    y_is_scalar = not _is_sequence(_to_python(y))
    return Vector(
        [
            (x if x_is_scalar else x[idx]) if cond else (y if y_is_scalar else y[idx])
            for idx, cond in enumerate(condition)
        ]
    )


def delete(values: Any, index: int, axis: int = 0) -> Any:
    if axis == 0:
        result = list(_to_python(values))
        del result[index]
        return array(result)
    matrix = as_2d_float_array(values, "values")
    return Matrix(
        [Vector([value for col_idx, value in enumerate(row) if col_idx != index]) for row in matrix]
    )


def take_rows(matrix: Any, indices: Any) -> Matrix:
    matrix = as_2d_float_array(matrix, "matrix")
    indices = [int(index) for index in indices]
    return Matrix([Vector(matrix[index]) for index in indices])


def take_vector(values: Any, indices: Any) -> Vector:
    values = as_1d_float_array(values, "values")
    indices = [int(index) for index in indices]
    return Vector([values[index] for index in indices])


def column(matrix: Any, index: int) -> Vector:
    matrix = as_2d_float_array(matrix, "matrix")
    return Vector([row[index] for row in matrix])


def scalar_multiply(values: Any, scalar: float) -> Any:
    return multiply(values, scalar)


def add(left: Any, right: Any) -> Any:
    return _binary_elementwise(left, right, lambda a, b: a + b)


def subtract(left: Any, right: Any) -> Any:
    return _binary_elementwise(left, right, lambda a, b: a - b)


def multiply(left: Any, right: Any) -> Any:
    return _binary_elementwise(left, right, lambda a, b: a * b)


def power(values: Any, exponent: float) -> Any:
    return _map_values(values, lambda value: value**exponent)


def _binary_elementwise(left: Any, right: Any, op: Any) -> Any:
    left = _to_python(left)
    right = _to_python(right)
    left_scalar = not _is_sequence(left)
    right_scalar = not _is_sequence(right)

    if left_scalar and right_scalar:
        return op(left, right)
    if _is_matrix_like(left) or _is_matrix_like(right):
        if left_scalar:
            right_matrix = as_2d_float_array(right, "right")
            return Matrix([Vector([op(left, value) for value in row]) for row in right_matrix])
        if right_scalar:
            left_matrix = as_2d_float_array(left, "left")
            return Matrix([Vector([op(value, right) for value in row]) for row in left_matrix])
        left_matrix = as_2d_float_array(left, "left")
        right_matrix = as_2d_float_array(right, "right")
        if left_matrix.shape != right_matrix.shape:
            raise ValueError("Matrices must have the same shape.")
        return Matrix(
            [
                Vector([op(a, b) for a, b in zip(left_row, right_row)])
                for left_row, right_row in zip(left_matrix, right_matrix)
            ]
        )
    if left_scalar:
        right_vector = as_1d_float_array(right, "right")
        return Vector([op(left, value) for value in right_vector])
    if right_scalar:
        left_vector = as_1d_float_array(left, "left")
        return Vector([op(value, right) for value in left_vector])
    left_vector = as_1d_float_array(left, "left")
    right_vector = as_1d_float_array(right, "right")
    if len(left_vector) != len(right_vector):
        raise ValueError("Vectors must have the same length.")
    return Vector([op(a, b) for a, b in zip(left_vector, right_vector)])


def _select_rows(matrix: Matrix, key: Any) -> Any:
    if isinstance(key, int):
        return matrix[key]
    if isinstance(key, slice):
        return Matrix([Vector(row) for row in list.__getitem__(matrix, key)])
    if _is_sequence(key):
        return Matrix([Vector(matrix[int(idx)]) for idx in key])
    raise TypeError("Unsupported row selector.")


def _select_cols(row: Vector, key: Any) -> Any:
    if isinstance(key, int):
        return row[key]
    if isinstance(key, slice):
        return Vector(row[key])
    if _is_sequence(key):
        return Vector([row[int(idx)] for idx in key])
    raise TypeError("Unsupported column selector.")


def random_seed(seed: int) -> None:
    random.seed(seed)


def random_normal(
    loc: float = 0.0,
    scale: float = 1.0,
    size: int | tuple[int, ...] | None = None,
) -> Any:
    def _one() -> float:
        return random.gauss(loc, scale)

    if size is None:
        return _one()
    if isinstance(size, int):
        return Vector([_one() for _ in range(size)])
    if len(size) == 1:
        return Vector([_one() for _ in range(size[0])])
    if len(size) == 2:
        rows, cols = size
        return Matrix([Vector([_one() for _ in range(cols)]) for _ in range(rows)])
    raise ValueError("Only scalar, 1D, and 2D random normal are supported.")


def random_randn(*shape: int) -> Any:
    return random_normal(loc=0.0, scale=1.0, size=shape)


def random_shuffle(values: Vector) -> None:
    random.shuffle(values)


def assert_allclose(actual: Any, desired: Any, atol: float, err_msg: str = "") -> None:
    if not all_close(actual, desired, atol=atol):
        raise AssertionError(err_msg or f"{actual} is not close to {desired}")


def assert_array_equal(actual: Any, desired: Any, err_msg: str = "") -> None:
    if _to_python(actual) != _to_python(desired):
        raise AssertionError(err_msg or f"{actual} != {desired}")
