"""Shared input-validation helpers for the ``part1`` package."""

from __future__ import annotations

import numpy as np


def _as_1d_float_array(values: np.ndarray, name: str) -> np.ndarray:
    """Convert a vector-like input to a 1D float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.ravel()
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array or a 2D column vector.")
    return array


def _as_2d_float_array(values: np.ndarray, name: str) -> np.ndarray:
    """Convert a matrix-like input to a 2D float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    return array
