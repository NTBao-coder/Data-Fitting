"""Data pipeline with leakage-safe fit/transform behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DataPipeline:
    """Handle missing values, categorical encoding, and numeric standardization."""

    numeric_indices: tuple[int, ...]
    categorical_indices: tuple[int, ...]
    _numeric_impute: dict[int, float] = field(default_factory=dict, init=False)
    _categorical_impute: dict[int, str] = field(default_factory=dict, init=False)
    _categories: dict[int, list[str]] = field(default_factory=dict, init=False)
    _numeric_mean: dict[int, float] = field(default_factory=dict, init=False)
    _numeric_std: dict[int, float] = field(default_factory=dict, init=False)
    _is_fitted: bool = field(default=False, init=False)

    def fit(self, X: np.ndarray) -> "DataPipeline":
        X = np.asarray(X, dtype=object)
        self._validate_input(X)

        for idx in self.numeric_indices:
            column = np.array(
                [float(v) for v in X[:, idx] if v is not None and v != ""],
                dtype=float,
            )
            impute_value = float(np.mean(column)) if column.size else 0.0
            self._numeric_impute[idx] = impute_value
            standardized_source = np.array(
                [float(v) if v is not None and v != "" else impute_value for v in X[:, idx]],
                dtype=float,
            )
            mean = float(np.mean(standardized_source))
            std = float(np.std(standardized_source))
            self._numeric_mean[idx] = mean
            self._numeric_std[idx] = std if std > 0 else 1.0

        for idx in self.categorical_indices:
            values = [
                str(v).strip()
                for v in X[:, idx]
                if v is not None and str(v).strip() != ""
            ]
            mode_value = max(set(values), key=values.count) if values else "missing"
            self._categorical_impute[idx] = mode_value
            filled_values = [
                str(v).strip() if v is not None and str(v).strip() != "" else mode_value
                for v in X[:, idx]
            ]
            self._categories[idx] = sorted(set(filled_values))

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform.")

        X = np.asarray(X, dtype=object)
        self._validate_input(X)

        transformed_columns: list[np.ndarray] = []

        for idx in self.numeric_indices:
            impute_value = self._numeric_impute[idx]
            mean = self._numeric_mean[idx]
            std = self._numeric_std[idx]
            filled = np.array(
                [float(v) if v is not None and v != "" else impute_value for v in X[:, idx]],
                dtype=float,
            )
            transformed_columns.append(((filled - mean) / std).reshape(-1, 1))

        for idx in self.categorical_indices:
            impute_value = self._categorical_impute[idx]
            values = [
                str(v).strip() if v is not None and str(v).strip() != "" else impute_value
                for v in X[:, idx]
            ]
            categories = self._categories[idx]
            encoded = np.zeros((X.shape[0], len(categories)), dtype=float)
            for row, value in enumerate(values):
                if value in categories:
                    encoded[row, categories.index(value)] = 1.0
            transformed_columns.append(encoded)

        if not transformed_columns:
            return np.empty((X.shape[0], 0), dtype=float)
        return np.hstack(transformed_columns)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def _validate_input(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        max_idx = X.shape[1] - 1
        for idx in self.numeric_indices + self.categorical_indices:
            if idx < 0 or idx > max_idx:
                raise ValueError("Feature index is out of bounds for X.")

