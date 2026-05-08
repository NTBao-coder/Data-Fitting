"""NumPy-first implementations for linear models and model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ModelType = Literal["ols", "ridge", "lasso"]
DEFAULT_LASSO_MAX_ITER = 2000


@dataclass(frozen=True)
class LinearModels:
    """Core linear-model algorithms built directly from mathematical formulas."""

    @staticmethod
    def _to_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        return X

    @staticmethod
    def _to_1d(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim > 2 or (y.ndim == 2 and y.shape[1] != 1):
            raise ValueError("y must be 1D or 2D with shape (n, 1).")
        return y.reshape(-1)

    @staticmethod
    def add_intercept(X: np.ndarray) -> np.ndarray:
        X = LinearModels._to_2d(X)
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack((ones, X))

    @staticmethod
    def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = LinearModels.add_intercept(X)
        y = LinearModels._to_1d(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        xtx = X.T @ X
        xty = X.T @ y
        beta = np.linalg.pinv(xtx) @ xty
        return beta

    @staticmethod
    def hat_matrix(X: np.ndarray) -> np.ndarray:
        X = LinearModels.add_intercept(X)
        xtx_inv = np.linalg.pinv(X.T @ X)
        return X @ xtx_inv @ X.T

    @staticmethod
    def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        X = LinearModels.add_intercept(X)
        y = LinearModels._to_1d(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        p = X.shape[1]
        penalty = np.eye(p, dtype=float)
        penalty[0, 0] = 0.0
        beta = np.linalg.pinv(X.T @ X + alpha * penalty) @ (X.T @ y)
        return beta

    @staticmethod
    def lasso_fit(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        max_iter: int = DEFAULT_LASSO_MAX_ITER,
        tol: float = 1e-6,
    ) -> np.ndarray:
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        X = LinearModels.add_intercept(X)
        y = LinearModels._to_1d(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        n_features = X.shape[1]
        beta = np.zeros(n_features, dtype=float)
        col_norm = np.sum(X * X, axis=0)

        for _ in range(max_iter):
            old_beta = beta.copy()
            for j in range(n_features):
                y_without_j = y - X @ beta + X[:, j] * beta[j]
                rho_j = np.dot(X[:, j], y_without_j)

                if j == 0:
                    beta[j] = rho_j / col_norm[j] if col_norm[j] > 0 else 0.0
                    continue

                threshold = alpha
                if rho_j < -threshold:
                    beta[j] = (rho_j + threshold) / col_norm[j]
                elif rho_j > threshold:
                    beta[j] = (rho_j - threshold) / col_norm[j]
                else:
                    beta[j] = 0.0

            if np.max(np.abs(beta - old_beta)) < tol:
                break

        return beta

    @staticmethod
    def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        X = LinearModels.add_intercept(X)
        beta = np.asarray(beta, dtype=float).reshape(-1)
        if X.shape[1] != beta.shape[0]:
            raise ValueError("Beta shape does not match transformed X.")
        return X @ beta

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = LinearModels._to_1d(y_true)
        y_pred = LinearModels._to_1d(y_pred)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have equal length.")
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = LinearModels._to_1d(y_true)
        y_pred = LinearModels._to_1d(y_pred)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have equal length.")
        total = np.sum((y_true - np.mean(y_true)) ** 2)
        if total == 0.0:
            return 1.0
        residual = np.sum((y_true - y_pred) ** 2)
        return float(1.0 - residual / total)

    @staticmethod
    def k_fold_cross_validation(
        X: np.ndarray,
        y: np.ndarray,
        k: int = 5,
        model: ModelType = "ols",
        alpha: float = 1.0,
        random_state: int = 42,
    ) -> float:
        X = LinearModels._to_2d(X)
        y = LinearModels._to_1d(y)
        n_samples = X.shape[0]
        if n_samples != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if k < 2 or k > n_samples:
            raise ValueError("k must be in [2, n_samples].")

        rng = np.random.default_rng(random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        folds = np.array_split(indices, k)

        fold_mse: list[float] = []
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if model == "ols":
                beta = LinearModels.ols_fit(X_train, y_train)
            elif model == "ridge":
                beta = LinearModels.ridge_fit(X_train, y_train, alpha=alpha)
            elif model == "lasso":
                beta = LinearModels.lasso_fit(X_train, y_train, alpha=alpha)
            else:
                raise ValueError("model must be one of: 'ols', 'ridge', 'lasso'.")

            y_pred = LinearModels.predict(X_test, beta)
            fold_mse.append(LinearModels.mse(y_test, y_pred))

        return float(np.mean(fold_mse))
