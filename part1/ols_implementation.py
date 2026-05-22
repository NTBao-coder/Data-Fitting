"""Core Ordinary Least Squares implementation.

The functions in this module implement the OLS normal-equation workflow with
NumPy matrix operations. The design matrix ``X`` is assumed to already include
an intercept column when the model requires one.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from part1._utils import _as_1d_float_array, _as_2d_float_array





def _validate_regression_shapes(X: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    """Validate common OLS input dimensions and return ``(n, p)``."""
    n_samples, n_features = X.shape
    if y.shape[0] != n_samples:
        raise ValueError("X and y must contain the same number of samples.")
    if n_samples <= n_features:
        raise ValueError("OLS requires n_samples > n_features for sigma2 inference.")
    return n_samples, n_features


def ols_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Estimate OLS coefficients and residual variance.

    Formula:
        beta_hat = (X.T @ X)^(-1) @ X.T @ y
        sigma_squared = RSS / (n - p)

    Parameters
    ----------
    X : np.ndarray
        Design matrix with shape (n_samples, n_features). Include an intercept
        column before calling this function if the model needs one.
    y : np.ndarray
        Target vector with shape (n_samples,) or (n_samples, 1).

    Returns
    -------
    Tuple[np.ndarray, float]
        Estimated coefficients with shape (n_features,) and unbiased residual
        variance estimate.
    """
    X = _as_2d_float_array(X, "X")
    y = _as_1d_float_array(y, "y")
    n_samples, n_features = _validate_regression_shapes(X, y)

    XTX = X.T @ X
    beta_hat = np.linalg.solve(XTX, X.T @ y)
    residuals = y - X @ beta_hat
    rss = residuals.T @ residuals
    sigma_squared = rss / (n_samples - n_features)

    return beta_hat, float(sigma_squared)


def hat_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute the OLS hat matrix.

    Formula:
        H = X @ (X.T @ X)^(-1) @ X.T

    The hat matrix maps observed values to fitted values:
        y_hat = H @ y

    It is symmetric and idempotent:
        H.T = H
        H @ H = H

    Parameters
    ----------
    X : np.ndarray
        Design matrix with shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Hat matrix with shape (n_samples, n_samples).
    """
    X = _as_2d_float_array(X, "X")
    n_samples, n_features = X.shape
    if n_samples < n_features:
        raise ValueError("X must have n_samples >= n_features.")

    H = X @ np.linalg.solve(X.T @ X, X.T)

    if not np.allclose(H.T, H, atol=1e-8):
        raise RuntimeError("Hat matrix is not symmetric — possible numerical issue.")
    if not np.allclose(H @ H, H, atol=1e-8):
        raise RuntimeError("Hat matrix is not idempotent — possible numerical issue.")

    return H


def model_metrics(y: np.ndarray, y_hat: np.ndarray, p: int) -> Dict[str, float]:
    """
    Compute common OLS model metrics.

    Formulas:
        RSS = sum((y - y_hat)^2)
        TSS = sum((y - mean(y))^2)
        R2 = 1 - RSS / TSS
        Adj R2 = 1 - (RSS / (n - p)) / (TSS / (n - 1))
        F = ((TSS - RSS) / (p - 1)) / (RSS / (n - p))

    Parameters
    ----------
    y : np.ndarray
        Observed target values.
    y_hat : np.ndarray
        Fitted target values.
    p : int
        Number of coefficients, including the intercept.

    Returns
    -------
    Dict[str, float]
        RSS, TSS, R2, adjusted R2, and F-statistic.
    """
    y = _as_1d_float_array(y, "y")
    y_hat = _as_1d_float_array(y_hat, "y_hat")

    if y.shape != y_hat.shape:
        raise ValueError("y and y_hat must have the same shape.")
    if p < 1:
        raise ValueError("p must be at least 1.")

    n_samples = y.shape[0]
    if n_samples <= p:
        raise ValueError("n_samples must be greater than p.")

    residuals = y - y_hat
    centered_y = y - np.mean(y)
    rss = float(residuals.T @ residuals)
    tss = float(centered_y.T @ centered_y)

    if np.isclose(tss, 0.0):
        r_squared = np.nan
        adjusted_r_squared = np.nan
        f_statistic = np.nan
    else:
        r_squared = 1.0 - rss / tss
        adjusted_r_squared = 1.0 - (rss / (n_samples - p)) / (tss / (n_samples - 1))
        if p == 1 or np.isclose(rss, 0.0):
            f_statistic = np.inf
        else:
            f_statistic = ((tss - rss) / (p - 1)) / (rss / (n_samples - p))

    return {
        "RSS": rss,
        "TSS": tss,
        "R2": float(r_squared),
        "adjusted_R2": float(adjusted_r_squared),
        "F_statistic": float(f_statistic),
    }


def coef_inference(
    X: np.ndarray,
    y: np.ndarray,
    beta_hat: np.ndarray,
    sigma2: float,
) -> pd.DataFrame:
    """
    Compute coefficient inference statistics.

    Formulas:
        Var(beta_hat) = sigma2 * (X.T @ X)^(-1)
        SE = sqrt(diag(Var(beta_hat)))
        t = beta_hat / SE
        p_value = 2 * (1 - t_cdf(abs(t), df=n-p))
        CI_95 = beta_hat +/- t_critical * SE

    Parameters
    ----------
    X : np.ndarray
        Design matrix with shape (n_samples, n_features).
    y : np.ndarray
        Target vector with shape (n_samples,) or (n_samples, 1).
    beta_hat : np.ndarray
        Estimated coefficients with shape (n_features,).
    sigma2 : float
        Estimated residual variance.

    Returns
    -------
    pd.DataFrame
        Table with coefficient, standard error, t-statistic, p-value, and 95%
        confidence interval columns.
    """
    X = _as_2d_float_array(X, "X")
    y = _as_1d_float_array(y, "y")
    beta_hat = _as_1d_float_array(beta_hat, "beta_hat")
    n_samples, n_features = _validate_regression_shapes(X, y)

    if beta_hat.shape[0] != n_features:
        raise ValueError("beta_hat length must match X.shape[1].")
    if sigma2 < 0:
        raise ValueError("sigma2 must be non-negative.")

    df_resid = n_samples - n_features
    XTX = X.T @ X
    cov_beta = sigma2 * np.linalg.solve(XTX, np.eye(n_features))
    std_error = np.sqrt(np.diag(cov_beta))
    t_stat = np.divide(
        beta_hat,
        std_error,
        out=np.full_like(beta_hat, np.nan, dtype=float),
        where=std_error > 0,
    )
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df_resid))
    t_critical = stats.t.ppf(0.975, df=df_resid)
    ci_lower = beta_hat - t_critical * std_error
    ci_upper = beta_hat + t_critical * std_error

    return pd.DataFrame(
        {
            "coefficient": beta_hat,
            "std_error": std_error,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    )
