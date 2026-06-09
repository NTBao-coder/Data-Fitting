"""Core Ordinary Least Squares implementation.

The functions in this module implement the OLS normal-equation workflow. The
design matrix ``X`` is assumed to already include an intercept column when the
model requires one.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from scipy import stats

from part1 import helper_function as hf


def ols_fit(X: hf.Array, y: hf.Array) -> Tuple[hf.Array, float]:
    """
    Estimate OLS coefficients and residual variance.

    Formula:
        beta_hat = (X.T @ X)^(-1) @ X.T @ y
        sigma_squared = RSS / (n - p)

    Parameters
    ----------
    X : array-like
        Design matrix with shape (n_samples, n_features). Include an intercept
        column before calling this function if the model needs one.
    y : array-like
        Target vector with shape (n_samples,) or (n_samples, 1).

    Returns
    -------
    Tuple[array, float]
        Estimated coefficients with shape (n_features,) and unbiased residual
        variance estimate.
    """
    X = hf.as_2d_float_array(X, "X")
    y = hf.as_1d_float_array(y, "y")
    n_samples, n_features = hf.validate_regression_shapes(X, y)

    XT = hf.transpose(X)
    XTX = hf.matmul(XT, X)
    beta_hat = hf.matmul(hf.matmul(hf.inverse(XTX), XT), y)
    residuals = hf.subtract(y, hf.matmul(X, beta_hat))
    rss = hf.dot(residuals, residuals)
    sigma_squared = rss / (n_samples - n_features)

    return beta_hat, float(sigma_squared)


def hat_matrix(X: hf.Array) -> hf.Array:
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
    X : array-like
        Design matrix with shape (n_samples, n_features).

    Returns
    -------
    array
        Hat matrix with shape (n_samples, n_samples).
    """
    X = hf.as_2d_float_array(X, "X")
    n_samples, n_features = X.shape
    if n_samples < n_features:
        raise ValueError("X must have n_samples >= n_features.")

    XT = hf.transpose(X)
    H = hf.matmul(hf.matmul(X, hf.inverse(hf.matmul(XT, X))), XT)

    assert hf.all_close(hf.transpose(H), H, atol=1e-8), "Hat matrix must be symmetric."
    assert hf.all_close(hf.matmul(H, H), H, atol=1e-8), "Hat matrix must be idempotent."

    return H


def model_metrics(y: hf.Array, y_hat: hf.Array, p: int) -> Dict[str, float]:
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
    y : array-like
        Observed target values.
    y_hat : array-like
        Fitted target values.
    p : int
        Number of coefficients, including the intercept.

    Returns
    -------
    Dict[str, float]
        RSS, TSS, R2, adjusted R2, and F-statistic.
    """
    y = hf.as_1d_float_array(y, "y")
    y_hat = hf.as_1d_float_array(y_hat, "y_hat")

    if y.shape != y_hat.shape:
        raise ValueError("y and y_hat must have the same shape.")
    if p < 1:
        raise ValueError("p must be at least 1.")

    n_samples = y.shape[0]
    if n_samples <= p:
        raise ValueError("n_samples must be greater than p.")

    residuals = hf.subtract(y, y_hat)
    centered_y = hf.subtract(y, hf.mean(y))
    rss = float(hf.dot(residuals, residuals))
    tss = float(hf.dot(centered_y, centered_y))

    if hf.is_close(tss, 0.0):
        r_squared = hf.NAN
        adjusted_r_squared = hf.NAN
        f_statistic = hf.NAN
    else:
        r_squared = 1.0 - rss / tss
        adjusted_r_squared = 1.0 - (rss / (n_samples - p)) / (tss / (n_samples - 1))
        if p == 1 or hf.is_close(rss, 0.0):
            f_statistic = hf.INF
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
    X: hf.Array,
    y: hf.Array,
    beta_hat: hf.Array,
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
    X : array-like
        Design matrix with shape (n_samples, n_features).
    y : array-like
        Target vector with shape (n_samples,) or (n_samples, 1).
    beta_hat : array-like
        Estimated coefficients with shape (n_features,).
    sigma2 : float
        Estimated residual variance.

    Returns
    -------
    pd.DataFrame
        Table with coefficient, standard error, t-statistic, p-value, and 95%
        confidence interval columns.
    """
    X = hf.as_2d_float_array(X, "X")
    y = hf.as_1d_float_array(y, "y")
    beta_hat = hf.as_1d_float_array(beta_hat, "beta_hat")
    n_samples, n_features = hf.validate_regression_shapes(X, y)

    if beta_hat.shape[0] != n_features:
        raise ValueError("beta_hat length must match X.shape[1].")
    if sigma2 < 0:
        raise ValueError("sigma2 must be non-negative.")

    df_resid = n_samples - n_features
    XT = hf.transpose(X)
    cov_beta = hf.scalar_multiply(hf.inverse(hf.matmul(XT, X)), sigma2)
    std_error = hf.sqrt(hf.diagonal(cov_beta))
    t_stat = hf.divide(
        beta_hat,
        std_error,
        out=hf.full_like(beta_hat, hf.NAN, dtype=float),
        where=[value > 0 for value in std_error],
    )
    p_value = 2.0 * (1.0 - stats.t.cdf(hf.absolute(t_stat), df=df_resid))
    t_critical = stats.t.ppf(0.975, df=df_resid)
    ci_lower = hf.subtract(beta_hat, hf.scalar_multiply(std_error, t_critical))
    ci_upper = hf.add(beta_hat, hf.scalar_multiply(std_error, t_critical))

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
