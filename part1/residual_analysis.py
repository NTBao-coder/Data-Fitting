"""Residual diagnostics and Monte Carlo simulation for OLS."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from part1.ols_implementation import hat_matrix


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


def residual_plots(
    X: np.ndarray,
    y: np.ndarray,
    beta_hat: np.ndarray,
) -> plt.Figure:
    """
    Create four standard OLS diagnostic plots.

    The returned figure contains:
    1. Residuals vs fitted values
    2. Normal Q-Q plot
    3. Scale-location plot
    4. Cook's distance by observation index

    Parameters
    ----------
    X : np.ndarray
        Design matrix with shape (n_samples, n_features).
    y : np.ndarray
        Target vector with shape (n_samples,) or (n_samples, 1).
    beta_hat : np.ndarray
        Estimated coefficients with shape (n_features,).

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the four diagnostic subplots.
    """
    X = _as_2d_float_array(X, "X")
    y = _as_1d_float_array(y, "y")
    beta_hat = _as_1d_float_array(beta_hat, "beta_hat")

    n_samples, n_features = X.shape
    if y.shape[0] != n_samples:
        raise ValueError("X and y must contain the same number of samples.")
    if beta_hat.shape[0] != n_features:
        raise ValueError("beta_hat length must match X.shape[1].")
    if n_samples <= n_features:
        raise ValueError("n_samples must be greater than n_features.")

    y_hat = X @ beta_hat
    residuals = y - y_hat
    leverage = np.diag(hat_matrix(X))
    rss = residuals.T @ residuals
    mse = rss / (n_samples - n_features)

    leverage_complement = np.clip(1.0 - leverage, 1e-12, None)
    if np.isclose(mse, 0.0):
        standardized_residuals = np.zeros_like(residuals, dtype=float)
    else:
        standardized_residuals = residuals / np.sqrt(mse * leverage_complement)

    scale_location = np.sqrt(np.abs(standardized_residuals))
    cooks_distance = (
        (standardized_residuals**2 / n_features)
        * (leverage / leverage_complement)
    )
    observation_index = np.arange(n_samples)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    sns.scatterplot(x=y_hat, y=residuals, ax=axes[0, 0])
    axes[0, 0].axhline(0.0, color="red", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Residuals vs Fitted")
    axes[0, 0].set_xlabel("Fitted values")
    axes[0, 0].set_ylabel("Residuals")

    stats.probplot(standardized_residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Normal Q-Q")

    sns.scatterplot(x=y_hat, y=scale_location, ax=axes[1, 0])
    axes[1, 0].set_title("Scale-Location")
    axes[1, 0].set_xlabel("Fitted values")
    axes[1, 0].set_ylabel("sqrt(|standardized residuals|)")

    axes[1, 1].stem(observation_index, cooks_distance, markerfmt=",")
    axes[1, 1].set_title("Cook's Distance")
    axes[1, 1].set_xlabel("Observation index")
    axes[1, 1].set_ylabel("Cook's distance")

    fig.tight_layout()
    return fig


def monte_carlo_gauss_markov(
    n_simulations: int,
    n_samples: int,
    true_beta: np.ndarray,
) -> Dict[str, np.ndarray | pd.DataFrame]:
    """
    Run a vectorized Monte Carlo simulation to demonstrate OLS unbiasedness.

    Data-generating process:
        y = X @ true_beta + epsilon
        epsilon ~ Normal(0, 1)

    Parameters
    ----------
    n_simulations : int
        Number of simulated datasets.
    n_samples : int
        Number of observations in each simulated dataset.
    true_beta : np.ndarray
        True coefficients. The first coefficient is treated as the intercept.

    Returns
    -------
    Dict[str, np.ndarray | pd.DataFrame]
        Contains all beta estimates, empirical mean estimates, empirical bias,
        and a summary table.
    """
    true_beta = _as_1d_float_array(true_beta, "true_beta")

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    n_features = true_beta.shape[0]
    if n_features < 1:
        raise ValueError("true_beta must contain at least one coefficient.")
    if n_samples <= n_features:
        raise ValueError("n_samples must be greater than len(true_beta).")

    non_intercept_features = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=(n_simulations, n_samples, n_features - 1),
    )
    intercept = np.ones((n_simulations, n_samples, 1), dtype=float)
    X = np.concatenate((intercept, non_intercept_features), axis=2)
    epsilon = np.random.normal(loc=0.0, scale=1.0, size=(n_simulations, n_samples))
    y = X @ true_beta + epsilon

    XT = np.swapaxes(X, 1, 2)
    XTX = XT @ X
    XTy = XT @ y[..., None]
    beta_hats = (np.linalg.inv(XTX) @ XTy).squeeze(axis=2)

    beta_mean = beta_hats.mean(axis=0)
    beta_bias = beta_mean - true_beta
    summary = pd.DataFrame(
        {
            "true_beta": true_beta,
            "mean_beta_hat": beta_mean,
            "bias": beta_bias,
        }
    )

    return {
        "beta_hats": beta_hats,
        "beta_mean": beta_mean,
        "beta_bias": beta_bias,
        "summary": summary,
    }
