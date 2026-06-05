"""Residual diagnostics and Monte Carlo simulation for OLS."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from part1 import helper_function as hf
from part1.ols_implementation import hat_matrix


def residual_plots(
    X: hf.Array,
    y: hf.Array,
    beta_hat: hf.Array,
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
    X : array-like
        Design matrix with shape (n_samples, n_features).
    y : array-like
        Target vector with shape (n_samples,) or (n_samples, 1).
    beta_hat : array-like
        Estimated coefficients with shape (n_features,).

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the four diagnostic subplots.
    """
    X = hf.as_2d_float_array(X, "X")
    y = hf.as_1d_float_array(y, "y")
    beta_hat = hf.as_1d_float_array(beta_hat, "beta_hat")

    n_samples, n_features = X.shape
    if y.shape[0] != n_samples:
        raise ValueError("X and y must contain the same number of samples.")
    if beta_hat.shape[0] != n_features:
        raise ValueError("beta_hat length must match X.shape[1].")
    if n_samples <= n_features:
        raise ValueError("n_samples must be greater than n_features.")

    y_hat = X @ beta_hat
    residuals = y - y_hat
    leverage = hf.diagonal(hat_matrix(X))
    rss = residuals.T @ residuals
    mse = rss / (n_samples - n_features)

    leverage_complement = hf.clip(1.0 - leverage, 1e-12, None)
    if hf.is_close(mse, 0.0):
        standardized_residuals = hf.zeros_like(residuals, dtype=float)
    else:
        standardized_residuals = residuals / hf.sqrt(mse * leverage_complement)

    scale_location = hf.sqrt(hf.absolute(standardized_residuals))
    cooks_distance = (
        (standardized_residuals**2 / n_features)
        * (leverage / leverage_complement)
    )
    observation_index = hf.arange(n_samples)

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

    axes[1, 1].stem(list(observation_index), list(cooks_distance), markerfmt=",")
    axes[1, 1].set_title("Cook's Distance")
    axes[1, 1].set_xlabel("Observation index")
    axes[1, 1].set_ylabel("Cook's distance")

    fig.tight_layout()
    return fig


def monte_carlo_gauss_markov(
    n_simulations: int,
    n_samples: int,
    true_beta: hf.Array,
) -> Dict[str, hf.Array | pd.DataFrame]:
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
    true_beta : array-like
        True coefficients. The first coefficient is treated as the intercept.

    Returns
    -------
    Dict[str, array | pd.DataFrame]
        Contains all beta estimates, empirical mean estimates, empirical bias,
        and a summary table.
    """
    true_beta = hf.as_1d_float_array(true_beta, "true_beta")

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    n_features = true_beta.shape[0]
    if n_features < 1:
        raise ValueError("true_beta must contain at least one coefficient.")
    if n_samples <= n_features:
        raise ValueError("n_samples must be greater than len(true_beta).")

    beta_hats_list = []
    for _ in range(n_simulations):
        if n_features > 1:
            non_intercept = hf.random_normal(loc=0.0, scale=1.0, size=(n_samples, n_features - 1))
            intercept = hf.ones((n_samples, 1), dtype=float)
            X = hf.concatenate([intercept, non_intercept], axis=1)
        else:
            X = hf.ones((n_samples, 1), dtype=float)
            
        epsilon = hf.random_normal(loc=0.0, scale=1.0, size=n_samples)
        y = X @ true_beta + epsilon
        
        beta_hat = hf.solve(X.T @ X, X.T @ y)
        beta_hats_list.append(beta_hat)

    beta_hats = hf.Matrix(beta_hats_list)

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
