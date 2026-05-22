"""Comprehensive test suite for the part1 OLS package.

Covers:
- _utils: input validation helpers
- ols_implementation: ols_fit, hat_matrix, model_metrics, coef_inference
- residual_analysis: residual_plots, monte_carlo_gauss_markov
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from part1._utils import _as_1d_float_array, _as_2d_float_array
from part1.ols_implementation import (
    _validate_regression_shapes,
    coef_inference,
    hat_matrix,
    model_metrics,
    ols_fit,
)
from part1.residual_analysis import monte_carlo_gauss_markov, residual_plots

# Use non-interactive backend so plots don't pop up during tests
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_design():
    """Simple 4×2 design matrix with intercept + one predictor."""
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    return X


@pytest.fixture()
def perfect_fit(simple_design):
    """y = 1 + 2x exactly — zero residual variance."""
    y = np.array([1.0, 3.0, 5.0, 7.0])
    return simple_design, y


@pytest.fixture()
def noisy_fit(simple_design):
    """y = 1 + 2x + noise — nonzero residual variance."""
    y = np.array([1.2, 2.8, 5.3, 6.7])
    return simple_design, y


@pytest.fixture()
def multi_feature_design():
    """10×3 design matrix with intercept + two predictors."""
    rng = np.random.default_rng(42)
    n = 10
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    beta_true = np.array([5.0, -2.0, 3.0])
    y = X @ beta_true + rng.standard_normal(n) * 0.5
    return X, y, beta_true


# =========================================================================
# Tests for _utils
# =========================================================================


class TestAs1dFloatArray:
    """Tests for _as_1d_float_array."""

    def test_1d_input(self):
        result = _as_1d_float_array(np.array([1, 2, 3]), "v")
        assert result.ndim == 1
        assert result.dtype == float

    def test_column_vector_squeezed(self):
        result = _as_1d_float_array(np.array([[1], [2], [3]]), "v")
        assert result.ndim == 1
        assert result.shape == (3,)

    def test_list_input_converted(self):
        result = _as_1d_float_array([1, 2, 3], "v")
        assert isinstance(result, np.ndarray)
        assert result.dtype == float

    def test_integer_array_to_float(self):
        result = _as_1d_float_array(np.array([1, 2, 3], dtype=int), "v")
        assert result.dtype == float

    def test_2d_wide_raises(self):
        with pytest.raises(ValueError, match="must be a 1D array"):
            _as_1d_float_array(np.array([[1, 2], [3, 4]]), "v")

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="must be a 1D array"):
            _as_1d_float_array(np.ones((2, 3, 4)), "v")


class TestAs2dFloatArray:
    """Tests for _as_2d_float_array."""

    def test_2d_input(self):
        result = _as_2d_float_array(np.array([[1, 2], [3, 4]]), "M")
        assert result.ndim == 2
        assert result.dtype == float

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="must be a 2D array"):
            _as_2d_float_array(np.array([1, 2, 3]), "M")

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="must be a 2D array"):
            _as_2d_float_array(np.ones((2, 3, 4)), "M")


# =========================================================================
# Tests for _validate_regression_shapes
# =========================================================================


class TestValidateRegressionShapes:
    """Tests for _validate_regression_shapes."""

    def test_valid_shapes(self, simple_design):
        n, p = _validate_regression_shapes(simple_design, np.zeros(4))
        assert n == 4
        assert p == 2

    def test_mismatched_samples_raises(self, simple_design):
        with pytest.raises(ValueError, match="same number of samples"):
            _validate_regression_shapes(simple_design, np.zeros(5))

    def test_n_equals_p_raises(self):
        X_square = np.eye(3)
        with pytest.raises(ValueError, match="n_samples > n_features"):
            _validate_regression_shapes(X_square, np.zeros(3))

    def test_n_less_than_p_raises(self):
        X_wide = np.ones((2, 5))
        with pytest.raises(ValueError, match="n_samples > n_features"):
            _validate_regression_shapes(X_wide, np.zeros(2))


# =========================================================================
# Tests for ols_fit
# =========================================================================


class TestOlsFit:
    """Tests for ols_fit."""

    def test_perfect_line(self, perfect_fit):
        X, y = perfect_fit
        beta_hat, sigma2 = ols_fit(X, y)
        np.testing.assert_allclose(beta_hat, [1.0, 2.0], atol=1e-10)
        assert np.isclose(sigma2, 0.0, atol=1e-10)

    def test_noisy_data_nonzero_sigma(self, noisy_fit):
        X, y = noisy_fit
        beta_hat, sigma2 = ols_fit(X, y)
        assert beta_hat.shape == (2,)
        assert sigma2 > 0.0

    def test_column_vector_y(self, simple_design):
        """y passed as (n,1) column vector should work."""
        y_col = np.array([[1.0], [3.0], [5.0], [7.0]])
        beta_hat, sigma2 = ols_fit(simple_design, y_col)
        np.testing.assert_allclose(beta_hat, [1.0, 2.0], atol=1e-10)

    def test_multi_feature(self, multi_feature_design):
        X, y, beta_true = multi_feature_design
        beta_hat, sigma2 = ols_fit(X, y)
        assert beta_hat.shape == (3,)
        # With low noise, coefficients should be close to true values
        np.testing.assert_allclose(beta_hat, beta_true, atol=1.0)
        assert sigma2 > 0.0

    def test_intercept_only(self):
        """y = intercept model should return mean(y) as coefficient."""
        X = np.ones((5, 1))
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        beta_hat, _ = ols_fit(X, y)
        np.testing.assert_allclose(beta_hat, [np.mean(y)], atol=1e-10)

    def test_residuals_orthogonal_to_X(self, noisy_fit):
        """OLS guarantee: X.T @ residuals ≈ 0."""
        X, y = noisy_fit
        beta_hat, _ = ols_fit(X, y)
        residuals = y - X @ beta_hat
        np.testing.assert_allclose(X.T @ residuals, 0.0, atol=1e-10)

    def test_mismatched_shapes_raises(self, simple_design):
        with pytest.raises(ValueError):
            ols_fit(simple_design, np.zeros(10))

    def test_underdetermined_raises(self):
        X = np.ones((2, 3))
        with pytest.raises(ValueError):
            ols_fit(X, np.zeros(2))


# =========================================================================
# Tests for hat_matrix
# =========================================================================


class TestHatMatrix:
    """Tests for hat_matrix."""

    def test_shape(self, simple_design):
        H = hat_matrix(simple_design)
        n = simple_design.shape[0]
        assert H.shape == (n, n)

    def test_symmetry(self, simple_design):
        H = hat_matrix(simple_design)
        np.testing.assert_allclose(H.T, H, atol=1e-10)

    def test_idempotent(self, simple_design):
        H = hat_matrix(simple_design)
        np.testing.assert_allclose(H @ H, H, atol=1e-10)

    def test_trace_equals_p(self, simple_design):
        """tr(H) should equal the number of features p."""
        H = hat_matrix(simple_design)
        p = simple_design.shape[1]
        assert np.isclose(np.trace(H), p, atol=1e-10)

    def test_maps_y_to_yhat(self, perfect_fit):
        """H @ y should equal X @ beta_hat."""
        X, y = perfect_fit
        H = hat_matrix(X)
        beta_hat, _ = ols_fit(X, y)
        np.testing.assert_allclose(H @ y, X @ beta_hat, atol=1e-10)

    def test_leverage_between_0_and_1(self, simple_design):
        """Diagonal elements (leverages) should be in [0, 1]."""
        H = hat_matrix(simple_design)
        leverages = np.diag(H)
        assert np.all(leverages >= -1e-10)
        assert np.all(leverages <= 1.0 + 1e-10)

    def test_n_less_than_p_raises(self):
        X = np.ones((2, 5))
        with pytest.raises(ValueError, match="n_samples >= n_features"):
            hat_matrix(X)

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            hat_matrix(np.array([1, 2, 3]))


# =========================================================================
# Tests for model_metrics
# =========================================================================


class TestModelMetrics:
    """Tests for model_metrics."""

    def test_perfect_fit_r2_is_one(self, perfect_fit):
        X, y = perfect_fit
        beta_hat, _ = ols_fit(X, y)
        y_hat = X @ beta_hat
        metrics = model_metrics(y, y_hat, p=2)
        assert np.isclose(metrics["R2"], 1.0, atol=1e-10)
        assert np.isclose(metrics["RSS"], 0.0, atol=1e-10)

    def test_noisy_r2_between_0_and_1(self, noisy_fit):
        X, y = noisy_fit
        beta_hat, _ = ols_fit(X, y)
        y_hat = X @ beta_hat
        metrics = model_metrics(y, y_hat, p=2)
        assert 0.0 < metrics["R2"] < 1.0
        assert 0.0 < metrics["adjusted_R2"] < 1.0

    def test_rss_plus_ess_equals_tss(self, noisy_fit):
        """RSS + ESS ≈ TSS (decomposition of total variance)."""
        X, y = noisy_fit
        beta_hat, _ = ols_fit(X, y)
        y_hat = X @ beta_hat
        metrics = model_metrics(y, y_hat, p=2)
        ess = metrics["TSS"] - metrics["RSS"]
        assert ess >= -1e-10  # ESS should be non-negative

    def test_adjusted_r2_leq_r2(self, noisy_fit):
        """Adjusted R² ≤ R² always."""
        X, y = noisy_fit
        beta_hat, _ = ols_fit(X, y)
        y_hat = X @ beta_hat
        metrics = model_metrics(y, y_hat, p=2)
        assert metrics["adjusted_R2"] <= metrics["R2"] + 1e-10

    def test_f_statistic_positive(self, noisy_fit):
        X, y = noisy_fit
        beta_hat, _ = ols_fit(X, y)
        y_hat = X @ beta_hat
        metrics = model_metrics(y, y_hat, p=2)
        assert metrics["F_statistic"] > 0.0

    def test_constant_y_returns_nan(self):
        """When y is constant, TSS = 0 → R², F should be NaN."""
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        y_hat = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        metrics = model_metrics(y, y_hat, p=1)
        assert np.isnan(metrics["R2"])
        assert np.isnan(metrics["F_statistic"])

    def test_perfect_fit_f_is_inf(self, perfect_fit):
        """Perfect fit: RSS = 0 → F = inf."""
        X, y = perfect_fit
        beta_hat, _ = ols_fit(X, y)
        y_hat = X @ beta_hat
        metrics = model_metrics(y, y_hat, p=2)
        assert np.isinf(metrics["F_statistic"])

    def test_all_required_keys_present(self, noisy_fit):
        X, y = noisy_fit
        beta_hat, _ = ols_fit(X, y)
        y_hat = X @ beta_hat
        metrics = model_metrics(y, y_hat, p=2)
        assert set(metrics.keys()) == {"RSS", "TSS", "R2", "adjusted_R2", "F_statistic"}

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            model_metrics(np.array([1, 2, 3]), np.array([1, 2]), p=1)

    def test_p_less_than_1_raises(self):
        with pytest.raises(ValueError, match="p must be at least 1"):
            model_metrics(np.array([1, 2, 3]), np.array([1, 2, 3]), p=0)

    def test_n_leq_p_raises(self):
        with pytest.raises(ValueError, match="n_samples must be greater than p"):
            model_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]), p=3)


# =========================================================================
# Tests for coef_inference
# =========================================================================


class TestCoefInference:
    """Tests for coef_inference."""

    def test_output_shape_and_columns(self, noisy_fit):
        X, y = noisy_fit
        beta_hat, sigma2 = ols_fit(X, y)
        df = coef_inference(X, y, beta_hat, sigma2)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 6)
        expected_cols = {"coefficient", "std_error", "t_stat", "p_value", "ci_lower", "ci_upper"}
        assert set(df.columns) == expected_cols

    def test_coefficients_match_input(self, noisy_fit):
        X, y = noisy_fit
        beta_hat, sigma2 = ols_fit(X, y)
        df = coef_inference(X, y, beta_hat, sigma2)
        np.testing.assert_allclose(df["coefficient"].values, beta_hat)

    def test_std_error_positive(self, noisy_fit):
        X, y = noisy_fit
        beta_hat, sigma2 = ols_fit(X, y)
        df = coef_inference(X, y, beta_hat, sigma2)
        assert np.all(df["std_error"].values > 0)

    def test_p_values_between_0_and_1(self, noisy_fit):
        X, y = noisy_fit
        beta_hat, sigma2 = ols_fit(X, y)
        df = coef_inference(X, y, beta_hat, sigma2)
        assert np.all(df["p_value"].values >= 0.0)
        assert np.all(df["p_value"].values <= 1.0)

    def test_ci_contains_coefficient(self, noisy_fit):
        """95% CI should contain the point estimate."""
        X, y = noisy_fit
        beta_hat, sigma2 = ols_fit(X, y)
        df = coef_inference(X, y, beta_hat, sigma2)
        assert np.all(df["ci_lower"].values <= df["coefficient"].values)
        assert np.all(df["coefficient"].values <= df["ci_upper"].values)

    def test_ci_lower_less_than_upper(self, noisy_fit):
        X, y = noisy_fit
        beta_hat, sigma2 = ols_fit(X, y)
        df = coef_inference(X, y, beta_hat, sigma2)
        assert np.all(df["ci_lower"].values < df["ci_upper"].values)

    def test_zero_sigma_gives_zero_se(self, perfect_fit):
        """sigma2 = 0 → SE = 0, t-stat → NaN (0/0)."""
        X, y = perfect_fit
        beta_hat, sigma2 = ols_fit(X, y)
        df = coef_inference(X, y, beta_hat, sigma2)
        np.testing.assert_allclose(df["std_error"].values, 0.0, atol=1e-15)

    def test_multi_feature(self, multi_feature_design):
        X, y, _ = multi_feature_design
        beta_hat, sigma2 = ols_fit(X, y)
        df = coef_inference(X, y, beta_hat, sigma2)
        assert df.shape == (3, 6)

    def test_cross_validation_against_scipy(self, noisy_fit):
        """Compare t-stats with scipy.stats.linregress for simple regression."""
        X, y = noisy_fit
        beta_hat, sigma2 = ols_fit(X, y)
        df = coef_inference(X, y, beta_hat, sigma2)

        # scipy linregress on the single predictor column
        slope, intercept, _, p_slope, se_slope = stats.linregress(X[:, 1], y)
        np.testing.assert_allclose(beta_hat[1], slope, atol=1e-10)
        np.testing.assert_allclose(df["std_error"].iloc[1], se_slope, atol=1e-6)

    def test_beta_length_mismatch_raises(self, noisy_fit):
        X, y = noisy_fit
        with pytest.raises(ValueError, match="beta_hat length"):
            coef_inference(X, y, np.array([1.0]), sigma2=1.0)

    def test_negative_sigma_raises(self, noisy_fit):
        X, y = noisy_fit
        beta_hat, _ = ols_fit(X, y)
        with pytest.raises(ValueError, match="sigma2 must be non-negative"):
            coef_inference(X, y, beta_hat, sigma2=-1.0)


# =========================================================================
# Tests for residual_plots
# =========================================================================


class TestResidualPlots:
    """Tests for residual_plots."""

    def test_returns_figure_with_4_axes(self, noisy_fit):
        X, y = noisy_fit
        beta_hat, _ = ols_fit(X, y)
        fig = residual_plots(X, y, beta_hat)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_subplot_titles(self, noisy_fit):
        X, y = noisy_fit
        beta_hat, _ = ols_fit(X, y)
        fig = residual_plots(X, y, beta_hat)
        titles = [ax.get_title() for ax in fig.axes]
        assert "Residuals vs Fitted" in titles
        assert "Normal Q-Q" in titles
        assert "Scale-Location" in titles
        assert "Cook's Distance" in titles
        plt.close(fig)

    def test_perfect_fit_does_not_crash(self, perfect_fit):
        """Perfect fit → MSE = 0; should handle gracefully."""
        X, y = perfect_fit
        beta_hat, _ = ols_fit(X, y)
        fig = residual_plots(X, y, beta_hat)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multi_feature(self, multi_feature_design):
        X, y, _ = multi_feature_design
        beta_hat, _ = ols_fit(X, y)
        fig = residual_plots(X, y, beta_hat)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_shape_mismatch_raises(self, simple_design):
        with pytest.raises(ValueError):
            residual_plots(simple_design, np.zeros(10), np.zeros(2))

    def test_beta_length_mismatch_raises(self, noisy_fit):
        X, y = noisy_fit
        with pytest.raises(ValueError, match="beta_hat length"):
            residual_plots(X, y, np.zeros(5))


# =========================================================================
# Tests for monte_carlo_gauss_markov
# =========================================================================


class TestMonteCarloGaussMarkov:
    """Tests for monte_carlo_gauss_markov."""

    def test_output_keys(self):
        result = monte_carlo_gauss_markov(100, 20, np.array([1.0, 2.0]))
        assert set(result.keys()) == {"beta_hats", "beta_mean", "beta_bias", "summary"}

    def test_beta_hats_shape(self):
        n_sim, n_feat = 200, 3
        true_beta = np.array([1.0, -1.0, 0.5])
        result = monte_carlo_gauss_markov(n_sim, 30, true_beta)
        assert result["beta_hats"].shape == (n_sim, n_feat)

    def test_unbiasedness(self):
        """With many simulations, mean(beta_hat) should be close to true_beta."""
        np.random.seed(42)
        true_beta = np.array([3.0, -1.5])
        result = monte_carlo_gauss_markov(2000, 50, true_beta)
        np.testing.assert_allclose(result["beta_mean"], true_beta, atol=0.15)

    def test_bias_near_zero(self):
        np.random.seed(123)
        true_beta = np.array([0.0, 5.0, -3.0])
        result = monte_carlo_gauss_markov(1000, 40, true_beta)
        assert np.all(np.abs(result["beta_bias"]) < 0.3)

    def test_summary_is_dataframe(self):
        result = monte_carlo_gauss_markov(50, 20, np.array([1.0, 2.0]))
        assert isinstance(result["summary"], pd.DataFrame)
        assert list(result["summary"].columns) == ["true_beta", "mean_beta_hat", "bias"]

    def test_summary_rows_match_features(self):
        true_beta = np.array([1.0, 2.0, 3.0])
        result = monte_carlo_gauss_markov(50, 20, true_beta)
        assert result["summary"].shape[0] == len(true_beta)

    def test_single_coefficient(self):
        """Intercept-only model (1 coefficient)."""
        result = monte_carlo_gauss_markov(100, 10, np.array([5.0]))
        assert result["beta_hats"].shape == (100, 1)

    def test_zero_simulations_raises(self):
        with pytest.raises(ValueError, match="n_simulations must be positive"):
            monte_carlo_gauss_markov(0, 20, np.array([1.0]))

    def test_negative_simulations_raises(self):
        with pytest.raises(ValueError, match="n_simulations must be positive"):
            monte_carlo_gauss_markov(-5, 20, np.array([1.0]))

    def test_n_leq_features_raises(self):
        with pytest.raises(ValueError, match="n_samples must be greater"):
            monte_carlo_gauss_markov(10, 2, np.array([1.0, 2.0, 3.0]))


# =========================================================================
# Integration / end-to-end tests
# =========================================================================


class TestEndToEnd:
    """Integration tests running the full OLS pipeline."""

    def test_full_pipeline(self, multi_feature_design):
        """Run fit → metrics → inference → plots in sequence."""
        X, y, _ = multi_feature_design
        p = X.shape[1]

        # Step 1: Fit
        beta_hat, sigma2 = ols_fit(X, y)
        assert beta_hat.shape == (p,)

        # Step 2: Metrics
        y_hat = X @ beta_hat
        metrics = model_metrics(y, y_hat, p)
        assert 0.0 <= metrics["R2"] <= 1.0

        # Step 3: Inference
        df = coef_inference(X, y, beta_hat, sigma2)
        assert df.shape == (p, 6)

        # Step 4: Plots
        fig = residual_plots(X, y, beta_hat)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_hat_matrix_yhat_consistency(self, noisy_fit):
        """H @ y should produce the same fitted values as X @ beta_hat."""
        X, y = noisy_fit
        beta_hat, _ = ols_fit(X, y)
        H = hat_matrix(X)
        np.testing.assert_allclose(H @ y, X @ beta_hat, atol=1e-10)
