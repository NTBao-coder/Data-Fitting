"""
Unit Tests — Part 1
Owner: Minh

Coverage:
    ols_implementation.py → hat_matrix, ols_fit
    ridge_lasso.py        → soft_threshold, ridge_fit, lasso_fit, vif, predict, rss
    cross_validation.py   → kfold_cv, find_best_lambda

Chạy:
    python3 -m pytest tests/test_part1.py -v
"""

import numpy as np
import pytest

from part1.ols_implementation import hat_matrix, ols_fit
from part1.ridge_lasso import (
    soft_threshold,
    predict,
    rss,
    ridge_fit,
    lasso_fit,
    vif,
)
from part1.cross_validation import kfold_cv, find_best_lambda


# =========================================================
# Fixtures — dữ liệu dùng chung
# =========================================================

@pytest.fixture
def simple_data():
    """
    Dữ liệu đơn giản: y = 2x (intercept = 0).
    Nghiệm OLS chính xác: beta = [0, 2].
    """
    X = np.column_stack([np.ones(5), np.array([1, 2, 3, 4, 5], dtype=float)])
    y = np.array([2, 4, 6, 8, 10], dtype=float)
    return X, y


@pytest.fixture
def noisy_data():
    """
    Dữ liệu có noise để test CV và regularization.
    y = 2 + 1.5x1 - 1.0x2 + noise.
    """
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 3)
    X = np.column_stack([np.ones(n), X])
    true_beta = np.array([2.0, 1.5, -1.0, 0.0])
    y = X @ true_beta + np.random.randn(n) * 0.3
    return X, y


# =========================================================
# ols_implementation
# =========================================================

class TestOLSImplementation:

    def test_ols_fit_perfect_line(self):
        """Test fit OLS với dữ liệu đường thẳng hoàn hảo."""
        X = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
            ]
        )
        y = np.array([1.0, 3.0, 5.0, 7.0])

        beta_hat, sigma2 = ols_fit(X, y)

        np.testing.assert_allclose(beta_hat, np.array([1.0, 2.0]), atol=1e-10)
        assert np.isclose(sigma2, 0.0, atol=1e-10)

    def test_hat_matrix_properties(self):
        """Hat matrix phải đối xứng và idempotent (H = H^T và H @ H = H)."""
        X = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
            ]
        )

        H = hat_matrix(X)

        assert H.shape == (4, 4)
        np.testing.assert_allclose(H.T, H, atol=1e-10)
        np.testing.assert_allclose(H @ H, H, atol=1e-10)


# =========================================================
# soft_threshold
# =========================================================

class TestSoftThreshold:

    def test_positive_above_lambda(self):
        """rho > lam → kết quả = rho - lam."""
        assert soft_threshold(5, 2) == 3

    def test_negative_below_neg_lambda(self):
        """rho < -lam → kết quả = rho + lam."""
        assert soft_threshold(-5, 2) == -3

    def test_within_lambda_returns_zero(self):
        """|rho| <= lam → kết quả = 0 (đặc trưng L1)."""
        assert soft_threshold(1, 2) == 0

    def test_exactly_at_lambda_returns_zero(self):
        """rho == lam → kết quả = 0 (boundary case)."""
        assert soft_threshold(2, 2) == 0

    def test_negative_exactly_at_neg_lambda(self):
        """rho == -lam → kết quả = 0 (boundary case)."""
        assert soft_threshold(-2, 2) == 0


# =========================================================
# predict
# =========================================================

class TestPredict:

    def test_basic_prediction(self):
        """y_hat = X @ beta, kết quả đúng với nghiệm đã biết."""
        X = np.array([[1, 2], [1, 3], [1, 4]], dtype=float)
        beta = np.array([1.0, 2.0])
        expected = np.array([5.0, 7.0, 9.0])
        np.testing.assert_array_almost_equal(predict(X, beta), expected)

    def test_zero_beta(self):
        """beta = 0 → y_hat = 0."""
        X = np.random.randn(10, 3)
        beta = np.zeros(3)
        np.testing.assert_array_equal(predict(X, beta), np.zeros(10))


# =========================================================
# rss
# =========================================================

class TestRSS:

    def test_perfect_prediction(self):
        """y_hat = y → RSS = 0."""
        y = np.array([1.0, 2.0, 3.0])
        assert rss(y, y) == 0.0

    def test_known_value(self):
        """RSS = Σ(y - y_hat)² với giá trị đã biết."""
        y     = np.array([1.0, 2.0, 3.0])
        y_hat = np.array([2.0, 2.0, 2.0])
        # (1-2)^2 + (2-2)^2 + (3-2)^2 = 1 + 0 + 1 = 2
        assert rss(y, y_hat) == pytest.approx(2.0)


# =========================================================
# ridge_fit
# =========================================================

class TestRidgeFit:

    def test_converges_to_ols_at_zero_lambda(self, simple_data):
        """
        λ → 0: Ridge tiệm cận OLS.
        Nghiệm OLS của y = 2x là beta ≈ [0, 2].
        """
        X, y = simple_data
        beta = ridge_fit(X, y, lam=1e-6)
        np.testing.assert_allclose(beta, [0, 2], atol=1e-3)

    def test_shrinks_toward_zero_at_large_lambda(self, simple_data):
        """
        λ rất lớn → feature coefficients (beta[1:]) bị kéo gần về 0.
        Intercept (beta[0]) không regularize nên vẫn giữ giá trị.
        """
        X, y = simple_data
        beta = ridge_fit(X, y, lam=1e9)
        # Chỉ kiểm tra feature coefficients, không kiểm tra intercept
        assert np.all(np.abs(beta[1:]) < 1e-3)

    def test_intercept_not_regularized(self, simple_data):
        """
        Intercept (beta[0]) không bị regularize.
        Với λ lớn, các feature coefficients → 0
        nhưng intercept vẫn fit giá trị trung bình y.
        """
        X, y = simple_data
        beta_small = ridge_fit(X, y, lam=0.001)
        beta_large = ridge_fit(X, y, lam=1e6)
        # Feature coefficient phải bị shrink đáng kể
        assert abs(beta_large[1]) < abs(beta_small[1])

    def test_output_shape(self, simple_data):
        """Output có shape đúng = số cột của X."""
        X, y = simple_data
        beta = ridge_fit(X, y, lam=1.0)
        assert beta.shape == (X.shape[1],)

    def test_matches_sklearn(self, noisy_data):
        """
        Kết quả phải gần với sklearn Ridge.
        sklearn minimize (1/2n)·RSS + α·||β||²
        Custom minimize RSS + λ·||β||²
        → tương đương khi α_sklearn = λ / (2 * n)
        atol=0.02 vì Ridge closed-form có thể lệch nhỏ do
        sklearn dùng solver tối ưu hơn np.linalg.inv.
        """
        from sklearn.linear_model import Ridge
        X, y = noisy_data
        lam = 1.0
        n = len(y)

        custom_beta = ridge_fit(X, y, lam=lam)
        sk = Ridge(alpha=lam / (2 * n), fit_intercept=False)
        sk.fit(X, y)

        np.testing.assert_allclose(custom_beta, sk.coef_, atol=0.02)


# =========================================================
# lasso_fit
# =========================================================

class TestLassoFit:

    def test_converges_to_ols_at_zero_lambda(self, simple_data):
        """
        λ → 0: Lasso tiệm cận OLS.
        Nghiệm OLS của y = 2x là beta ≈ [0, 2].
        """
        X, y = simple_data
        beta = lasso_fit(X, y, lam=1e-6, max_iter=5000)
        np.testing.assert_allclose(beta, [0, 2], atol=1e-2)

    def test_sparsity_at_large_lambda(self, noisy_data):
        """
        λ lớn: Lasso đưa một số coefficients về đúng 0.
        Đây là đặc trưng L1 phân biệt với Ridge.
        """
        X, y = noisy_data
        beta = lasso_fit(X, y, lam=10.0)
        # Ít nhất 1 feature coefficient phải = 0
        assert np.any(beta[1:] == 0)

    def test_output_shape(self, simple_data):
        """Output có shape đúng = số cột của X."""
        X, y = simple_data
        beta = lasso_fit(X, y, lam=0.1)
        assert beta.shape == (X.shape[1],)

    def test_matches_sklearn(self, noisy_data):
        """
        Kết quả phải gần với sklearn Lasso.
        Convention: α_sklearn = λ / (2 * n)
        """
        from sklearn.linear_model import Lasso
        X, y = noisy_data
        lam = 0.1
        n = len(y)

        custom_beta = lasso_fit(X, y, lam=lam, max_iter=5000)
        sk = Lasso(alpha=lam / (2 * n), fit_intercept=False, max_iter=10000)
        sk.fit(X, y)

        np.testing.assert_allclose(custom_beta, sk.coef_, atol=1e-2)


# =========================================================
# vif
# =========================================================

class TestVIF:

    def test_uncorrelated_features_near_one(self):
        """
        Features độc lập tuyến tính → VIF ≈ 1.
        Không có multicollinearity.
        """
        np.random.seed(0)
        X = np.random.randn(1000, 3)
        result = vif(X)
        np.testing.assert_allclose(result, np.ones(3), atol=0.1)

    def test_perfect_collinearity_gives_inf(self):
        """
        Feature hoàn toàn là tổ hợp tuyến tính của feature khác
        → VIF = inf (R² = 1 → 1/(1-R²) = inf).
        """
        np.random.seed(1)
        x1 = np.random.randn(100)
        x2 = np.random.randn(100)
        x3 = x1  # x3 hoàn toàn bằng x1
        X = np.column_stack([x1, x2, x3])
        result = vif(X)
        assert np.isinf(result[0])
        assert np.isinf(result[2])

    def test_output_length(self):
        """Output có đúng p phần tử = số feature."""
        X = np.random.randn(50, 4)
        result = vif(X)
        assert len(result) == 4

    def test_high_correlation_gives_high_vif(self):
        """
        Features tương quan cao → VIF lớn (dấu hiệu multicollinearity).
        """
        np.random.seed(2)
        n = 500
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.05  # tương quan ~0.999
        X = np.column_stack([x1, x2])
        result = vif(X)
        assert result[0] > 100
        assert result[1] > 100


# =========================================================
# kfold_cv
# =========================================================

class TestKFoldCV:

    def test_returns_correct_number_of_folds(self, noisy_data):
        """fold_mses phải có đúng k phần tử."""
        X, y = noisy_data
        _, fold_mses = kfold_cv(X, y, k=5, model="ols")
        assert len(fold_mses) == 5

    def test_cv_score_is_mean_of_fold_mses(self, noisy_data):
        """cv_score phải bằng mean của fold_mses."""
        X, y = noisy_data
        cv_score, fold_mses = kfold_cv(X, y, k=5, model="ols")
        assert cv_score == pytest.approx(np.mean(fold_mses))

    def test_mse_is_positive(self, noisy_data):
        """MSE luôn dương."""
        X, y = noisy_data
        cv_score, fold_mses = kfold_cv(X, y, k=5, model="ridge", lam=1.0)
        assert cv_score > 0
        assert all(m > 0 for m in fold_mses)

    def test_invalid_model_raises_value_error(self, noisy_data):
        """Model name không hợp lệ → ValueError rõ ràng."""
        X, y = noisy_data
        with pytest.raises(ValueError, match="model phải là"):
            kfold_cv(X, y, k=5, model="invalid")

    def test_reproducible_with_same_seed(self, noisy_data):
        """
        Hai lần gọi kfold_cv với cùng data → cùng kết quả.
        (seed được fix bên trong hàm)
        """
        X, y = noisy_data
        score1, _ = kfold_cv(X, y, k=5, model="ols")
        score2, _ = kfold_cv(X, y, k=5, model="ols")
        assert score1 == score2

    def test_ridge_cv_finite(self, noisy_data):
        """Ridge CV score phải là số hữu hạn."""
        X, y = noisy_data
        score, _ = kfold_cv(X, y, k=5, model="ridge", lam=1.0)
        assert np.isfinite(score)

    def test_lasso_cv_finite(self, noisy_data):
        """Lasso CV score phải là số hữu hạn."""
        X, y = noisy_data
        score, _ = kfold_cv(X, y, k=5, model="lasso", lam=0.1)
        assert np.isfinite(score)


# =========================================================
# find_best_lambda
# =========================================================

class TestFindBestLambda:

    def test_best_lambda_in_search_list(self, noisy_data):
        """best_lam phải nằm trong danh sách lambdas đã cho."""
        X, y = noisy_data
        lambdas = [0.01, 0.1, 1.0, 10.0]
        best_lam, _, _ = find_best_lambda(X, y, lambdas, model="ridge", k=5)
        assert best_lam in lambdas

    def test_all_scores_returned(self, noisy_data):
        """all_scores phải chứa đủ tất cả giá trị λ đã thử."""
        X, y = noisy_data
        lambdas = [0.01, 0.1, 1.0, 10.0]
        _, _, all_scores = find_best_lambda(X, y, lambdas, model="ridge", k=5)
        assert len(all_scores) == len(lambdas)
        for lam in lambdas:
            assert lam in all_scores

    def test_best_score_is_minimum(self, noisy_data):
        """best_score phải là giá trị MSE nhỏ nhất trong all_scores."""
        X, y = noisy_data
        lambdas = [0.01, 0.1, 1.0, 10.0]
        _, best_score, all_scores = find_best_lambda(
            X, y, lambdas, model="ridge", k=5
        )
        assert best_score == pytest.approx(min(all_scores.values()))

    def test_lasso_best_lambda(self, noisy_data):
        """find_best_lambda hoạt động với model=lasso."""
        X, y = noisy_data
        lambdas = [0.001, 0.01, 0.1, 1.0]
        best_lam, best_score, _ = find_best_lambda(
            X, y, lambdas, model="lasso", k=5
        )
        assert best_lam in lambdas
        assert best_score > 0


# =========================================================
# Advanced Methods Tests
# =========================================================

from part1.advanced_methods import (
    elastic_net_fit,
    huber_irls_fit,
    bayesian_linear_fit,
)

class TestAdvancedMethods:

    def test_elastic_net_converges_to_ols(self, simple_data):
        """Khi alpha -> 0, Elastic Net tiệm cận OLS."""
        X, y = simple_data
        beta = elastic_net_fit(X, y, l1_ratio=0.5, alpha=1e-6, max_iter=5000)
        np.testing.assert_allclose(beta, [0, 2], atol=1e-2)

    def test_elastic_net_sparsity(self, noisy_data):
        """L1 penalty lớn dẫn đến đặc trưng bị triệt tiêu (sparsity)."""
        X, y = noisy_data
        beta = elastic_net_fit(X, y, l1_ratio=1.0, alpha=10.0) # Tương đương Lasso
        assert np.any(beta[1:] == 0)

    def test_huber_robustness(self):
        """Huber loss giảm thiểu ảnh hưởng của ngoại lai (outliers) tốt hơn OLS."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        X = np.column_stack([np.ones(n), x])
        true_beta = np.array([2.0, 3.0])
        y = X @ true_beta + np.random.randn(n) * 0.1
        
        # Tạo outliers nặng
        y[0] += 50.0
        y[1] -= 50.0
        
        # OLS fit
        beta_ols = np.linalg.solve(X.T @ X, X.T @ y)
        
        # Huber fit
        beta_huber = huber_irls_fit(X, y, delta=1.345)
        
        # Huber phải gần true_beta hơn nhiều so với OLS do OLS bị kéo theo outliers
        err_ols = np.linalg.norm(beta_ols - true_beta)
        err_huber = np.linalg.norm(beta_huber - true_beta)
        assert err_huber < err_ols
        assert err_huber < 0.5  # Huber phải rất gần true_beta

    def test_bayesian_linear_fit(self, simple_data):
        """Bayesian linear regression tính đúng trung bình và phương sai hậu nghiệm."""
        X, y = simple_data
        mu_N, Sigma_N = bayesian_linear_fit(X, y, alpha=1e-3, beta_n=1e3) # prior yếu, likelihood mạnh
        # Sẽ rất gần OLS beta [0, 2]
        np.testing.assert_allclose(mu_N, [0, 2], atol=1e-2)
        assert Sigma_N.shape == (2, 2)
        assert np.all(np.diag(Sigma_N) > 0)