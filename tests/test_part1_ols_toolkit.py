import unittest

import numpy as np

from Group_01.part1.ols_toolkit import LinearModels


class TestLinearModels(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.array([[1.0], [2.0], [3.0], [4.0]])
        self.y = np.array([3.0, 5.0, 7.0, 9.0])  # y = 1 + 2x

    def test_ols_fit_recovers_known_line(self) -> None:
        beta = LinearModels.ols_fit(self.X, self.y)
        np.testing.assert_allclose(beta, np.array([1.0, 2.0]), atol=1e-8)

    def test_hat_matrix_is_idempotent(self) -> None:
        H = LinearModels.hat_matrix(self.X)
        np.testing.assert_allclose(H @ H, H, atol=1e-8)

    def test_ridge_shrinks_coefficients(self) -> None:
        beta_ols = LinearModels.ols_fit(self.X, self.y)
        beta_ridge = LinearModels.ridge_fit(self.X, self.y, alpha=10.0)
        self.assertLess(abs(beta_ridge[1]), abs(beta_ols[1]))

    def test_lasso_prefers_sparse_solution(self) -> None:
        X = np.array(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        )
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        beta = LinearModels.lasso_fit(X, y, alpha=0.5)
        self.assertAlmostEqual(beta[2], 0.0, places=6)

    def test_metrics_mse_and_r2_known_values(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        self.assertAlmostEqual(LinearModels.mse(y_true, y_pred), 1.0 / 3.0, places=8)
        self.assertAlmostEqual(LinearModels.r2_score(y_true, y_pred), 0.5, places=8)

    def test_cross_validation_is_reproducible(self) -> None:
        score_1 = LinearModels.k_fold_cross_validation(
            self.X,
            self.y,
            k=2,
            model="ols",
            random_state=7,
        )
        score_2 = LinearModels.k_fold_cross_validation(
            self.X,
            self.y,
            k=2,
            model="ols",
            random_state=7,
        )
        self.assertAlmostEqual(score_1, score_2, places=12)


if __name__ == "__main__":
    unittest.main()

