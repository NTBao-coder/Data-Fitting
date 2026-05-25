import numpy as np
import sys
import os

# Đảm bảo import được từ part1
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from part1.advanced_methods import bayesian_linear_fit


class BayesianRegressor:
    """
    Wrapper class cho Bayesian Linear Regression hướng đối tượng.
    """

    def __init__(self, alpha=1.0, beta_n=1.0):
        self.alpha = alpha
        self.beta_n = beta_n
        self.mu_N = None
        self.Sigma_N = None

    def fit(self, X, y):
        # Đảm bảo X là matrix numpy, y là vector numpy
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        # Gọi hàm fit toán học ở part1
        self.mu_N, self.Sigma_N = bayesian_linear_fit(
            X_arr, y_arr, alpha=self.alpha, beta_n=self.beta_n
        )
        return self

    def predict(self, X, return_std=False):
        X_arr = np.asarray(X, dtype=float)

        # Dự đoán trung bình hậu nghiệm: y_hat = X @ mu_N
        y_hat = X_arr @ self.mu_N

        if return_std:
            # Phương sai dự đoán = 1/beta_n + diag(X @ Sigma_N @ X.T)
            # Tối ưu hóa tính diag: np.sum((X @ Sigma_N) * X, axis=1)
            var_pred = (1.0 / self.beta_n) + np.sum(
                (X_arr @ self.Sigma_N) * X_arr, axis=1
            )
            std_pred = np.sqrt(np.maximum(var_pred, 1e-12))
            return y_hat, std_pred

        return y_hat

    def get_credible_intervals(self, X, confidence=0.95):
        """
        Tính khoảng tin cậy Bayesian (credible intervals) cho dự đoán.
        """
        y_hat, std_pred = self.predict(X, return_std=True)

        # Với phân phối chuẩn, khoảng tin cậy 95% có z-score là 1.96
        # Tổng quát hơn có thể dùng scipy stats, nhưng dùng numpy cố định z=1.96
        # cho 95% là chuẩn và không phụ thuộc nhiều thư viện.
        if np.isclose(confidence, 0.95):
            z = 1.96
        else:
            from scipy import stats

            z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)

        lower_bound = y_hat - z * std_pred
        upper_bound = y_hat + z * std_pred

        return lower_bound, upper_bound
