import numpy as np


def predict(X, beta):
    """
    Dự đoán giá trị y_hat bằng công thức:
    y_hat = X @ beta
    """

    return X @ beta


def rss(y, y_hat):
    """
    Tính RSS (Residual Sum of Squares)

    RSS = Σ(y - y_hat)^2
    """

    return np.sum((y - y_hat) ** 2)


def ridge_fit(X, y, lam=1.0):
    """
    Ridge Regression sử dụng closed-form solution:

    beta = (X^T X + λI)^(-1) X^T y

    Lưu ý:
    Không regularize intercept.
    """

    # n = số samples
    # p = số features
    n, p = X.shape

    # Tạo ma trận đơn vị kích thước p x p
    I = np.eye(p)

    # Không regularize intercept
    I[0, 0] = 0

    # Công thức Ridge Regression
    # Dùng np.linalg.solve thay vì inv() để ổn định số hơn
    # khi ma trận (X^T X + λI) gần singular.
    # solve(A, b) tính A^{-1} b mà không cần tính nghịch đảo tường minh.
    beta = np.linalg.solve(X.T @ X + lam * I, X.T @ y)

    return beta


def soft_threshold(rho, lam):
    """
    Soft-thresholding operator cho Lasso.

    Nếu |rho| <= lambda:
        coefficient bị kéo về 0
    """

    # Nếu rho lớn hơn lambda
    # shrink về gần 0
    if rho > lam:
        return rho - lam

    # Nếu rho âm và nhỏ hơn -lambda
    # shrink theo chiều ngược lại
    elif rho < -lam:
        return rho + lam

    # Nếu |rho| <= lambda
    # coefficient bị kéo về đúng 0
    else:
        return 0


def lasso_fit(X, y, lam=1.0, max_iter=1000, tol=1e-4):
    """
    Lasso Regression bằng Coordinate Descent.

    Lasso sử dụng L1 regularization:
        RSS + λΣ|beta|

    Đặc điểm:
    Có khả năng đưa coefficient về đúng 0
    -> tự động chọn feature.

    Lưu ý về convention:
    Implementation này minimize: RSS + λ|β|
    sklearn Lasso minimize: (1/2n) * RSS + α|β|
    => Tương đương khi α_sklearn = λ / (2 * n)
    """

    # n = số samples
    # p = số features
    n, p = X.shape

    # Khởi tạo beta ban đầu = 0
    beta = np.zeros(p)

    # Lặp tối đa max_iter lần
    for _ in range(max_iter):

        # Lưu beta cũ để kiểm tra hội tụ
        beta_old = beta.copy()

        # Coordinate Descent:
        # cập nhật từng coefficient beta_j
        for j in range(p):

            # Không regularize intercept
            if j == 0:
                beta[j] = np.sum(y - (predict(X, beta) - X[:, j] * beta[j])) / np.sum(X[:, j] ** 2)
                continue

            # Prediction hiện tại
            y_predict = predict(X, beta)

            # Loại bỏ contribution hiện tại của feature j
            # để cập nhật riêng beta_j
            residual = y - (y_predict - X[:, j] * beta[j])

            # Đo mức độ liên quan giữa feature j và residual
            rho = np.sum(X[:, j] * residual)

            # Update beta_j bằng soft-thresholding
            beta[j] = (soft_threshold(rho, lam) / np.sum(X[:, j] ** 2))

        # Nếu beta gần như không đổi nữa
        # thì model đã hội tụ
        if np.linalg.norm(beta - beta_old) < tol:
            break

    return beta


def vif(X):
    """
    Tính Variance Inflation Factor (VIF)
    cho từng feature.

    VIF dùng để phát hiện multicollinearity.
    """

    # n = số samples
    # p = số features
    n, p = X.shape

    # List lưu VIF của từng feature
    vif_values = []

    # Tính VIF cho từng feature
    for j in range(p):

        # Chọn feature hiện tại làm target
        y_j = X[:, j]

        # Xóa feature j khỏi dataset
        # để dùng các feature còn lại predict nó
        X_rest = np.delete(X, j, axis=1)

        # Fit OLS:
        # X_rest -> y_j
        beta = np.linalg.lstsq(X_rest, y_j, rcond=None)[0]

        # Prediction của feature j
        y_hat = predict(X_rest, beta)

        # Tính R^2
        r2 = 1 - (np.sum((y_j - y_hat) ** 2) / np.sum((y_j - np.mean(y_j)) ** 2))

        # Tránh chia cho 0 nếu R^2 ≈ 1
        if np.isclose(r2, 1):
            vif_values.append(np.inf)

        else:
            # Công thức:
            # VIF = 1 / (1 - R^2)
            vif_values.append(1 / (1 - r2))

    return np.array(vif_values)

# =========================================================
# Demo / Testing Section
# - Sinh dữ liệu giả lập
# - Train Ridge/Lasso tự code
# - Tính RSS và VIF
# - Kiểm chứng kết quả với sklearn
# =========================================================
if __name__ == "__main__":

    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso

    # Đặt seed để kết quả random có thể reproducible
    np.random.seed(42)

    # Tạo dữ liệu giả lập:
    # 100 samples, 3 features
    X = np.random.randn(100, 3)

    # Thêm intercept column (cột toàn số 1)
    X = np.column_stack([np.ones(100), X])

    # Beta thật dùng để sinh dữ liệu
    # y = 5 + 2x1 - 3x2 + x3 + noise
    true_beta = np.array([5, 2, -3, 1])

    # Sinh target y
    # thêm noise để giống dữ liệu thực tế
    y = X @ true_beta + np.random.randn(100)

    # =========================
    # CUSTOM IMPLEMENTATION
    # =========================

    # Train Ridge Regression tự code
    ridge_beta = ridge_fit(X, y, lam=1)

    # Train Lasso Regression tự code
    lasso_beta = lasso_fit(X, y, lam=1)

    # In beta thật
    print("True beta:")
    print(true_beta)

    # In kết quả Ridge
    print("\nCustom Ridge beta:")
    print(ridge_beta)

    # In kết quả Lasso
    print("\nCustom Lasso beta:")
    print(lasso_beta)

    # Tính RSS Ridge
    print("\nRSS Ridge:")
    print(rss(y, predict(X, ridge_beta)))

    # Tính RSS Lasso
    print("\nRSS Lasso:")
    print(rss(y, predict(X, lasso_beta)))

    # Tính VIF
    # Không truyền intercept column vào VIF
    print("\nVIF:")
    print(vif(X[:, 1:]))

    # =========================
    # SKLEARN VERIFICATION
    # =========================

    print("\n========== SKLEARN VERIFICATION ==========")

    # Ridge sklearn
    ridge_sklearn = Ridge(
        alpha=1.0,
        fit_intercept=False
    )

    # Train Ridge sklearn
    ridge_sklearn.fit(X, y)

    # In coefficients sklearn Ridge
    print("\nSklearn Ridge beta:")
    print(ridge_sklearn.coef_)

    # Lasso sklearn
    # sklearn minimize (1/2n)*RSS + alpha*|beta|
    # Custom minimize RSS + lam*|beta|
    # => alpha_sklearn = lam / (2 * n)
    n_samples = X.shape[0]
    lasso_sklearn = Lasso(
        alpha=1.0 / (2 * n_samples),
        fit_intercept=False,
        max_iter=10000
    )

    # Train Lasso sklearn
    lasso_sklearn.fit(X, y)

    # In coefficients sklearn Lasso
    print("\nSklearn Lasso beta (alpha=lam/(2n), tương đương custom lam=1.0):")
    print(lasso_sklearn.coef_)

    # Nhận xét nhanh
    print("\nObservation:")
    print(
        "Custom implementation gives coefficients "
        "similar to sklearn."
        )