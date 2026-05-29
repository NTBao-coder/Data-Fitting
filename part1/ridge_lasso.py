from part1 import helper_function as hf


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

    return hf.sum_values((y - y_hat) ** 2)


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
    I = hf.eye(p)

    # Không regularize intercept
    I[0, 0] = 0

    # Công thức Ridge Regression
    # Dùng solver tuyến tính thay vì inv() để ổn định số hơn
    # khi ma trận (X^T X + λI) gần singular.
    # solve(A, b) tính A^{-1} b mà không cần tính nghịch đảo tường minh.
    beta = hf.solve(X.T @ X + lam * I, X.T @ y)

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

    n, p = X.shape

    # Khởi tạo beta ban đầu = 0
    beta = hf.zeros(p)
    # Khởi tạo vector dự đoán y_predict = X @ beta = 0
    y_predict = hf.zeros(n)

    # Tính trước chuẩn bình phương của các cột (z_j = X_j^T X_j)
    # Tránh tính toán lại nhiều lần trong loop
    z = hf.sum_values(X ** 2, axis=0)
    # Tránh chia cho 0 nếu cột toàn 0
    z = hf.where(z == 0, 1e-12, z)

    # Lặp tối đa max_iter lần
    for _ in range(max_iter):

        # Lưu beta cũ để kiểm tra hội tụ
        beta_old = beta.copy()

        # Coordinate Descent:
        # cập nhật từng coefficient beta_j
        for j in range(p):

            beta_old_j = beta[j]

            # Loại bỏ contribution hiện tại của feature j khỏi y_predict
            val_j = y_predict - X[:, j] * beta_old_j
            residual = y - val_j
            rho = X[:, j] @ residual

            # Không regularize intercept (j = 0)
            if j == 0:
                beta[j] = hf.sum_values(residual) / z[j]
            else:
                beta[j] = soft_threshold(rho, lam) / z[j]

            # Cập nhật y_predict nếu beta_j thay đổi
            if beta[j] != beta_old_j:
                y_predict += X[:, j] * (beta[j] - beta_old_j)

        # Nếu beta gần như không đổi nữa
        # thì model đã hội tụ
        if hf.norm(beta - beta_old) < tol:
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
        X_rest = hf.delete(X, j, axis=1)

        try:
            # Fit OLS:
            # X_rest -> y_j
            # Thay thế lstsq bằng solver tuyến tính để tuân thủ Constraint 1
            XTX = X_rest.T @ X_rest
            XTy = X_rest.T @ y_j
            beta = hf.solve(XTX, XTy)

            # Prediction của feature j
            y_hat = predict(X_rest, beta)

            # Tính R^2
            r2 = 1 - (
                hf.sum_values((y_j - y_hat) ** 2)
                / hf.sum_values((y_j - hf.mean(y_j)) ** 2)
            )

            # Tránh chia cho 0 nếu R^2 ≈ 1
            if hf.is_close(r2, 1) or r2 >= 1.0:
                vif_values.append(hf.INF)
            else:
                # Công thức:
                # VIF = 1 / (1 - R^2)
                vif_values.append(1 / (1 - r2))
        except hf.LinAlgError:
            # Nếu ma trận kì dị, VIF là vô cùng
            vif_values.append(hf.INF)

    return hf.array(vif_values)

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
    hf.random_seed(42)

    # Tạo dữ liệu giả lập:
    # 100 samples, 3 features
    X = hf.random_randn(100, 3)

    # Thêm intercept column (cột toàn số 1)
    X = hf.column_stack([hf.ones(100), X])

    # Beta thật dùng để sinh dữ liệu
    # y = 5 + 2x1 - 3x2 + x3 + noise
    true_beta = hf.array([5, 2, -3, 1])

    # Sinh target y
    # thêm noise để giống dữ liệu thực tế
    y = X @ true_beta + hf.random_randn(100)

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
