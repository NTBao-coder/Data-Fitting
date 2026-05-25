import numpy as np

try:
    from part1.ridge_lasso import soft_threshold, predict
except ModuleNotFoundError:
    from ridge_lasso import soft_threshold, predict


def elastic_net_fit(X, y, l1_ratio=0.5, alpha=1.0, max_iter=1000, tol=1e-4):
    """
    Elastic Net Regression bằng Coordinate Descent.

    Minimize: RSS + λ1 * ||β||_1 + 0.5 * λ2 * ||β||_2^2
    với:
        λ1 = alpha * l1_ratio
        λ2 = alpha * (1 - l1_ratio)

    Lưu ý: Không phạt (regularize) intercept (cột đầu tiên).
    """
    n, p = X.shape
    beta = np.zeros(p)
    y_predict = np.zeros(n)

    # Tính trước chuẩn bình phương của các cột (z_j = X_j^T X_j)
    z = np.sum(X ** 2, axis=0)
    z = np.where(z == 0, 1e-12, z)

    lam1 = alpha * l1_ratio
    lam2 = alpha * (1 - l1_ratio)

    for _ in range(max_iter):
        beta_old = beta.copy()

        for j in range(p):
            beta_old_j = beta[j]

            # Loại bỏ contribution hiện tại của feature j
            val_j = y_predict - X[:, j] * beta_old_j
            residual = y - val_j
            rho = X[:, j] @ residual

            # Không regularize intercept
            if j == 0:
                beta[j] = np.sum(residual) / z[j]
            else:
                beta[j] = soft_threshold(rho, lam1) / (z[j] + lam2)

            # Cập nhật y_predict nếu beta_j thay đổi
            if beta[j] != beta_old_j:
                y_predict += X[:, j] * (beta[j] - beta_old_j)

        # Kiểm tra hội tụ
        if np.linalg.norm(beta - beta_old) < tol:
            break

    return beta


def huber_irls_fit(X, y, delta=1.345, max_iter=100, tol=1e-5):
    """
    Robust Regression sử dụng Huber Loss thông qua thuật toán IRLS.

    Tham số:
        delta: Ngưỡng chia giữa Huber L2 và L1 loss (mặc định 1.345)
    """
    n, p = X.shape
    
    # Khởi động beta bằng OLS ban đầu
    try:
        beta = np.linalg.solve(X.T @ X, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.zeros(p)

    for _ in range(max_iter):
        beta_old = beta.copy()
        
        # Tính residuals
        residuals = y - X @ beta
        abs_res = np.abs(residuals)
        
        # Tính trọng số w_i cho từng dòng
        # Tránh chia cho 0
        abs_res_safe = np.where(abs_res == 0, 1e-12, abs_res)
        w = np.where(abs_res_safe <= delta, 1.0, delta / abs_res_safe)
        
        # Thực hiện weighted OLS: beta = (X^T W X)^(-1) X^T W y
        # Để tối ưu hóa nhân trực tiếp:
        # X^T W X = (X.T * w) @ X
        # X^T W y = X.T @ (w * y)
        XTWX = (X.T * w) @ X
        XTWy = X.T @ (w * y)
        
        try:
            beta = np.linalg.solve(XTWX, XTWy)
        except np.linalg.LinAlgError:
            # Thêm ridge nhỏ nếu ma trận bị suy biến
            beta = np.linalg.solve(XTWX + 1e-6 * np.eye(p), XTWy)

        if np.linalg.norm(beta - beta_old) < tol:
            break

    return beta


def bayesian_linear_fit(X, y, alpha=1.0, beta_n=1.0):
    """
    Bayesian Linear Regression.

    Prior: P(beta) = N(0, alpha^(-1) * I)
    Likelihood: P(y | X, beta) = N(X @ beta, beta_n^(-1) * I)

    Trả về:
        mu_N    : Vector kỳ vọng hậu nghiệm (Posterior Mean)
        Sigma_N : Ma trận phương sai hậu nghiệm (Posterior Covariance)
    """
    n, p = X.shape
    
    # Covariance matrix: Sigma_N = (alpha * I + beta_n * X^T X)^(-1)
    XTX = X.T @ X
    A = alpha * np.eye(p) + beta_n * XTX
    Sigma_N = np.linalg.inv(A)
    
    # Mean vector: mu_N = beta_n * Sigma_N @ X.T @ y
    mu_N = beta_n * Sigma_N @ X.T @ y
    
    return mu_N, Sigma_N
