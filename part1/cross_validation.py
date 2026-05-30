from part1 import helper_function as hf
try:
    from part1.ridge_lasso import ridge_fit, lasso_fit, predict
except ModuleNotFoundError:
    from ridge_lasso import ridge_fit, lasso_fit, predict


# =========================================================
# K-Fold Cross-Validation
# =========================================================

def kfold_cv(X, y, k=5, model="ols", lam=1.0):
    """
    K-Fold Cross-Validation.

    Chia dữ liệu thành k phần (folds) bằng nhau.
    Mỗi lần dùng k-1 phần để train, 1 phần để validate.
    Lặp k lần, mỗi lần dùng một fold khác nhau làm validation.
    Trả về CV score trung bình (MSE).

    CV(k) = (1/k) * Σ MSE_i

    Tham số:
        X     : ma trận features (đã thêm cột intercept nếu cần)
        y     : vector target
        k     : số folds (khuyến nghị 5 hoặc 10)
        model : "ols" | "ridge" | "lasso"
        lam   : tham số regularization λ (chỉ dùng cho ridge/lasso)

    Trả về:
        cv_score  : MSE trung bình qua k folds
        fold_mses : list MSE của từng fold (để phân tích thêm)
    """

    n = len(y)

    # Tạo index ngẫu nhiên để shuffle trước khi chia fold
    # Tránh bias nếu data có thứ tự (ví dụ: sorted theo salary)
    indices = hf.arange(n)
    hf.random_seed(42)
    hf.random_shuffle(indices)

    # Chia indices thành k phần (gần) bằng nhau
    # array_split tự xử lý khi n không chia hết cho k
    folds = hf.array_split(indices, k)

    fold_mses = []

    for i in range(k):

        # Fold i là validation set
        val_idx = folds[i]

        # Các fold còn lại là train set
        train_idx = hf.concatenate([folds[j] for j in range(k) if j != i])

        # Convert indices to list of integers to be compatible with numpy arrays
        train_idx_int = [int(idx) for idx in train_idx]
        val_idx_int = [int(idx) for idx in val_idx]

        # Tách train / validation
        X_train, y_train = X[train_idx_int], y[train_idx_int]
        X_val,   y_val   = X[val_idx_int],   y[val_idx_int]

        # Train model theo loại được chọn
        if model == "ols":
            beta = _ols_fit(X_train, y_train)

        elif model == "ridge":
            beta = ridge_fit(X_train, y_train, lam=lam)

        elif model == "lasso":
            beta = lasso_fit(X_train, y_train, lam=lam)

        else:
            raise ValueError(f"model phải là 'ols', 'ridge' hoặc 'lasso'. Nhận được: '{model}'")

        # Dự đoán trên validation set
        y_hat = predict(X_val, beta)

        # Tính MSE của fold này
        mse = hf.mean((y_val - y_hat) ** 2)
        fold_mses.append(mse)

    # CV score = trung bình MSE qua tất cả k folds
    cv_score = hf.mean(fold_mses)

    return cv_score, fold_mses


def _ols_fit(X, y):
    """
    OLS internal helper dùng trong CV.

    beta = (X^T X)^{-1} X^T y
    """
    return hf.solve(X.T @ X, X.T @ y)


def find_best_lambda(X, y, lambdas, model="ridge", k=5):
    """
    Dùng K-Fold CV để chọn λ tối ưu cho Ridge hoặc Lasso.

    Thử từng giá trị λ trong danh sách `lambdas`,
    chọn λ cho CV score (MSE) nhỏ nhất.

    Tham số:
        X       : ma trận features
        y       : vector target
        lambdas : danh sách các giá trị λ cần thử
        model   : "ridge" hoặc "lasso"
        k       : số folds

    Trả về:
        best_lam    : λ tốt nhất
        best_score  : CV score tương ứng
        all_scores  : dict {lam: cv_score} để vẽ đồ thị
    """

    all_scores = {}

    for lam in lambdas:
        cv_score, _ = kfold_cv(X, y, k=k, model=model, lam=lam)
        all_scores[lam] = cv_score

    # Chọn lambda có MSE thấp nhất
    best_lam = min(all_scores, key=all_scores.get)
    best_score = all_scores[best_lam]

    return best_lam, best_score, all_scores


# =========================================================
# Demo / Testing Section
# =========================================================
if __name__ == "__main__":

    from sklearn.linear_model import RidgeCV, LassoCV

    hf.random_seed(42)

    # Tạo dữ liệu giả lập
    n, p = 200, 4
    X = hf.random_randn(n, p)
    X = hf.column_stack([hf.ones(n), X])

    # y = 3 + 1.5x1 - 2x2 + 0.5x3 + 0*x4 + noise
    true_beta = hf.array([3.0, 1.5, -2.0, 0.5, 0.0])
    y = X @ true_beta + hf.random_randn(n) * 0.5

    print("=" * 50)
    print("K-FOLD CROSS-VALIDATION DEMO")
    print("=" * 50)

    # -------------------------------------------------------
    # 1. So sánh OLS / Ridge / Lasso với k=5
    # -------------------------------------------------------
    print("\n--- CV Score (MSE) với k=5 ---")

    score_ols, folds_ols = kfold_cv(X, y, k=5, model="ols")
    print(f"OLS   CV MSE : {score_ols:.6f}")
    print(f"  Fold MSEs  : {[round(m, 4) for m in folds_ols]}")

    score_ridge, folds_ridge = kfold_cv(X, y, k=5, model="ridge", lam=1.0)
    print(f"\nRidge CV MSE (λ=1.0): {score_ridge:.6f}")
    print(f"  Fold MSEs          : {[round(m, 4) for m in folds_ridge]}")

    score_lasso, folds_lasso = kfold_cv(X, y, k=5, model="lasso", lam=0.01)
    print(f"\nLasso CV MSE (λ=0.01): {score_lasso:.6f}")
    print(f"  Fold MSEs           : {[round(m, 4) for m in folds_lasso]}")

    # -------------------------------------------------------
    # 2. Tìm lambda tốt nhất cho Ridge và Lasso
    # -------------------------------------------------------
    print("\n--- Tìm λ tối ưu ---")

    lambdas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

    best_lam_ridge, best_score_ridge, scores_ridge = find_best_lambda(
        X, y, lambdas, model="ridge", k=5
    )
    print(f"\nRidge — Best λ: {best_lam_ridge}, CV MSE: {best_score_ridge:.6f}")
    print("  All scores:", {k: round(v, 4) for k, v in scores_ridge.items()})

    best_lam_lasso, best_score_lasso, scores_lasso = find_best_lambda(
        X, y, lambdas, model="lasso", k=5
    )
    print(f"\nLasso — Best λ: {best_lam_lasso}, CV MSE: {best_score_lasso:.6f}")
    print("  All scores:", {k: round(v, 4) for k, v in scores_lasso.items()})

    # -------------------------------------------------------
    # 3. Kiểm chứng với sklearn RidgeCV
    # -------------------------------------------------------
    print("\n--- SKLEARN VERIFICATION ---")

    # sklearn RidgeCV dùng Leave-One-Out mặc định
    # Để so sánh apples-to-apples, dùng cv=5
    alphas = hf.array(lambdas)
    ridge_cv_sk = RidgeCV(alphas=alphas, fit_intercept=False, cv=5)
    ridge_cv_sk.fit(X, y)
    print(f"\nSklearn RidgeCV — Best α: {ridge_cv_sk.alpha_}")
    print(f"Custom  Ridge CV — Best λ: {best_lam_ridge}")
    print(
        "\n[Convention Note] Hai kết quả KHÔNG so sánh trực tiếp được vì khác convention:\n"
        "  - sklearn minimize : (1/2n) * RSS + α * ||β||²\n"
        "  - Custom minimize  : RSS + λ * ||β||²\n"
        "  => Tương đương khi α_sklearn = λ_custom / (2 * n)\n"
        f"  => Với n={len(y)}, λ_custom=1.0 tương đương α_sklearn={1.0 / (2 * len(y)):.5f}\n"
        "  Để so sánh apples-to-apples, cần rescale lambdas trước khi truyền vào RidgeCV."
    )
