import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import math
import random

# Đảm bảo import được từ thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from part1 import helper_function as hf
from part1.ols_implementation import ols_fit, coef_inference
from part1.ridge_lasso import ridge_fit, lasso_fit, vif, predict
from part1.cross_validation import find_best_lambda
from part1.residual_analysis import residual_plots
from part2.advanced_methods import BayesianRegressor
from part2.data_pipeline import NBADataPipeline


def calculate_metrics(y_true_raw, y_pred_log):
    """
    Tính toán các chỉ số MAE, RMSE, R2 trên thang đo gốc (USD).
    Nghịch đảo log transform: exp(y)
    """
    y_true = hf.as_1d_float_array(y_true_raw, "y_true")
    y_pred = hf.Vector([math.exp(val) for val in y_pred_log])

    diff = y_true - y_pred
    mae = sum(abs(v) for v in diff) / len(diff)
    rmse = math.sqrt(sum(v**2 for v in diff) / len(diff))

    mean_y = sum(y_true) / len(y_true)
    tss = sum((v - mean_y)**2 for v in y_true)
    rss = sum(v**2 for v in diff)
    r2 = 1.0 - (rss / tss) if tss != 0 else hf.NAN

    return mae, rmse, r2


def run_vif_feature_selection(X_train_matrix, feature_names, threshold=10.0):
    """
    Loại bỏ đa cộng tuyến tuần tự dựa trên VIF.
    Trả về danh sách các đặc trưng giữ lại.
    """
    # Copy để không ảnh hưởng dữ liệu gốc
    X = hf.Matrix([hf.Vector(row) for row in X_train_matrix])
    current_features = list(feature_names)

    while True:
        # Tính VIF cho các đặc trưng hiện tại
        vif_vals = vif(X)

        max_idx = 0
        max_vif = vif_vals[0]
        for idx, val in enumerate(vif_vals):
            if val > max_vif:
                max_vif = val
                max_idx = idx

        if max_vif > threshold and max_vif != hf.INF:
            # Loại bỏ đặc trưng có VIF cao nhất
            dropped = current_features.pop(max_idx)
            X = hf.delete(X, max_idx, axis=1)
        elif max_vif == hf.INF:
            # Loại bỏ cột đa cộng tuyến hoàn hảo đầu tiên tìm thấy
            dropped = current_features.pop(max_idx)
            X = hf.delete(X, max_idx, axis=1)
        else:
            break

    return current_features, X


def plot_ridge_trace(X_train_bias, y_train, lambdas, output_path="ridge_trace.png"):
    """Vẽ biểu đồ Ridge Trace thể hiện sự co rút của các hệ số."""
    coefs = []
    # Bỏ qua intercept khi vẽ trace (cột 0)
    for lam in lambdas:
        beta = ridge_fit(X_train_bias, y_train, lam=lam)
        coefs.append(beta[1:])

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, coefs)
    plt.xscale("log")
    plt.xlabel("Regularization parameter Lambda (log scale)")
    plt.ylabel("Coefficients")
    plt.title("Ridge Trace Analysis")
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Đã lưu Ridge Trace tại: {output_path}")


# =====================================================================
# CÁC HÀM ĐƯỢC TÁCH RIÊNG CHO TỪNG MÔ HÌNH
# =====================================================================

def run_ols_basic(X_train_bias, y_train_arr, X_test_bias, y_test_raw):
    print("\nHuấn luyện Model 1: OLS Cơ Bản...")
    beta_ols, sigma2_ols = ols_fit(X_train_bias, y_train_arr)
    y_pred_ols_log = X_test_bias @ beta_ols
    mae_ols, rmse_ols, r2_ols = calculate_metrics(y_test_raw, y_pred_ols_log)
    
    result = {
        "MAE": mae_ols,
        "RMSE": rmse_ols,
        "R2": r2_ols,
        "coefficients": beta_ols,
    }
    return result, beta_ols, sigma2_ols


def run_ols_selected(X_train_arr, y_train_arr, X_test_arr, feature_names, y_test_raw, X_train, X_test):
    print("\nHuấn luyện Model 2: OLS Chọn Biến...")
    # Bước a: Loại bỏ đa cộng tuyến bằng VIF
    selected_feats, X_train_selected = run_vif_feature_selection(
        X_train_arr, feature_names, threshold=5.0
    )
    print(f"  - VIF lọc giảm số lượng đặc trưng từ {len(feature_names)} xuống {len(selected_feats)}")

    # Thêm bias cho tập rút gọn
    X_train_sel_bias = hf.column_stack([hf.ones(X_train_selected.shape[0])] + list(X_train_selected.T))

    # Bước b: Fit OLS và loại các biến có p-value > 0.05
    beta_sel, sigma2_sel = ols_fit(X_train_sel_bias, y_train_arr)
    inf_df = coef_inference(X_train_sel_bias, y_train_arr, beta_sel, sigma2_sel)

    # Lọc biến có ý nghĩa thống kê (p-value <= 0.05). Intercept (chỉ số 0) luôn giữ lại.
    significant_indices = [0]  # Intercept
    for idx in range(1, len(beta_sel)):
        p_val = inf_df.loc[idx, "p_value"]
        if p_val <= 0.05:
            significant_indices.append(idx)

    # Chọn lại đặc trưng cuối cùng
    # index trong selected_feats dịch chuyển đi 1 đơn vị do bias
    final_selected_feats_idx = [i - 1 for i in significant_indices if i > 0]
    final_selected_feats = [selected_feats[i] for i in final_selected_feats_idx]
    print(f"  - p-value lọc giảm tiếp số lượng đặc trưng xuống {len(final_selected_feats)}: {final_selected_feats}")

    # Chuẩn bị dữ liệu cuối cùng cho OLS rút gọn
    X_train_final = hf.as_2d_float_array(X_train[final_selected_feats].values, "X_train_final")
    X_test_final = hf.as_2d_float_array(X_test[final_selected_feats].values, "X_test_final")

    X_train_final_bias = hf.column_stack([hf.ones(X_train_final.shape[0])] + list(X_train_final.T))
    X_test_final_bias = hf.column_stack([hf.ones(X_test_final.shape[0])] + list(X_test_final.T))

    beta_ols_sel, sigma2_ols_sel = ols_fit(X_train_final_bias, y_train_arr)
    y_pred_ols_sel_log = X_test_final_bias @ beta_ols_sel
    mae_sel, rmse_sel, r2_sel = calculate_metrics(y_test_raw, y_pred_ols_sel_log)
    
    result = {
        "MAE": mae_sel,
        "RMSE": rmse_sel,
        "R2": r2_sel,
        "coefficients": beta_ols_sel,
        "features": final_selected_feats,
    }
    return result, beta_ols_sel, sigma2_ols_sel, X_train_final_bias

def run_ridge_regression(X_train_bias, y_train_arr, X_test_bias, y_test_raw):
    print("\nHuấn luyện Model 3: Ridge Regression...")
    lambdas = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    # Vẽ Ridge Trace
    plot_ridge_trace(X_train_bias, y_train_arr, lambdas, "ridge_trace.png")

    # Tìm lambda tối ưu
    best_lam_ridge, _, _ = find_best_lambda(X_train_bias, y_train_arr, lambdas, model="ridge", k=5)
    print(f"  - Ridge: Lambda tối ưu chọn bởi 5-Fold CV: {best_lam_ridge}")

    beta_ridge = ridge_fit(X_train_bias, y_train_arr, lam=best_lam_ridge)
    y_pred_ridge_log = X_test_bias @ beta_ridge
    mae_ridge, rmse_ridge, r2_ridge = calculate_metrics(y_test_raw, y_pred_ridge_log)
    
    result = {
        "MAE": mae_ridge,
        "RMSE": rmse_ridge,
        "R2": r2_ridge,
        "coefficients": beta_ridge,
        "best_lambda": best_lam_ridge,
    }
    return result, beta_ridge


def run_lasso_regression(X_train_bias, y_train_arr, X_test_bias, y_test_raw):
    print("\nHuấn luyện Model 4: Lasso Regression...")
    lasso_lambdas = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    best_lam_lasso, _, _ = find_best_lambda(X_train_bias, y_train_arr, lasso_lambdas, model="lasso", k=5)
    print(f"  - Lasso: Lambda tối ưu chọn bởi 5-Fold CV: {best_lam_lasso}")

    beta_lasso = lasso_fit(X_train_bias, y_train_arr, lam=best_lam_lasso)
    
    # [Debug Lasso] In ra số lượng đặc trưng bị loại bỏ (hệ số = 0)
    non_zero_feats = sum(1 for b in beta_lasso[1:] if abs(b) > 1e-5)
    total_feats = len(beta_lasso) - 1
    print(f"  - Debug Lasso: Model đã tự động thu gọn (shrink) {total_feats - non_zero_feats} / {total_feats} đặc trưng về 0.")
    
    y_pred_lasso_log = X_test_bias @ beta_lasso
    mae_lasso, rmse_lasso, r2_lasso = calculate_metrics(y_test_raw, y_pred_lasso_log)
    
    result = {
        "MAE": mae_lasso,
        "RMSE": rmse_lasso,
        "R2": r2_lasso,
        "coefficients": beta_lasso,
        "best_lambda": best_lam_lasso,
    }
    return result, beta_lasso


def run_bayesian_regression(X_train_bias, y_train_arr, X_test_bias, y_test_raw):
    print("\nHuấn luyện Model 5: Bayesian Linear Regression...")
    bayesian_model = BayesianRegressor(alpha=1.0, beta_n=1.0)
    bayesian_model.fit(X_train_bias, y_train_arr)
    
    # [Debug Bayesian] In ra thông tin về Posterior Covariance
    print(f"  - Debug Bayesian: Tính toán Posterior Mean (mu_N) và Covariance (Sigma_N) với kích thước {bayesian_model.Sigma_N.shape}.")
    
    sigma_n = bayesian_model.Sigma_N
    avg_variance = sum(sigma_n[i][i] for i in range(len(sigma_n))) / len(sigma_n)
    print(f"  - Debug Bayesian: Phương sai hậu nghiệm trung bình của các hệ số (độ bất định): {avg_variance:.4e}")

    y_pred_bayes_log = bayesian_model.predict(X_test_bias)
    mae_bayes, rmse_bayes, r2_bayes = calculate_metrics(y_test_raw, y_pred_bayes_log)
    
    result = {
        "MAE": mae_bayes,
        "RMSE": rmse_bayes,
        "R2": r2_bayes,
        "coefficients": bayesian_model.mu_N,
    }
    return result, bayesian_model

def generate_polynomial_features(X_matrix, feature_names, poly_features=['PTS', 'TRB', 'AST', 'MP', 'USG%']):
    """
    Tạo các đặc trưng đa thức thuần Python NHƯNG chỉ áp dụng cho các biến liên tục quan trọng.
    (Việc tạo đa thức cho One-hot encoding là vô nghĩa và làm mảng phình to, gây treo máy vì Python thuần tính toán chậm).
    """
    n_samples = len(X_matrix)
    n_features = len(X_matrix[0])
    
    poly_indices = []
    for feat in poly_features:
        if feat in feature_names:
            poly_indices.append(feature_names.index(feat))
            
    X_poly = [[] for _ in range(n_samples)]
    
    for i in range(n_samples):
        # 1. Đặc trưng gốc
        for j in range(n_features):
            X_poly[i].append(X_matrix[i][j])
            
        # 2. Đặc trưng bậc 2 và tương tác (chỉ áp dụng trên các biến được chọn)
        for idx1 in range(len(poly_indices)):
            for idx2 in range(idx1, len(poly_indices)):
                v1 = X_matrix[i][poly_indices[idx1]]
                v2 = X_matrix[i][poly_indices[idx2]]
                X_poly[i].append(v1 * v2)
                    
    return hf.Matrix([hf.Vector(row) for row in X_poly])


def run_polynomial_regression(X_train_arr, y_train_arr, X_test_arr, y_test_raw, feature_names):
    print(f"\nHuấn luyện Model 6: Polynomial Regression (Ridge trên biến liên tục)...")
    
    # Chỉ bành trướng đa thức trên các chỉ số chính để tránh tạo ra 600+ features gây treo RAM/CPU.
    poly_feats = ['PTS', 'TRB', 'AST', 'MP', 'USG%']
    X_train_poly = generate_polynomial_features(X_train_arr, feature_names, poly_feats)
    X_test_poly = generate_polynomial_features(X_test_arr, feature_names, poly_feats)

    X_train_poly_bias = hf.column_stack([hf.ones(X_train_poly.shape[0])] + list(X_train_poly.T))
    X_test_poly_bias = hf.column_stack([hf.ones(X_test_poly.shape[0])] + list(X_test_poly.T))

    poly_lambdas = [0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]
    best_lam_poly, _, _ = find_best_lambda(X_train_poly_bias, y_train_arr, poly_lambdas, model="ridge", k=5)
    print(f"  - Polynomial: Lambda tối ưu chọn bởi 5-Fold CV: {best_lam_poly}")

    beta_poly = ridge_fit(X_train_poly_bias, y_train_arr, lam=best_lam_poly)
    y_pred_poly_log = X_test_poly_bias @ beta_poly
    mae_poly, rmse_poly, r2_poly = calculate_metrics(y_test_raw, y_pred_poly_log)
    
    result = {
        "MAE": mae_poly,
        "RMSE": rmse_poly,
        "R2": r2_poly,
        "coefficients": beta_poly,
        "best_lambda": best_lam_poly,
    }
    return result, beta_poly, X_train_poly_bias


def main():
    print("=" * 60)
    print("CHƯƠNG TRÌNH SO SÁNH CÁC MÔ HÌNH HỒI QUY TRÊN DỮ LIỆU NBA")
    print("=" * 60)

    # 1. Đọc dữ liệu
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "nba_salary_raw.csv"
    )
    if not os.path.exists(data_path):
        data_path = os.path.join(
            os.path.dirname(__file__), "data", "nba_salaries.csv"
        )

    print(f"1. Đang tải bộ dữ liệu từ: {data_path}")
    df = pd.read_csv(data_path)

    # 2. Chia tập Train/Test (80/20) với seed cố định
    print("2. Chia dữ liệu thành tập Train (80%) và Test (20%) với seed 42")
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    y_train_raw = train_df["Salary"].tolist()
    y_test_raw = test_df["Salary"].tolist()

    # 3. Chạy OOP DataPipeline
    print("3. Tiền xử lý dữ liệu qua NBADataPipeline...")
    pipeline = NBADataPipeline()
    X_train, y_train, X_test, y_test = pipeline.process_pipeline(
        train_df, test_df
    )
    feature_names = X_train.columns.tolist()

    X_train_arr = hf.as_2d_float_array(X_train.values, "X_train")
    y_train_arr = hf.as_1d_float_array(y_train.values, "y_train")
    X_test_arr = hf.as_2d_float_array(X_test.values, "X_test")

    # Thêm cột bias (intercept)
    X_train_bias = hf.column_stack([hf.ones(X_train_arr.shape[0])] + list(X_train_arr.T))
    X_test_bias = hf.column_stack([hf.ones(X_test_arr.shape[0])] + list(X_test_arr.T))

    results = {}

    # GỌI CÁC HÀM MÔ HÌNH ĐÃ ĐƯỢC TÁCH RỜI
    results["OLS Cơ Bản"], beta_ols, _ = run_ols_basic(
        X_train_bias, y_train_arr, X_test_bias, y_test_raw
    )
    
    results["OLS Chọn Biến"], beta_ols_sel, sigma2_ols_sel, X_train_final_bias = run_ols_selected(
        X_train_arr, y_train_arr, X_test_arr, feature_names, y_test_raw, X_train, X_test
    )
    
    results["Polynomial Regression"], beta_poly, X_train_poly_bias = run_polynomial_regression(
        X_train_arr, y_train_arr, X_test_arr, y_test_raw, feature_names
    )
    
    results["Ridge Regression"], beta_ridge = run_ridge_regression(
        X_train_bias, y_train_arr, X_test_bias, y_test_raw
    )
    
    results["Lasso Regression"], beta_lasso = run_lasso_regression(
        X_train_bias, y_train_arr, X_test_bias, y_test_raw
    )
    
    results["Bayesian Regression"], bayesian_model = run_bayesian_regression(
        X_train_bias, y_train_arr, X_test_bias, y_test_raw
    )

    # ----------------------------------------------------
    # 4. Hiển thị bảng so sánh kết quả
    # ----------------------------------------------------
    print("\n" + "=" * 60)
    print("KẾT QUẢ SO SÁNH CÁC MÔ HÌNH (ĐÁNH GIÁ TRÊN TẬP TEST - ĐƠN VỊ USD)")
    print("=" * 60)

    comparison_rows = []
    for model_name, metrics in results.items():
        comparison_rows.append(
            {
                "Mô hình": model_name,
                "MAE (USD)": f"{metrics['MAE']:,.2f}",
                "RMSE (USD)": f"{metrics['RMSE']:,.2f}",
                "R2 Score": f"{metrics['R2']:.4f}",
                "Ghi chú/Siêu tham số": (
                    f"Lambda={metrics['best_lambda']}"
                    if "best_lambda" in metrics
                    else (
                        f"{len(metrics['features'])} đặc trưng"
                        if "features" in metrics
                        else "Mặc định"
                    )
                ),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    print(comparison_df.to_string(index=False))
    
    csv_path = "model_comparison_results.csv"
    comparison_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Đã xuất bảng kết quả so sánh đầy đủ ra file: {csv_path}")

    # 5. Phân tích chẩn đoán phần dư cho mô hình tốt nhất
    best_model_name = min(results, key=lambda k: results[k]["RMSE"])
    print(f"\n🏆 Mô hình tốt nhất dựa trên RMSE: {best_model_name}")

    if best_model_name == "OLS Chọn Biến":
        best_X_train = X_train_final_bias
    elif best_model_name == "Polynomial Regression":
        best_X_train = X_train_poly_bias
    else:
        best_X_train = X_train_bias

    best_beta = results[best_model_name]["coefficients"]

    fig = residual_plots(best_X_train, y_train_arr, best_beta)
    diag_path = "residuals_diagnostic.png"
    fig.savefig(diag_path)
    plt.close(fig)
    print(f"Đã vẽ và lưu chẩn đoán phần dư của mô hình tốt nhất tại: {diag_path}")

    # 6. Đồ thị dự đoán khoảng tin cậy Bayesian 
    print("Tạo biểu đồ khoảng tin cậy Bayesian cho 20 cầu thủ ngẫu nhiên trong tập Test...")
    random.seed(42)
    sample_indices = random.sample(range(len(y_test_raw)), 20)

    y_test_sample_raw = [y_test_raw[i] for i in sample_indices]
    X_test_sample_bias = hf.Matrix([X_test_bias[i] for i in sample_indices])

    lower_log, upper_log = bayesian_model.get_credible_intervals(X_test_sample_bias, confidence=0.95)
    y_pred_sample_log = bayesian_model.predict(X_test_sample_bias)

    y_pred_sample_raw = [math.exp(val) for val in y_pred_sample_log]
    lower_raw = [math.exp(val) for val in lower_log]
    upper_raw = [math.exp(val) for val in upper_log]
    player_names = [test_df.iloc[i]["Player Name"] for i in sample_indices]

    plt.figure(figsize=(12, 7))
    x_axis = range(20)
    
    yerr_lower = [(pred - low) / 1e6 for pred, low in zip(y_pred_sample_raw, lower_raw)]
    yerr_upper = [(up - pred) / 1e6 for pred, up in zip(y_pred_sample_raw, upper_raw)]
    
    y_pred_sample_raw_scaled = [val / 1e6 for val in y_pred_sample_raw]
    y_test_sample_raw_scaled = [val / 1e6 for val in y_test_sample_raw]

    plt.errorbar(
        x_axis, y_pred_sample_raw_scaled, yerr=[yerr_lower, yerr_upper],
        fmt="o", color="royalblue", ecolor="lightblue", capsize=5, elinewidth=2,
        label="Dự đoán & Khoảng tin cậy 95% (Bayesian)",
    )
    plt.scatter(
        x_axis, y_test_sample_raw_scaled, color="red", marker="x", s=80, label="Giá trị thực tế"
    )
    plt.xticks(x_axis, player_names, rotation=45, ha="right")
    plt.ylabel("Salary (Millions USD)")
    plt.title("Bayesian Linear Regression: 95% Credible Prediction Intervals")
    plt.legend()
    plt.tight_layout()
    bayes_interval_path = "bayesian_prediction_intervals.png"
    plt.savefig(bayes_interval_path)
    plt.close()
    print(f"✅ Đã lưu đồ thị khoảng tin cậy Bayesian tại: {bayes_interval_path}")

    # 7. Giải thích đặc trưng quan trọng của OLS chọn biến
    print("\nÝ NGHĨA THỐNG KÊ & PHÂN TÍCH HỆ SỐ HỒI QUY (OLS CHỌN BIẾN):")
    final_feats = results["OLS Chọn Biến"]["features"]
    sig_sel_ols = results["OLS Chọn Biến"]["coefficients"]
    inf_sel_df = coef_inference(X_train_final_bias, y_train_arr, sig_sel_ols, sigma2_ols_sel)

    feat_importance = pd.DataFrame(
        {
            "Đặc trưng": ["Intercept"] + final_feats,
            "Hệ số (Beta)": sig_sel_ols,
            "Sai số chuẩn": inf_sel_df["std_error"],
            "t-stat": inf_sel_df["t_stat"],
            "p-value": inf_sel_df["p_value"],
        }
    )
    print(feat_importance.to_string(index=False))


if __name__ == "__main__":
    main()
