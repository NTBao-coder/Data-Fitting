import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Đảm bảo import được từ thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    y_true = np.asarray(y_true_raw, dtype=float)
    y_pred = np.exp(np.asarray(y_pred_log, dtype=float))

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (rss / tss) if tss != 0 else np.nan

    return mae, rmse, r2


def run_vif_feature_selection(X_train, feature_names, threshold=10.0):
    """
    Loại bỏ đa cộng tuyến tuần tự dựa trên VIF.
    Trả về danh sách các đặc trưng giữ lại.
    """
    X = X_train.copy()
    current_features = list(feature_names)

    while True:
        # Tính VIF cho các đặc trưng hiện tại
        vif_vals = vif(X)

        max_idx = np.argmax(vif_vals)
        max_vif = vif_vals[max_idx]

        if max_vif > threshold and not np.isinf(max_vif):
            # Loại bỏ đặc trưng có VIF cao nhất
            dropped = current_features.pop(max_idx)
            X = np.delete(X, max_idx, axis=1)
            # print(f"VIF Selection: Loại bỏ '{dropped}' (VIF = {max_vif:.2f})")
        elif np.isinf(max_vif):
            # Loại bỏ cột đa cộng tuyến hoàn hảo đầu tiên tìm thấy
            dropped = current_features.pop(max_idx)
            X = np.delete(X, max_idx, axis=1)
            # print(f"VIF Selection: Loại bỏ '{dropped}' do đa cộng tuyến hoàn hảo (VIF = inf)")
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

    coefs = np.array(coefs)

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


def main():
    print("=" * 60)
    print("CHƯƠNG TRÌNH SO SÁNH CÁC MÔ HÌNH HỒI QUY TRÊN DỮ LIỆU NBA")
    print("=" * 60)

    # 1. Đọc dữ liệu
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "nba_salary_raw.csv"
    )
    if not os.path.exists(data_path):
        # Fallback sang file nba_salaries nếu không có
        data_path = os.path.join(
            os.path.dirname(__file__), "data", "nba_salaries.csv"
        )

    print(f"1. Đang tải bộ dữ liệu từ: {data_path}")
    df = pd.read_csv(data_path)

    # 2. Chia tập Train/Test (80/20) với seed cố định
    print("2. Chia dữ liệu thành tập Train (80%) và Test (20%) với seed 42")
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # Lưu lương gốc để đánh giá (vì pipeline transform sẽ biến đổi hoặc tách Salary)
    y_train_raw = train_df["Salary"].values
    y_test_raw = test_df["Salary"].values

    # 3. Chạy OOP DataPipeline
    print("3. Tiền xử lý dữ liệu qua NBADataPipeline...")
    pipeline = NBADataPipeline()
    X_train, y_train, X_test, y_test = pipeline.process_pipeline(
        train_df, test_df
    )

    feature_names = X_train.columns.tolist()

    # Chuyển sang numpy array
    X_train_arr = X_train.values
    y_train_arr = y_train.values
    X_test_arr = X_test.values
    y_test_arr = y_test.values

    # Thêm cột bias (intercept) cho OLS, Ridge, Lasso, Bayesian
    X_train_bias = np.column_stack([np.ones(X_train_arr.shape[0]), X_train_arr])
    X_test_bias = np.column_stack([np.ones(X_test_arr.shape[0]), X_test_arr])

    # Lưu kết quả so sánh
    results = {}

    # ----------------------------------------------------
    # Model 1: OLS cơ bản
    # ----------------------------------------------------
    print("\nHuấn luyện Model 1: OLS Cơ Bản...")
    beta_ols, sigma2_ols = ols_fit(X_train_bias, y_train_arr)
    y_pred_ols_log = X_test_bias @ beta_ols
    mae_ols, rmse_ols, r2_ols = calculate_metrics(y_test_raw, y_pred_ols_log)
    results["OLS Cơ Bản"] = {
        "MAE": mae_ols,
        "RMSE": rmse_ols,
        "R2": r2_ols,
        "coefficients": beta_ols,
    }

    # ----------------------------------------------------
    # Model 2: OLS chọn biến (VIF + p-value)
    # ----------------------------------------------------
    print("Huấn luyện Model 2: OLS Chọn Biến...")
    # Bước a: Loại bỏ đa cộng tuyến bằng VIF
    selected_feats, X_train_selected = run_vif_feature_selection(
        X_train_arr, feature_names, threshold=5.0
    )
    print(
        f"  - VIF lọc giảm số lượng đặc trưng từ {len(feature_names)} xuống {len(selected_feats)}"
    )

    # Thêm bias cho tập rút gọn
    X_train_sel_bias = np.column_stack(
        [np.ones(X_train_selected.shape[0]), X_train_selected]
    )

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
    print(
        f"  - p-value lọc giảm tiếp số lượng đặc trưng xuống {len(final_selected_feats)}: {final_selected_feats}"
    )

    # Chuẩn bị dữ liệu cuối cùng cho OLS rút gọn
    X_train_final = X_train[final_selected_feats].values
    X_test_final = X_test[final_selected_feats].values

    X_train_final_bias = np.column_stack(
        [np.ones(X_train_final.shape[0]), X_train_final]
    )
    X_test_final_bias = np.column_stack(
        [np.ones(X_test_final.shape[0]), X_test_final]
    )

    beta_ols_sel, sigma2_ols_sel = ols_fit(X_train_final_bias, y_train_arr)
    y_pred_ols_sel_log = X_test_final_bias @ beta_ols_sel
    mae_sel, rmse_sel, r2_sel = calculate_metrics(
        y_test_raw, y_pred_ols_sel_log
    )
    results["OLS Chọn Biến"] = {
        "MAE": mae_sel,
        "RMSE": rmse_sel,
        "R2": r2_sel,
        "coefficients": beta_ols_sel,
        "features": final_selected_feats,
    }

    # ----------------------------------------------------
    # Model 3: Ridge Regression (Chọn lambda qua CV)
    # ----------------------------------------------------
    print("Huấn luyện Model 3: Ridge Regression...")
    lambdas = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    # Vẽ Ridge Trace
    plot_ridge_trace(X_train_bias, y_train_arr, lambdas, "ridge_trace.png")

    # Tìm lambda tối ưu
    best_lam_ridge, _, _ = find_best_lambda(
        X_train_bias, y_train_arr, lambdas, model="ridge", k=5
    )
    print(f"  - Ridge: Lambda tối ưu chọn bởi 5-Fold CV: {best_lam_ridge}")

    beta_ridge = ridge_fit(X_train_bias, y_train_arr, lam=best_lam_ridge)
    y_pred_ridge_log = X_test_bias @ beta_ridge
    mae_ridge, rmse_ridge, r2_ridge = calculate_metrics(
        y_test_raw, y_pred_ridge_log
    )
    results["Ridge Regression"] = {
        "MAE": mae_ridge,
        "RMSE": rmse_ridge,
        "R2": r2_ridge,
        "coefficients": beta_ridge,
        "best_lambda": best_lam_ridge,
    }

    # ----------------------------------------------------
    # Model 4: Lasso Regression (Chọn lambda qua CV)
    # ----------------------------------------------------
    print("Huấn luyện Model 4: Lasso Regression...")
    lasso_lambdas = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    best_lam_lasso, _, _ = find_best_lambda(
        X_train_bias, y_train_arr, lasso_lambdas, model="lasso", k=5
    )
    print(f"  - Lasso: Lambda tối ưu chọn bởi 5-Fold CV: {best_lam_lasso}")

    beta_lasso = lasso_fit(X_train_bias, y_train_arr, lam=best_lam_lasso)
    y_pred_lasso_log = X_test_bias @ beta_lasso
    mae_lasso, rmse_lasso, r2_lasso = calculate_metrics(
        y_test_raw, y_pred_lasso_log
    )
    results["Lasso Regression"] = {
        "MAE": mae_lasso,
        "RMSE": rmse_lasso,
        "R2": r2_lasso,
        "coefficients": beta_lasso,
        "best_lambda": best_lam_lasso,
    }

    # ----------------------------------------------------
    # Model 5: Bayesian Linear Regression (Bonus)
    # ----------------------------------------------------
    print("Huấn luyện Model 5: Bayesian Linear Regression...")
    bayesian_model = BayesianRegressor(alpha=1.0, beta_n=1.0)
    bayesian_model.fit(X_train_bias, y_train_arr)
    y_pred_bayes_log = bayesian_model.predict(X_test_bias)
    mae_bayes, rmse_bayes, r2_bayes = calculate_metrics(
        y_test_raw, y_pred_bayes_log
    )
    results["Bayesian Regression"] = {
        "MAE": mae_bayes,
        "RMSE": rmse_bayes,
        "R2": r2_bayes,
        "coefficients": bayesian_model.mu_N,
    }

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

    # 5. Phân tích chẩn đoán phần dư cho mô hình tốt nhất
    # Tìm mô hình có RMSE nhỏ nhất làm mô hình tốt nhất
    best_model_name = min(results, key=lambda k: results[k]["RMSE"])
    print(f"\n🏆 Mô hình tốt nhất dựa trên RMSE: {best_model_name}")

    if best_model_name == "OLS Chọn Biến":
        best_X_train = X_train_final_bias
        best_y_train = y_train_arr
    else:
        best_X_train = X_train_bias
        best_y_train = y_train_arr

    best_beta = results[best_model_name]["coefficients"]

    # Vẽ và lưu chẩn đoán phần dư trên tập Train
    fig = residual_plots(best_X_train, best_y_train, best_beta)
    diag_path = "residuals_diagnostic.png"
    fig.savefig(diag_path)
    plt.close(fig)
    print(f"Đã vẽ và lưu chẩn đoán phần dư của mô hình tốt nhất tại: {diag_path}")

    # 6. Đồ thị dự đoán khoảng tin cậy Bayesian (Chọn ngẫu nhiên 20 cầu thủ để hiển thị rõ)
    print(
        "Tạo biểu đồ khoảng tin cậy Bayesian cho 20 cầu thủ ngẫu nhiên trong tập Test..."
    )
    np.random.seed(42)
    sample_indices = np.random.choice(len(y_test_raw), 20, replace=False)

    y_test_sample_raw = y_test_raw[sample_indices]
    X_test_sample_bias = X_test_bias[sample_indices]

    # Tính khoảng tin cậy 95% trên log scale, sau đó đổi sang thang đo gốc
    lower_log, upper_log = bayesian_model.get_credible_intervals(
        X_test_sample_bias, confidence=0.95
    )
    y_pred_sample_log = bayesian_model.predict(X_test_sample_bias)

    y_pred_sample_raw = np.exp(y_pred_sample_log)
    lower_raw = np.exp(lower_log)
    upper_raw = np.exp(upper_log)

    # Lấy tên cầu thủ mẫu
    player_names = test_df.iloc[sample_indices]["Player Name"].values

    plt.figure(figsize=(12, 7))
    x_axis = np.arange(20)
    plt.errorbar(
        x_axis,
        y_pred_sample_raw / 1e6,
        yerr=[
            (y_pred_sample_raw - lower_raw) / 1e6,
            (upper_raw - y_pred_sample_raw) / 1e6,
        ],
        fmt="o",
        color="royalblue",
        ecolor="lightblue",
        capsize=5,
        elinewidth=2,
        label="Dự đoán & Khoảng tin cậy 95% (Bayesian)",
    )
    plt.scatter(
        x_axis, y_test_sample_raw / 1e6, color="red", marker="x", s=80, label="Giá trị thực tế"
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
    coefs_selected = results["OLS Chọn Biến"]["coefficients"][1:]
    intercept_selected = results["OLS Chọn Biến"]["coefficients"][0]

    # Tính toán lại Standard Errors & p-values cho bảng kết quả hoàn chỉnh trên tập Train
    sig_sel_ols = results["OLS Chọn Biến"]["coefficients"]
    inf_sel_df = coef_inference(
        X_train_final_bias, y_train_arr, sig_sel_ols, sigma2_ols_sel
    )

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

    print("\n--- Nhận xét ý nghĩa các đặc trưng cốt lõi (theo thang đo log):")
    print(
        "- Các biến liên quan tới hiệu suất ghi bàn và đóng góp tổng thể (PTS, MP, VORP, USG%) "
        "thường có tác động tích cực lớn nhất đến mức lương của cầu thủ."
    )
    print(
        "- Do biến mục tiêu được log-transform, một hệ số Beta_j cho biết: khi đặc trưng j tăng 1 đơn vị chuẩn hóa (Z-score), "
        "mức lương của cầu thủ dự kiến tăng trung bình exp(Beta_j) lần."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
