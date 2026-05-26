import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_pipeline import NBADataPipeline 
from part1.ols_implementation import ols_fit
from part1.ridge_lasso import ridge_fit, lasso_fit

def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    """Hàm chia tập dữ liệu train/test"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    test_samples = int(n_samples * test_size)
    
    test_idx = indices[:test_samples]
    train_idx = indices[test_samples:]
    
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def evaluate_model(y_true, y_pred, model_name="Model"):
    """ Hàm tính toán các metrics: MAE, RMSE, R^2 """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    mae = np.mean(np.abs(y_true_np - y_pred_np))
    rmse = np.sqrt(np.mean((y_true_np - y_pred_np) ** 2))
    
    ss_res = np.sum((y_true_np - y_pred_np) ** 2)
    ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print(f"--- Kết quả cho {model_name} ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2:  {r2:.4f}\n")
    
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}

def main():
    # 1. Đọc dữ liệu
    df = pd.read_csv('data/merged_nba_data.csv')
    
    # Tách X (features) và y (target)
    X = df.drop(columns=['Salary', 'Player', 'PERSON_ID']) 
    y = df['Salary']

    # 2. Train/Test Split (Tỷ lệ 80/20, theo test_size=0.2 đã gán)
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Chống Data Leakage với DataPipeline
    pipeline = NBADataPipeline()
    pipeline.fit(X_train)
    X_train_processed = pipeline.transform(X_train).astype(float).to_numpy()
    X_test_processed = pipeline.transform(X_test).astype(float).to_numpy()

    # 4. Log-transform biến mục tiêu
    y_train_log = np.log(y_train)
    
    results = []

    # ====================================================
    # MÔ HÌNH 1: OLS Cơ bản
    # ====================================================
    beta_hat_ols, _ = ols_fit(X_train_processed, y_train_log)
    
    # Dự đoán
    y_pred_log_ols = X_test_processed @ beta_hat_ols
    y_pred_ols = np.exp(np.array(y_pred_log_ols, dtype=float))
    
    res_ols = evaluate_model(y_test, y_pred_ols, "OLS Basic")
    results.append(res_ols)

    # ====================================================
    # MÔ HÌNH 2: Ridge Regression
    # ====================================================
    # Giả sử lamda tốt nhất (cv) là 0.1
    best_lam_ridge = 0.1 
    beta_hat_ridge = ridge_fit(X_train_processed, y_train_log, best_lam_ridge)
    
    y_pred_log_ridge = X_test_processed @ beta_hat_ridge
    y_pred_ridge = np.exp(np.array(y_pred_log_ridge, dtype=float))
    
    res_ridge = evaluate_model(y_test, y_pred_ridge, "Ridge Regression")
    results.append(res_ridge)

    # ====================================================
    # MÔ HÌNH 3: Lasso Regression
    # ====================================================
    best_lam_lasso = 0.05
    beta_hat_lasso = lasso_fit(X_train_processed, y_train_log, best_lam_lasso)
    
    y_pred_log_lasso = X_test_processed @ beta_hat_lasso
    y_pred_lasso = np.exp(np.array(y_pred_log_lasso, dtype=float))
    
    res_lasso = evaluate_model(y_test, y_pred_lasso, "Lasso Regression")
    results.append(res_lasso)

    # 5. Xuất bảng so sánh tổng hợp
    results_df = pd.DataFrame(results)
    print("====== BẢNG SO SÁNH TỔNG HỢP MÔ HÌNH ======")
    print(results_df.to_markdown(index=False))

if __name__ == "__main__":
    main()