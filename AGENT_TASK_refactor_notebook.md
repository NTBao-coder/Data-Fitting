# AGENT TASK — Refactor Notebook: Thay Thế Thư Viện Bằng Custom Implementations

## 0. Tổng Quan Nhiệm Vụ

**Mục tiêu:** Tái cấu trúc notebook `Data_Fitting_Linear.ipynb` (Kaggle NBA 2022-23)
theo đúng bố cục cell hiện có, **thay thế toàn bộ** các hàm từ `sklearn` và `statsmodels`
bằng các hàm custom do người dùng cung cấp.

**Đầu vào:** File notebook gốc `Data_Fitting_Linear.ipynb`  
**Đầu ra:** File notebook mới `Data_Fitting_Custom.ipynb` — chạy được trên Kaggle,
**không có bất kỳ import nào từ `sklearn.linear_model` hoặc `statsmodels`**.

**Nguyên tắc quan trọng nhất:**
> Không được thay đổi **luồng xử lý**, **thứ tự cell**, **tiêu đề markdown**,
> hay **biến số đầu ra** (`results` DataFrame, `best`, `df_clean`, v.v.).
> Chỉ thay đổi phần **implementation bên trong** từng cell.

---

## 1. Cấu Trúc Notebook Gốc (Cell-by-Cell Reference)

Dưới đây là bản đồ đầy đủ các cell trong notebook gốc và vai trò của từng cell.
Agent phải **giữ nguyên thứ tự và nhãn** này.

| # | Loại | Nhãn / Tiêu đề | Nội dung hiện tại |
|---|------|----------------|-------------------|
| MD-0 | `markdown` | `# Dự báo lương cầu thủ NBA 2022-23` | Tiêu đề, pipeline tổng quan |
| C-1 | `code` | `CELL 1 — Import thư viện` | `import sklearn`, `statsmodels`, `pandas`, `numpy`, v.v. |
| C-2 | `code` | `CELL 2 — Đọc dữ liệu` | `os.walk('/kaggle/input')` |
| C-3 | `code` | `CELL 3 — Load CSV` | `pd.read_csv(...)`, `df.head()` |
| C-4 | `code` | `CELL 4 — Làm sạch dữ liệu` | Lọc `GP >= 20`, drop NaN → `df_clean` |
| C-5 | `code` | `CELL 5 — Chuẩn bị features` | Định nghĩa `FEATURES`, log-transform `Salary` → `y` |
| C-6 | `code` | `CELL 6 — Train/test split + chuẩn hóa` | `train_test_split`, `StandardScaler` |
| C-7 | `code` | `CELL 7 — EDA: Heatmap tương quan` | `seaborn.heatmap` trên correlation matrix |
| C-8 | `code` | `CELL 8 — OLS cơ bản` | `statsmodels.OLS` / `sklearn.LinearRegression` → fit full model |
| C-9 | `code` | `CELL 9 — OLS chọn biến` | Forward/backward selection hoặc p-value filter |
| C-10 | `code` | `CELL 10 — Ridge Regression` | `sklearn.RidgeCV` |
| C-11 | `code` | `CELL 11 — Lasso Regression` | `sklearn.LassoCV` |
| C-12 | `code` | `CELL 12 — Huber Robust Regression` | `sklearn.SVR` với kernel RBF |
| C-13 | `code` | `CELL 13 — Bayesian Linear Regression` | `sklearn.BayesianRidge` |
| C-14 | `code` | `CELL 14 — Bảng so sánh mô hình` | Build DataFrame `results` với cột `Mô hình`, `R² test`, `CV RMSE` |
| C-15 | `code` | `CELL 15 — Biểu đồ so sánh` | Bar chart `R² test` và `CV RMSE` các model |
| MD-16 | `markdown` | `## Bước 6 — Kết luận` | Hướng dẫn điền số liệu |
| C-17 | `code` | `CELL 17 — In tóm tắt kết quả` | Print bảng kết quả cuối |

---

## 2. Quy Tắc Thay Thế (Replacement Rules)

### 2.1 Những gì được GIỮ NGUYÊN ✅

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler     # chỉ dùng để chuẩn hóa X
from sklearn.model_selection import train_test_split # chỉ dùng để split
```

> **Lý do:** `StandardScaler` và `train_test_split` là utility thuần túy,
> không liên quan đến việc fit model — được phép giữ lại.

---

### 2.2 Những gì cần THAY THẾ ❌→✅

Bảng dưới liệt kê từng dòng/hàm cần loại bỏ và hàm custom thay thế tương ứng.
Các hàm custom sẽ được người dùng cung cấp dưới dạng **placeholder block**
(xem Mục 3).

| Thư viện gốc | Hàm/Class bị loại bỏ | Hàm custom thay thế | Cell áp dụng |
|---|---|---|---|
| `statsmodels.api` | `sm.OLS(...).fit()` | `ols_fit(X, y)` | C-1, C-8, C-9 |
| `statsmodels` | `variance_inflation_factor` | `vif(X)` | C-1, C-9 |
| `sklearn.linear_model` | `LinearRegression` | `ols_fit(X, y)` | C-1, C-8 |
| `sklearn.linear_model` | `RidgeCV` | `ridge_fit(X, y, lambdas)` | C-1, C-10 |
| `sklearn.linear_model` | `LassoCV` | `lasso_fit(X, y, lambdas)` | C-1, C-11 |
| `sklearn.linear_model` | `BayesianRidge` | `bayesian_linear_fit(X, y)` | C-1, C-13 |
| `sklearn.svm` | `huber_irls_fit(X, y)` | `huber_irls_fit(X, y, delta)` | C-1, C-12 |
| `sklearn.model_selection` | `cross_val_score`, `KFold` | `kfold_cv(X, y, k, model, lam)` | C-8 đến C-13 |
| `sklearn.metrics` | `r2_score`, `mean_squared_error` | `model_metrics(y_true, y_pred, k)` | C-8 đến C-14 |
| `statsmodels` | `.summary()`, `.pvalues`, `.tvalues` | `coef_inference(X, y, beta_hat, sigma2_hat)` | C-8, C-9 |

---

## 3. Ký Hiệu Placeholder

Tại mỗi vị trí cần inject custom code, agent phải chèn block sau
và **KHÔNG được tự điền code vào**:

```python
# ╔══════════════════════════════════════════════════════════════╗
# ║  [PLACEHOLDER: <TÊN_HÀM>]                                   ║
# ║  Người dùng sẽ cung cấp implementation tại đây.             ║
# ║  Signature yêu cầu:                                         ║
# ║  <MÔ_TẢ_SIGNATURE>                                          ║
# ╚══════════════════════════════════════════════════════════════╝
```

---

## 4. Đặc Tả Chi Tiết Từng Cell Cần Refactor

---

### CELL 1 — Import Thư Viện (refactored)

**Yêu cầu:** Xóa toàn bộ import từ `sklearn.linear_model`, `statsmodels`.
Giữ lại các import utility. Thêm placeholder cho toàn bộ custom library block.

**Cấu trúc cell sau refactor:**

```python
# ============================================================
# CELL 1 — Import thư viện
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════╗
# ║  [PLACEHOLDER: CUSTOM_LIBRARY_BLOCK]                        ║
# ║  Người dùng paste toàn bộ custom functions vào đây:         ║
# ║  - ols_fit(X, y) → Tuple[np.ndarray, float]                 ║
# ║  - hat_matrix(X) → np.ndarray                               ║
# ║  - coef_inference(X, y, beta_hat, sigma2) → pd.DataFrame    ║
# ║  - vif(X) → np.ndarray                                      ║
# ║  - model_metrics(y, y_hat, p) → dict                        ║
# ║  - kfold_cv(X, y, k, model, lam) → Tuple[float, list]       ║
# ║  - find_best_lambda(X, y, lambdas, model, k) → Tuple        ║
# ║  - ridge_fit(X, y, lam) → np.ndarray                        ║
# ║  - lasso_fit(X, y, lam, max_iter, tol) → np.ndarray         ║
# ║  - huber_irls_fit(X, y, delta) → np.ndarray                 ║
# ║  - bayesian_linear_fit(X, y, alpha, beta_n) → Tuple         ║
# ║  - residual_plots(X, y, beta_hat) → plt.Figure              ║
# ╚══════════════════════════════════════════════════════════════╝

print('✓ Import thành công!')
```

---

### CELL 8 — OLS Cơ Bản (refactored)

**Yêu cầu:** Thay `statsmodels.OLS` và `LinearRegression` bằng `ols_fit`.
Giữ nguyên tên biến đầu ra: `y_pred_ols`, `r2_ols`, `rmse_cv_ols`.

**Cấu trúc cell sau refactor:**

```python
# ============================================================
# CELL 8 — OLS cơ bản (full model, tất cả features)
# ============================================================

# --- Fit model ---
# ╔══════════════════════════════════════════════════════════════╗
# ║  [PLACEHOLDER: ols_fit]                                     ║
# ║  beta_ols, sigma2_ols = ols_fit(X_train, y_train)           ║
# ║  Signature:                                                  ║
# ║    ols_fit(X: np.ndarray, y: np.ndarray) -> Tuple           ║
# ║    Returns: beta_hat, sigma_squared                          ║
# ╚══════════════════════════════════════════════════════════════╝

# --- Dự đoán trên tập test ---
# y_train_hat = X_train @ beta_ols
# X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])
# y_pred_ols = X_test_aug @ beta_ols

# --- Tính R² test ---
# ╔══════════════════════════════════════════════════════════════╗
# ║  [PLACEHOLDER: model_metrics]                               ║
# ║  metrics_ols = model_metrics(y_test, y_pred_ols, p=len(FEATURES)+1)
# ║  Signature:                                                  ║
# ║    model_metrics(y, y_hat, p) -> dict                       ║
# ║    Returns: {'RSS': float, 'TSS': float, 'R2': float,       ║
# ║              'adjusted_R2': float, 'F_statistic': float}    ║
# ╚══════════════════════════════════════════════════════════════╝
# r2_ols = metrics_ols['R2']

# --- CV RMSE (5-fold) ---
# ╔══════════════════════════════════════════════════════════════╗
# ║  [PLACEHOLDER: kfold_cv]                                    ║
# ║  mse_cv_ols, _ = kfold_cv(X_train, y_train, k=5, model='ols') ║
# ║  rmse_cv_ols = np.sqrt(mse_cv_ols)                           ║
# ║  Signature:                                                  ║
# ║    kfold_cv(X, y, k, model, lam) -> tuple                    ║
# ╚══════════════════════════════════════════════════════════════╝

# --- In hệ số và kiểm định ---
# ╔══════════════════════════════════════════════════════════════╗
# ║  [PLACEHOLDER: coef_inference]                              ║
# ║  inf_ols = coef_inference(X_train, y_train, beta_ols, sigma2_ols) ║
# ║  Signature:                                                  ║
# ║    coef_inference(X, y, beta_hat, sigma2) -> pd.DataFrame   ║
# ║    Returns: pd.DataFrame with columns: coefficient,         ║
# ║             std_error, t_stat, p_value, ci_lower, ci_upper  ║
# ╚══════════════════════════════════════════════════════════════╝

# --- Hat Matrix & kiểm tra idempotent ---
# ╔══════════════════════════════════════════════════════════════╗
# ║  [PLACEHOLDER: hat_matrix]                                  ║
# ║  H = hat_matrix(X_train)                                    ║
# ║  Signature:                                                  ║
# ║    hat_matrix(X: np.ndarray) -> np.ndarray  # (n, n)       ║
# ║  Kiểm tra: np.allclose(H @ H, H) phải là True              ║
# ╚══════════════════════════════════════════════════════════════╝

# --- Phân tích phần dư ---
# ╔══════════════════════════════════════════════════════════════╗
# ║  [PLACEHOLDER: residual_plots]                              ║
# ║  residual_plots(X_train, y_train, beta_ols)                 ║
# ║  Signature:                                                  ║
# ║    residual_plots(X, y, beta_hat) -> plt.Figure             ║
# ║    Hiển thị 4 subplots: Residuals vs Fitted, Q-Q,           ║
# ║    Scale-Location, Cook's Distance                          ║
# ╚══════════════════════════════════════════════════════════════╝

print(f'OLS cơ bản  →  R² test = {r2_ols:.4f}  |  CV RMSE = {rmse_cv_ols:.4f}')
```

---

### CELL 9 — OLS Chọn Biến (refactored)

**Yêu cầu:** Thay forward selection dùng `statsmodels` bằng logic dùng `ols_fit` + `vif`.
Giữ nguyên biến đầu ra: `FEATURES_SELECTED`, `y_pred_ols_sel`, `r2_ols_sel`, `rmse_cv_ols_sel`.

**Cấu trúc cell sau refactor:**

```python
# ============================================================
# CELL 9 — OLS chọn biến (VIF filter + p-value filter)
# ============================================================

# Bước 1: Loại biến đa cộng tuyến cao (VIF > ngưỡng)
# ╔══════════════════════════════════════════════════════════════╗
# ║  [PLACEHOLDER: vif]                                         ║
# ║  vif_values = vif(X_train)                                  ║
# ║  Signature:                                                  ║
# ║    vif(X: np.ndarray) -> np.ndarray                         ║
# ║  FEATURES_VIF = [FEATURES[i] for i in range(len(FEATURES)) if vif_values[i] < 10] ║
# ╚══════════════════════════════════════════════════════════════╝

# Bước 2: Fit OLS trên tập biến đã lọc
# ╔══════════════════════════════════════════════════════════════╗
# ║  [PLACEHOLDER: ols_fit + coef_inference]                    ║
# ║  Dùng coef_inference để lọc biến có p-value < 0.05         ║
# ║  → FEATURES_SELECTED: list[str]                             ║
# ║  → ols_sel_result = ols_fit(X_train_sel, y_train)           ║
# ╚══════════════════════════════════════════════════════════════╝

# r2_ols_sel   = ...  # float
# rmse_cv_ols_sel = ...  # float từ kfold_cv

print(f'OLS chọn biến  →  R² test = {r2_ols_sel:.4f}  |  CV RMSE = {rmse_cv_ols_sel:.4f}')
print(f'Biến được chọn: {FEATURES_SELECTED}')
```

---

### CELL 10 — Ridge Regression (refactored)

**Yêu cầu:** Thay `sklearn.RidgeCV` bằng `ridge_fit`.
Giữ nguyên biến đầu ra: `y_pred_ridge`, `r2_ridge`, `rmse_cv_ridge`, `lambda_ridge`.

**Cấu trúc cell sau refactor:**

```python
# ============================================================
# CELL 10 — Ridge Regression (L2 regularization)
# ============================================================
LAMBDAS = np.logspace(-3, 4, 100)

# ╔══════════════════════════════════════════════════════════════╗
# ║  [PLACEHOLDER: ridge_fit]                                   ║
# ║  lambda_ridge, _, all_scores = find_best_lambda(X_train, y_train, LAMBDAS, model='ridge') ║
# ║  beta_ridge = ridge_fit(X_train, y_train, lam=lambda_ridge) ║
# ║  Signature:                                                  ║
# ║    find_best_lambda(X, y, lambdas, model, k) -> Tuple        ║
# ║    ridge_fit(X, y, lam) -> np.ndarray                        ║
# ╚══════════════════════════════════════════════════════════════╝

# Vẽ biểu đồ CV Scores theo Lambda
# plt.figure(figsize=(9, 4))
# plt.plot(list(all_scores.keys()), list(all_scores.values()), marker='o')
# plt.axvline(lambda_ridge, ls='--', color='red', label='λ*')
# plt.xscale('log'); plt.xlabel('λ (log scale)'); plt.ylabel('CV MSE')
# plt.title('Ridge Lambda Optimization'); plt.legend(); plt.tight_layout(); plt.show()

# r2_ridge      = ...
# rmse_cv_ridge = ...

print(f'Ridge  →  λ* = {lambda_ridge:.4f}  |  R² test = {r2_ridge:.4f}  |  CV RMSE = {rmse_cv_ridge:.4f}')
```

---

### CELL 11 — Lasso Regression (refactored)

**Yêu cầu:** Thay `sklearn.LassoCV` bằng `lasso_fit`.
Giữ nguyên: `y_pred_lasso`, `r2_lasso`, `rmse_cv_lasso`, `lambda_lasso`.

**Cấu trúc cell sau refactor:**

```python
# ============================================================
# CELL 11 — Lasso Regression (L1 regularization)
# ============================================================

# ╔══════════════════════════════════════════════════════════════╗
# ║  [PLACEHOLDER: lasso_fit]                                   ║
# ║  lambda_lasso, _, _ = find_best_lambda(X_train, y_train, LAMBDAS, model='lasso') ║
# ║  beta_lasso = lasso_fit(X_train, y_train, lam=lambda_lasso) ║
# ║  Signature:                                                  ║
# ║    find_best_lambda(X, y, lambdas, model, k) -> Tuple        ║
# ║    lasso_fit(X, y, lam, max_iter, tol) -> np.ndarray         ║
# ╚══════════════════════════════════════════════════════════════╝

# r2_lasso      = ...
# rmse_cv_lasso = ...
# n_nonzero     = np.sum(beta_lasso != 0)

print(f'Lasso  →  λ* = {lambda_lasso:.4f}  |  Biến được chọn: {n_nonzero}')
print(f'Lasso  →  R² test = {r2_lasso:.4f}  |  CV RMSE = {rmse_cv_lasso:.4f}')
```

---

### CELL 12 — Huber Robust Regression (refactored)

**Yêu cầu:** Thay `sklearn.SVR` bằng `huber_irls_fit`.
Giữ nguyên: `y_pred_svr`, `r2_huber`, `rmse_cv_huber`.

**Cấu trúc cell sau refactor:**

```python
# ============================================================
# CELL 12 — Huber Robust Regression (RBF kernel)
# ============================================================

# ╔══════════════════════════════════════════════════════════════╗
# ║  [PLACEHOLDER: huber_irls_fit]                              ║
# ║  beta_huber = huber_irls_fit(X_train, y_train,              ║
# ║                               gamma='scale', C=1.0)         ║
# ║  Signature:                                                  ║
# ║    huber_irls_fit(X, y, delta) -> dict                   ║
# ║    Returns: {                                                ║
# ║      'predict': Callable[[np.ndarray], np.ndarray],         ║
# ║      'y_hat': np.ndarray                                     ║
# ║    }                                                         ║
# ╚══════════════════════════════════════════════════════════════╝

# y_pred_svr   = beta_huber['predict'](X_test)
# r2_huber       = ...
# rmse_cv_huber  = ...

print(f'Huber Robust Regression  →  R² test = {r2_huber:.4f}  |  CV RMSE = {rmse_cv_huber:.4f}')
```

---

### CELL 13 — Bayesian Linear Regression (refactored)

**Yêu cầu:** Thay `sklearn.BayesianRidge` bằng `bayesian_linear_fit`.
Giữ nguyên: `y_pred_bayes`, `r2_bayes`, `rmse_cv_bayes`.

**Cấu trúc cell sau refactor:**

```python
# ============================================================
# CELL 13 — Bayesian Linear Regression Regression
# ============================================================

# ╔══════════════════════════════════════════════════════════════╗
# ║  [PLACEHOLDER: bayesian_linear_fit]                          ║
# ║  beta_bayes, sigma_bayes = bayesian_linear_fit(X_train, y_train) ║
# ║  Signature:                                                  ║
# ║    bayesian_linear_fit(X, y, alpha, beta_n) -> Tuple         ║
# ╚══════════════════════════════════════════════════════════════╝

# y_pred_bayes   = X_test_aug @ beta_bayes
# r2_bayes       = ...
# rmse_cv_bayes  = ...

print(f'Bayesian Linear Regression  →  R² test = {r2_bayes:.4f}  |  CV RMSE = {rmse_cv_bayes:.4f}')
```

---

### CELL 14 — Bảng So Sánh Mô Hình

**Yêu cầu:** Giữ nguyên **toàn bộ** code, chỉ đảm bảo các biến đầu vào
(`r2_ols`, `rmse_cv_ols`, v.v.) đã được tính đúng từ các cell trên.
DataFrame `results` phải có đúng cấu trúc sau:

```python
# ============================================================
# CELL 14 — Bảng so sánh các mô hình
# ============================================================
results = pd.DataFrame({
    'Mô hình' : ['OLS cơ bản', 'OLS chọn biến', 'Ridge', 'Lasso',
                  'Huber Robust Regression', 'Bayesian Linear Regression'],
    'R² test' : [r2_ols, r2_ols_sel, r2_ridge, r2_lasso, r2_huber, r2_bayes],
    'CV RMSE' : [rmse_cv_ols, rmse_cv_ols_sel, rmse_cv_ridge,
                  rmse_cv_lasso, rmse_cv_huber, rmse_cv_bayes]
})
results = results.sort_values('CV RMSE').reset_index(drop=True)
best    = results['CV RMSE'].idxmin()

print(results[['Mô hình', 'R² test', 'CV RMSE']].to_string(index=False))
print(f'\n→ Mô hình tốt nhất (CV RMSE nhỏ nhất): {results.loc[best, "Mô hình"]}')
```

> ⚠️ **Không thay đổi cell này.** Chỉ đảm bảo tên biến đầu vào nhất quán.

---

## 5. Quy Ước Đặt Tên Biến Bắt Buộc

Để CELL 14 và CELL 17 hoạt động đúng, agent **phải** dùng chính xác các tên biến sau:

| Biến | Kiểu | Mô tả |
|------|------|-------|
| `df_clean` | `pd.DataFrame` | Dữ liệu sau khi lọc `GP >= 20`, drop NaN |
| `FEATURES` | `list[str]` | `['Age', 'GP', 'PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'MP']` |
| `FEATURES_SELECTED` | `list[str]` | Tập biến sau khi chọn lọc ở CELL 9 |
| `X_train`, `X_test` | `np.ndarray` | Đã chuẩn hóa bằng `StandardScaler` |
| `y_train`, `y_test` | `np.ndarray` | `log(Salary)` |
| `r2_ols` | `float` | R² test của OLS cơ bản |
| `r2_ols_sel` | `float` | R² test của OLS chọn biến |
| `r2_ridge` | `float` | R² test của Ridge |
| `r2_lasso` | `float` | R² test của Lasso |
| `r2_huber` | `float` | R² test của Huber Robust Regression |
| `r2_bayes` | `float` | R² test của Bayesian Linear Regression |
| `rmse_cv_ols` | `float` | 5-fold CV RMSE của OLS cơ bản |
| `rmse_cv_ols_sel` | `float` | 5-fold CV RMSE của OLS chọn biến |
| `rmse_cv_ridge` | `float` | 5-fold CV RMSE của Ridge |
| `rmse_cv_lasso` | `float` | 5-fold CV RMSE của Lasso |
| `rmse_cv_huber` | `float` | 5-fold CV RMSE của Huber Robust Regression |
| `rmse_cv_bayes` | `float` | 5-fold CV RMSE của Bayesian Linear Regression |
| `results` | `pd.DataFrame` | Bảng so sánh (CELL 14) |
| `best` | `int` | Index dòng tốt nhất trong `results` |

---

## 6. Giao Diện (Signature) Bắt Buộc của Các Hàm Custom

Agent phải giả định người dùng sẽ cung cấp các hàm với **đúng signature** sau.
Khi gọi hàm, agent phải gọi đúng như mô tả — không thêm tham số tùy tiện.

```python
# ─── Core OLS ────────────────────────────────────────────────
def ols_fit(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Returns: beta_hat, sigma_squared
    """

def hat_matrix(X: np.ndarray) -> np.ndarray:
    """
    Returns: Hat matrix H (n, n)
    """

def coef_inference(X: np.ndarray, y: np.ndarray, beta_hat: np.ndarray, sigma2: float) -> pd.DataFrame:
    """
    Returns: DataFrame with coefficient, std_error, t_stat, p_value, ci_lower, ci_upper
    """

# ─── Metrics & CV ────────────────────────────────────────────
def model_metrics(y: np.ndarray, y_hat: np.ndarray, p: int) -> dict:
    """
    Returns: {'RSS': float, 'TSS': float, 'R2': float, 'adjusted_R2': float, 'F_statistic': float}
    """

def kfold_cv(X: np.ndarray, y: np.ndarray, k: int = 5, model: str = "ols", lam: float = 1.0) -> tuple:
    """
    Returns: cv_score (MSE), fold_mses
    """

def find_best_lambda(X: np.ndarray, y: np.ndarray, lambdas: list, model: str = "ridge", k: int = 5) -> tuple:
    """
    Returns: best_lam, best_score, all_scores
    """

# ─── Diagnostics ─────────────────────────────────────────────
def vif(X: np.ndarray) -> np.ndarray:
    """
    Returns: array of VIF values
    """

def residual_plots(X: np.ndarray, y: np.ndarray, beta_hat: np.ndarray) -> None:
    """
    Returns: matplotlib Figure
    """

# ─── Regularization ──────────────────────────────────────────
def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """
    Returns: beta
    """

def lasso_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0, max_iter: int = 1000, tol: float = 1e-4) -> np.ndarray:
    """
    Returns: beta
    """

def bayesian_linear_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, beta_n: float = 1.0) -> tuple:
    """
    Returns: mu_N, Sigma_N
    """

def huber_irls_fit(X: np.ndarray, y: np.ndarray, delta: float = 1.345, max_iter: int = 100, tol: float = 1e-5) -> np.ndarray:
    """
    Returns: beta
    """
```

---

## 7. Các Ràng Buộc Bổ Sung

### 7.1 Môi trường Kaggle
- Notebook phải chạy được với kernel Python 3.12 trên Kaggle.
- **Không** dùng `pip install` trong cell.
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy` là sẵn có.

### 7.2 Xử lý R² từ custom `model_metrics`
Hàm `model_metrics` trả về dict. Lấy R² bằng:
```python
r2_ols = model_metrics(y_test, y_pred_ols, k=len(FEATURES))['R2']
```
**Không** dùng `sklearn.metrics.r2_score`.

### 7.3 kfold_cv wrapper cho từng model
Mỗi model có thể có signature khác nhau. Dùng `functools.partial` hoặc
lambda để chuẩn hóa về `model_fn(X_tr, y_tr) -> dict có key 'beta'`:
```python
import functools
ridge_fn = functools.partial(ridge_fit, lambdas=LAMBDAS)
rmse_cv_ridge = kfold_cv(X_train, y_train, model_fn=ridge_fn, K=5)
```
Đối với SVR và Bayesian Linear Regression, `kfold_cv` cần wrapper:
```python
def _svr_wrapper(X_tr, y_tr):
    res = huber_irls_fit(X_tr, y_tr)
    # Tạo 'beta' giả để kfold_cv tính y_pred bằng res['predict']
    res['predict_fn'] = res['predict']
    return res
```
> Agent hãy ghi chú rõ trong cell nếu cần điều chỉnh logic wrapper.

### 7.4 Thống nhất augmented matrix
Các hàm custom giả định `X` đầu vào **đã có cột 1** nếu mô hình cần intercept.
Do đó, sau khi dùng `StandardScaler`, cần thêm cột intercept bằng `np.column_stack([np.ones(len(X)), X])` trước khi truyền vào mô hình.

### 7.5 Kiểm tra hat matrix
Sau khi gọi `hat_matrix`, CELL 8 phải in:
```python
H = hat_matrix(X_train)
print('H idempotent:', np.allclose(H @ H, H, atol=1e-6))   # phải là True
print('tr(H) = k+1:', np.isclose(np.trace(H), len(FEATURES)+1, atol=1e-4))
```

---

## 8. Checklist Kiểm Tra Trước Khi Giao Nộp

Agent phải tự kiểm tra từng hạng mục sau trước khi xuất notebook:

- [ ] Không còn `import statsmodels` ở bất kỳ cell nào
- [ ] Không còn `import sklearn.linear_model` ở bất kỳ cell nào  
- [ ] Không còn `import sklearn.svm` ở bất kỳ cell nào
- [ ] Không còn `cross_val_score` hay `KFold` từ sklearn ở bất kỳ cell nào
- [ ] Không còn `r2_score`, `mean_squared_error` từ sklearn ở bất kỳ cell nào
- [ ] Tất cả 11 placeholder block xuất hiện đúng vị trí trong notebook
- [ ] CELL 14 giữ nguyên hoàn toàn, không chỉnh sửa
- [ ] CELL 17 giữ nguyên hoàn toàn, không chỉnh sửa
- [ ] Tất cả tên biến trong Mục 5 được sử dụng nhất quán
- [ ] Notebook chạy lần lượt từ C-1 đến C-17 không lỗi (với placeholder được điền)
- [ ] `hat_matrix` idempotent check in ra `True`
- [ ] Bảng `results` có đúng 6 dòng và 3 cột: `Mô hình`, `R² test`, `CV RMSE`

---

## 9. Ví Dụ Output Mong Đợi (CELL 14)

Sau khi refactor và điền custom functions, CELL 14 phải in ra bảng
có dạng tương tự (số liệu thực tế phụ thuộc implementation):

```
       Mô hình  R² test  CV RMSE
    OLS cơ bản   0.7100   0.6506
 OLS chọn biến   0.2236   0.9784
         Ridge   0.7076   0.6510
         Lasso   0.7130   0.6502
    Huber Robust Regression   0.7180   0.6472
Bayesian Linear Regression   0.7087   0.6496

→ Mô hình tốt nhất (CV RMSE nhỏ nhất): Huber Robust Regression
```

---

*File này là đặc tả tác vụ (task specification) dành cho agent.
Người dùng sẽ cung cấp file custom functions riêng để điền vào các placeholder.*
