# 🚀 Kế Hoạch Kỹ Thuật Tổng Thể (Technical Execution Plan)

**Dự án:** Data Fitting và Phương Pháp OLS (Project 2)
**Bộ dữ liệu:** NBA Player Salaries - Dự đoán mức lương dựa trên chỉ số thể chất và thi đấu.
**Mục tiêu cốt lõi:** Cài đặt thuật toán hồi quy tuyến tính từ đầu (from scratch) bằng đại số tuyến tính, xây dựng pipeline tiền xử lý dữ liệu hướng đối tượng (OOP) và ngăn chặn tuyệt đối hiện tượng rò rỉ dữ liệu (Data Leakage).

---

## 🛠 1. Tiêu Chuẩn Môi Trường & Luồng Làm Việc
* **Quản lý môi trường:** Sử dụng `micromamba` để khởi tạo không gian làm việc Python 3.10+, khắc phục triệt để lỗi phân tách `PATH` trên hệ điều hành macOS (Apple Silicon).
* **Thư viện cho phép:** `numpy`, `pandas`, `matplotlib`, `seaborn`. 
* [cite_start]**Hạn chế thư viện:** Tuyệt đối không sử dụng `sklearn.linear_model.LinearRegression` hay `numpy.linalg.lstsq` cho các thuật toán lõi của mô hình OLS[cite: 40, 47]. Thư viện `scikit-learn` chỉ được dùng để kiểm chứng kết quả và chia tập dữ liệu.
* [cite_start]**Tính tái lập (Reproducibility):** Cố định `random_state = 42` (hoặc seed tương tự) trong mọi thao tác chia tập Train/Test, khởi tạo K-Fold và mô phỏng Monte Carlo[cite: 352].

---

## 🧮 2. Phần 1: Cài Đặt Toán Học & Lõi Thuật Toán (Core Implementation)
**Mục tiêu:** Xây dựng bộ thư viện hồi quy tuyến tính nội bộ với độ chính xác ma trận tương đương với các thư viện tiêu chuẩn.

### 2.1. Ordinary Least Squares (OLS) & Ma trận mũ (Hat Matrix)
* [cite_start]**Thuật toán:** Tính toán vector hệ số $\hat{\beta} = (X^T X)^{-1} X^T y$ và ma trận chiếu $H = X(X^T X)^{-1} X^T$ bằng phép nhân ma trận `numpy` thuần túy[cite: 83, 91].
* **Kiểm chứng:** Xác nhận tính chất lũy đẳng (idempotent) của ma trận $H$ ($H^2 = H$).
* [cite_start]**Đo lường:** Cài đặt các hàm tính $R^2$, $\overline{R}^2$, F-statistic và t-statistic để suy diễn thống kê cho từng hệ số[cite: 117, 121, 126].

### 2.2. Regularization (Ridge & Lasso)
* [cite_start]**Ridge Regression:** Thêm tham số phạt $L_2$ vào đường chéo chính: $\hat{\beta}_{ridge} = (X^T X + \lambda I)^{-1} X^T y$[cite: 145]. Vẽ đồ thị Ridge Trace để quan sát sự co rút của các hệ số.
* [cite_start]**Lasso Regression:** Cài đặt thuật toán **Coordinate Descent** để tối ưu hóa hàm mất mát $L_1$, do Lasso không có nghiệm đóng (closed-form solution)[cite: 147, 150].

### 2.3. Cross-Validation & Monte Carlo
* [cite_start]**K-Fold CV:** Tự xây dựng hàm chia tập dữ liệu thành $k$ phần (khuyến nghị $k=5$) để tìm siêu tham số $\lambda$ tối ưu cho Ridge và Lasso[cite: 158].
* [cite_start]**Mô phỏng Gauss-Markov:** Sinh ngẫu nhiên hàng ngàn mẫu dữ liệu nhiễu, tính $\hat{\beta}$ cho từng mẫu để chứng minh tính không chệch $\mathbb{E}[\hat{\beta}] = \beta$ bằng thực nghiệm[cite: 100, 172].

---

## 📊 3. Phần 2: Pipeline Dữ Liệu Thực Tế (Real-World Application)
**Mục tiêu:** Áp dụng hệ thống toán học vào bộ dữ liệu thực tế thỏa mãn các điều kiện khắt khe về kích thước và missing values.

### 3.1. Ràng Buộc Dữ Liệu
* **Đầu vào:** File `nba_salary_raw.csv`. [cite_start]Dữ liệu đáp ứng điều kiện $n \ge 200$, $p \ge 3$ và có cột chứa $\ge 5\%$ giá trị khuyết tự nhiên[cite: 184, 186, 188].
* [cite_start]**Biến mục tiêu (Target):** Bài toán hồi quy liên tục[cite: 187]. Áp dụng phép biến đổi Logarit $y = \ln(\text{Salary})$ để xử lý phân phối lệch phải (right-skewed) của mức lương NBA.

### 3.2. Tiền Xử Lý Hướng Đối Tượng (OOP DataPipeline)
Thiết kế class `NBADataPipeline` tuân thủ nghiêm ngặt nguyên tắc ngăn chặn rò rỉ dữ liệu:
1.  **Giai đoạn Học (`.fit()`):** Chỉ trích xuất các tham số (mean, $\mu$, $\sigma$, cấu trúc Dummy variables, mô hình K-NN) từ tập **Train**.
2.  **Giai đoạn Biến đổi (`.transform()`):**
    * [cite_start]Sử dụng **K-NN Imputation (MV4)** để điền khuyết các chỉ số thi đấu[cite: 222]. Quá trình này bắt buộc phải thực hiện *sau khi* chuẩn hóa (Standardization) để thuật toán tính khoảng cách không bị sai lệch.
    * Xử lý biến phân loại bằng `get_dummies` với tham số `drop_first=True` để né bẫy biến giả (Dummy Variable Trap), ngăn chặn hiện tượng đa cộng tuyến hoàn hảo khiến ma trận $X^T X$ bị suy biến.
    * Căn chỉnh cột (Alignment) bằng `.reindex()` để đảm bảo tập Test có ma trận thiết kế cấu trúc chính xác như tập Train.

### 3.3. Đánh Giá & Phân Tích
* [cite_start]Chạy 3 mô hình so sánh: OLS cơ bản, OLS chọn biến (lọc theo VIF), và Ridge/Lasso[cite: 251].
* [cite_start]Chẩn đoán phần dư: Vẽ 4 biểu đồ (Residuals vs Fitted, Q-Q Plot, Scale-Location, Cook's Distance) trên mô hình tốt nhất để kiểm tra các giả thiết Gauss-Markov[cite: 153, 154, 155, 156].
* Dịch ngược dự đoán: Áp dụng hàm $\exp(\hat{y})$ trước khi tính toán các chỉ số lỗi cuối cùng (MAE, RMSE) để đưa kết quả về lại đơn vị USD thực tế.

---

## 📦 4. Cấu Trúc Bàn Giao Kỹ Thuật (Deliverables Architecture)
[cite_start]Hệ thống thư mục bắt buộc phải tuân thủ thiết kế sau để đáp ứng tiêu chí chấm điểm tự động[cite: 307]:

```text
📦 Group_<ID>_Project2/
 ┣ 📂 part1/
 ┃ ┣ 📜 ols_implementation.py        # Core OLS & metrics (No sklearn)
 ┃ ┣ 📜 ridge_lasso.py               # Regularization & Coordinate Descent
 ┃ ┣ 📜 residual_analysis.py         # Diagnostic plots
 ┃ ┗ 📜 cross_validation.py          # K-fold CV implementation
 ┣ 📂 part2/
 ┃ ┣ 📂 data/
 ┃ ┃ ┗ 📜 nba_salary_raw.csv         # Raw dataset
 ┃ ┣ 📜 data_pipeline.py             # OOP Pipeline (Fit/Transform)
 ┃ ┗ 📜 model_comparison.py          # Training, Evaluation & Metrics
 ┣ 📂 report/
 ┃ ┣ 📜 report.tex                   # LaTeX Source
 ┃ ┗ 📜 report.pdf                   # Compiled PDF Document
 ┣ 📜 part1_notebook.ipynb           # Theoretical demo & Monte Carlo
 ┣ 📜 part2_notebook.ipynb           # Pipeline execution & Results interpretation
 ┣ 📜 requirements.txt               # Dependencies list
 ┗ 📜 README.md                      # Reproducibility instructions