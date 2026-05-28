# 📋 ĐẶC TẢ YÊU CẦU ĐỒ ÁN 2: DATA FITTING VÀ OLS

[cite_start]**Đơn vị:** Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên (FIT-HCMUS)[cite: 1, 9].
[cite_start]**Môn học:** Toán Ứng Dụng và Thống Kê (MTH00051)[cite: 4, 10].
[cite_start]**Hạn chót nộp bài (Hard Deadline):** 30/5/2026, trước 23:59[cite: 378].
[cite_start]**Trọng số điểm:** Tổng 10 điểm (Phần 1: 52%, Phần 2: 48%, Bonus: +0.5)[cite: 362, 363].

---

## 🛑 1. CÁC RÀNG BUỘC KỸ THUẬT CỐT LÕI (NGHIÊM CẤM VI PHẠM)
* [cite_start]**Ngôn ngữ & Môi trường:** Python 3.10+[cite: 36, 345]. 
* [cite_start]**Hạn chế thư viện:** Các hàm như `sklearn.linear_model.LinearRegression` hay `numpy.linalg.lstsq` **chỉ được dùng để kiểm chứng**[cite: 46, 47]. [cite_start]Thuật toán chính bắt buộc phải tự cài đặt bằng ma trận thông qua `NumPy`[cite: 37, 47].
* **Tính tái lập (Reproducibility):** Mọi kết quả phải tái lập được. [cite_start]Bắt buộc đặt `random_state` hoặc `seed` cụ thể trong toàn bộ mã nguồn[cite: 352].
* [cite_start]**Kiểm thử (Testing):** Mỗi hàm cài đặt phải có ít nhất 2 unit test kiểm tra kết quả trên dữ liệu đã biết[cite: 353].

---

## 🧮 2. PHẦN 1: LÝ THUYẾT VÀ CÀI ĐẶT THUẬT TOÁN (6.0 Điểm)
[cite_start]**Mục tiêu:** Trình bày lý thuyết toán học và code Python minh họa các tính chất của OLS[cite: 53, 54].

### 2.1. Cài đặt các hàm toán học cốt lõi
* [cite_start]**`ols_fit(X, y)`**: Tính toán vector hệ số $\hat{\beta} = (X^T X)^{-1} X^T y$ và phương sai nhiễu $\hat{\sigma}^2$[cite: 83, 113, 172].
* **`hat_matrix(X)`**: Tính ma trận chiếu $H = X(X^T X)^{-1} X^T$ và kiểm tra tính lũy đẳng $H^2 = H$[cite: 91, 92, 172].
* **`model_metrics(y, y_hat, p)`**: Tính toán các chỉ số RSS, TSS, hệ số xác định $R^2$, $\overline{R}^2$, và kiểm định F[cite: 117, 121, 128, 172].
* [cite_start]**`coef_inference(...)`**: Tính sai số chuẩn (standard errors), t-statistics, p-values và khoảng tin cậy 95%[cite: 126, 172].

### 2.2. Các kỹ thuật nâng cao & Xác thực
* **Xử lý đa cộng tuyến (`vif(X)`)**: Cài đặt hàm tính hệ số phóng đại phương sai (VIF) cho từng biến[cite: 134, 172].
* [cite_start]**Regularization (`ridge_fit(...)`)**: Cài đặt Hồi quy Ridge $\hat{\beta}_{ridge} = (X^T X + \lambda I)^{-1} X^T y$ và vẽ biểu đồ Ridge Trace[cite: 145, 172].
* [cite_start]**Phân tích phần dư (`residual_plots(...)`)**: Vẽ đầy đủ 4 biểu đồ: Residuals vs Fitted, Q-Q Plot, Scale-Location, và Cook's Distance[cite: 153, 154, 155, 156, 172].
* [cite_start]**Cross-Validation (`kfold_cv(...)`)**: Cài đặt k-fold CV để tính điểm CV score[cite: 158, 159, 160, 172].
* [cite_start]**Mô phỏng Monte Carlo**: Sinh dữ liệu giả lập để kiểm chứng Định lý Gauss-Markov: OLS là ước lượng không chệch $\mathbb{E}[\hat{\beta}] = \beta$ và có phương sai nhỏ nhất[cite: 100, 101, 102, 172].

---

## 📊 3. PHẦN 2: ỨNG DỤNG DỮ LIỆU THỰC TẾ (5.5 Điểm)
[cite_start]**Mục tiêu:** Ứng dụng data fitting vào dữ liệu thực tiễn với pipeline hoàn chỉnh[cite: 181].

### 3.1. Ràng buộc về Bộ Dữ Liệu
* [cite_start]**Tính chất:** Dữ liệu thực tế (real-world), không dùng dữ liệu tổng hợp hay toy dataset (như Iris, Boston)[cite: 185].
* **Biến mục tiêu:** Phải là biến liên tục (bài toán hồi quy)[cite: 187].
* [cite_start]**Missing Values:** Phải có ít nhất một cột chứa $\ge 5\%$ giá trị bị thiếu[cite: 186].
* [cite_start]**Kích thước:** Số dòng $n \ge 200$, số đặc trưng $p \ge 3$[cite: 188].

### 3.2. Tiền xử lý Dữ liệu (Data Pipeline)
* **Khảo sát (EDA):** Thống kê mô tả, vẽ phân phối, heatmap tương quan, phân tích missing values và outliers[cite: 208, 209, 210, 212, 213].
* [cite_start]**Xử lý Missing Values:** Chọn phương pháp phù hợp (Mean/Median, Regression, k-NN imputation) và **phải giải thích lý do** dựa trên cơ chế khuyết (MCAR, MAR, MNAR)[cite: 218, 221, 222, 225].
* [cite_start]**OOP Pipeline:** Viết class `DataPipeline` với các phương thức phân tách rõ ràng: học thông số trên tập train (`.fit()`) và áp dụng lên tập test (`.transform()`)[cite: 283].

### 3.3. Xây dựng và Đánh giá Mô hình
* **Huấn luyện:** Phải xây dựng và so sánh ít nhất 3 mô hình: (1) OLS cơ bản, (2) OLS chọn biến (lọc theo p-value/VIF), (3) Ridge hoặc Lasso (chọn siêu tham số $\lambda$ qua CV)[cite: 251, 283].
* [cite_start]**Đánh giá:** Tính MAE, RMSE, $R^2$ trên tập test riêng biệt (không được dùng trong lúc huấn luyện)[cite: 253, 254].
* [cite_start]**Giải thích:** Vẽ biểu đồ Feature importance (hệ số hồi quy) và giải thích kết quả theo ngữ cảnh thực tế của bộ dữ liệu[cite: 283].

---

## 🔮 4. PHẦN NÂNG CAO - BONUS (+0.5 Điểm)
Cài đặt và cấu hình một trong các mô hình nâng cao để so sánh với OLS thông thường:
* **Kernel Ridge Regression**: Mở rộng phi tuyến với ma trận Gram $K_{ij}$ và hàm kernel (ví dụ: RBF)[cite: 262, 264, 266, 268].
* [cite_start]**Bayesian Linear Regression**: Đặt phân phối tiên nghiệm (Prior) và tính toán phân phối hậu nghiệm (Posterior), cung cấp khoảng tin cậy Bayesian (credible intervals) cho dự đoán[cite: 271, 275, 279, 280].

---

## 📦 5. CẤU TRÚC NỘP BÀI BẮT BUỘC
[cite_start]File nén zip nộp trên Moodle bắt buộc phải có kiến trúc thư mục chính xác như sau[cite: 307, 308, 377]:

```text
Group_<ID>/
 ├── README.md
 ├── requirements.txt
 ├── report/
 │    ├── report.tex
 │    └── report.pdf
 ├── part1/
 │    ├── ols_implementation.py
 │    ├── ridge_lasso.py
 │    ├── residual_analysis.py
 │    └── cross_validation.py
 ├── part1_notebook.ipynb
 ├── part2/
 │    ├── data/
 │    │    └── <ten_dataset>.csv
 │    ├── data_pipeline.py
 │    ├── model_comparison.py
 │    └── advanced_methods.py      # (Nếu có làm phần Bonus)
 └── part2_notebook.ipynb