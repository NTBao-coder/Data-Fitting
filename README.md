# Dự Án 2: Data Fitting và Phương Pháp OLS (Dự Đoán Lương Cầu Thủ NBA)

Dự án này triển khai các thuật toán hồi quy tuyến tính từ đầu (from scratch) bằng `NumPy` mà không sử dụng các thư viện hồi quy có sẵn (như `scikit-learn` hay `numpy.linalg.lstsq` cho lõi hồi quy). Hệ thống được áp dụng trên tập dữ liệu thực tế về các chỉ số thi đấu và lương của cầu thủ NBA mùa giải 2022-23 để thực hiện tiền xử lý dữ liệu và so sánh mô hình.

---

## 📦 Kiến Trúc Thư Mục Bàn Giao

Thư mục dự án tuân thủ nghiêm ngặt cấu trúc được đặc tả trong yêu cầu đồ án:

```text
Group_<ID>/
 ├── README.md                     # Hướng dẫn tái lập và thông tin dự án
 ├── requirements.txt              # Danh sách thư viện phụ thuộc
 ├── part1/                        # Phần 1: Lý thuyết và cài đặt cốt lõi
 │    ├── __init__.py
 │    ├── ols_implementation.py    # Thuật toán OLS thuần túy & suy diễn thống kê
 │    ├── ridge_lasso.py           # Ridge, Lasso (Coordinate Descent) & VIF
 │    ├── residual_analysis.py     # Vẽ đồ thị chẩn đoán residuals & Monte Carlo
 │    ├── cross_validation.py      # K-Fold CV tự cài đặt cho Ridge/Lasso
 │    └── advanced_methods.py      # Elastic Net, Huber Robust Regression, Bayesian Fit
 ├── part1_notebook.ipynb          # Notebook trình diễn thuật toán & Monte Carlo
 ├── part2/                        # Phần 2: Tiền xử lý & Ứng dụng dữ liệu thực tế
 │    ├── data/
 │    │    └── nba_salary_raw.csv  # Dataset chính gốc
 │    ├── data_pipeline.py         # OOP Data Pipeline (Fit & Transform độc lập)
 │    ├── model_comparison.py      # Quy trình huấn luyện & đánh giá so sánh mô hình
 │    └── advanced_methods.py      # Wrapper Bayesian Linear Regression cho dữ liệu NBA
 └── part2_notebook.ipynb          # Notebook chạy thực tế, phân tích EDA & kết quả
```

---

## 🛠️ Hướng Dẫn Cài Đặt Môi Trường

Khởi tạo môi trường Python 3.10 bằng `micromamba` hoặc `conda` để tránh các xung đột đường dẫn trên macOS/Linux:

```bash
# 1. Tạo môi trường mới
micromamba create -n toan_udtk python=3.10 -y

# 2. Kích hoạt môi trường
micromamba activate toan_udtk

# 3. Cài đặt các thư viện phụ thuộc từ file requirements.txt
pip install -r requirements.txt
```

---

## 🚀 Cách Chạy Dự Án

### 1. Chạy các Unit Tests kiểm tra độ chính xác toán học
Chúng tôi đã xây dựng đầy đủ 39 unit tests kiểm tra tính đúng đắn của OLS, Ridge, Lasso, VIF, CV, Elastic Net, Huber Robust và Bayesian Regression. Để chạy toàn bộ test suite:
```bash
python -m pytest tests/test_part1.py -v
```

### 2. Chạy quy trình so sánh mô hình trên dữ liệu thực tế
Để chạy toàn bộ quy trình tiền xử lý, huấn luyện 5 mô hình, xuất các bảng chỉ số đánh giá MAE/RMSE/$R^2$ (đã dịch ngược log-transform về USD) và vẽ các đồ thị chẩn đoán:
```bash
python part2/model_comparison.py
```

Sau khi chạy xong, các tệp đồ thị sau sẽ được sinh ra ở thư mục gốc:
- `ridge_trace.png`: Đường co rút hệ số khi thay đổi tham số phạt $\lambda$ trong Ridge.
- `residuals_diagnostic.png`: 4 đồ thị chẩn đoán giả định Gauss-Markov cho mô hình tốt nhất.
- `bayesian_prediction_intervals.png`: Khoảng tin cậy Bayesian 95% cho 20 cầu thủ ngẫu nhiên.

### 3. Xem các Notebook phân tích
Sử dụng Jupyter Notebook hoặc VS Code để mở các tệp:
- `part1_notebook.ipynb`: Chứa demo OLS ma trận, kiểm chứng tính lũy đẳng của Hat Matrix, Ridge trace và mô phỏng Monte Carlo kiểm chứng OLS là BLUE.
- `part2_notebook.ipynb`: Chứa phân tích EDA (phân phối lương lệch phải, lý do biến đổi log), thực thi Pipeline không rò rỉ dữ liệu, và kết quả so sánh đối chiếu giữa 5 mô hình.
