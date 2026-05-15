# 🚀 Bảng Phân Công Nhiệm Vụ Nước Rút (Sprint 2 Tuần)

**Môn học:** Toán Ứng Dụng và Thống Kê (Project 2 - OLS)
**Bộ dữ liệu:** NBA Player Salaries (Kaggle + `nba_api`) - Mục tiêu: $y = \ln(Salary)$
**Thời gian thực hiện:** 2 Tuần (Hard Deadline).
**Nhóm trưởng:** Bảo (Leader / Reviewer chính).
**Thành viên:** Hoàng, Dương, Minh, Linh.

**Quy tắc sinh tồn:** 1. Push code mỗi ngày. Review PR trong vòng 12 giờ.
2. Không đợi code hoàn hảo mới mở PR, mở "Draft PR" ngay khi bắt đầu làm để Leader theo dõi tiến độ.

---

## 🌿 Quy Tắc Làm Việc Trên GitHub (Git Workflow)
1. **Tuyệt đối không push trực tiếp lên nhánh `main`.**
2. **Branch Naming:** Tạo nhánh mới từ `main` theo cú pháp: `<tên_thành_viên>/<tên_task>`. *(Ví dụ: `hoang/part1-ols-core`, `linh/part2-api-scrape`, `minh/data-pipeline`)*
3. **Môi trường:** Đảm bảo chạy trên Python 3.10+ (macOS/Unix ưu tiên). Đồng bộ các thư viện qua file `requirements.txt`.
4. **Pull Request (PR):** Hoàn thành task -> Mở PR -> Gắn tag Reviewer (Bảo) -> Approve mới được Merge.

---

## 🏁 Lịch Trình Thực Hiện (14 Ngày)

### Tuần 1: Xây móng & Hoàn thiện Data Pipeline
**Mục tiêu:** Nhóm Lý thuyết xong 80% thuật toán lõi. Nhóm Dữ liệu xuất được file CSV sạch và chạy qua class `DataPipeline` thành công.

| Phân công | Chi tiết công việc | File làm việc chính (Ownership) |
| :--- | :--- | :--- |
| **Bảo (Leader)** | Setup Github Repo/Ruleset. Viết skeleton code cho các file. Trực ban duyệt PR liên tục. Code mô phỏng Monte Carlo kiểm chứng $\mathbb{E}[\hat{\beta}] = \beta$ ngay sau khi Hoàng hoàn thành `ols_fit`. | Repo config, `part1_notebook.ipynb` |
| **Hoàng (Toán 1)** | Code toàn bộ `ols_fit`, `hat_matrix`, metrics ($R^2$, F-test) và t-stat. Nhận thêm code 4 biểu đồ phân tích phần dư (do đã có sẵn $y - \hat{y}$). | `part1/ols_implementation.py`, `part1/residual_analysis.py` |
| **Dương (Toán 2)** | Code `vif` tính đa cộng tuyến, `ridge_fit` và **`lasso_fit` (Coordinate Descent)**. Tách hàm `kfold_cv` ra file độc lập. | `part1/ridge_lasso.py`, `part1/cross_validation.py` |
| **Linh (Data 1)** | Kéo `nba_api`, merge với Kaggle xuất ra CSV tổng. Chạy EDA nhanh (Heatmap) tìm ra các cột bị missing $\ge 5\%$. | `part2/data/`, `eda.ipynb` |
| **Minh (Data 2)** | Viết class `DataPipeline` (Imputation, Log-transform, phân tách fit/transform). **Đồng thời phối hợp với Linh để validate quy tắc "One Truth" ngay khi có file CSV tổng.** | `part2/data_pipeline.py` |

### Tuần 2: Huấn luyện, Biện luận & Ghép Báo Cáo
**Mục tiêu:** Chốt mô hình, ráp Notebook và hoàn thiện file PDF bằng LaTeX. 

| Phân công | Chi tiết công việc | File làm việc chính (Ownership) |
| :--- | :--- | :--- |
| **Bảo (Leader)** | **Nhiệm vụ Reviewer:** Gom code vào 2 file Notebook nộp bài. Tổng kiểm tra Data Leakage chặn cuối. Chạy Unit test toàn bộ repo. | `part1_notebook.ipynb`, `part2_notebook.ipynb` |
| **Hoàng (Toán 1)** | Dịch toán học Part 1 sang mã LaTeX. Chèn các công thức chứng minh và đồ thị của Phần 1 vào báo cáo. | `report/report.tex` (Owner Part 1) |
| **Linh (Data 1)** | **Chủ trì Model Training:** Import OLS/Ridge/Lasso đã code để train. Đánh giá MAE/RMSE trên tập test (lưu ý convert ngược hàm Log sang USD). | `part2/model_comparison.py` (Owner file) |
| **Dương (Toán 2)** | Hỗ trợ kỹ thuật cho Linh: Đảm bảo class Ridge/Lasso tự code ở Part 1 tương thích và chạy ổn định trong file `model_comparison.py` của Part 2. | `part2/model_comparison.py` (Support) |
| **Minh (Data 2)** | Vẽ biểu đồ Feature Importance từ kết quả của Linh. Chủ trì viết báo cáo LaTeX Phần 2: Giải thích kết quả mô hình theo tư duy kinh tế/thể thao. | `report/report.tex` (Owner Part 2) |

---

## 🛡️ Khung Đánh Giá Chất Lượng (Quality Assurance & Logic Check)
Bất kỳ Pull Request (PR) nào vi phạm các điểm dưới đây sẽ bị Reject bởi Leader.

1. **Sự nhất quán của Dữ liệu (The "One Truth" Rule):** Số lượng dòng dữ liệu ($n$) đưa vào tập Train bắt buộc phải bằng tổng số dòng sau khi đã loại bỏ/điền Missing Values ở Part 2.2.
2. **Bức tường Data Leakage (The "Iron Wall" Rule):** Imputation hoặc Standardization chỉ được dùng `.fit()` trên **X_train**. Tập **X_test** chỉ được phép dùng `.transform()`.
3. **Liên kết Toán học - Code:** Công thức tính ma trận $\hat{\beta} = (X^T X)^{-1} X^T y$ và $H$ phải được code bằng Numpy chuẩn xác. Tuyệt đối không dùng vòng lặp `for` để tính toán ma trận.
4. **Xử lý Biến Mục Tiêu (Salary):** Phải log-transform lương ($y' = \ln(y)$) khi train. Tuy nhiên, kết quả MAE/RMSE cuối cùng báo cáo cho giảng viên phải được dịch ngược bằng $\exp(y')$ để hiển thị đúng đơn vị USD.

---

## 📚 Hệ Sinh Thái Kiến Thức Cốt Lõi (Mandatory Knowledge)
Mỗi thành viên **bắt buộc** phải tự nghiên cứu các khái niệm dưới đây để phục vụ bảo vệ đồ án/vấn đáp:

| Phụ trách | Các Từ Khóa Cốt Lõi Cần Nắm Vững |
| :--- | :--- |
| **Bảo (Leader)** | Bias-Variance Tradeoff, Monte Carlo Simulation properties, Data Leakage prevention, OLS Assumptions (Gauss-Markov Theorem). |
| **Hoàng (Toán 1)** | Ordinary Least Squares (OLS) derivation, Hat Matrix Idempotence, $R^2$ vs Adjusted $R^2$, Residual Properties. |
| **Dương (Toán 2)** | Ridge/Lasso Regularization ($\lambda$ penalty term, Coordinate Descent), K-Fold Cross Validation mechanics, Multicollinearity, VIF. |
| **Linh (Data 1)**| API Rate Limiting, JSON Parsing, Pandas Merge/Join logic, Missing Not At Random (MNAR) vs Missing Completely At Random (MCAR). |
| **Minh (Data 2)**| KNN Imputation vs Mean Imputation, One-Hot Encoding vs Label Encoding, Dummy Variable Trap, Standardization vs Normalization. |

---

## 📦 Danh Mục Bàn Giao Bắt Buộc (Mandatory Github Deliverables)
Cấu trúc nhánh `main` khi tới deadline bắt buộc phải đúng như sau:

```text
📦 Group_<ID>_Project2/
 ┣ 📂 part1/
 ┃ ┣ 📜 ols_implementation.py        # Code tự viết các hàm toán học OLS
 ┃ ┣ 📜 ridge_lasso.py               # Code Ridge/Lasso (bao gồm thuật toán Coordinate Descent)
 ┃ ┣ 📜 residual_analysis.py         # Code vẽ biểu đồ phần dư
 ┃ ┗ 📜 cross_validation.py          # Code chia tập k-fold CV
 ┣ 📂 part2/
 ┃ ┣ 📂 data/
 ┃ ┃ ┗ 📜 nba_salary_raw.csv         # File dữ liệu tổng (Lưu ý giới hạn 50MB của Github)
 ┃ ┣ 📜 data_pipeline.py             # Class DataPipeline xử lý dữ liệu
 ┃ ┗ 📜 model_comparison.py          # Script chạy test và đánh giá
 ┣ 📂 report/
 ┃ ┣ 📜 report.tex                   # Mã nguồn LaTeX
 ┃ ┗ 📜 report.pdf                   # Báo cáo cuối