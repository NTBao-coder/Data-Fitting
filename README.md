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
2. **Branch Naming:** Tạo nhánh mới từ `main` theo cú pháp: `<tên_thành_viên>/<tên_task>`. *(Ví dụ: `hieu/part1-ols-core`, `huy/part2-api-scrape`)*
3. **Môi trường:** Đảm bảo chạy trên Python 3.10+ (macOS/Unix ưu tiên). Đồng bộ các thư viện qua file `requirements.txt`.
4. **Pull Request (PR):** Hoàn thành task -> Mở PR -> Gắn tag Reviewer (Bảo) -> Approve mới được Merge.

---

## 🏁 Lịch Trình Thực Hiện (14 Ngày)

### Tuần 1: Xây móng & Hoàn thiện Data Pipeline
**Mục tiêu:** Nhóm Lý thuyết xong 80% thuật toán lõi. Nhóm Dữ liệu xuất được file CSV sạch và chạy qua class `DataPipeline` thành công.

| Phân công | Chi tiết công việc | File làm việc |
| :--- | :--- | :--- |
| **Bảo (Leader)** | Setup Github Repo/Ruleset. Viết bộ khung rỗng cho file code. Code mô phỏng Monte Carlo kiểm chứng $\mathbb{E}[\hat{\beta}] = \beta$ ngay khi Hoàng code xong `ols_fit`. | Repo config, `part1_notebook.ipynb` |
| **Hoàng (Toán 1)** | Cài đặt toàn bộ `ols_fit`, `hat_matrix`, các hàm metrics (RSS, TSS, $R^2$, F-test) và kiểm định t-stat. Không dùng sklearn. | `part1/ols_implementation.py` |
| **Dương (Toán 2)** | Cài đặt `vif`, `ridge_fit` và **`lasso_fit` (bằng Coordinate Descent)**. Cài đặt hàm `kfold_cv` ra file riêng. Code 4 biểu đồ phân tích phần dư. | `part1/ridge_lasso.py`, `part1/residual_analysis.py`, `part1/cross_validation.py` |
| **Linh (Data 1)** | Kéo `nba_api`, merge với Kaggle xuất ra CSV tổng. Chạy EDA nhanh (Heatmap) tìm ra các cột bị missing $\ge 5\%$. | `part2/data/`, `eda.ipynb` |
| **Minh (Data 2)** | Hoàn thiện class `DataPipeline`: Tích hợp Imputation (KNN/Regression), Log-transform cho Salary. Tách bạch `.fit()` và `.transform()`. | `part2/data_pipeline.py` |

### Tuần 2: Huấn luyện, Biện luận & Ghép Báo Cáo
**Mục tiêu:** Chốt mô hình, ráp Notebook và hoàn thiện file PDF bằng LaTeX. 

| Phân công | Chi tiết công việc | File làm việc |
| :--- | :--- | :--- |
| **Bảo (Leader)** | Gom code vào 2 file Notebook nộp bài. Tổng kiểm tra data leakage. Chạy Unit test cho toàn bộ thư mục. | `part1_notebook.ipynb`, `part2_notebook.ipynb` |
| **Hoàng + Bảo** | Dịch toán học Part 1 sang mã LaTeX. Chèn công thức chứng minh, chèn đồ thị Ridge Trace và phân tích phần dư vào báo cáo. | `report/report.tex` |
| **Linh + Bảo + Dương** | Train 3 mô hình: OLS cơ bản, OLS lọc biến, và Ridge/Lasso. Đánh giá MAE/RMSE trên tập test (nhớ convert ngược hàm Log). | `part2/model_comparison.py` |
| **Minh + Hoàng + Bảo** | Nhận kết quả train, vẽ biểu đồ Feature Importance. Viết phần LaTeX giải thích kết quả mô hình theo tư duy kinh tế thể thao. | `report/report.tex` |
| **Cả nhóm** | Đọc chéo báo cáo để bắt lỗi chính tả, chuẩn hóa trích dẫn tài liệu tham khảo, xuất PDF và nén file `.zip` theo định dạng của GV. | `report.pdf`, `README.md` |

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
| **Hoàng (Toán 1)** | Ordinary Least Squares (OLS) derivation, Hat Matrix Idempotence, Multicollinearity, VIF, $R^2$ vs Adjusted $R^2$. |
| **Dương (Toán 2)** | Ridge/Lasso Regularization ($\lambda$ penalty term, Coordinate Descent), K-Fold Cross Validation mechanics, Residual Analysis (Heteroskedasticity). |
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
 ┃ ┗ 📜 report.pdf                   # Báo cáo cuối cùng
 ┣ 📜 part1_notebook.ipynb           # Notebook demo Phần 1
 ┣ 📜 part2_notebook.ipynb           # Notebook demo Phần 2
 ┣ 📜 requirements.txt               # Thư viện (numpy, pandas, matplotlib, nba_api...)
 ┣ 📜 .cursorrules                   # Cấu hình AI Assistant
 ┗ 📜 README.md                      # Phải chứa phần "How to Reproduce" hướng dẫn chạy code