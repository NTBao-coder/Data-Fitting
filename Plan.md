# 🚀 Bảng Phân Công Nhiệm Vụ Nước Rút (Sprint 2 Tuần)

**Môn học:** Toán Ứng Dụng và Thống Kê (Project 2 - OLS)
**Bộ dữ liệu:** NBA Player Salaries (Kaggle + `nba_api`) - Mục tiêu: y = ln(Salary)
**Thời gian thực hiện:** 2 Tuần (Hard Deadline).
**Nhóm trưởng:** Bảo (Leader / Reviewer / Organizer).
**Thành viên:** Linh, Minh, Dương, Hoàng.

**Quy tắc sinh tồn:** 1. Push code mỗi ngày. Leader review PR trong vòng tối đa 12 giờ (SLA 12h).
2. Không đợi code hoàn hảo mới mở PR, mở "Draft PR" ngay khi bắt đầu làm để Leader theo dõi tiến độ.
3. **Daily Standup:** Mỗi ngày 18:00 (15 phút) trên nhóm chat cập nhật 3 mục: Done / Blocked / Todo.

---

## 🌿 Quy Tắc Làm Việc Trên GitHub (Git Workflow)
1. **Tuyệt đối không push trực tiếp lên nhánh `main`.**
2. **Branch Naming:** Tạo nhánh từ `main` theo cú pháp: `<tên_thành_viên>/<tên_task>`. *(Ví dụ: `linh/ols-impl`, `duong/ridge-lasso`, `Dương/data-pipeline`, `hoang/api-scrape`)*
3. **Môi trường:** Đảm bảo chạy trên Python 3.10+ (macOS/Unix ưu tiên). Đồng bộ các thư viện qua `requirements.txt`.
4. **Pull Request (PR):** Phải sử dụng mẫu kiểm tra (tạo file `.github/pull_request_template.md` chứa checklist QA). Gắn tag Reviewer (Bảo), Approve mới được Merge.

---

## 🏁 Lịch Trình Thực Hiện (14 Ngày)

### Tuần 1: Xây móng & Hoàn thiện Data Pipeline
**Mục tiêu:** Nhóm Lý thuyết xong 80% thuật toán lõi. Nhóm Dữ liệu xuất được file CSV sạch và chạy qua class `DataPipeline`.

| Phân công | Chi tiết công việc | File làm việc chính (Owner) |
| :--- | :--- | :--- |
| **Bảo (Leader)** | Setup repo + skeleton code + PR Template. Tuyệt đối không code tính năng. Tập trung Review PR liên tục. | Repo config |
| **Linh (Toán 1)** | Code `ols_fit`, `hat_matrix`, metrics, t-stat. Tự code residual plot (3-4 biểu đồ). Nhận thêm code mô phỏng Monte Carlo E[beta_hat] = beta. | `part1/ols_implementation.py`, `part1/residual_analysis.py` |
| **Minh (Toán 2)** | Code `vif`, `ridge_fit`, và `lasso_fit` (Coordinate Descent). Code `kfold_cv`. Tập trung hoàn thành trước ngày 7. | `part1/ridge_lasso.py`, `part1/cross_validation.py` |
| **Hoàng (Data 1)** | Kéo `nba_api`, merge Kaggle xuất CSV tổng. Chạy EDA nhanh (Heatmap) tìm cột missing >= 5%. | `part2/data/`, `eda.ipynb` |
| **Dương (Data 2)** | Code `DataPipeline` (phân tách fit/transform). Validate quy tắc "One Truth" với Hoàng. *(Nếu xong sớm Day 5: Giúp A code test cases cho Ridge/Lasso)*. | `part2/data_pipeline.py` |

### Tuần 2: Huấn luyện, Biện luận & Ghép Báo Cáo
**Mục tiêu:** Chốt mô hình, ráp Notebook và hoàn thiện file PDF bằng LaTeX.

| Phân công | Chi tiết công việc | File làm việc chính (Owner) |
| :--- | :--- | :--- |
| **Bảo (Leader)** | Chạy Sanity Check toàn bộ code. Kiểm tra Data Leakage chặn cuối. **Chủ trì thiết kế Trang bìa (Cover), Mục lục (TOC) và chuẩn hóa định dạng (Format) tổng thể của file LaTeX.** | Toàn bộ Repo, `report/report.tex` (Formatting) |
| **Linh (Toán 1)** | Dịch toán học Part 1 sang LaTeX. **Chủ trì `part1_notebook.ipynb`** (demo kết quả OLS + Monte Carlo). Tự đóng góp 1-2 tài liệu tham khảo phần OLS. | `report/report.tex` (Part 1), `part1_notebook.ipynb` |
| **Hoàng (Data 1)** | Chủ trì Model Training. **Chủ trì `part2_notebook.ipynb`** (demo DataPipeline + KQ Train). Code Unit Test cho Part 2. Tự đóng góp 1-2 tài liệu tham khảo phần API/Modeling. | `part2/model_comparison.py`, `part2_notebook.ipynb` |
| **Minh (Toán 2)** | Tối ưu hóa thuật toán Ridge/Lasso nếu chạy chậm. Chủ trì maintain và code Unit Test cho Part 1. Tự đóng góp 1-2 tài liệu tham khảo phần Regularization. | `tests/test_part1.py` |
| **Dương (Data 2)** | Vẽ biểu đồ Feature Importance từ KQ của Hoàng. **Chủ trì LaTeX Part 2 và viết phần Kết luận (Conclusion) cho toàn bộ báo cáo**. Tự đóng góp 1-2 tài liệu tham khảo phần Imputation/Encoding. | `report/report.tex` (Part 2 & Conclusion) |

### 🎯 Checkpoint Ngày 11: Hoàng merge xong model_comparison.py. Linh merge xong part1_notebook.ipynb. Bảo bắt đầu sanity check tổng.

---

## 🛡️ Khung Đánh Giá Chất Lượng (Quality Assurance & Logic Check)
Bất kỳ PR nào vi phạm sẽ bị Leader Reject.

1. **Sự nhất quán Dữ liệu (The "One Truth"):** Kích thước (n) của tập Train bắt buộc phải bằng tổng số dòng sau xử lý Missing Values ở Part 2.2.
2. **Bức tường Data Leakage:** Imputation/Standardization chỉ dùng `.fit()` trên **X_train**. Tập **X_test** chỉ được phép dùng `.transform()` nhằm triệt tiêu hoàn toàn rủi ro rò rỉ dữ liệu (Data Leakage).
3. **Liên kết Toán học:** Tính beta_hat = (X^T X)^-1 X^T y và ma trận H bằng Numpy chuẩn xác. Cấm dùng vòng lặp `for`.
4. **Xử lý Biến Mục Tiêu:** Log-transform lương (y' = ln(Salary)) khi train. Dịch ngược bằng exp(y') khi tính toán MAE/RMSE cuối cùng.
5. **Tiêu chuẩn Học thuật (Report Rules):** Báo cáo bắt buộc phải đủ Trang bìa, Mục lục, Kết luận và Danh mục tài liệu tham khảo có tối thiểu 5 nguồn uy tín (sách, bài báo khoa học, nguồn tài liệu chính thống của giải đấu).

---

## 📚 Hệ Sinh Thái Kiến Thức Cốt Lõi (Mandatory Knowledge per Member)
Mỗi thành viên **bắt buộc** phải nắm vững các khối kiến thức tương ứng với phân công thực tế để phục vụ quá trình bảo vệ vấn đáp trước giảng viên:

| Thành viên | Khối Kiến Thức Bắt Buộc Phải Ôn Tập |
| :--- | :--- |
| **Bảo (Leader)** | Bias-Variance Tradeoff, Monte Carlo Simulation properties, Data Leakage prevention, OLS Assumptions (Gauss-Markov Theorem). |
| **Linh (Toán 1)** | Chứng minh công thức OLS (OLS derivation), tính chất ma trận hình chiếc mũ (Hat Matrix Idempotence), mô phỏng lặp Monte Carlo. |
| **Minh (Toán 2)** | Cơ chế thu hẹp biến của Ridge/Lasso regularization, thuật toán tối ưu Coordinate Descent cho Lasso, nguyên lý K-Fold Cross Validation. |
| **Hoàng (Data 1)**| Các cơ chế khuyết dữ liệu (MCAR, MAR, MNAR), tư duy xử lý và ghép nối bảng (Pandas merge/join), kỹ thuật gọi API và xử lý giới hạn băng thông (Rate Limiting). |
| **Dương (Data 2)**| Thuật toán điền khuyết KNN Imputation, mã hóa biến phân loại (Categorical Encoding), bẫy biến giả (Dummy Variable Trap), phân biệt chuẩn hóa dữ liệu (Standardization vs Normalization). |

---

## 🚨 Kế Hoạch Dự Phòng (Backup Plan)
Để đảm bảo đúng tiến độ Hard Deadline 14 ngày, toàn nhóm tuân thủ các phương án ứng phó rủi ro sau:
* **Rủi ro Toán học:** Nếu A bị trễ tiến độ (delay) thuật toán Lasso Coordinate Descent, Linh lập tức tạm dừng việc khác để support 50% khối lượng code Ridge/Lasso.
* **Rủi ro Dữ liệu:** Nếu Hoàng bị kẹt ở khâu gọi API/Merge dữ liệu, A sẽ nhận lại file CSV thô để chạy tạm Model Comparison, Hoàng chuyển sang fix bug API sau.

---

## 📦 Danh Mục Bàn Giao Bắt Buộc (Mandatory Github Deliverables)
1 file = 1 Owner. Owner là người chịu trách nhiệm chính nếu có bug.

```text
📦 Group_<ID>_Project2/
 ┣ 📂 part1/
 ┃ ┣ 📜 ols_implementation.py        # Owner: Linh
 ┃ ┣ 📜 ridge_lasso.py               # Owner: A
 ┃ ┣ 📜 residual_analysis.py         # Owner: Linh
 ┃ ┗ 📜 cross_validation.py          # Owner: A
 ┣ 📂 part2/
 ┃ ┣ 📂 data/                        # Owner: Hoàng
 ┃ ┃ ┗ 📜 nba_salary_raw.csv         
 ┃ ┣ 📜 data_pipeline.py             # Owner: Dương
 ┃ ┗ 📜 model_comparison.py          # Owner: Hoàng
 ┣ 📂 report/                        # Owner: Dương (Part 2 & Conclusion), Linh (Part 1), Bảo (Format)
 ┃ ┣ 📜 report.tex                   
 ┃ ┗ 📜 report.pdf                   
 ┣ 📜 part1_notebook.ipynb           # Owner: Linh
 ┣ 📜 part2_notebook.ipynb           # Owner: Hoàng
 ┣ 📜 requirements.txt               # Owner: Bảo (Duyệt cuối)
 ┣ 📜 README.md                      # Owner: Bảo
 ┗ 📂 .github/
   ┗ 📜 pull_request_template.md     # Owner: Bảo