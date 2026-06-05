#PHẦN 2: ỨNG DỤNG DATA FITTING VÀO DỮ LIỆU THỰC TẾ

**1. Cơ sở lý thuyết:**

**2.** **Áp dụng Data Fitting vào dữ liệu:**

**2.1. Tìm dữ liệu:**

\- Nguồn dữ liệu: <https://www.kaggle.com/datasets/jamiewelsh2/nba-player-salaries-2022-23-season>

\- Thông tin cơ bản của dữ liệu: Dữ liệu về lương và thông số của các cầu thủ trong giải đấu bóng rổ nhà nghề Mỹ (NBA), mùa giải 2022 – 2023.

\- Merge dữ liệu: Dữ liệu của **file .csv** trên Kaggle chỉ chứa thông tin chủ yếu về tiền lương và một vài thông số khác, tuy nhiên, không chứa đủ và không cập nhật các thông số về chỉ số của các cầu thủ ấy trên sân. Dữ liệu trên **nba\_api** lại chứa các chỉ số thi đấu trên sân nhưng không công bố về tiền lương. Vì thế, để nguồn dữ liệu được đầy đủ và để làm cho mô hình chính xác hơn, cần hợp nhất 2 file với nhau.

**2.2.** **Quy mô dữ liệu:** Bộ dữ liệu trên gồm 467 quan trắc (467 cầu thủ) tương ứng với 467 dòng, 51 đặc trưng (cột) tương ứng với 51 cột. Ý nghĩa của từng cột:

- **Player Name**: Tên cầu thủ.
- **Position**: Vị trí thi đấu (PG, SG, PF, C...).
- **Salary**: Mức lương tính bằng USD.
- **Team**: Đội bóng chủ quản.
- **Age**: Tuổi của cầu thủ (thường tỉ lệ thuận với kinh nghiệm và lương).
- **GP (Games Played)**: Số trận đã thi đấu.
- **GS (Games Started)**: Số trận đá chính.
- **MP (Minutes Played)**: Số phút thi đấu mỗi trận.
- **FG, FGA, FG%**: Các chỉ số về cú ném thành công và tỉ lệ ném chính xác.
- **3P, 3PA, 3P%**: Các chỉ số về ném 3 điểm.
- **2P, 2PA, 2P%**: Các chỉ số về ném 2 điểm.
- **eFG%:** Tỉ lệ ném rổ hiệu quả.
- **FT, FTA, FT%**: Các chỉ số về ném phạt.
- **ORB, DRB, TRB**: Các chỉ số về bắt bóng bật bảng (Rebounds): Rebounds tấn công, Rebounds phòng thủ, tổng số Rebounds
- **AST**: Kiến tạo.
- **STL**: Cướp bóng.
- **BLK**: Chắn bóng.
- **TOV**: Mất bóng.
- **PF:** Cầu thủ tiên phong chính.
- **PTS**: Điểm số ghi được trung bình mỗi trận.
- **Total Minute:** Số phút đã chơi.
- **PER:** Chỉ số hiệu quả (số lần chơi tích cực trừ đi số sai lầm).
- **TS%:** Tỉ lệ ném bóng thực tế.
- **USG%:** Tỉ lệ sử dụng bóng.
- **WS:** Số trận thắng có đóng góp của cầu thủ.
- **BPM (Box Plus/Minus):** Điểm mà cầu thủ tạo ra cho mỗi 100 lần kiểm soát bóng.
- **VORP:** Giá trị đóng góp của 1 cầu thủ với 1 cầu thủ dự bị.

Trong các biến này, ta chọn **biến mục tiêu** là **Salary** – Lương của cầu thủ. Các biến còn lại sẽ là biến đầu vào.

**2.3.** **Khảo sát, phân tích dữ liệu**

**2.3.1.** **Merge** **file dữ liệu từ Kaggle với nba\_api**

- Mục đích: Để có đầy đủ thông tin về chỉ số và mức lương của một cầu thủ, tạo thành một bộ dữ liệu hoàn chỉnh để xử lý.
- Thiết lập: Trong file `fetch\_and\_merge.py`, xây dựng hàm `build\_raw\_dataset()`: Lấy dữ liệu từ `nba\_api` -> Đọc dữ liệu của file .csv lấy từ Kaggle -> Hợp nhất dữ liệu giữa 2 file bằng phép Inner Join.

**2.3.2.** **Kiểm tra và xử lý dữ liệu bị trùng lặp và nhiễu**

- Mục đích: Loại bỏ các cầu thủ có dữ liệu trùng lặp và dữ liệu nhiễu (các cầu thủ thi đấu quá ít trận hoặc quá ít số phút) để làm cho dữ liệu sạch và dễ xử lý hơn.
- Thiết lập: Trong file `fetch\_and\_merge.py`, xây dựng hàm `build\_raw\_dataset()`: Loại bỏ các dòng trùng lặp dựa trên tên cầu thủ -> Lọc dữ liệu nhiễu (loại bỏ các cầu thủ thi đấu ít hơn 5 trận hoặc ít hơn 50 phút, vì các cầu thủ này thường sẽ không đóng góp nhiều vào trong giải đấu nên nếu thêm vào sẽ gây nhiễu dữ liệu).

**2.3.3.** **Thực hiện EDA cho dữ liệu**
- Mục đích: Hiểu rõ hơn về dữ liệu, các đặc trưng của dữ liệu và mối quan hệ giữa các biến số trong dữ liệu, từ đó đưa ra những nhận định và quyết định đúng đắn trong việc xử lý dữ liệu và xây dựng mô hình.

*a.* *Phân tích kích thước dữ liệu và tỉ lệ missing values:*
- Trong file `part2_notebook.ipynb`, sử dụng hàm `df.shape` để thống kê kích thước dữ liệu (số dòng và số cột); sử dụng hàm .isnull().mean() để thống kê tỉ lệ missing values cho từng biến.
- Dùng hàm sns.histplot() để trực quan phân phối mức lương. Dựa vào biểu đồ, ta thấy rằng mức lương của các cầu thủ có sự chênh lệch đáng kể, không có sự phân phối chuẩn hay đều. Tuy nhiên, nếu lấy logarit của mức lương, ta sẽ được sự phân phối gần như chuẩn.
&rarr Kết luận: Việc đưa biến mục tiêu (Salary) về dạng logarit sẽ đưa phân phối về dạng gần chuẩn, làm giảm sự chênh lệch mức lương giữa các cầu thủ và giúp mô hình dự đoán chính xác hơn.

*b.* *Thống kê mô tả:* mean, median, min, max, std, quartiles (25%, 50%, 75%).
- Trong file `part2_notebook.ipynb`, sử dụng hàm `df.describe()` để thống kê mô tả các biến số trong dữ liệu. Hàm này sẽ trả về các thông số cần thiết trong thống kê mô tả

*c.* *Phân phối từng biến:* Vẽ histogram và boxplot cho các biến số trong dữ liệu. Một vài biến ảnh hưởng tới biến mục tiêu (Salary):
- Age: Có phân phối lệch trái, cho thấy tuổi càng thấp càng có lương cao, tuy nhiên không có sự chênh lệch lớn.
- GP: Có phân phối hơi lệch trái, cho thấy đa số cầu thủ sẽ chơi nhiều trận đấu để cống hiến cho đội bóng. Một vài cầu thủ do chấn thương hoặc các lý do khác nên số trận đấu không nhiều.
- MP: Có phân phối đối xứng gần chuẩn, cho thấy thời gian thi đấu của các cầu thủ khá đồng đều, trải dài từ khoảng hơn 5 phút - gần 40 phút.
- PTS: Có phân phối lệch phải, cho thấy số điểm tập trung nhiều ở mức từ 0 - 10 điểm, rất ít cầu thủ ghi được nhiều điểm mỗi trận.

*d.* *Ma trận tương quan (heatmap):*
- Trong file `part2_notebook.ipynb`, sử dụng hàm sns.heatmap() để trực quan ma trận tương quan, với tham số `annot=True` để hiển thị giá trị tương quan trên mỗi ô, `cmap='coolwarm'` để tô màu theo giá trị tương quan (từ -1 đến 1).
- Nhận xét:
    - Ma trận này là một ma trận đối xứng qua đường chéo chính (đường chéo gồm toàn giá trị 1.0).
    - Thang độ màu chạy từ $-1.0$ (đỏ đậm - tương quan nghịch biến hoàn toàn) đến $1.0$ (xanh đậm tương quan đồng biến hoàn toàn). Các vùng màu nhạt (gần mức $0.0$) thể hiện hai biến độc lập hoặc có quan hệ tuyến tính rất yếu.
    - Dựa vào ma trận, ta có thể thấy một vài biến có độ tương quan yếu với biến mục tiêu Salary (FG%, 3P%, 2P%, ...); những biến có độ tương quan mạnh với nhau - hay còn gọi là đa cộng tuyến (ví dụ: GP và GS, 3P và 3PA, 2P và 2PA, ...). Sự tồn tại của các cặp biến có độ tương quan cực cao này sẽ làm cho ma trận $X^T X$ gần suy biến, khiến phương sai của các hệ số ước lượng $\hat{\beta}$ tăng. Do đó, nhóm dự kiến sẽ thực hiện loại bỏ các cột không cần thiết và có tương quan cao với nhau để làm giảm đa cộng tuyến, cũng như làm cho ma trận tương quan trở nên dễ nhìn và dễ phân tích hơn.

**2.4.** **Tiền xử lý dữ liệu**
**2.4.1.** **Chia Train/Test từ dữ liệu**
- Mục đích: Chia dữ liệu ra thành 2 tập **Train** và **Test**. 
   - Tập **Train** mang nhiệm vụ "dạy" cho mô hình để tìm ra các quy luật, hệ số, đường xu hướng; đánh giá mức độ ảnh hưởng của các biến lên biến mục tiêu; đánh giá độ phù hợp của mô hình với dữ liệu. 
   - Tập **Test** có được giấu đi trong quá trình huấn luyện. Sau khi mô hình học xong, ta sẽ đưa tập **Test** vào để kiểm thử khả năng dự đoán, tổng quát hoá của mô hình với dữ liệu, đánh giá xem mô hình có đúng với thực tế hay không.
   - Nếu không chia ra thành 2 tập khác biệt, mô hình sẽ có hiện tượng Overfitting: mô hình chỉ học thuộc các dữ liệu đã train, không thể hoặc có hiệu suất kém trong việc xử lý dữ liệu mới chưa từng gặp.
   - Ngoài ra, cần đặt **random seed** để có thể chia dữ liệu 1 cách ngẫu nhiên, từ đó đảm bảo tính khách quan và tái lặp kết quả. Để có thể tái lập lại kết quả, nhóm đã thiết lập seed = 42 và dùng nó trong tất cả các bước xử lý dữ liệu và xây dựng mô hình.
- Thiết lập:
   - Trong file `part2_notebook.ipynb`, chia dữ liệu ra làm 2 tập **Train** (80%) và **Test** (20%) thông qua hàm `sample()`, và đặt **random_seed = 42**.
   ```python
   train_df = df.sample(frac=0.8, random_state=42)
   test_df = df.drop(train_df.index)
   ```
   - Trong file `data_pipeline.py`, nhận 2 mảnh **Train** và **Test** từ vừa truyền vào, sau đó thực hiên fit và transform 2 mảnh dữ liệu này bằng cách gọi lần lượt các hàm trong class `NBADataPipeline`.
   ```python
   pipeline = NBADataPipeline()
   X_train, y_train, X_test, y_test = pipeline.process_pipeline(train_df, test_df)
   ```

**2.4.2.** **Encode dữ liệu chuỗi**
- Mục đích: Xử lý các biến có dữ liệu dạng chuỗi, mã hóa chúng về dạng số để mô hình có thể hiểu và xử lý được. (Ví dụ: Trong 'Position', vị trí PG sẽ được mã hóa thành 0, SG thành 1, C thành 2...)
- Thiết lập: Trong file `data_pipeline.py`:
  - Về phía hàm fit(): Mô hình sẽ được dạy bằng cách học cách mã hóa các cột có chứa dữ liệu là kiểu chuỗi thành các cột nhị phân chứa giá trị 0 hoặc 1 (tạo biến Dummy). Cầu thủ thuộc đội nào thì cột tương ứng với đội đó sẽ nhận giá trị là 1 và các cột khác nhận giá trị 0.
  - Về phía hàm transform(): Test "kiến thức" mã hóa mà mô hình đã được học. Các đội bóng được mã hóa như thế nào thông qua hàm fit(), thì qua hàm transform(), các đội bóng tương ứng cũng sẽ được mã hóa tương tự như vậy.
  Tuy nhiên, sẽ có conflict xảy ra: Tập Train không chứa tất cả các tên đội -> Team sẽ có các tên đội lạ -> không thể mã hóa được các cột tương ứng -> Gây ra lỗi.
  Để khắc phục, nhóm dùng hàm reindex() tái chỉ số các cột chưa có chỉ số, dựa vào danh sách đội bóng gốc, ép bảng dữ liệu của Test phải định hình lại sao cho có đầy đủ tất cả các đội bóng như danh sách.
  Nếu các cột này bị lỗi -> Cơ chế xử lý lỗi giúp tạo một cột phụ, và điền số 0 cho toàn bộ, tránh việc gây lỗi hệ thống, mô hình vẫn chạy bình thường.

**2.4.3.** **Chuẩn hóa**
- Mục đích: Giúp đưa các biến về cùng một thang đo, giảm thiểu ảnh hưởng của các biến có giá trị lớn, đảm bảo tính nhất quán và cho các dữ liệu tuân theo một quy tắc nhất định.
- Thiết lập:
   - Tại hàm fit(): Tính toán giá trị trung bình (mean) và độ lệch chuẩn (std) cho từng biến số. Sau đó, áp dụng công thức chuẩn hóa Z-score.
   - Tại hàm transform(): Áp dụng giá trị trung bình và độ lệch chuẩn đã tính ở trên cho tập Test.

**2.4.4.** **Xử lý Missing Values**
- Mục đích: Điền các thông số còn khuyết thiếu của dữ liệu.
- Thiết lập:
   - Thuật toán: Cần phải chọn một thuật toán hợp lý để xử lý Missing Values.
     - Listwise deletion: Xóa toàn bộ hàng bị khuyết dữ liệu -> Mất dữ liệu, lãng phí tài nguyên, làm mô hình hoạt động không tốt.
     - 



    
  








