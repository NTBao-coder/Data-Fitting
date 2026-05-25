import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class NBADataPipeline:
    def __init__(self):
        self.scaler_params = {} 
        self.categorical_columns = ['Position', 'Team']
        # Loại bỏ các cột không dùng làm đặc trưng hồi quy (Y và ID)
        self.exclude_numeric = ['Unnamed: 0', 'PERSON_ID', 'TEAM_ID', 'Salary', 'Log_Salary']
        self.numeric_features = []
        self.dummy_columns = []
        self.final_feature_columns = []
        
        # Sử dụng K-NN Imputation (MV4) thay vì Mean Imputation để lấy điểm tối đa
        self.imputer = KNNImputer(n_neighbors=5, weights='distance')

    def fit(self, X):
        """
        Bước 'Học': Tính toán các thông số từ tập dữ liệu huấn luyện (Train set).
        """
        # 1. Tự động xác định các cột số hợp lệ làm đặc trưng X
        numeric_df = X.select_dtypes(include=[np.number])
        self.numeric_features = [c for c in numeric_df.columns if c not in self.exclude_numeric]
        
        # 2. Học thông số chuẩn hóa Z-score: (x - mu) / sigma
        # Lưu ý: Tính toán bỏ qua giá trị NaN tự động
        for col in self.numeric_features:
            self.scaler_params[col] = {
                'mean': X[col].mean(),
                'std': X[col].std()
            }
            
        # 3. Scale tạm thời tập Train để fit thuật toán KNN (KNN nhạy cảm với scale)
        X_temp = X[self.numeric_features].copy()
        for col in self.numeric_features:
            std = self.scaler_params[col]['std']
            if std != 0:
                X_temp[col] = (X_temp[col] - self.scaler_params[col]['mean']) / std
                
        # Học mô hình điền khuyết dựa trên khoảng cách giữa các cầu thủ
        self.imputer.fit(X_temp)
            
        # 4. Học cấu trúc các cột phân loại (Dummy columns) từ tập Train
        X_dummy = pd.get_dummies(X[self.categorical_columns], columns=self.categorical_columns, drop_first=True)
        self.dummy_columns = X_dummy.columns.tolist()
        
        # Danh sách các đặc trưng cuối cùng
        self.final_feature_columns = self.numeric_features + self.dummy_columns
        print(f"Pipeline: Đã học xong thông số từ tập Train. Tổng số đặc trưng đầu vào: {len(self.final_feature_columns)}")

    def transform(self, X):
        """
        Bước 'Biến đổi': Áp dụng thông số đã học lên dữ liệu mới (Train/Test).
        """
        X_clean = X.copy()

        # 1. Feature Engineering: Biến đổi target y = ln(Salary)
        if 'Salary' in X_clean.columns:
            X_clean['Log_Salary'] = np.log(X_clean['Salary'])
            
        # 2. Chuẩn hóa dữ liệu số TRƯỚC KHI điền khuyết
        for col in self.numeric_features:
            if col in X_clean.columns:
                mu = self.scaler_params[col]['mean']
                sigma = self.scaler_params[col]['std']
                if sigma != 0:
                    X_clean[col] = (X_clean[col] - mu) / sigma

        # 3. Điền khuyết bằng K-NN Imputer (Thay thế giá trị NaN bằng các láng giềng gần nhất)
        X_clean[self.numeric_features] = self.imputer.transform(X_clean[self.numeric_features])

        # 4. Mã hóa biến phân loại & Căn chỉnh cột (Alignment)
        if set(self.categorical_columns).issubset(X_clean.columns):
            X_dummy = pd.get_dummies(X_clean[self.categorical_columns], columns=self.categorical_columns, drop_first=True)
            # Ép tập Test phải có đúng các cột dummy giống tập Train (thiếu thì điền 0)
            X_dummy = X_dummy.reindex(columns=self.dummy_columns, fill_value=0)
        else:
            # Nếu tập đưa vào không có cột categorical (ví dụ test case nhỏ), tạo ma trận 0
            X_dummy = pd.DataFrame(0, index=X_clean.index, columns=self.dummy_columns)
        
        # 5. Kết hợp các đặc trưng
        X_out = pd.concat([X_clean[self.numeric_features], X_dummy], axis=1)
        
        # 6. Tách biến mục tiêu y (nếu có) để phục vụ cho mô hình hồi quy
        y_out = None
        if 'Log_Salary' in X_clean.columns:
            y_out = X_clean['Log_Salary']
            
        return X_out, y_out

    def process_pipeline(self, train_df, test_df):
        """Hàm tiện ích chạy trọn gói quy trình"""
        self.fit(train_df)
        X_train_processed, y_train = self.transform(train_df)
        X_test_processed, y_test = self.transform(test_df)
        return X_train_processed, y_train, X_test_processed, y_test


if __name__ == "__main__":
    # --- CHẠY THỬ NGHIỆM ĐỘC LẬP ---
    try:
        # Cập nhật đường dẫn file chuẩn xác theo cấu trúc thư mục
        df = pd.read_csv('data/nba_salary_raw.csv')
        
        # Chia tập dữ liệu thành 80% Train và 20% Test
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)

        # Khởi tạo và thực thi quy trình
        pipeline = NBADataPipeline()
        X_train, y_train, X_test, y_test = pipeline.process_pipeline(train_df, test_df)

        print("\n--- KẾT QUẢ KIỂM TRA PIPELINE ---")
        print(f"Kích thước X_Train sau xử lý: {X_train.shape} | Y_Train: {y_train.shape}")
        print(f"Kích thước X_Test sau xử lý:  {X_test.shape} | Y_Test: {y_test.shape}")
        print("Test 1: Số lượng cột Train và Test khớp hoàn toàn? ->", X_train.shape[1] == X_test.shape[1])
        print("Test 2: Còn NaN trong X_Train không? ->", X_train.isna().sum().sum() == 0)
        
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file. Hãy chạy script này từ thư mục 'part2' hoặc kiểm tra lại file 'data/nba_salary_raw.csv'.")