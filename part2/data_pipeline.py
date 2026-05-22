
import pandas as pd
import numpy as np

class NBADataPipeline:
    def __init__(self):
        self.mean_values = None
        self.scaler_params = {} 
        self.categorical_columns = ['Position', 'Team']
        # Loại bỏ các cột ID hoặc cột định danh không dùng làm đặc trưng hồi quy
        self.exclude_numeric = ['Unnamed: 0', 'PERSON_ID', 'TEAM_ID', 'Salary']
        self.numeric_features = []
        self.dummy_columns = []
        self.final_feature_columns = []

    def fit(self, X):
        """
        Bước 'Học': Tính toán các thông số từ tập dữ liệu huấn luyện (Train set).
        """
        # 1. Tự động xác định các cột số hợp lệ làm đặc trưng
        numeric_df = X.select_dtypes(include=[np.number])
        self.numeric_features = [c for c in numeric_df.columns if c not in self.exclude_numeric]
        
        # 2. Học giá trị trung bình để xử lý khuyết (Imputation)
        self.mean_values = X[self.numeric_features].mean()
        
        # 3. Học thông số chuẩn hóa (Standardization: (x - mu) / sigma)
        for col in self.numeric_features:
            self.scaler_params[col] = {
                'mean': X[col].mean(),
                'std': X[col].std()
            }
            
        # 4. Học cấu trúc các cột phân loại (Dummy columns) từ tập Train
        X_dummy = pd.get_dummies(X[self.categorical_columns], columns=self.categorical_columns, drop_first=True)
        self.dummy_columns = X_dummy.columns.tolist()
        
        # Danh sách các đặc trưng cuối cùng sẽ đưa vào mô hình
        self.final_feature_columns = self.numeric_features + self.dummy_columns
        print(f"Pipeline: Đã học xong thông số từ tập Train. Tổng số đặc trưng đầu vào: {len(self.final_feature_columns)}")

    def transform(self, X):
        """
        Bước 'Biến đổi': Áp dụng thông số đã học lên dữ liệu mới (Train/Test).
        """
        X_clean = X.copy()

        # 1. Feature Engineering: Biến đổi target y = ln(Salary) như kế hoạch nhóm
        if 'Salary' in X_clean.columns:
            X_clean['Log_Salary'] = np.log(X_clean['Salary'])
            
        # 2. Xử lý Missing Values (Đặc biệt là cột FT% thiếu 5.03%)
        for col, value in self.mean_values.items():
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna(value)

        # 3. Chuẩn hóa dữ liệu số (Quan trọng cho Ridge/Lasso)
        for col in self.numeric_features:
            if col in X_clean.columns:
                mu = self.scaler_params[col]['mean']
                sigma = self.scaler_params[col]['std']
                if sigma != 0:
                    X_clean[col] = (X_clean[col] - mu) / sigma

        # 4. Mã hóa biến phân loại & Căn chỉnh cột (Alignment) giữa Train và Test
        X_dummy = pd.get_dummies(X_clean[self.categorical_columns], columns=self.categorical_columns, drop_first=True)
        # Ép tập dữ liệu mới phải có đúng các cột dummy giống tập Train (thiếu thì điền 0)
        X_dummy = X_dummy.reindex(columns=self.dummy_columns, fill_value=0)
        
        # 5. Kết hợp các đặc trưng số và đặc trưng dummy, loại bỏ các cột chữ thừa
        X_out = pd.concat([X_clean[self.numeric_features], X_dummy], axis=1)
        
        # Giữ lại biến mục tiêu nếu có
        if 'Log_Salary' in X_clean.columns:
            X_out['Log_Salary'] = X_clean['Log_Salary']
        elif 'Salary' in X_clean.columns:
            X_out['Salary'] = X_clean['Salary']
            
        return X_out

    def process_pipeline(self, train_df, test_df):
        """Hàm tiện ích chạy trọn gói quy trình"""
        self.fit(train_df)
        train_processed = self.transform(train_df)
        test_processed = self.transform(test_df)
        return train_processed, test_processed


if __name__ == "__main__":
    # --- CHẠY THỬ NGHIỆM ĐỘC LẬP ---
    try:
        # Đọc file dữ liệu mới do Hoàng merge từ API
        df = pd.read_csv('merged_nba_data.csv')
        
        # Chia tập dữ liệu thành 80% Train và 20% Test
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)

        # Khởi tạo và thực thi quy trình làm sạch dữ liệu
        pipeline = NBADataPipeline()
        train_final, test_final = pipeline.process_pipeline(train_df, test_df)

        print("\n--- KẾT QUẢ KIỂM TRA PIPELINE ---")
        print("Kích thước tập Train sau xử lý:", train_final.shape)
        print("Kích thước tập Test sau xử lý:", test_final.shape)
        print("Số lượng cột Train và Test có khớp hoàn toàn không:", train_final.shape[1] == test_final.shape[1])
        print("Còn cột dạng chữ (Object) nào sót lại không:", train_final.dtypes.isin([object]).any())
        print("\nBiến mục tiêu Log_Salary của 5 dòng đầu:")
        print(train_final['Log_Salary'].head())
        
    except FileNotFoundError:
        print("Lưu ý: Hãy đảm bảo file 'merged_nba_data.csv' nằm cùng thư mục để chạy test!")