
import pandas as pd
import numpy as np

class NBADataPipeline:
    def __init__(self):
        self.mean_values = None
        self.scaler_params = {} 
        self.categorical_columns = ['Position', 'Team']
        
        # 1. DANH SÁCH LOẠI BỎ BIẾN 
        self.exclude_columns = [
            'Unnamed: 0', 'Player', 'Player-additional', 'PLAYER_SLUG',
            'FIRST_NAME', 'LAST_NAME', 'DISPLAY_LAST_COMMA_FIRST', 'DISPLAY_FI_LAST',
            'BIRTHDATE', 'SCHOOL', 'COUNTRY', 'LAST_AFFILIATION', 'PLAYERCODE',
            'PERSON_ID', 'TEAM_ID', 'TEAM_NAME', 'TEAM_ABBREVIATION', 'TEAM_CODE', 'TEAM_CITY',
            'FROM_YEAR', 'TO_YEAR', 'JERSEY', 'POSITION', 'ROSTERSTATUS',
            'GAMES_PLAYED_CURRENT_SEASON_FLAG', 'DLEAGUE_FLAG', 'NBA_FLAG', 'GAMES_PLAYED_FLAG',
            'DRAFT_YEAR', 'DRAFT_ROUND', 'DRAFT_NUMBER', 'GREATEST_75_FLAG',
            'Salary', 'Log_Salary'
        ]
        
        self.numeric_features = []
        self.dummy_columns = []
        self.final_feature_columns = []
        self.is_fitted = False # Cờ kiểm tra an toàn hệ thống

    def fit(self, X):
        """
        Bước 'Học': Tính toán và lưu trữ các thông số thống kê từ tập huấn luyện (Train set).
        """
        X_clean = X.copy()
        
        # 2. Tự động lọc ra các cột số thực sự mang giá trị chuyên môn toán học
        numeric_df = X_clean.select_dtypes(include=[np.number])
        self.numeric_features = [c for c in numeric_df.columns if c not in self.exclude_columns]
        
        # Tính giá trị trung vị/trung bình để gán cho các ô dữ liệu bị khuyết (Imputation)
        self.mean_values = X_clean[self.numeric_features].mean()
        
        # 3. Học thông số để chuẩn hóa Z-score (Standardization)
        for col in self.numeric_features:
            col_std = X_clean[col].std()
            # Tránh lỗi chia cho 0 nếu cột đó là một hằng số (phương sai = 0)
            if pd.isna(col_std) or col_std == 0:
                col_std = 1.0
                
            self.scaler_params[col] = {
                'mean': X_clean[col].mean(),
                'std': col_std
            }
            
        # 4. Học cấu trúc các cột phân loại (One-Hot Encoding)
        # Chỉ lấy các cột phân loại được chỉ định, loại bỏ hoàn toàn các cột chữ định danh khác
        X_dummy = pd.get_dummies(X_clean[self.categorical_columns], columns=self.categorical_columns, drop_first=True)
        self.dummy_columns = X_dummy.columns.tolist()
        
        # Tập hợp danh sách thuộc tính chuẩn chỉnh cuối cùng
        self.final_feature_columns = self.numeric_features + self.dummy_columns
        self.is_fitted = True
        print(f"Pipeline: Học thông số hoàn tất! Tổng số đặc trưng thực tế đưa vào mô hình: {len(self.final_feature_columns)}")

    def transform(self, X):
        """
        Bước 'Biến đổi': Áp dụng các thông số đã học ở tập Train lên dữ liệu mới (Train/Test).
        """
        # KIỂM TRA AN TOÀN: Bắt lỗi nếu chưa gọi fit nhằm chống rò rỉ dữ liệu (Data Leakage)
        if not self.is_fitted:
            raise RuntimeError("LỖI HỆ THỐNG: Bạn không thể gọi hàm .transform() trước khi gọi hàm .fit() trên tập Train!")
            
        X_clean = X.copy()

        # 1. Thực hiện biến đổi mục tiêu như mô tả trong báo cáo: y = ln(Salary)
        if 'Salary' in X_clean.columns:
            X_clean['Log_Salary'] = np.log(X_clean['Salary'])
            
        # 2. Xử lý Missing Values một cách đồng bộ (Imputation)
        for col, value in self.mean_values.items():
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna(value)

        # 3. Thực thi chuẩn hóa dữ liệu số (Standardization)
        for col in self.numeric_features:
            if col in X_clean.columns:
                mu = self.scaler_params[col]['mean']
                sigma = self.scaler_params[col]['std']
                X_clean[col] = (X_clean[col] - mu) / sigma

        # 4. Mã hóa biến phân loại và đồng bộ hóa cấu trúc cột giữa Train và Test
        X_dummy = pd.get_dummies(X_clean[self.categorical_columns], columns=self.categorical_columns, drop_first=True)
        # Cực kỳ quan trọng: Reindex giúp tập Test luôn có đúng số lượng và thứ tự cột giống tập Train
        X_dummy = X_dummy.reindex(columns=self.dummy_columns, fill_value=0)
        
        # 5. Gom các mảng đặc trưng số và đặc trưng dummy lại thành ma trận hoàn chỉnh
        X_out = pd.concat([X_clean[self.numeric_features], X_dummy], axis=1)
        
        # Giữ lại biến mục tiêu để các file model thực hiện huấn luyện/đánh giá hiệu năng
        if 'Log_Salary' in X_clean.columns:
            X_out['Log_Salary'] = X_clean['Log_Salary']
        elif 'Salary' in X_clean.columns:
            X_out['Salary'] = X_clean['Salary']
            
        return X_out

    def process_pipeline(self, train_df, test_df):
        """Hàm tiện ích chạy trọn gói quy trình làm sạch dữ liệu nhanh chóng"""
        self.fit(train_df)
        train_processed = self.transform(train_df)
        test_processed = self.transform(test_df)
        return train_processed, test_processed


if __name__ == "__main__":
    # --- KHU VỰC CHẠY KIỂM THỬ ĐỘC LẬP ---
    try:
        # Thử đọc file dữ liệu NBA thực tế
        df = pd.read_csv('merged_nba_data.csv')
        
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)

        pipeline = NBADataPipeline()
        train_final, test_final = pipeline.process_pipeline(train_df, test_df)

        print("\n--- KIỂM TRA ---")
        print("Kích thước ma trận Train đầu ra:", train_final.shape)
        print("Kích thước ma trận Test đầu ra:", test_final.shape)
        print("Số lượng cột Train và Test đồng nhất hoàn toàn:", train_final.shape[1] == test_final.shape[1])
        
        # Kiểm tra xem còn sót lại cột dạng chuỗi (Object) nào gây crash mô hình không
        remaining_objects = train_final.select_dtypes(include=[object]).columns.tolist()
        print("Số lượng cột chữ (Object) còn sót lại:", len(remaining_objects))
        if remaining_objects:
            print("Cảnh báo các cột sót:", remaining_objects)
        else:
            print(" Dữ liệu hoàn toàn sạch sẽ, không còn biến nhiễu hay chữ định danh rác!")
            
    except FileNotFoundError:
        print("Thông báo: Hãy để file 'merged_nba_data.csv' chung thư mục nếu muốn chạy thử nghiệm độc lập.")