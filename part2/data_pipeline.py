import pandas as pd
import numpy as np

class RigorousNBAPipeline:
    def __init__(self, vif_threshold=10.0):
        self.vif_threshold = vif_threshold
        # Từ điển lưu trữ trạng thái (State) học được từ tập Train
        self.imputation_values = {}  # Lưu giá trị thay thế Missing Values
        self.encoded_columns = []    # Lưu cấu trúc cột One-hot Encoding
        self.scaler_params = {}      # Lưu mean (mu) và std (sigma) để chuẩn hóa
        self.vif_passed_features = [] # Danh sách các cột vượt qua bài test Đa cộng tuyến
        
        self.categorical_features = ['Position', 'Team'] # Các cột cần encode
        self.exclude_cols = ['Player', 'Salary', 'Log_Salary', 'Unnamed: 0'] # ID và Target
        
        self.is_fitted = False

    def _base_feature_engineering(self, df):
        """Bước 0: Tạo biến mới độc lập (Không gây Data Leakage vì tính trên từng dòng)"""
        X = df.copy()
        
        # 1. Advanced Metrics (TS%)
        if set(['PTS', 'FGA', 'FTA']).issubset(X.columns):
            X['TS_pct'] = X['PTS'] / (2 * (X['FGA'] + 0.44 * X['FTA']) + 1e-9)
        
        # 2. Domain Knowledge (Rookie / Superstar Clusters)
        if 'Age' in X.columns:
            X['is_rookie'] = (X['Age'] <= 23).astype(int)
        if set(['PTS', 'MP']).issubset(X.columns):
            X['is_superstar'] = (((X['PTS'] / (X['MP'] + 1e-9)) * 36 > 20) & (X['MP'] > 2000)).astype(int)

        # 3. Chuẩn hóa Per 36 Minutes và HỦY BỎ cột thô
        count_stats = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV']
        if 'MP' in X.columns:
            for col in count_stats:
                if col in X.columns:
                    X[f'{col}_per36'] = (X[col] / (X['MP'] + 1e-9)) * 36
            X.drop(columns=[c for c in count_stats if c in X.columns], inplace=True)
            
        return X

    def fit(self, X_train):
        """BƯỚC HỌC: Chỉ tính toán tham số từ X_train và cất vào State"""
        X = self._base_feature_engineering(X_train)
        
        # Lọc ra danh sách biến số học thực sự
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_features = [c for c in numeric_cols if c not in self.exclude_cols]

        X_temp = X.copy()

        # THỨ TỰ 1: Học Missing Values (Tính Mean của tập Train)
        for col in self.numeric_features:
            mean_val = X_temp[col].mean()
            self.imputation_values[col] = mean_val
            X_temp[col] = X_temp[col].fillna(mean_val) # Điền nội bộ để tính bước sau

        # THỨ TỰ 2: Học cấu trúc Encoding (Ghi nhớ các Category có trong Train)
        if set(self.categorical_features).issubset(X_temp.columns):
            X_dummy = pd.get_dummies(X_temp[self.categorical_features], drop_first=True)
            self.encoded_columns = X_dummy.columns.tolist()

        # THỨ TỰ 3: Học Standardization (Tính mu và sigma của tập Train)
        for col in self.numeric_features:
            mu = X_temp[col].mean()
            sigma = X_temp[col].std()
            sigma = 1.0 if pd.isna(sigma) or sigma == 0 else sigma
            
            self.scaler_params[col] = {'mu': mu, 'sigma': sigma}
            X_temp[col] = (X_temp[col] - mu) / sigma # Chuẩn hóa nội bộ

        # THỨ TỰ 4: Học VIF (Lọc đa cộng tuyến trên dữ liệu Train ĐÃ CHUẨN HÓA)
        corr_matrix = X_temp[self.numeric_features].corr().values
        inv_corr = np.linalg.pinv(corr_matrix) # Dùng Pseudo-inverse để an toàn
        vifs = np.diag(inv_corr)
        
        self.vif_passed_features = [
            col for col, vif in zip(self.numeric_features, vifs) if vif <= self.vif_threshold
        ]

        self.is_fitted = True

    def transform(self, X_data):
        """BƯỚC BIẾN ĐỔI: Áp đặt tham số đã học lên bất kỳ tập dữ liệu nào (Train/Test)"""
        if not self.is_fitted:
            raise RuntimeError("Lỗi: Phải gọi hàm fit() trước khi transform()")
            
        X = self._base_feature_engineering(X_data)
        X_out = pd.DataFrame(index=X.index)

        # THỨ TỰ 1: Xử lý Missing Values (Áp dụng Mean của Train)
        for col in self.numeric_features:
            if col in X.columns:
                X[col] = X[col].fillna(self.imputation_values[col])

        # THỨ TỰ 2: Encoding (Đồng bộ cấu trúc cột với Train)
        if set(self.categorical_features).issubset(X.columns):
            X_dummy = pd.get_dummies(X[self.categorical_features], drop_first=True)
            # Hàm reindex cực kỳ quan trọng: 
            # - Nếu Test thiếu category -> tạo cột toàn 0
            # - Nếu Test dư category lạ -> loại bỏ cột đó
            X_dummy = X_dummy.reindex(columns=self.encoded_columns, fill_value=0)
        else:
            # Fallback nếu df không có cột category
            X_dummy = pd.DataFrame(0, index=X.index, columns=self.encoded_columns)

        # THỨ TỰ 3: Standardization (Trừ mu_train và chia sigma_train)
        # Tối ưu: Chỉ cần chuẩn hóa những cột số đã vượt qua bài test VIF
        for col in self.vif_passed_features:
            mu = self.scaler_params[col]['mu']
            sigma = self.scaler_params[col]['sigma']
            X_out[col] = (X[col] - mu) / sigma

        # THỨ TỰ 4: Lắp ráp ma trận cuối cùng (Đã trừ khử VIF + Đã Encode)
        X_final = pd.concat([X_out[self.vif_passed_features], X_dummy], axis=1)
        
        # Xử lý biến mục tiêu (Nếu có)
        if 'Salary' in X.columns:
            X_final['Log_Salary'] = np.log(X['Salary'])
            
        return X_final