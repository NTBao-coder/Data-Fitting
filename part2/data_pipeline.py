import pandas as pd
import numpy as np

class NBADataPipeline:
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
        self.train_knn_source = None

    def _knn_impute(self, target_df, source_df, k=5):
        """
        Điền khuyết (Imputation) bằng thuật toán K-Nearest Neighbors tự code bằng NumPy.
        """
        target = target_df.copy()
        source = source_df.values
        
        for i in range(len(target)):
            row = target.iloc[i].values
            missing_mask = np.isnan(row)
            
            if not missing_mask.any():
                continue
                
            # Tính bình phương khoảng cách
            diff = source - row
            diff_sq = np.nan_to_num(diff**2)
            
            # Số lượng thuộc tính hợp lệ (cả 2 đều không NaN)
            valid_cols_count = np.sum(~np.isnan(source) & ~np.isnan(row), axis=1)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                dist = np.sqrt(np.sum(diff_sq, axis=1) / valid_cols_count)
            
            dist[valid_cols_count == 0] = np.inf
            
            for col_idx in np.where(missing_mask)[0]:
                col_source = source[:, col_idx]
                valid_source_mask = ~np.isnan(col_source)
                
                valid_dists = dist.copy()
                valid_dists[~valid_source_mask] = np.inf
                
                nearest_idx = np.argsort(valid_dists)[:k]
                
                if np.isinf(valid_dists[nearest_idx[0]]):
                    target.iloc[i, col_idx] = np.nanmean(col_source)
                else:
                    k_nearest_vals = col_source[nearest_idx]
                    k_nearest_vals = k_nearest_vals[~np.isinf(valid_dists[nearest_idx])]
                    if len(k_nearest_vals) > 0:
                        target.iloc[i, col_idx] = np.mean(k_nearest_vals)
                    else:
                        target.iloc[i, col_idx] = np.nanmean(col_source)
                        
        return target

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

        # 1. Encode: Học cấu trúc biến Dummy
        if set(self.categorical_features).issubset(X_temp.columns):
            X_dummy = pd.get_dummies(X_temp[self.categorical_features], drop_first=True, dtype=int)
            self.encoded_columns = X_dummy.columns.tolist()
        else:
            X_dummy = pd.DataFrame(index=X_temp.index)
            self.encoded_columns = []

        # 2. Normalize (Học): Tính mu, sigma và chuẩn hóa Z-score
        valid_features = []
        for col in self.numeric_features:
            mu = X_temp[col].mean()
            sigma = X_temp[col].std()
            if pd.isna(sigma) or sigma == 0:
                continue # Loại bỏ biến hằng số
            
            self.scaler_params[col] = {'mu': mu, 'sigma': sigma}
            X_temp[col] = (X_temp[col] - mu) / sigma # Chuẩn hóa nội bộ
            valid_features.append(col)
            
        self.numeric_features = valid_features

        # 3. KNN Imputer (Học & Điền khuyết Train)
        # Điền khuyết trên tập đã chuẩn hóa, tập Train mượn chính nó làm source
        self.train_knn_source = X_temp[self.numeric_features].copy()
        X_temp[self.numeric_features] = self._knn_impute(X_temp[self.numeric_features], self.train_knn_source, k=5)
        # Cập nhật lại train_knn_source thành dữ liệu đã SẠCH BONG NaN
        self.train_knn_source = X_temp[self.numeric_features].copy()
        
        # Ghi nhận Mean phòng hờ nếu KNN gặp lỗi
        for col in self.numeric_features:
            self.imputation_values[col] = X_temp[col].mean()

        # 4. VIF (Lọc): Chỉ lọc trên các biến số (đã sạch NaN)
        corr_matrix = X_temp[self.numeric_features].corr().values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
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

        # 1. Encode: Tạo Dummy, reindex khớp với Train
        if set(self.categorical_features).issubset(X.columns):
            X_dummy = pd.get_dummies(X[self.categorical_features], drop_first=True, dtype=int)
            X_dummy = X_dummy.reindex(columns=self.encoded_columns, fill_value=0)
        else:
            X_dummy = pd.DataFrame(0, index=X.index, columns=self.encoded_columns)

        # 2. Normalize (Áp dụng): Lấy mu, sigma của Train áp lên tập mới
        for col in self.numeric_features:
            mu = self.scaler_params[col]['mu']
            sigma = self.scaler_params[col]['sigma']
            X_out[col] = (X[col] - mu) / sigma

        # 3. KNN Imputer (Điền khuyết tập mới dựa trên Train)
        X_out[self.numeric_features] = self._knn_impute(X_out[self.numeric_features], self.train_knn_source, k=5)
        
        # Fallback lấp đầy bằng Mean nếu KNN vẫn chừa lại NaN
        for col in self.numeric_features:
            if X_out[col].isnull().any():
                X_out[col] = X_out[col].fillna(self.imputation_values[col])

        # 4. Lọc cột số và lắp ráp ma trận cuối cùng
        # Chỉ lấy các biến số học đã pass VIF + toàn bộ biến Dummy
        X_final = pd.concat([X_out[self.vif_passed_features], X_dummy], axis=1)
        
        # Xử lý biến mục tiêu (Nếu có)
        if 'Salary' in X.columns:
            X_final['Log_Salary'] = np.log(X['Salary'])
            
        return X_final

    def process_pipeline(self, train_df, test_df):
        """Hàm gộp tiện ích để chạy cả fit và transform cho Train/Test giống cấu trúc cũ"""
        self.fit(train_df)
        
        X_train_processed = self.transform(train_df)
        X_test_processed = self.transform(test_df)
        
        if 'Log_Salary' in X_train_processed.columns:
            y_train = X_train_processed.pop('Log_Salary')
        else:
            y_train = train_df['Salary']
            
        if 'Log_Salary' in X_test_processed.columns:
            y_test = X_test_processed.pop('Log_Salary')
        else:
            y_test = test_df['Salary']
            
        return X_train_processed, y_train, X_test_processed, y_test