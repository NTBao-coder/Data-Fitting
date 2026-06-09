import pandas as pd
import math
import sys
import os

# Đảm bảo có thể import các module thuần Python từ part1
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from part1 import helper_function as hf
from part1.ridge_lasso import vif

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
        Điền dữ liệu khuyết bằng K-Nearest Neighbors thuần Python.
        """
        target = target_df.copy()
        
        # Chuyển DataFrame sang list thuần để xử lý tốc độ cao
        source_vals = source_df.values.tolist()
        target_vals = target.values.tolist()
        
        for i in range(len(target_vals)):
            row = target_vals[i]
            
            # Tìm danh sách các cột bị khuyết trong dòng này
            missing_cols = [col_idx for col_idx, val in enumerate(row) if pd.isna(val)]
            if not missing_cols:
                continue
                
            # Tính khoảng cách Euclidean đến tất cả các dòng trong source
            distances = []
            for s_idx, s_row in enumerate(source_vals):
                diff_sq_sum = 0.0
                valid_count = 0
                for j in range(len(row)):
                    if not pd.isna(row[j]) and not pd.isna(s_row[j]):
                        diff_sq_sum += (s_row[j] - row[j]) ** 2
                        valid_count += 1
                
                if valid_count > 0:
                    dist = math.sqrt(diff_sq_sum / valid_count)
                else:
                    dist = float('inf')
                distances.append(dist)
                
            # Điền khuyết từng cột
            for col_idx in missing_cols:
                # Lọc các láng giềng có giá trị hợp lệ ở cột này
                valid_neighbors = [(distances[s_idx], source_vals[s_idx][col_idx]) 
                                   for s_idx in range(len(source_vals)) 
                                   if not pd.isna(source_vals[s_idx][col_idx])]
                
                # Sắp xếp theo khoảng cách
                valid_neighbors.sort(key=lambda x: x[0])
                
                # Lấy k láng giềng gần nhất (khoảng cách khác vô cực)
                top_k = [val for d, val in valid_neighbors[:k] if d != float('inf')]
                
                if top_k:
                    target.iloc[i, col_idx] = sum(top_k) / len(top_k)
                else:
                    # Fallback (dùng trung bình toàn cột)
                    col_values = [s_row[col_idx] for s_row in source_vals if not pd.isna(s_row[col_idx])]
                    if col_values:
                        target.iloc[i, col_idx] = sum(col_values) / len(col_values)
                    else:
                        target.iloc[i, col_idx] = 0.0
                        
        return target

    def _base_feature_engineering(self, df):
        """Bước 0: Tạo biến mới độc lập"""
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
        
        # Lọc ra danh sách biến số học thực sự (thay thế np.number bằng string 'number' của Pandas)
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
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

        # 4. VIF (Lọc tuần tự): Chỉ lọc trên các biến số
        print("Đang chạy VIF tuần tự để lọc đa cộng tuyến...")
        X_vif_input = hf.Matrix([hf.Vector(row) for row in X_temp[self.numeric_features].values.tolist()])
        current_features = list(self.numeric_features)
        
        while True:
            vif_vals = vif(X_vif_input)
            
            max_idx = 0
            max_vif = vif_vals[0]
            for idx, val in enumerate(vif_vals):
                if val > max_vif:
                    max_vif = val
                    max_idx = idx
            
            if max_vif > self.vif_threshold and max_vif != hf.INF:
                dropped = current_features.pop(max_idx)
                X_vif_input = hf.delete(X_vif_input, max_idx, axis=1)
                print(f"  -> Loại bỏ '{dropped}' (VIF = {max_vif:.2f})")
            elif max_vif == hf.INF:
                dropped = current_features.pop(max_idx)
                X_vif_input = hf.delete(X_vif_input, max_idx, axis=1)
                print(f"  -> Loại bỏ '{dropped}' (VIF = INF)")
            else:
                break
                
        self.vif_passed_features = current_features

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
        X_final = pd.concat([X_out[self.vif_passed_features], X_dummy], axis=1)
        
        # Xử lý biến mục tiêu (Nếu có) bằng math.log thay cho np.log
        if 'Salary' in X.columns:
            X_final['Log_Salary'] = X['Salary'].apply(math.log)
            
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