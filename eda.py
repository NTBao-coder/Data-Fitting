import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo
import time

# 1. Đọc file dữ liệu từ Kaggle trước
# Lưu ý: Cập nhật lại đường dẫn tới thư mục giải nén của bạn
kaggle_file_path = 'D:\\TUDTK\\TUDTK - Đồ án 2\\archive\\nba_salaries.csv'
df_kaggle = pd.read_csv(kaggle_file_path)

# 2. Lấy danh sách cầu thủ từ file Kaggle để kéo API
player_col = 'Player Name' if 'Player Name' in df_kaggle.columns else 'Player'
kaggle_player_names = df_kaggle[player_col].dropna().unique()

all_nba_players = players.get_players()
nba_players_dict = {p['full_name'].lower(): p['id'] for p in all_nba_players}

player_ids = []
for name in kaggle_player_names:
    if name.lower() in nba_players_dict:
        player_ids.append(nba_players_dict[name.lower()])

print(f"Tìm thấy ID của {len(player_ids)}/{len(kaggle_player_names)} cầu thủ trong danh sách.")

# 3. Kéo dữ liệu từ NBA API cho toàn bộ cầu thủ tìm được
player_data_list = []
print("Bắt đầu kéo dữ liệu từ API...")
for i, p_id in enumerate(player_ids):
    try:
        # Gọi endpoint lấy thông tin cầu thủ
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=p_id)
        df_info = player_info.get_data_frames()[0]
        player_data_list.append(df_info)
        time.sleep(0.6) # Sleep để tránh bị API chặn do rate limit
        if (i + 1) % 50 == 0:
            print(f" Đã kéo dữ liệu {i + 1}/{len(player_ids)} cầu thủ")
    except Exception as e:
        print(f"Lỗi khi kéo dữ liệu cầu thủ ID {p_id}: {e}")

# Gộp dữ liệu từ API thành 1 DataFrame duy nhất
df_api = pd.concat(player_data_list, ignore_index=True)
print("Kích thước dữ liệu API:", df_api.shape)

# 3. Merge dữ liệu
# Kiểm tra xem tên cột chứa tên cầu thủ ở 2 df là gì (thường là 'DISPLAY_FIRST_LAST' bên API và 'Player Name' bên Kaggle)
# Giả sử ta thống nhất đổi tên cột về 'Player' để merge
df_api.rename(columns={'DISPLAY_FIRST_LAST': 'Player'}, inplace=True)
if 'Player Name' in df_kaggle.columns:
    df_kaggle.rename(columns={'Player Name': 'Player'}, inplace=True)

# Thực hiện Inner Join để lấy các dòng khớp nhau ở cả 2 bảng
df_merged = pd.merge(df_kaggle, df_api, on='Player', how='inner')
print("Kích thước dữ liệu sau khi merge:", df_merged.shape)

# Xuất dữ liệu đã merge ra file CSV
merged_file_path = 'data/merged_nba_data.csv'
df_merged.to_csv(merged_file_path, index=False)
print(f"Đã xuất dữ liệu merge ra file: {merged_file_path}")

# 4. Tính toán tỉ lệ Missing Values theo từng cột
missing_percentages = (df_merged.isnull().sum() / len(df_merged)) * 100

# Lọc ra các cột có tỉ lệ missing >= 5%
cols_with_missing_over_5 = missing_percentages[missing_percentages >= 5].sort_values(ascending=False)

print("Các cột có missing value >= 5%:\n", cols_with_missing_over_5)

# 5. Trực quan hóa bằng Heatmap
# Vẽ heatmap để quan sát sự phân bố của các giá trị bị thiếu (missing values heatmap)
plt.figure(figsize=(12, 6))
sns.heatmap(df_merged.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Heatmap phân bố Missing Values trong bộ dữ liệu')
plt.show()

# (Tùy chọn) Vẽ Heatmap ma trận tương quan cho các biến số (Numeric variables)
# Đây cũng là một yêu cầu trong phần EDA của đồ án
numeric_df = df_merged.select_dtypes(include=[np.number])
plt.figure(figsize=(15, 10))
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Ma trận tương quan (Correlation Heatmap)')
plt.show()