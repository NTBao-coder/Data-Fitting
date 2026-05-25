import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

def build_raw_dataset():
    print("1. Đang kéo dữ liệu từ nba_api (Mùa giải 2022-23)...")
    # Lấy toàn bộ thống kê của giải đấu trong 1 lần gọi API
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season='2022-23').get_data_frames()[0]
    
    # Giữ lại các cột quan trọng. Cột FG3_PCT (Tỉ lệ ném 3đ) và FT_PCT (Ném phạt) 
    # chắc chắn sẽ có Missing Values tự nhiên (NaN) cho các cầu thủ không thực hiện cú ném nào.
    api_cols = ['PLAYER_NAME', 'AGE', 'GP', 'MIN', 'PTS', 'AST', 'REB', 'FG3_PCT', 'FT_PCT', 'TOV']
    df_api = stats[api_cols]

    print("2. Đang đọc file Kaggle Salary...")
    try:
        # Tên file này có thể thay đổi tùy vào file bạn tải từ Kaggle
        df_salary = pd.read_csv("kaggle_salary_data.csv") 
        
        # Chuẩn hóa tên cột để trùng khớp khóa chính (Primary Key)
        # Sửa lại 'Player Name' và 'Salary' cho khớp với file Kaggle thực tế của bạn
        df_salary = df_salary.rename(columns={'Player Name': 'PLAYER_NAME', 'Salary': 'SALARY'})
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file Kaggle. Vui lòng kiểm tra lại đường dẫn.")
        return

    print("3. Đang hợp nhất dữ liệu (Merge)...")
    # Sử dụng Inner Join để đảm bảo chỉ giữ lại những cầu thủ có cả dữ liệu chỉ số và lương
    df_final = pd.merge(df_api, df_salary[['PLAYER_NAME', 'SALARY']], on='PLAYER_NAME', how='inner')

    print("4. Xuất file cho DataPipeline...")
    output_path = "nba_salary_raw.csv"
    df_final.to_csv(output_path, index=False)
    print(f"✅ Hoàn tất! File đã được lưu tại: {output_path}")
    print(f"📊 Kích thước dữ liệu: {df_final.shape[0]} dòng, {df_final.shape[1]} cột.")

if __name__ == "__main__":
    build_raw_dataset()