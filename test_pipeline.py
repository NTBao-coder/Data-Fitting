import pandas as pd
import sys
from part2.data_pipeline import NBADataPipeline

sys.stdout.reconfigure(encoding='utf-8')

try:
    df = pd.read_csv('part2/data/nba_salary_raw.csv')
except:
    df = pd.read_csv('part2/data/nba_salaries.csv')

train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

pipeline = NBADataPipeline()
X_train, y_train, X_test, y_test = pipeline.process_pipeline(train_df, test_df)

print(f"X_train shape: {X_train.shape}")
print("Columns:", X_train.columns.tolist())
