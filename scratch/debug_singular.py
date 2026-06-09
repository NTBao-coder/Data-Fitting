import pandas as pd
import numpy as np
from part2.data_pipeline import NBADataPipeline

def main():
    df = pd.read_csv('part2/data/nba_salary_raw.csv')
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    pipeline = NBADataPipeline()
    X_train, y_train, X_test, y_test = pipeline.process_pipeline(train_df, test_df)
    
    print("X_train shape:", X_train.shape)
    
    # Check for zero variance columns
    variances = X_train.var()
    zero_var_cols = variances[variances == 0].index.tolist()
    print("Zero variance columns:", zero_var_cols)
    
    # Add bias
    X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train.values]
    
    # Check rank
    rank = np.linalg.matrix_rank(X_train_bias)
    print("Matrix rank:", rank, "Number of columns:", X_train_bias.shape[1])
    
    try:
        XTX = X_train_bias.T @ X_train_bias
        inv = np.linalg.inv(XTX)
        print("Invertible!")
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()
