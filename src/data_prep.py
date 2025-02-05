import pandas as pd
import numpy as np
import os

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")
# train_data = pd.read_csv("data/raw/train_data.csv")
# test_data = pd.read_csv("data/raw/test_data.csv")

def fill_missing_with_mean(df):
    try:
        for col in df.columns:
            df.fillna({col: df[col].mean()}, inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Error filling missing values with mean: {e}")

# train_processed_data = fill_missing_with_median(train_data)
# test_processed_data = fill_missing_with_median(test_data)

# data_path = os.path.join("data", "processed")
# os.makedirs(data_path, exist_ok=True)

def save_data(data: pd.DataFrame, filepath: str) -> None:
    try:
        data.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")
# train_processed_data.to_csv(os.path.join(data_path, "train_processed_data.csv"), index=False)
# test_processed_data.to_csv(os.path.join(data_path, "test_processed_data.csv"), index=False)

def main():
    raw_data_path = os.path.join("data", "raw")
    processed_data_path = os.path.join("data", "processed")
    try:
        train_data = load_data(os.path.join(raw_data_path, "train_data.csv"))
        test_data = load_data(os.path.join(raw_data_path, "test_data.csv"))
        train_processed_data = fill_missing_with_mean(train_data)
        test_processed_data = fill_missing_with_mean(test_data)
        os.makedirs(processed_data_path, exist_ok=True)
        save_data(train_processed_data, os.path.join(processed_data_path, "train_processed_data.csv"))
        save_data(test_processed_data, os.path.join(processed_data_path, "test_processed_data.csv"))
    except Exception as e:
        raise Exception(f"Error in data pre-processing: {e}")


if __name__ == "__main__":
    main()