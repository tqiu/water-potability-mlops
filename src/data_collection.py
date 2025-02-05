import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml


def load_params(filepath: str) -> float:
    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}: {e}")
# test_size = yaml.safe_load(open("params.yaml"))["data_collection"]["test_size"]

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")
# data = pd.read_csv("water_potability.csv")

def split_data(data: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return train_test_split(data, test_size=test_size, random_state=42)
    except ValueError as e:
        raise ValueError(f"Error splitting data: {e}")
# train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

# data_path = os.path.join("data", "raw")
# os.makedirs(data_path, exist_ok=True)

def save_data(data: pd.DataFrame, filepath: str) -> None:
    try:
        data.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")
# train_data.to_csv(os.path.join(data_path, "train_data.csv"), index=False)
# test_data.to_csv(os.path.join(data_path, "test_data.csv"), index=False)

def main():
    params_filepath = "params.yaml"
    raw_data_path = os.path.join("data", "raw")
    data_filepath = r"C:\Users\qiutu\Documents\courses\Data_Thinkers\water_potability_dvc\water_potability.csv"

    try:
        data = load_data(data_filepath)
        test_size = load_params(params_filepath)
        train_data, test_data = split_data(data, test_size)
        os.makedirs(raw_data_path, exist_ok=True)
        save_data(train_data, os.path.join(raw_data_path, "train_data.csv"))
        save_data(test_data, os.path.join(raw_data_path, "test_data.csv"))
    except Exception as e:
        raise Exception(f"Error in data collection: {e}")


if __name__ == "__main__":
    main()