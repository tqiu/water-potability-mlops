import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

def load_params(filepath: str) -> int:
    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"], params["model_building"]["max_depth"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}: {e}")
# n_estimators = yaml.safe_load(open("params.yaml"))["model_building"]["n_estimators"]

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")
# train_data = pd.read_csv("data/processed/train_processed_data.csv")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop("Potability", axis=1)
        y = data["Potability"]
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")
# X_train = train_data.drop("Potability", axis=1)
# y_train = train_data["Potability"]

def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        clf.fit(X, y)
        return clf
    except Exception as e:
        raise Exception(f"Error training model: {e}")
# clf = RandomForestClassifier(n_estimators=n_estimators)
# clf.fit(X_train, y_train)

def save_model(model: RandomForestClassifier, filepath: str) -> None:
    try:
        with open(filepath, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error saving model to {filepath}: {e}")
# pickle.dump(clf, open("model.pkl", "wb"))

def main():
    processed_data_filepath = "data/processed/train_processed_data.csv"
    params_filepath = "params.yaml"
    model_filepath = "models/model.pkl"

    try:
        n_estimators, max_depth = load_params(params_filepath)
        train_data = load_data(processed_data_filepath)
        X_train, y_train = prepare_data(train_data)
        clf = train_model(X_train, y_train, n_estimators, max_depth)
        save_model(clf, model_filepath)
    except Exception as e:
        raise Exception(f"Error in model building: {e}")
    

if __name__ == "__main__":
    main()