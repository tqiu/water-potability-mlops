import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.metrics import accuracy_score, f1_score
import yaml
import mlflow
from mlflow.models import infer_signature
import dagshub


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")
# test_data = pd.read_csv("data/processed/test_processed_data.csv")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop("Potability", axis=1)
        y = data["Potability"]
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")
# X_test = test_data.drop("Potability", axis=1)
# y_test = test_data["Potability"]

def load_model(filepath: str):
    try:
        with open(filepath, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}: {e}")
# model = pickle.load(open("model.pkl", "rb"))

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        metrics_dict = {"accuracy": accuracy, "f1_score": f1}
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")    
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# f1_score = f1_score(y_test, y_pred)

def save_metrics(metrics_dict: dict, filepath: str) -> None:
    try:
        with open(filepath, "w") as f:
            json.dump(metrics_dict, f, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {filepath}: {e}")

def main():
    try:
        # dagshub.init(repo_owner='tqiu', repo_name='water-potability-mlops', mlflow=True)
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("Final experiment")

        test_data_filepath = "data/processed/test_processed_data.csv"
        model_filepath = "models/model.pkl"
        metrics_filepath = "reports/metrics.json"
        params_filepath = "params.yaml"

        test_data = load_data(test_data_filepath)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_filepath)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, metrics_filepath)

        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(model, "model", signature=infer_signature(X_test, y_test))
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(__file__)

            with open(params_filepath, "r") as file:
                params = yaml.safe_load(file)
            test_size = params["data_collection"]["test_size"]
            n_estimators = params["model_building"]["n_estimators"]
            max_depth = params["model_building"]["max_depth"]
            mlflow.log_params({"test_size": test_size, "n_estimators": n_estimators, "max_depth": max_depth})

            # save run_id and model info to a json file
            run_info = {"run_id": run.info.run_id, "model_name": "best-model"}
            run_info_filepath = "reports/run_info.json"
            with open(run_info_filepath, "w") as f:
                json.dump(run_info, f, indent=4)

    except Exception as e:
        raise Exception(f"Error in model evaluation: {e}")
    

if __name__ == "__main__":
    main()
