# %%
import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

dagshub.init(repo_owner='tqiu', repo_name='water-potability-mlops', mlflow=True)

mlflow.set_experiment("Experiment 4")


# %%
data = pd.read_csv(r"C:\Users\qiutu\Documents\courses\Data_Thinkers\water_potability_dvc\water_potability.csv")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
# print(train_data.shape, test_data.shape)

# %%
def fill_missing_with_mean(df):
    for col in df.columns:
        df.fillna({col: df[col].mean()}, inplace=True)
    return df

train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

X_train = train_processed_data.drop("Potability", axis=1)
y_train = train_processed_data["Potability"]

X_test = test_processed_data.drop("Potability", axis=1)
y_test = test_processed_data["Potability"]

# model training
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

rf = RandomForestClassifier(random_state=42)
param_dist = {
    "n_estimators": [100, 200, 300, 500, 1000],
    "max_depth": [3, 5, 7, 10],
}
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=5, 
                                   random_state=42, n_jobs=-1)


with mlflow.start_run():
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    signature = infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(best_model, "model", signature=signature)
    mlflow.log_params(random_search.best_params_)


    for i, params in enumerate(random_search.cv_results_["params"]):
        with mlflow.start_run(run_name=f"Params-combo-{i}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", random_search.cv_results_["mean_test_score"][i])

    
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

    mlflow.set_tags({"author": "Tuoling Qiu", "model": "RF"})

    # log dataset
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)
    mlflow.log_input(train_df, "train_data")
    mlflow.log_input(test_df, "test_data")

    mlflow.log_artifact(__file__)
    

