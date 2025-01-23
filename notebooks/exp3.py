# %%
import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

dagshub.init(repo_owner='tqiu', repo_name='water-potability-mlops', mlflow=True)

mlflow.set_experiment("Experiment 3")


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

# compare multiple models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    "XGBoost": XGBClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}


signature = infer_signature(X_train, y_train)
with mlflow.start_run():
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name, nested=True):
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "model", signature=signature)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

            # log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.savefig(f"reports/figures/confusion_matrix_{model_name}.png")
            mlflow.log_artifact(f"reports/figures/confusion_matrix_{model_name}.png")

            mlflow.set_tags({"author": "Tuoling Qiu", "model": model_name})

    # log dataset
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)
    mlflow.log_input(train_df, "train_data")
    mlflow.log_input(test_df, "test_data")

    mlflow.log_artifact(__file__)
    

