# %%
import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

dagshub.init(repo_owner='tqiu', repo_name='water-potability-mlops', mlflow=True)

mlflow.set_experiment("Experiment 1")


# %%
data = pd.read_csv(r"C:\Users\qiutu\Documents\courses\Data_Thinkers\water_potability_dvc\water_potability.csv")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
# print(train_data.shape, test_data.shape)

# %%
def fill_missing_with_median(df):
    for col in df.columns:
        df.fillna({col: df[col].median()}, inplace=True)
    return df

train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# %%
from sklearn.ensemble import RandomForestClassifier
import pickle
X_train = train_processed_data.drop("Potability", axis=1)
y_train = train_processed_data["Potability"]

n_estimators = 100
with mlflow.start_run():
    clf = RandomForestClassifier(random_state=42, n_estimators=n_estimators)
    clf.fit(X_train, y_train)
          
    mlflow.log_param("n_estimators", n_estimators)

    # log model
    signature = infer_signature(X_train, clf.predict(X_train))
    mlflow.sklearn.log_model(clf, "model", signature=signature)
    
    # %%
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    X_test = test_processed_data.drop("Potability", axis=1)
    y_test = test_processed_data["Potability"]

    y_pred = clf.predict(X_test) 

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
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # log dataset
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)
    mlflow.log_input(train_df, "train_data")
    mlflow.log_input(test_df, "test_data")

    mlflow.log_artifact(__file__)
    
    mlflow.set_tags({"author": "Tuoling Qiu", "model": "RF"})

