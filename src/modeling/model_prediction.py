import mlflow
import pandas as pd
import dagshub
import json

dagshub.init(repo_owner='tqiu', repo_name='water-potability-mlops', mlflow=True)


client = mlflow.tracking.MlflowClient()

# Fetch the latest versions of the model for all stages
model_name = "best-model"

try:
    latest_versions = client.get_latest_versions(name=model_name)
    if latest_versions:
        latest_version = latest_versions[0]
        version_number = latest_version.version
        run_id = latest_version.run_id
        print(f"Latest version: {version_number}, Run ID: {run_id}")

        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        # model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

        input_data = pd.DataFrame({
            "ph": [7.1],
            "Hardness": [200.0],
            "Solids": [200.0],
            "Chloramines": [10.0],
            "Sulfate": [300.0],
            "Conductivity": [400.0],
            "Organic_carbon": [10.0],
            "Trihalomethanes": [50.0],
            "Turbidity": [5.0]
        })

        prediction = model.predict(input_data)
        print(f"Prediction: {prediction}")
    else:
        print("No model versions found.")
except Exception as e:
    raise Exception(f"Error fetching model: {e}")



