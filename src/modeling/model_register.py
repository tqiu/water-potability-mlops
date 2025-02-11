import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import json
import os

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("Final experiment")


# load run_id and model name from a json file
with open("reports/run_info.json", "r") as f:
    run_info = json.load(f)
run_id = run_info["run_id"]
model_name = run_info["model_name"]

# Register the model and create a new version
model_uri = f"runs:/{run_id}/model"

client = MlflowClient()
# client.create_registered_model(model_name)
# model_version = client.create_model_version(
#     name=model_name,
#     source=model_uri,
#     run_id=run_id
# )
reg = mlflow.register_model(model_uri, model_name)

# Transition to "Staging"
client.transition_model_version_stage(
    name=model_name,
    version=reg.version,
    stage="Staging"
)

