import mlflow
import pandas as pd
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Water Potability Prediction",
    description="Predict if water is potable or not",
)

mlflow.set_tracking_uri("https://dagshub.com/tqiu/water-potability-mlops.mlflow")

model_name = "best-model"
model_version = 4
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

class Water(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.get("/")
def home():
    return {"message": "Welcome to the Water Potability Prediction API!"}

@app.post("/predict")
def predict(water: Water):
    sample = pd.DataFrame([water.model_dump()])
    prediction = model.predict(sample)

    if prediction[0] == 1:
        return {"prediction": "Water is potable"}
    else:
        return {"prediction": "Water is not potable"}



