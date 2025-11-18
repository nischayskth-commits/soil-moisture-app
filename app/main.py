from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Soil Moisture Predictor")

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model (path relative to project root)
MODEL_PATH = os.path.join("models", "soil_moisture_rf.pkl")
model = joblib.load(MODEL_PATH)

class InputData(BaseModel):
    temperature: float = Field(..., example=25.0)
    humidity: float = Field(..., example=60.0)
    rainfall: float = Field(..., example=4.0)
    soil_type: int = Field(..., example=1, ge=0, le=2)

class Prediction(BaseModel):
    moisture: float
    category: str

def moisture_category(val: float) -> str:
    if val < 15:
        return "Low (Dry)"
    if val < 30:
        return "Medium (Optimal)"
    return "High (Wet)"

@app.post("/predict", response_model=Prediction)
def predict(input: InputData):
    # build a DataFrame with column names so sklearn is happy
    df = pd.DataFrame([{
        "temperature": input.temperature,
        "humidity": input.humidity,
        "rainfall": input.rainfall,
        "soil_type": input.soil_type
    }])
    pred = float(round(model.predict(df)[0], 2))
    return {"moisture": pred, "category": moisture_category(pred)}
@app.get("/")
def root():
    return {"message": "Soil Moisture Predictor API. Use POST /predict"}
