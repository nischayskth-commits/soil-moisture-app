# app/main.py
from fastapi import FastAPI, File, UploadFile, Query
from pydantic import BaseModel, Field
import joblib, os, csv, time, requests
import numpy as np, pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO

app = FastAPI(title="Soil Moisture Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
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
    explanation: str = ""
    recommendation: str = ""

def moisture_category(val: float) -> str:
    if val < 15:
        return "Low (Dry)"
    if val < 30:
        return "Medium (Optimal)"
    return "High (Wet)"

@app.post("/predict", response_model=Prediction)
def predict(input: InputData):
    # prepare df with same columns used during training
    df = pd.DataFrame([{
        "temperature": input.temperature,
        "humidity": input.humidity,
        "rainfall": input.rainfall,
        "soil_type": input.soil_type
    }])
    pred_val = float(round(model.predict(df)[0], 2))
    category = moisture_category(pred_val)

    # explanation rules
    reasons = []
    if input.humidity >= 75:
        reasons.append("High humidity increases moisture.")
    if input.rainfall >= 5:
        reasons.append("Recent rainfall increases moisture.")
    if input.temperature >= 35:
        reasons.append("High temperature causes moisture loss.")
    if input.soil_type == 0:
        reasons.append("Sandy soil drains quickly.")
    if input.soil_type == 2:
        reasons.append("Clay retains more water.")
    explanation = " ".join(reasons) or "Inputs are balanced; no single factor dominates."

    # recommendation
    if category.startswith("Low"):
        recommendation = "Irrigate soon."
    elif category.startswith("Medium"):
        recommendation = "Moisture is acceptable; monitor regularly."
    else:
        recommendation = "Soil is wet; avoid irrigation and check drainage."

    # log to CSV
    try:
        log_path = "predictions.csv"
        exists = os.path.exists(log_path)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["timestamp","temperature","humidity","rainfall","soil_type","moisture","category","explanation","recommendation"])
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), input.temperature, input.humidity, input.rainfall, input.soil_type, pred_val, category, explanation, recommendation])
    except Exception as e:
        print("Warning: failed to log:", e)

    return {"moisture": pred_val, "category": category, "explanation": explanation, "recommendation": recommendation}

# IMAGE UPLOAD: classifies soil type using color heuristics
def classify_soil_from_image(img: Image.Image) -> int:
    img = img.resize((60,60))
    arr = np.array(img.convert("RGB"))
    avg = arr.mean(axis=(0,1))
    r,g,b = avg
    lightness = arr.mean()
    yellowness = (r + g) / 2 - b
    # simple heuristic
    if lightness > 160 and yellowness > 10:
        return 0  # sandy
    if (r - b) > 10 and lightness < 140:
        return 2  # clay
    return 1  # loam

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    content = await file.read()
    try:
        img = Image.open(BytesIO(content))
    except Exception:
        return {"error": "Invalid image file."}
    soil_type = classify_soil_from_image(img)
    labels = {0: "Sandy", 1: "Loam", 2: "Clay"}
    return {"soil_type": int(soil_type), "soil_label": labels[int(soil_type)]}

# Optional server-side weather proxy if you set OWM API key in env (safer than exposing key client-side)
OWM_KEY = os.getenv("OWM_API_KEY", "")
@app.get("/weather")
def get_weather(city: str):
    if not OWM_KEY:
        return {"error":"No API key set on server."}
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OWM_KEY}&units=metric"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return {"error":"Weather fetch failed"}
    j = r.json()
    temp = j["main"]["temp"]
    humidity = j["main"]["humidity"]
    rain = 0
    if "rain" in j:
        rain = j["rain"].get("1h", j["rain"].get("3h", 0))
    return {"temp": temp, "humidity": humidity, "rain": rain}

@app.get("/")
def root():
    return {"message":"Soil Moisture Predictor API. Use /predict and /upload."}
