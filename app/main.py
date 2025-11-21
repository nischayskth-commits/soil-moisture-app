# app/main.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
import joblib, os, csv, time, requests
import numpy as np, pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import traceback

app = FastAPI(title="Soil Moisture Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# SAFE MODEL LOADING
# -----------------------------
MODEL_PATH = os.path.join("models", "soil_moisture_rf.pkl")
model = None

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("\nMODEL LOADED SUCCESSFULLY:", MODEL_PATH)
            return True
        else:
            print("\nMODEL NOT FOUND:", MODEL_PATH)
            return False
    except Exception as e:
        print("\nMODEL LOAD ERROR:", e)
        traceback.print_exc()
        model = None
        return False

# Load on startup
load_model()

# -----------------------------
# DATA MODELS
# -----------------------------
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

# -----------------------------
# CATEGORY HELPER
# -----------------------------
def moisture_category(val: float) -> str:
    if val < 5:
        return "Low (Dry)"
    if val < 12:
        return "Medium (Optimal)"
    return "High (Wet)"

# -----------------------------
# DEBUG ENDPOINT (for testing)
# -----------------------------
@app.get("/_debug_model")
def debug_model():
    info = {
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None
    }

    if model is None:
        return info

    try:
        sample = pd.DataFrame([
            {"temperature": 34, "humidity": 30, "rainfall": 0, "soil_type": 0},
            {"temperature": 22, "humidity": 80, "rainfall": 15, "soil_type": 2},
            {"temperature": 27, "humidity": 60, "rainfall": 4, "soil_type": 1}
        ])
        preds = model.predict(sample)
        info["sample_preds"] = [float(i) for i in preds]
    except Exception as e:
        info["sample_error"] = str(e)

    return info

# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@app.post("/predict", response_model=Prediction)
def predict(input: InputData):

    if model is None:
        load_model()
        if model is None:
            return {
                "moisture": 0.0,
                "category": "Model Not Loaded",
                "explanation": "Model file could not be loaded.",
                "recommendation": "Fix model loading."
            }

    # DataFrame with correct column order
    df = pd.DataFrame([{
        "temperature": float(input.temperature),
        "humidity": float(input.humidity),
        "rainfall": float(input.rainfall),
        "soil_type": int(input.soil_type)
    }])

    try:
        pred_val = float(round(model.predict(df)[0], 2))
    except Exception as e:
        print("PREDICT ERROR:", e)
        traceback.print_exc()
        return {
            "moisture": 0.0,
            "category": "Prediction Failed",
            "explanation": "See server logs.",
            "recommendation": ""
        }

    category = moisture_category(pred_val)

    # Explanation rules
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

    # Recommendations
    if category.startswith("Low"):
        recommendation = "Irrigate soon."
    elif category.startswith("Medium"):
        recommendation = "Moisture is acceptable; monitor regularly."
    else:
        recommendation = "Soil is wet; avoid irrigation and check drainage."

    # Log predictions to CSV
    try:
        log_file = "predictions.csv"
        write_header = not os.path.exists(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "timestamp", "temperature", "humidity", "rainfall", "soil_type",
                    "moisture", "category", "explanation", "recommendation"
                ])
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                input.temperature, input.humidity, input.rainfall, input.soil_type,
                pred_val, category, explanation, recommendation
            ])
    except Exception as e:
        print("CSV LOG ERROR:", e)

    return {
        "moisture": pred_val,
        "category": category,
        "explanation": explanation,
        "recommendation": recommendation
    }

# -----------------------------
# IMAGE UPLOAD / SOIL TYPE CLASSIFICATION (existing heuristic)
# -----------------------------
def classify_soil_from_image(img: Image.Image) -> int:
    """
    Very small heuristic classifier that guesses soil type from color averages.
    Returns 0 (sandy), 1 (loam), 2 (clay)
    """
    img = img.resize((60, 60))
    arr = np.array(img.convert("RGB"))
    avg = arr.mean(axis=(0, 1))
    r, g, b = avg
    lightness = arr.mean()
    yellowness = (r + g) / 2 - b

    if lightness > 160 and yellowness > 10:
        return 0  # sandy
    if (r - b) > 10 and lightness < 140:
        return 2  # clay
    return 1  # loam

# -----------------------------
# SOIL vs NON-SOIL DETECTION (COLOR HEURISTIC)
# -----------------------------
def is_soil_image(img: Image.Image) -> float:
    """
    Detect if the image looks like soil based on color statistics.
    Returns a confidence value between 0.0 and 1.0.
    """
    try:
        img = img.resize((80, 80)).convert("RGB")
        arr = np.array(img)

        # average R,G,B values
        r = float(arr[:, :, 0].mean())
        g = float(arr[:, :, 1].mean())
        b = float(arr[:, :, 2].mean())

        # scoring rules
        brown_score = 0.0

        # Rule 1: brownish color pattern (R > G > B)
        if r > g and g > b:
            brown_score += 0.4

        # Rule 2: low blue increases soil likelihood
        if b < 120:
            brown_score += 0.3

        # Rule 3: earthy brightness range
        brightness = (r + g + b) / 3.0
        if 50 < brightness < 180:
            brown_score += 0.3

        confidence = min(1.0, brown_score)
        return float(confidence)
    except Exception:
        return 0.0

@app.post("/check-soil")
async def check_soil(file: UploadFile = File(...)):
    """
    Returns whether the uploaded image looks like soil.
    Uses color heuristics; no TensorFlow required.
    JSON: { "soil": bool, "confidence": float }
    """
    try:
        image_bytes = await file.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        confidence = is_soil_image(img)

        # threshold (adjust if you want stricter/looser detection)
        THRESHOLD = 0.60
        is_soil = confidence >= THRESHOLD

        return {"soil": bool(is_soil), "confidence": float(confidence)}
    except Exception as e:
        # return an error payload so frontend can show message
        return {"error": "Invalid image", "detail": str(e)}

# -----------------------------
# /upload endpoint (uses check first then classifies)
# -----------------------------
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload image, validate it's soil (using check), then classify soil type with heuristic.
    Returns: { soil_type: int, soil_label: str, confidence: float } on success
             { error: "not_soil", confidence: float } if check fails
             { error: "Invalid image", detail: ... } on image decode error
    """
    try:
        content = await file.read()
    except Exception as e:
        return {"error": "Invalid image read", "detail": str(e)}

    # Validate it's an image and looks like soil
    try:
        img = Image.open(BytesIO(content)).convert("RGB")
    except Exception as e:
        return {"error": "Invalid image", "detail": str(e)}

    try:
        confidence = is_soil_image(img)
        if confidence < 0.60:
            return {"error": "not_soil", "confidence": float(confidence)}
    except Exception as e:
        return {"error": "check_failed", "detail": str(e)}

    # classify soil type (using original heuristic)
    try:
        soil_type = classify_soil_from_image(img)
        labels = {0: "Sandy", 1: "Loam", 2: "Clay"}
        return {
            "soil_type": int(soil_type),
            "soil_label": labels.get(int(soil_type), "Unknown"),
            "confidence": float(confidence)
        }
    except Exception as e:
        return {"error": "classification_failed", "detail": str(e)}

# -----------------------------
# WEATHER API
# -----------------------------
OWM_KEY = os.getenv("OWM_API_KEY", "")

@app.get("/weather")
def get_weather(city: str):
    if not OWM_KEY:
        return {"error": "No API key set on server."}

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OWM_KEY}&units=metric"
    r = requests.get(url, timeout=10)

    if r.status_code != 200:
        return {"error": "Weather fetch failed"}

    data = r.json()
    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    rain = data.get("rain", {}).get("1h", 0)

    return {"temp": temp, "humidity": humidity, "rain": rain}

# -----------------------------
# ROOT
# -----------------------------
@app.get("/")
def root():
    return {"message": "Soil Moisture Predictor API. Use /predict, /upload, and /weather."}
