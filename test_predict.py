# test_predict.py
import os, joblib, pandas as pd
MODEL_PATH = os.path.join("models", "soil_moisture_rf.pkl")
if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found at " + MODEL_PATH)
pipeline = joblib.load(MODEL_PATH)
X = pd.DataFrame([
  {"temperature":5.0, "humidity":10.0, "rainfall":0.0, "soil_type":0},
  {"temperature":35.0, "humidity":90.0, "rainfall":20.0, "soil_type":2}
])
preds = pipeline.predict(X)
print("Predictions:", preds.tolist())
