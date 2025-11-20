# train_model.py
# Saves: models/soil_moisture_rf.pkl and data/soil_moisture.csv
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Config - tune these to make training faster or stronger
N_SAMPLES = 2000            # increase if you want a larger dataset
RF_ESTIMATORS = 80          # number of trees (80 is moderate)
USE_RANDOM_FOREST = True    # set False to use Ridge (fast)

ROOT = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def generate_synthetic_data(n=N_SAMPLES, seed=0):
    np.random.seed(seed)
    temperature = np.random.normal(25, 8, n)
    humidity = np.clip(np.random.normal(60, 20, n), 0, 100)
    rainfall = np.clip(np.random.exponential(3.0, n), 0, 200)
    soil_type = np.random.choice([0,1,2], n, p=[0.35,0.45,0.2])

    # Create moisture with interactions and noise (range 0-60)
    moisture = (
        0.4*rainfall + 
        0.25*humidity - 
        0.35*temperature + 
        (soil_type==2)*8 - (soil_type==0)*4 +
        10*np.sin(humidity/20) +
        np.random.normal(0,5,n)
    )
    moisture = np.clip(moisture, 0, 60)
    df = pd.DataFrame({
        "temperature": temperature.round(2),
        "humidity": humidity.round(2),
        "rainfall": rainfall.round(2),
        "soil_type": soil_type.astype(int),
        "moisture": moisture.round(2)
    })
    return df

def train_and_save(df):
    X = df[["temperature","humidity","rainfall","soil_type"]]
    y = df["moisture"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    if USE_RANDOM_FOREST:
        model = RandomForestRegressor(n_estimators=RF_ESTIMATORS, random_state=42, n_jobs=-1)
    else:
        model = Ridge(alpha=1.0)

    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    print("Training model...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Done. Test MSE: {mse:.3f}, R2: {r2:.3f}")

    model_path = os.path.join(MODELS_DIR, "soil_moisture_rf.pkl")
    joblib.dump(pipeline, model_path)
    print("Saved model to:", model_path)
    return pipeline, model_path

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_synthetic_data()
    csv_path = os.path.join(DATA_DIR, "soil_moisture.csv")
    df.to_csv(csv_path, index=False)
    print("Saved dataset to:", csv_path)
    pipeline, model_path = train_and_save(df)

    # Show sample predictions to sanity-check
    sample = pd.DataFrame([
        {"temperature":5.0,  "humidity":10.0, "rainfall":0.0,  "soil_type":0},
        {"temperature":35.0, "humidity":90.0, "rainfall":20.0, "soil_type":2}
    ])
    print("SAMPLE INPUTS:")
    print(sample)
    print("SAMPLE PREDICTIONS:", pipeline.predict(sample).tolist())
