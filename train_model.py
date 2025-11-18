import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/soil_moisture.csv")

# Select features and target
X = df[["temperature", "humidity", "rainfall", "soil_type"]]
y = df["moisture"]

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
# RMSE (manual calculation because your sklearn version does not support squared=False)
rmse = mean_squared_error(y_test, pred) ** 0.5
print("RMSE:", rmse)
print("R2 Score:", r2_score(y_test, pred))


# Save the trained model to the models folder
joblib.dump(model, "models/soil_moisture_rf.pkl")
print("Model saved to models/soil_moisture_rf.pkl")
