import numpy as np
import pandas as pd

np.random.seed(0)
n = 2000
temperature = np.random.normal(25, 6, n)
humidity = np.clip(np.random.normal(60, 15, n), 5, 100)
rainfall = np.clip(np.random.exponential(1.5, n), 0, 100)
soil_type = np.random.choice([0,1,2], n, p=[0.4,0.4,0.2])

moisture = (
    0.3*rainfall +
    0.2*humidity -
    0.4*temperature +
    (soil_type==2)*5 - (soil_type==0)*3
) + np.random.normal(0,3,n)

moisture = np.clip(moisture, 0, 60)

df = pd.DataFrame({
    "temperature": temperature.round(2),
    "humidity": humidity.round(2),
    "rainfall": rainfall.round(2),
    "soil_type": soil_type,
    "moisture": moisture.round(2)
})

df.to_csv("data/soil_moisture.csv", index=False)
print("Saved data/soil_moisture.csv")
