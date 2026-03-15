from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Car Insurance Claim Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model/model.pkl")

class CustomerData(BaseModel):
    Gender: int
    Age: int
    Driving_License: int
    Region_Code: float
    Previously_Insured: int
    Vehicle_Age: int
    Vehicle_Damage: int
    Annual_Premium: float
    Policy_Sales_Channel: float
    Vintage: int

@app.get("/")
def home():
    return {"message": "Car Insurance Claim Prediction API is running!"}

@app.post("/predict")
def predict(data: CustomerData):
    features = np.array([[
        data.Gender, data.Age, data.Driving_License,
        data.Region_Code, data.Previously_Insured,
        data.Vehicle_Age, data.Vehicle_Damage,
        data.Annual_Premium, data.Policy_Sales_Channel,
        data.Vintage
    ]])
    result = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if probability > 0.7:
        risk = "High"
    elif probability > 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "prediction": bool(result),
        "claim": "YES" if result == 1 else "NO",
        "probability": round(float(probability) * 100, 2),
        "risk_level": risk
    }