from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import uvicorn

# ==========================================
# 1. CONFIG & APP
# ==========================================
app = FastAPI(title="NeuroGuard API")

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. LOAD MODEL
# ==========================================
try:
    artifact = joblib.load('model.pkl')
    model = artifact['model']
    base_threshold = artifact['base_threshold']
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Model file not found. Please run train.py first.")
    model = None

# ==========================================
# 3. DATA MODELS
# ==========================================
class PatientData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

# ==========================================
# 4. ENDPOINTS
# ==========================================
@app.get("/")
def home():
    return {"message": "NeuroGuard Stroke Prediction API is Running"}

@app.post("/predict")
def predict_stroke(data: PatientData):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert Input to DataFrame
    # Handle BMI=0 as NaN for Imputation
    bmi_val = np.nan if data.bmi == 0 else data.bmi
    
    input_df = pd.DataFrame([{
        'gender': data.gender,
        'age': data.age,
        'hypertension': data.hypertension,
        'heart_disease': data.heart_disease,
        'ever_married': data.ever_married,
        'work_type': data.work_type,
        'Residence_type': data.Residence_type,
        'avg_glucose_level': data.avg_glucose_level,
        'bmi': bmi_val,
        'smoking_status': data.smoking_status
    }])

    # Predict Probability
    try:
        prob = model.predict_proba(input_df)[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Smart Prediction Logic (Dynamic Thresholds)
    adjusted_threshold = base_threshold
    risk_factors = []

    if data.age > 60:
        adjusted_threshold -= 0.05
        risk_factors.append("Advanced Age (>60)")
    
    if data.avg_glucose_level > 200:
        adjusted_threshold -= 0.05
        risk_factors.append("High Glucose Levels (>200)")
        
    if data.hypertension == 1:
        risk_factors.append("History of Hypertension")
        
    if data.heart_disease == 1:
        risk_factors.append("History of Heart Disease")

    is_high_risk = prob >= adjusted_threshold

    return {
        "probability": float(prob),
        "threshold": float(adjusted_threshold),
        "is_high_risk": bool(is_high_risk),
        "risk_factors": risk_factors
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
