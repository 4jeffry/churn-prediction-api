
import os
import joblib
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import tensorflow as tf

app = FastAPI(title="Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
xgb_model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
kmeans = joblib.load("models/kmeans.pkl")
scaler_seg = joblib.load("models/scaler_seg.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
nn_model = tf.keras.models.load_model("models/nn_model.keras")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-3.6-flash"
    ":generateContent?key=" + GEMINI_API_KEY
)

class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    features: dict

class BatchRequest(BaseModel):
    customers: List[CustomerData]

def get_risk_label(segment, churn_proba):
    if churn_proba >= 0.6:
        return "High Risk"
    elif churn_proba >= 0.3:
        return "Medium Risk"
    else:
        return "Low Risk"

def get_llm_explanation(tenure, monthly, churn_proba, risk_label):
    prompt = f"""Kamu adalah AI business analyst untuk tim customer retention.

Data customer:
- Tenure: {tenure} bulan
- Monthly Charges: ${monthly:.0f}
- Churn Probability: {churn_proba*100:.1f}%
- Risk Segment: {risk_label}

Berikan analisis singkat dalam 3 kalimat:
1. Status risiko customer ini
2. Faktor utama yang mempengaruhi
3. Rekomendasi aksi konkret untuk tim retention

Gunakan bahasa Indonesia yang profesional."""

    try:
        response = requests.post(
            GEMINI_URL,
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=10
        )
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"LLM unavailable: {str(e)}"

def predict_single(customer: CustomerData):
    # Build feature vector
    input_dict = {col: 0 for col in feature_cols}
    input_dict["tenure"] = customer.tenure
    input_dict["MonthlyCharges"] = customer.MonthlyCharges
    input_dict["TotalCharges"] = customer.TotalCharges
    for k, v in customer.features.items():
        if k in input_dict:
            input_dict[k] = v

    X = np.array([[input_dict[col] for col in feature_cols]])
    X_scaled = scaler.transform(X)

    # XGBoost prediction
    xgb_proba = float(xgb_model.predict_proba(X_scaled)[0][1])
    xgb_pred = int(xgb_proba >= 0.5)

    # NN prediction
    nn_proba = float(nn_model.predict(X_scaled, verbose=0)[0][0])

    # Best model = XGBoost (sudah terbukti F1 lebih tinggi)
    churn_proba = xgb_proba

    # Segmentation
    seg_input = np.array([[customer.tenure, customer.MonthlyCharges, customer.TotalCharges]])
    seg_scaled = scaler_seg.transform(seg_input)
    segment = int(kmeans.predict(seg_scaled)[0])
    risk_label = get_risk_label(segment, churn_proba)

    # LLM Explanation
    explanation = get_llm_explanation(
        customer.tenure, customer.MonthlyCharges, churn_proba, risk_label
    )

    return {
        "churn_probability": round(churn_proba * 100, 2),
        "churn_prediction": "Churn" if xgb_pred == 1 else "No Churn",
        "xgb_probability": round(xgb_proba * 100, 2),
        "nn_probability": round(nn_proba * 100, 2),
        "risk_segment": risk_label,
        "ai_explanation": explanation
    }

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: CustomerData):
    return predict_single(customer)

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    results = []
    for i, customer in enumerate(request.customers):
        result = predict_single(customer)
        result["customer_index"] = i
        results.append(result)

    high_risk = [r for r in results if r["risk_segment"] == "High Risk"]
    medium_risk = [r for r in results if r["risk_segment"] == "Medium Risk"]
    low_risk = [r for r in results if r["risk_segment"] == "Low Risk"]

    return {
        "total_customers": len(results),
        "summary": {
            "high_risk": len(high_risk),
            "medium_risk": len(medium_risk),
            "low_risk": len(low_risk),
            "avg_churn_probability": round(
                sum(r["churn_probability"] for r in results) / len(results), 2
            )
        },
        "predictions": results
    }
