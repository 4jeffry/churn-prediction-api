
import os
import joblib
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

xgb_model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
kmeans = joblib.load("models/kmeans.pkl")
scaler_seg = joblib.load("models/scaler_seg.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
BASE_URL = "https://generativelanguage.googleapis.com"
ENDPOINT = "/v1beta/models/gemini-3.6-flash:generateContent"
GEMINI_URL = BASE_URL + ENDPOINT + "?key=" + GEMINI_API_KEY

class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    features: dict = {}

class BatchRequest(BaseModel):
    customers: List[CustomerData]

def get_risk_label(churn_proba):
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
    input_dict = {col: 0 for col in feature_cols}
    input_dict["tenure"] = customer.tenure
    input_dict["MonthlyCharges"] = customer.MonthlyCharges
    input_dict["TotalCharges"] = customer.TotalCharges
    for k, v in customer.features.items():
        if k in input_dict:
            input_dict[k] = v

    X = np.array([[input_dict[col] for col in feature_cols]])
    X_scaled = scaler.transform(X)

    churn_proba = float(xgb_model.predict_proba(X_scaled)[0][1])
    churn_pred = int(churn_proba >= 0.5)
    risk_label = get_risk_label(churn_proba)

    explanation = get_llm_explanation(
        customer.tenure, customer.MonthlyCharges, churn_proba, risk_label
    )

    return {
        "churn_probability": round(churn_proba * 100, 2),
        "churn_prediction": "Churn" if churn_pred == 1 else "No Churn",
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
