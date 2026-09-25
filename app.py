
import os
import joblib
import numpy as np
import requests
import resend
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from monitor import check_drift

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

# API Keys & URLs
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
BASE_URL = "https://generativelanguage.googleapis.com"
ENDPOINT = "/v1beta/models/gemini-3.6-flash:generateContent"
GEMINI_URL = f"{BASE_URL}{ENDPOINT}?key={GEMINI_API_KEY}"

resend.api_key = os.environ.get("RESEND_API_KEY", "")

# Buffer untuk simpan recent predictions (in-memory)
recent_predictions_buffer = []

class CustomerData(BaseModel):
    customer_id: Optional[str] = "Unknown-ID"
    customer_name: Optional[str] = "Klien"
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    features: dict = {}

class BatchRequest(BaseModel):
    customers: List[CustomerData]

class EmailReportRequest(BaseModel):
    target_email: str
    customer_id: str
    customer_name: str
    risk_segment: str
    churn_probability: float
    ai_explanation: str

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
- Monthly Charges: Rp {monthly:.0f}
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

# Alert Internal Otomatis (Jika risiko > 60%)
def send_internal_alert(customer_id, customer_name, churn_proba, explanation):
    if not resend.api_key:
        return

    html_body = f"""
    <div style="font-family: sans-serif; padding: 20px;">
        <h2 style="color: #e74c3c;">🚨 Alert: Risiko Klien Keluar Tinggi</h2>
        <p>Sistem mendeteksi klien dengan risiko churn tinggi:</p>
        <ul>
            <li><strong>ID Klien:</strong> {customer_id}</li>
            <li><strong>Nama:</strong> {customer_name}</li>
            <li><strong>Risiko:</strong> <span style="color: red;">{churn_proba}%</span></li>
        </ul>
        <h3>Insight AI:</h3>
        <p>{explanation}</p>
    </div>
    """
    
    try:
        resend.Emails.send({
            "from": "ixiera AI System <contact@ixiera.id>",
            "to": ["contact@ixiera.id"], 
            "subject": f"Action Required: High Risk Client [{customer_id}]",
            "html": html_body,
        })
    except Exception as e:
        print(f"Gagal mengirim internal alert: {e}")

def predict_single(customer: CustomerData, background_tasks: BackgroundTasks = None):
    input_dict = {col: 0 for col in feature_cols}
    input_dict["tenure"] = customer.tenure
    input_dict["MonthlyCharges"] = customer.MonthlyCharges
    input_dict["TotalCharges"] = customer.TotalCharges
    for k, v in customer.features.items():
        if k in input_dict:
            input_dict[k] = v

    X = np.array([[input_dict[col] for col in feature_cols]])
    X_scaled = scaler.transform(X)

    recent_predictions_buffer.append(input_dict)
    if len(recent_predictions_buffer) > 500:
        recent_predictions_buffer.pop(0)

    churn_proba = float(xgb_model.predict_proba(X_scaled)[0][1])
    churn_pred = int(churn_proba >= 0.5)
    risk_label = get_risk_label(churn_proba)

    explanation = get_llm_explanation(
        customer.tenure, customer.MonthlyCharges, churn_proba, risk_label
    )
    
    churn_probability_pct = round(churn_proba * 100, 2)

    # Trigger alert internal jika bahaya
    if risk_label == "High Risk" and background_tasks is not None:
        background_tasks.add_task(
            send_internal_alert, 
            customer.customer_id, 
            customer.customer_name, 
            churn_probability_pct, 
            explanation
        )

    return {
        "customer_id": customer.customer_id,
        "churn_probability": churn_probability_pct,
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
def predict(customer: CustomerData, background_tasks: BackgroundTasks):
    return predict_single(customer, background_tasks)

@app.post("/predict/batch")
def predict_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    results = []
    for i, customer in enumerate(request.customers):
        result = predict_single(customer, background_tasks)
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

# Endpoint Baru: Kirim Laporan ke Email User via Frontend
@app.post("/send-report")
def send_report_to_user(req: EmailReportRequest, background_tasks: BackgroundTasks):
    def send_email():
        if not resend.api_key:
            return
            
        html_body = f"""
        <div style="font-family: sans-serif; padding: 20px; color: #333;">
            <h2 style="color: #0a0a0a;">Laporan Analisis Retensi Klien | ixiera.id</h2>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
                <p><strong>ID Klien:</strong> {req.customer_id}</p>
                <p><strong>Nama:</strong> {req.customer_name}</p>
                <p><strong>Status Risiko:</strong> <span style="font-weight: bold;">{req.risk_segment}</span></p>
                <p><strong>Potensi Keluar:</strong> <span style="color: red; font-weight: bold;">{req.churn_probability}%</span></p>
            </div>
            <h3>Saran Strategi Retensi (AI Insight):</h3>
            <p style="line-height: 1.6;">{req.ai_explanation}</p>
        </div>
        """
        try:
            resend.Emails.send({
                "from": "ixiera AI System <contact@ixiera.id>",
                "to": [req.target_email],
                "subject": f"Hasil Analisis Retensi - {req.customer_name}",
                "html": html_body,
            })
        except Exception as e:
            print(f"Gagal kirim laporan ke user: {e}")

    background_tasks.add_task(send_email)
    return {"message": "Email queued for sending"}

@app.get("/monitoring/drift")
def monitoring_drift():
    if len(recent_predictions_buffer) < 50:
        return {
            "status": "insufficient_data",
            "message": f"Butuh minimal 50 predictions",
            "current_count": len(recent_predictions_buffer)
        }
    return check_drift(recent_predictions_buffer)

@app.get("/monitoring/status")
def monitoring_status():
    return {
        "total_predictions_buffered": len(recent_predictions_buffer),
        "buffer_capacity": 500,
        "model": "XGBoost",
        "status": "active"
    }
