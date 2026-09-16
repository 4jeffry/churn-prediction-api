
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

feature_cols = joblib.load("models/feature_cols.pkl")
reference_df = pd.read_csv("data/reference.csv")

def check_drift(current_data: list[dict]) -> dict:
    """
    current_data: list of dict dari request yang masuk ke API
    """
    try:
        current_df = pd.DataFrame(current_data)

        # Pastikan kolom sama dengan reference
        for col in feature_cols:
            if col not in current_df.columns:
                current_df[col] = 0
        current_df = current_df[feature_cols]

        report = Report(metrics=[DatasetDriftMetric()])
        report.run(reference_data=reference_df, current_data=current_df)
        result = report.as_dict()

        drift_detected = result["metrics"][0]["result"]["dataset_drift"]
        drifted_features = [
            k for k, v in result["metrics"][0]["result"].get("drift_by_columns", {}).items()
            if v.get("drift_detected", False)
        ]

        return {
            "drift_detected": drift_detected,
            "features_drifted": drifted_features,
            "total_features": len(feature_cols),
            "checked_at": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "drift_detected": None,
            "error": str(e),
            "checked_at": datetime.now().isoformat()
        }
