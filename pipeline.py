"""
End-to-End Component Screening Pipeline Wrapper
Integrates Data Validation, Module A Anomaly Detection, Module B Drift Prediction,
Risk Engine, and Explainability Generator.
"""

import pandas as pd
from typing import Tuple, Dict, Any

from data.ingestion import load_burnin_dataset
from data.validation import validate_burnin_dataset
from models.anomaly_module import run_anomaly_module_a
from models.drift_predictor import DriftPredictor168h
from models.drift_evaluator import evaluate_drift_and_safety, compute_model_performance_metrics
from engine.risk_engine import evaluate_composite_risk
from engine.explainer import attach_explanations_to_df
from engine.audit_logger import log_screening_run

def run_full_screening_pipeline(
    source: Any,
    dataset_name: str = "Uploaded Dataset"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs the complete 5-step screening pipeline on an input CSV path or DataFrame.
    Returns (processed_df, execution_summary).
    """
    # 1. Ingestion & Validation
    df_raw = load_burnin_dataset(source)
    df_clean, val_summary = validate_burnin_dataset(df_raw)
    
    if len(df_clean) == 0:
        return df_clean, {"error": "No valid component records found after validation.", "validation": val_summary}
        
    # 2. Module A: Peer Anomaly Detection
    df_mod_a = run_anomaly_module_a(df_clean)
    
    # 3. Module B: Drift Prediction
    predictor = DriftPredictor168h()
    # Fit if ground truth is available
    if "Value_168h" in df_mod_a.columns:
        try:
            predictor.fit(df_mod_a)
        except Exception:
            pass
            
    df_mod_b = predictor.predict(df_mod_a)
    df_mod_b = evaluate_drift_and_safety(df_mod_b)
    
    # 4. Risk Engine & Explainability
    df_risk = evaluate_composite_risk(df_mod_b)
    df_final = attach_explanations_to_df(df_risk)
    
    # 5. Metrics & Audit Logging
    perf_metrics = compute_model_performance_metrics(df_final)
    
    pass_count = int((df_final["Screening_Status"] == "PASS").sum())
    review_count = int((df_final["Screening_Status"] == "REVIEW").sum())
    flag_count = int((df_final["Screening_Status"] == "FLAG (HIGH RISK)").sum())
    
    log_entry = log_screening_run(
        dataset_name=dataset_name,
        total_components=len(df_final),
        flagged_count=flag_count,
        review_count=review_count,
        pass_count=pass_count,
        metrics=perf_metrics
    )
    
    execution_summary = {
        "dataset_name": dataset_name,
        "total_processed": len(df_final),
        "validation_summary": val_summary,
        "screening_counts": {
            "PASS": pass_count,
            "REVIEW": review_count,
            "FLAG": flag_count
        },
        "performance_metrics": perf_metrics,
        "audit_entry": log_entry
    }
    
    return df_final, execution_summary
