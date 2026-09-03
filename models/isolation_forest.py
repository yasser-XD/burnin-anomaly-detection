"""
Isolation Forest Outlier Detection Model (Module A)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Tuple
from config import ISOLATION_FOREST_CONTAMINATION, RANDOM_STATE

def train_run_isolation_forest(
    df: pd.DataFrame,
    feature_cols: list = ["Value_0h", "Value_24h"],
    contamination: float = ISOLATION_FOREST_CONTAMINATION
) -> Tuple[pd.DataFrame, IsolationForest]:
    """
    Fits an Isolation Forest model on component time-point measurements
    and outputs anomaly scores and binary outlier predictions (-1: anomaly, 1: normal).
    """
    result_df = df.copy()
    X = result_df[feature_cols].fillna(0)
    
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_estimators=100
    )
    
    # Fit & predict
    predictions = iso_forest.fit_predict(X)  # -1 = anomaly, 1 = normal
    raw_scores = iso_forest.score_samples(X) # lower score = more anomalous
    
    # Normalize score between 0.0 (normal) and 1.0 (highly anomalous)
    min_score, max_score = raw_scores.min(), raw_scores.max()
    normalized_scores = 1.0 - (raw_scores - min_score) / (max_score - min_score + 1e-8)
    
    result_df["IsoForest_Prediction"] = np.where(predictions == -1, "ANOMALY", "NORMAL")
    result_df["IsoForest_Score"] = normalized_scores.round(3)
    
    return result_df, iso_forest
