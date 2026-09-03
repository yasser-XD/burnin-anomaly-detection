"""
Configuration settings for AI-Driven Anomaly Detection in Component Burn-In & Screening
(ISRO Problem Statement ID: 26170)
"""
import os
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data_store"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Threshold & Screening Rules
ROBUST_Z_SCORE_THRESHOLD = 3.0  # Outlier flag if |z| > 3.0 relative to lot
SAFETY_SLOPE_MARGIN_RATIO = 0.80  # Safety threshold set to 80% of Datasheet Limit

# Time points in burn-in cycle (hours)
TIME_POINTS = [0, 24, 96, 168]
EARLY_TIME_POINTS = [0, 24]

# Model Parameters
ISOLATION_FOREST_CONTAMINATION = 0.05
RANDOM_STATE = 42

# Risk Categories
RISK_PASS = "PASS"
RISK_REVIEW = "REVIEW"
RISK_FLAG = "FLAG (HIGH RISK)"
