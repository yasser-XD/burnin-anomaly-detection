"""
Data Validation Module (FR-02)
Detects missing, corrupted, non-numeric, or duplicate component measurements.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

REQUIRED_COLUMNS = [
    "Component_ID", "Lot_ID", "Parameter",
    "Value_0h", "Value_24h", "Value_96h", "Value_168h",
    "Datasheet_Limit"
]

NUMERIC_COLUMNS = ["Value_0h", "Value_24h", "Value_96h", "Value_168h", "Datasheet_Limit"]

def validate_burnin_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates dataset structure, checks for missing/corrupted values, physical limits, and duplicates.
    Returns (cleaned_df, validation_summary).
    """
    summary = {
        "total_input_rows": len(df),
        "missing_values_count": 0,
        "duplicate_components_count": 0,
        "invalid_numeric_count": 0,
        "negative_values_count": 0,
        "is_valid": True,
        "warnings": []
    }
    
    cleaned_df = df.copy()
    
    # Check missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in cleaned_df.columns]
    if missing_cols:
        summary["is_valid"] = False
        summary["warnings"].append(f"Missing required columns: {missing_cols}")
        return cleaned_df, summary
        
    # Check for missing values in numeric fields
    null_mask = cleaned_df[NUMERIC_COLUMNS].isnull().any(axis=1)
    summary["missing_values_count"] = int(null_mask.sum())
    if summary["missing_values_count"] > 0:
        summary["warnings"].append(f"Found {summary['missing_values_count']} rows with missing values.")
        
    # Coerce numeric columns
    for col in NUMERIC_COLUMNS:
        invalid_mask = pd.to_numeric(cleaned_df[col], errors='coerce').isnull() & cleaned_df[col].notnull()
        summary["invalid_numeric_count"] += int(invalid_mask.sum())
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
    # Check for negative parameter values (physical impossibility for leakage current, delay)
    neg_mask = (cleaned_df[NUMERIC_COLUMNS] < 0).any(axis=1)
    summary["negative_values_count"] = int(neg_mask.sum())
    if summary["negative_values_count"] > 0:
        summary["warnings"].append(f"Found {summary['negative_values_count']} rows with negative values.")
        
    # Check duplicates
    dup_mask = cleaned_df.duplicated(subset=["Component_ID"], keep='first')
    summary["duplicate_components_count"] = int(dup_mask.sum())
    if summary["duplicate_components_count"] > 0:
        summary["warnings"].append(f"Found {summary['duplicate_components_count']} duplicate Component_IDs. Dropping duplicates.")
        cleaned_df = cleaned_df.drop_duplicates(subset=["Component_ID"], keep='first')
        
    # Clean rows with unusable missing key measurements (e.g. Value_0h or Value_24h)
    essential_mask = cleaned_df["Value_0h"].notnull() & cleaned_df["Value_24h"].notnull() & cleaned_df["Datasheet_Limit"].notnull()
    cleaned_df = cleaned_df[essential_mask].reset_index(drop=True)
    
    summary["total_valid_rows"] = len(cleaned_df)
    return cleaned_df, summary
