"""
Robust Statistics & Peer-Relative Outlier Detection (Module A)
Calculates Lot Median, Median Absolute Deviation (MAD), and Robust Z-Scores.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

def compute_lot_robust_stats(df: pd.DataFrame, target_col: str = "Value_24h") -> pd.DataFrame:
    """
    Computes lot-level median and MAD for peer-relative normalization.
    
    Robust Z-Score formula:
    Z_robust = (Value - Lot_Median) / (MAD * 1.4826)
    """
    result_df = df.copy()
    
    # Calculate Lot Median and Lot MAD
    lot_groups = result_df.groupby("Lot_ID")[target_col]
    
    medians = lot_groups.transform("median")
    
    # Calculate MAD: median(|x - median|)
    def get_mad(series):
        med = series.median()
        return np.median(np.abs(series - med))
        
    mads = lot_groups.transform(get_mad)
    
    # Avoid division by zero: if MAD is 0 (all components identical), fallback to standard std or small epsilon
    mads = np.where(mads == 0, 1e-6, mads)
    
    # Robust Z-score: scaling factor 1.4826 makes MAD equivalent to standard deviation for normal distributions
    robust_z = (result_df[target_col] - medians) / (mads * 1.4826)
    
    result_df[f"{target_col}_Lot_Median"] = medians.round(3)
    result_df[f"{target_col}_Lot_MAD"] = mads.round(3)
    result_df[f"{target_col}_Robust_Z"] = robust_z.round(3)
    
    return result_df
