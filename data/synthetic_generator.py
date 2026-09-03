"""
Synthetic Benchmark Generator for Component Burn-In & Screening Data
Simulates realistic time-series electrical measurements across manufacturing lots,
including normal population, peer anomalies, and early deteriorating components.
"""

import numpy as np
import pandas as pd
from typing import Optional
from config import DATA_DIR, RANDOM_STATE

def generate_synthetic_burnin_data(
    num_lots: int = 5,
    components_per_lot: int = 40,
    seed: int = RANDOM_STATE,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates a synthetic dataset simulating burn-in screening for electronic components.
    
    Parameters:
        num_lots: Number of distinct manufacturing lots.
        components_per_lot: Number of components in each lot.
        seed: Random seed for reproducibility.
        save_path: Optional path to save CSV.
    """
    np.random.seed(seed)
    records = []
    
    datasheet_limit = 50.0  # uA max leakage current limit
    
    comp_counter = 1000
    for lot_idx in range(1, num_lots + 1):
        lot_id = f"LOT_{chr(64 + lot_idx)}"
        
        # Lot baseline mean (e.g., 9.5 to 11.5 uA)
        lot_base_mean = np.random.uniform(9.5, 11.5)
        lot_base_std = np.random.uniform(0.8, 1.4)
        
        for c in range(components_per_lot):
            comp_counter += 1
            comp_id = f"C{comp_counter}"
            
            # Determine component behavior profile
            rand_val = np.random.random()
            
            if rand_val < 0.82:
                # Normal Component: low initial value, minor expected drift
                v0 = np.random.normal(lot_base_mean, lot_base_std)
                v0 = max(2.0, v0)
                drift_rate = np.random.uniform(0.01, 0.05)  # uA per hour
                noise_scale = 0.5
                
                v24 = v0 + (24 * drift_rate) + np.random.normal(0, noise_scale)
                v96 = v0 + (96 * drift_rate) + np.random.normal(0, noise_scale)
                v168 = v0 + (168 * drift_rate) + np.random.normal(0, noise_scale)
                label = "NORMAL"
                
            elif rand_val < 0.91:
                # Peer Anomaly: significantly higher baseline than lot peers, but within datasheet limit
                # E.g., Lot mean ~10 uA, this component starts at 38-42 uA (Limit is 50 uA)
                v0 = np.random.uniform(36.0, 42.0)
                drift_rate = np.random.uniform(0.02, 0.06)
                v24 = v0 + (24 * drift_rate) + np.random.normal(0, 0.5)
                v96 = v0 + (96 * drift_rate) + np.random.normal(0, 0.5)
                v168 = v0 + (168 * drift_rate) + np.random.normal(0, 0.5)
                label = "PEER_ANOMALY"
                
            else:
                # Early Deteriorating Component (Drift Defect):
                # Starts normal at 0h (~11 uA), but drifts aggressively, breaching limit at 168h (~55 uA)
                v0 = np.random.normal(lot_base_mean, lot_base_std)
                v0 = max(2.0, v0)
                # Rapid non-linear or steep linear drift
                v24 = v0 + np.random.uniform(6.0, 10.0)
                v96 = v0 + np.random.uniform(22.0, 30.0)
                v168 = v0 + np.random.uniform(42.0, 52.0)
                label = "DRIFT_DEFECT"
            
            # Ensure physical lower bound (non-negative)
            v0 = round(max(0.1, v0), 2)
            v24 = round(max(0.1, v24), 2)
            v96 = round(max(0.1, v96), 2)
            v168 = round(max(0.1, v168), 2)
            
            records.append({
                "Component_ID": comp_id,
                "Lot_ID": lot_id,
                "Parameter": "Leakage_Current_uA",
                "Value_0h": v0,
                "Value_24h": v24,
                "Value_96h": v96,
                "Value_168h": v168,
                "Datasheet_Limit": datasheet_limit,
                "Ground_Truth_Label": label
            })

    df = pd.DataFrame(records)
    
    if save_path:
        df.to_csv(save_path, index=False)
        
    return df

if __name__ == "__main__":
    out_file = DATA_DIR / "sample_burnin_data.csv"
    df = generate_synthetic_burnin_data(save_path=str(out_file))
    print(f"Generated {len(df)} synthetic component records saved to {out_file}")
    print(df.head(10))
