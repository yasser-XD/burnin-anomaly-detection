"""
Data Ingestion Module for Component Burn-In Test Data
"""

import pandas as pd
from typing import Union, Tuple, Dict, Any
from pathlib import Path

REQUIRED_COLUMNS = [
    "Component_ID", "Lot_ID", "Parameter",
    "Value_0h", "Value_24h", "Value_96h", "Value_168h",
    "Datasheet_Limit"
]

def load_burnin_dataset(file_source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """
    Loads raw component test dataset from CSV path or pandas DataFrame.
    """
    if isinstance(file_source, (str, Path)):
        df = pd.read_csv(file_source)
    elif isinstance(file_source, pd.DataFrame):
        df = file_source.copy()
    else:
        raise ValueError("Unsupported data source. Provide CSV file path or pandas DataFrame.")
        
    # Strip column whitespaces
    df.columns = [c.strip() for c in df.columns]
    return df
