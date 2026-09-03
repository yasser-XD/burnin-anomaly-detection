"""
Time-Series Drift Prediction Engine (Module B)
Predicts 168h parameter value using early burn-in measurements (0h and 24h).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, Any
from config import RANDOM_STATE

class DriftPredictor168h:
    """
    Predicts 168h parameter value using 0h and 24h measurements and calculated early slope.
    """
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
        self.is_fitted = False

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers early trajectory features:
        - Value_0h
        - Value_24h
        - Early_Drift_24h = Value_24h - Value_0h
        - Early_Slope_per_hr = (Value_24h - Value_0h) / 24.0
        """
        X = pd.DataFrame()
        X["Value_0h"] = df["Value_0h"]
        X["Value_24h"] = df["Value_24h"]
        X["Early_Drift_24h"] = df["Value_24h"] - df["Value_0h"]
        X["Early_Slope_per_hr"] = X["Early_Drift_24h"] / 24.0
        return X

    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Fits the regressor if ground truth Value_168h is available in the dataset.
        """
        if "Value_168h" not in df.columns:
            raise ValueError("Training requires ground truth 'Value_168h' in DataFrame.")

        valid_df = df.dropna(subset=["Value_0h", "Value_24h", "Value_168h"])
        X = self._extract_features(valid_df)
        y = valid_df["Value_168h"]

        self.model.fit(X, y)
        self.is_fitted = True

        y_pred = self.model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        return {"train_mae": round(mae, 4), "train_rmse": round(rmse, 4), "train_r2": round(r2, 4)}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts 168h value for components. Fallback to analytical extrapolation if model isn't pre-trained.
        """
        res_df = df.copy()
        X = self._extract_features(res_df)

        if self.is_fitted:
            preds = self.model.predict(X)
        else:
            # Analytical physics-based extrapolation fallback:
            # Linear extrapolation: 0h + (slope * 168h) with slight decay assumption
            preds = X["Value_0h"] + (X["Early_Slope_per_hr"] * 168.0)

        res_df["Predicted_Value_168h"] = np.round(preds, 2)
        res_df["Predicted_Total_Drift"] = (res_df["Predicted_Value_168h"] - res_df["Value_0h"]).round(2)
        res_df["Predicted_Drift_Rate_per_hr"] = (res_df["Predicted_Total_Drift"] / 168.0).round(4)

        return res_df
