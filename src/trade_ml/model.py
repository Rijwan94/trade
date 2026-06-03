from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit


FEATURE_COLUMNS = [
    "atr_14",
    "volatility_20",
    "return_1",
    "return_5",
    "volume_z20",
    "dist_to_supply",
    "dist_to_demand",
    "hammer",
    "shooting_star",
    "doji",
    "trend_strength",
    "breakout_up",
    "breakout_down",
]


@dataclass
class DualTaskModel:
    classifier: CalibratedClassifierCV = CalibratedClassifierCV(RandomForestClassifier(n_estimators=200, random_state=42), method="sigmoid", cv=3)
    regressor: GradientBoostingRegressor = GradientBoostingRegressor(random_state=42)

    def fit(self, frame: pd.DataFrame, horizon: int) -> None:
        x, y_direction, y_price = prepare_training_data(frame, horizon)
        self.classifier.fit(x, y_direction)
        self.regressor.fit(x, y_price)

    def predict(self, latest_features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probabilities = self.classifier.predict_proba(latest_features[FEATURE_COLUMNS])
        direction_prob = probabilities[:, 1]
        direction = (direction_prob >= 0.5).astype(int)
        future_price = self.regressor.predict(latest_features[FEATURE_COLUMNS])
        return direction, direction_prob, future_price


def prepare_training_data(frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    target_future_close = frame["close"].shift(-horizon)
    y_direction = (target_future_close > frame["close"]).astype(int)
    y_price = target_future_close

    out = frame.copy()
    out["target_direction"] = y_direction
    out["target_price"] = y_price
    out = out.dropna().reset_index(drop=True)

    x = out[FEATURE_COLUMNS]
    return x, out["target_direction"], out["target_price"]


def time_series_splits(frame: pd.DataFrame, n_splits: int = 5):
    x = frame[FEATURE_COLUMNS]
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(x):
        yield train_idx, test_idx
