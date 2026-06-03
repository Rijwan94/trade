from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from .model import FEATURE_COLUMNS, DualTaskModel, prepare_training_data, time_series_splits


@dataclass
class FoldResult:
    accuracy: float
    f1: float
    mae: float
    rmse: float


def regime_buckets(frame: pd.DataFrame) -> pd.Series:
    vol = frame["volatility_20"].fillna(frame["volatility_20"].median())
    return pd.qcut(vol.rank(method="first"), q=3, labels=["low", "mid", "high"])


def evaluate_walk_forward(frame: pd.DataFrame, horizon: int) -> dict:
    x, y_direction, y_price = prepare_training_data(frame, horizon)
    rows = []

    for train_idx, test_idx in time_series_splits(frame.loc[x.index].copy()):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_dir_train, y_dir_test = y_direction.iloc[train_idx], y_direction.iloc[test_idx]
        y_price_train, y_price_test = y_price.iloc[train_idx], y_price.iloc[test_idx]

        model = DualTaskModel()
        model.classifier.fit(x_train, y_dir_train)
        model.regressor.fit(x_train, y_price_train)

        dir_pred = model.classifier.predict(x_test)
        price_pred = model.regressor.predict(x_test)

        rows.append(
            FoldResult(
                accuracy=accuracy_score(y_dir_test, dir_pred),
                f1=f1_score(y_dir_test, dir_pred, zero_division=0),
                mae=mean_absolute_error(y_price_test, price_pred),
                rmse=np.sqrt(mean_squared_error(y_price_test, price_pred)),
            )
        )

    averages = {
        "accuracy": float(np.mean([r.accuracy for r in rows])),
        "f1": float(np.mean([r.f1 for r in rows])),
        "mae": float(np.mean([r.mae for r in rows])),
        "rmse": float(np.mean([r.rmse for r in rows])),
    }

    model = DualTaskModel()
    model.fit(frame, horizon)
    probs = model.classifier.predict_proba(x)[:, 1]
    calib_brier = float(np.mean((probs - y_direction) ** 2))

    regimes = regime_buckets(frame.loc[x.index])
    regime_metrics = {}
    for regime in regimes.unique():
        mask = regimes == regime
        if mask.sum() == 0:
            continue
        y_reg = y_direction[mask]
        pred_reg = (probs[mask] >= 0.5).astype(int)
        regime_metrics[str(regime)] = {
            "accuracy": float(accuracy_score(y_reg, pred_reg)),
            "f1": float(f1_score(y_reg, pred_reg, zero_division=0)),
        }

    averages["calibration_brier"] = calib_brier
    averages["regime_metrics"] = regime_metrics
    return averages
