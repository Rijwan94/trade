from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import make_feature_table
from .model import FEATURE_COLUMNS, DualTaskModel, prepare_training_data


@dataclass
class BacktestResult:
    trades: int
    hit_rate: float
    avg_pnl: float
    sharpe_like: float


def run_backtest(frame: pd.DataFrame, horizon: int, confidence_threshold: float) -> BacktestResult:
    if set(FEATURE_COLUMNS).issubset(frame.columns):
        features = frame.copy()
    else:
        features = make_feature_table(frame)
    x, y_direction, _ = prepare_training_data(features, horizon)

    split = int(len(x) * 0.8)
    model = DualTaskModel()
    model.classifier.fit(x.iloc[:split], y_direction.iloc[:split])

    probs = model.classifier.predict_proba(x.iloc[split:])[:, 1]
    preds = (probs >= 0.5).astype(int)
    actual = y_direction.iloc[split:].to_numpy()

    taken = probs >= confidence_threshold
    if taken.sum() == 0:
        return BacktestResult(trades=0, hit_rate=0.0, avg_pnl=0.0, sharpe_like=0.0)

    pnl = np.where(preds[taken] == actual[taken], 1.0, -1.0)
    sharpe_like = float(np.mean(pnl) / (np.std(pnl) + 1e-8))
    return BacktestResult(
        trades=int(taken.sum()),
        hit_rate=float((preds[taken] == actual[taken]).mean()),
        avg_pnl=float(np.mean(pnl)),
        sharpe_like=sharpe_like,
    )
