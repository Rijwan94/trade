from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .inference import InferenceService


@dataclass
class DriftMonitor:
    window: int
    recent_confidence: list[float] = field(default_factory=list)

    def update(self, confidence: float) -> None:
        self.recent_confidence.append(confidence)
        if len(self.recent_confidence) > self.window:
            self.recent_confidence.pop(0)

    def drift_score(self) -> float:
        if len(self.recent_confidence) < 20:
            return 0.0
        arr = np.array(self.recent_confidence)
        return float(abs(np.mean(arr[: len(arr)//2]) - np.mean(arr[len(arr)//2 :])))


@dataclass
class PaperTrader:
    service: InferenceService
    drift_monitor: DriftMonitor

    def on_new_market_data(self, asset: str, frame: pd.DataFrame) -> dict:
        prediction = self.service.predict(asset, frame)
        self.drift_monitor.update(prediction["confidence"])
        prediction["drift_score"] = self.drift_monitor.drift_score()
        prediction["paper_trade_signal"] = prediction["prediction_accepted"]
        return prediction
