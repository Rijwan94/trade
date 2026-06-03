from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import ModelTargets
from .features import make_feature_table
from .model import DualTaskModel


@dataclass
class MetricsMonitor:
    total_predictions: int = 0
    failovers: int = 0
    latency_ms: list[float] = field(default_factory=list)

    def record_latency(self, ms: float) -> None:
        self.latency_ms.append(ms)
        self.total_predictions += 1

    def snapshot(self) -> dict[str, Any]:
        p95 = 0.0
        if self.latency_ms:
            sorted_l = sorted(self.latency_ms)
            p95 = sorted_l[int(0.95 * (len(sorted_l) - 1))]
        return {
            "total_predictions": self.total_predictions,
            "failovers": self.failovers,
            "p95_latency_ms": p95,
        }


@dataclass
class FeatureCache:
    latest: dict[str, pd.DataFrame] = field(default_factory=dict)

    def put(self, asset: str, features: pd.DataFrame) -> None:
        self.latest[asset] = features

    def get(self, asset: str) -> pd.DataFrame | None:
        return self.latest.get(asset)


@dataclass
class InferenceService:
    model: DualTaskModel
    targets: ModelTargets
    cache: FeatureCache = field(default_factory=FeatureCache)
    monitor: MetricsMonitor = field(default_factory=MetricsMonitor)

    def predict(self, asset: str, market_frame: pd.DataFrame) -> dict[str, Any]:
        started = time.perf_counter()

        try:
            features = make_feature_table(market_frame)
            self.cache.put(asset, features)
        except Exception:
            self.monitor.failovers += 1
            features = self.cache.get(asset)
            if features is None:
                raise

        latest = features.tail(1)
        direction, confidence, future_price = self.model.predict(latest)

        latency_ms = (time.perf_counter() - started) * 1000
        self.monitor.record_latency(latency_ms)

        passes_confidence = float(confidence[0]) >= self.targets.confidence_threshold
        return {
            "asset": asset,
            "direction": int(direction[0]),
            "confidence": float(confidence[0]),
            "future_price": float(future_price[0]),
            "prediction_accepted": passes_confidence,
            "latency_ms": latency_ms,
            "latency_within_budget": latency_ms <= self.targets.max_inference_latency_ms,
            "monitor": self.monitor.snapshot(),
        }
