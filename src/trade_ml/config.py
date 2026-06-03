from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ModelTargets:
    assets: Tuple[str, ...] = ("XAUUSD", "XAGUSD")
    horizon_bars: int = 5
    max_inference_latency_ms: int = 50
    min_direction_f1: float = 0.62
    max_price_rmse: float = 1.5
    confidence_threshold: float = 0.60
    drift_window: int = 500
    extra: dict = field(default_factory=dict)
