from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticProvider:
    rows: int = 500
    seed: int = 7

    def _make(self, asset: str) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + hash(asset) % 1000)
        ts = pd.date_range("2024-01-01", periods=self.rows, freq="5min")
        base = 2000 if asset == "XAUUSD" else 25
        noise = rng.normal(0, base * 0.0007, size=self.rows).cumsum()
        close = base + noise
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        spread = np.abs(rng.normal(base * 0.0004, base * 0.0002, size=self.rows))
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        volume = rng.integers(100, 2000, size=self.rows)
        return pd.DataFrame(
            {
                "timestamp": ts,
                "asset": asset,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def get_historical(self, asset: str, start=None, end=None) -> pd.DataFrame:
        frame = self._make(asset)
        if start is not None:
            frame = frame[frame["timestamp"] >= start]
        if end is not None:
            frame = frame[frame["timestamp"] <= end]
        return frame.reset_index(drop=True)

    def get_live(self, asset: str) -> pd.DataFrame:
        return self._make(asset).tail(250).reset_index(drop=True)
