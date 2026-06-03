from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "asset", "open", "high", "low", "close", "volume"]


class DataProvider(Protocol):
    def get_historical(self, asset: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame: ...

    def get_live(self, asset: str) -> pd.DataFrame: ...


@dataclass
class DataAuditor:
    required_columns: tuple[str, ...] = tuple(REQUIRED_COLUMNS)

    def validate(self, frame: pd.DataFrame) -> None:
        missing = set(self.required_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Input market data frame is empty")
        if not frame["timestamp"].is_monotonic_increasing:
            raise ValueError("timestamp must be sorted ascending")
        if frame[list(self.required_columns)].isna().any().any():
            raise ValueError("Market data contains null values in required columns")


@dataclass
class UnifiedDataPipeline:
    primary: DataProvider
    fallback: Optional[DataProvider] = None
    auditor: DataAuditor = field(default_factory=DataAuditor)

    def load_historical(
        self,
        asset: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        frame = self._call_with_failover("historical", asset, start, end)
        self.auditor.validate(frame)
        return frame.reset_index(drop=True)

    def load_live(self, asset: str) -> pd.DataFrame:
        frame = self._call_with_failover("live", asset)
        self.auditor.validate(frame)
        return frame.reset_index(drop=True)

    def _call_with_failover(self, mode: str, asset: str, *args) -> pd.DataFrame:
        try:
            if mode == "historical":
                return self.primary.get_historical(asset, *args)
            return self.primary.get_live(asset)
        except Exception:
            if self.fallback is None:
                raise
            if mode == "historical":
                return self.fallback.get_historical(asset, *args)
            return self.fallback.get_live(asset)
