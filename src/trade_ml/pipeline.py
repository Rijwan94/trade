from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .backtest import run_backtest
from .config import ModelTargets
from .data import UnifiedDataPipeline
from .evaluation import evaluate_walk_forward
from .features import make_feature_table
from .inference import InferenceService
from .model import DualTaskModel
from .paper import DriftMonitor, PaperTrader


@dataclass
class TradingMLSystem:
    targets: ModelTargets
    data_pipeline: UnifiedDataPipeline

    def train_for_asset(self, asset: str) -> dict[str, Any]:
        raw = self.data_pipeline.load_historical(asset)
        features = make_feature_table(raw)

        model = DualTaskModel()
        model.fit(features, self.targets.horizon_bars)

        metrics = evaluate_walk_forward(features, self.targets.horizon_bars)
        backtest = run_backtest(features, self.targets.horizon_bars, self.targets.confidence_threshold)

        service = InferenceService(model=model, targets=self.targets)
        paper = PaperTrader(service=service, drift_monitor=DriftMonitor(window=self.targets.drift_window))

        latest_live = self.data_pipeline.load_live(asset)
        prediction = paper.on_new_market_data(asset, latest_live)

        return {
            "asset": asset,
            "targets": self.targets,
            "metrics": metrics,
            "backtest": backtest,
            "latest_paper_prediction": prediction,
        }

    def train_all_assets(self) -> dict[str, dict[str, Any]]:
        return {asset: self.train_for_asset(asset) for asset in self.targets.assets}
