from __future__ import annotations

import json

from .config import ModelTargets
from .data import UnifiedDataPipeline
from .pipeline import TradingMLSystem
from .providers import SyntheticProvider


def run() -> dict:
    targets = ModelTargets()
    provider = SyntheticProvider()
    data_pipeline = UnifiedDataPipeline(primary=provider, fallback=provider)
    system = TradingMLSystem(targets=targets, data_pipeline=data_pipeline)
    results = system.train_all_assets()
    return results


if __name__ == "__main__":
    output = run()
    printable = {
        asset: {
            "metrics": result["metrics"],
            "backtest": result["backtest"].__dict__,
            "latest_paper_prediction": {
                k: v
                for k, v in result["latest_paper_prediction"].items()
                if k not in {"monitor"}
            },
        }
        for asset, result in output.items()
    }
    print(json.dumps(printable, indent=2, default=str))
