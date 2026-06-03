import unittest

from trade_ml.config import ModelTargets
from trade_ml.data import UnifiedDataPipeline
from trade_ml.pipeline import TradingMLSystem
from trade_ml.providers import SyntheticProvider


class TestPipeline(unittest.TestCase):
    def test_train_all_assets(self):
        targets = ModelTargets(horizon_bars=3)
        provider = SyntheticProvider(rows=300)
        pipeline = UnifiedDataPipeline(primary=provider, fallback=provider)
        system = TradingMLSystem(targets=targets, data_pipeline=pipeline)

        results = system.train_all_assets()
        self.assertIn("XAUUSD", results)
        self.assertIn("XAGUSD", results)

        xau = results["XAUUSD"]
        self.assertIn("metrics", xau)
        self.assertIn("f1", xau["metrics"])
        self.assertIn("backtest", xau)
        self.assertGreaterEqual(xau["latest_paper_prediction"]["confidence"], 0.0)
        self.assertLessEqual(xau["latest_paper_prediction"]["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
