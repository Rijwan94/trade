# trade

## ML trading system

This repository now includes a production-style baseline pipeline in `/tmp/workspace/Rijwan94/trade/src/trade_ml` for:

- Direction classification and future price regression for **XAUUSD** and **XAGUSD**
- Unified historical/live data loading with failover
- Feature engineering for OHLCV, ATR, volatility, supply-demand distance, candle and chart pattern proxies
- Time-series CV + walk-forward evaluation
- Risk filtering via confidence thresholds
- Low-latency inference service with feature cache and runtime monitoring
- Backtesting and paper-trading hooks with drift scoring

### Quick start

```bash
cd /tmp/workspace/Rijwan94/trade
python -m pip install -e .
python -m trade_ml.main
python -m unittest discover -s tests -v
```

> Note: the included provider is synthetic (`SyntheticProvider`). Replace with your broker/feed provider implementing `get_historical` and `get_live`.
