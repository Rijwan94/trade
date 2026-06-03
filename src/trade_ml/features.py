from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            (frame["high"] - frame["low"]).abs(),
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _volatility(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    return frame["close"].pct_change().rolling(window, min_periods=window).std()


def _supply_demand(frame: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    rolling_high = frame["high"].rolling(lookback, min_periods=lookback).max()
    rolling_low = frame["low"].rolling(lookback, min_periods=lookback).min()
    return pd.DataFrame(
        {
            "dist_to_supply": (rolling_high - frame["close"]) / frame["close"],
            "dist_to_demand": (frame["close"] - rolling_low) / frame["close"],
        }
    )


def _candle_patterns(frame: pd.DataFrame) -> pd.DataFrame:
    body = frame["close"] - frame["open"]
    candle_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    upper_shadow = frame["high"] - frame[["open", "close"]].max(axis=1)
    lower_shadow = frame[["open", "close"]].min(axis=1) - frame["low"]

    hammer = ((lower_shadow > 2 * body.abs()) & (upper_shadow < body.abs())).astype(int)
    shooting_star = ((upper_shadow > 2 * body.abs()) & (lower_shadow < body.abs())).astype(int)
    doji = (body.abs() / candle_range < 0.1).astype(int)

    return pd.DataFrame(
        {
            "hammer": hammer,
            "shooting_star": shooting_star,
            "doji": doji,
        }
    )


def _chart_patterns(frame: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    sma_fast = frame["close"].rolling(10, min_periods=10).mean()
    sma_slow = frame["close"].rolling(window, min_periods=window).mean()
    trend_strength = (sma_fast - sma_slow) / frame["close"]
    breakout_up = (frame["close"] > frame["high"].rolling(window, min_periods=window).max().shift(1)).astype(int)
    breakout_down = (frame["close"] < frame["low"].rolling(window, min_periods=window).min().shift(1)).astype(int)

    return pd.DataFrame(
        {
            "trend_strength": trend_strength,
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
        }
    )


def make_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    # All features only use t and t-1... history, never future rows.
    out = frame.copy()
    out["atr_14"] = _atr(out, 14)
    out["volatility_20"] = _volatility(out, 20)
    out["return_1"] = out["close"].pct_change(1)
    out["return_5"] = out["close"].pct_change(5)
    out["volume_z20"] = (out["volume"] - out["volume"].rolling(20, min_periods=20).mean()) / out["volume"].rolling(20, min_periods=20).std()
    out = pd.concat([out, _supply_demand(out), _candle_patterns(out), _chart_patterns(out)], axis=1)
    return out.dropna().reset_index(drop=True)
