"""Feature engine for computing intraday feature vectors per symbol."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import structlog

from trader.core.events import BarEvent
from trader.core.time_utils import minutes_since_open, session_fraction
from trader.features.indicators import (
    compute_returns,
    compute_vwap,
    distance_from_vwap,
    momentum,
    opening_range,
    orb_breakout,
    relative_volume,
    rolling_volatility,
    rolling_zscore,
    time_bucket,
)
from trader.features.registry import FEATURE_VERSION

logger = structlog.get_logger(__name__)


class FeatureEngine:
    """Computes rolling intraday features per symbol from bar data.

    Maintains an in-memory rolling window of bars per symbol and computes
    a feature vector on demand. Designed to be called from both live pipeline
    workers and backtest replay.
    """

    def __init__(self, max_bars: int = 400) -> None:
        self._max_bars = max_bars
        self._bars: dict[str, list[dict]] = {}

    def add_bar(self, event: BarEvent) -> None:
        """Add a bar event to the rolling window for its symbol."""
        if event.symbol not in self._bars:
            self._bars[event.symbol] = []
        self._bars[event.symbol].append(event.model_dump())
        if len(self._bars[event.symbol]) > self._max_bars:
            self._bars[event.symbol] = self._bars[event.symbol][-self._max_bars:]

    def add_bars_bulk(self, symbol: str, bars: list[dict]) -> None:
        """Add multiple bars at once (for backtest initialization)."""
        self._bars[symbol] = bars[-self._max_bars:]

    def compute_features(self, symbol: str, timestamp: datetime | None = None) -> dict:
        """Compute the full feature vector for a symbol.

        Returns a dict of feature_name -> float suitable for model input and DB storage.
        """
        bars = self._bars.get(symbol, [])
        if len(bars) < 2:
            logger.debug("insufficient_bars", symbol=symbol, count=len(bars))
            return self._empty_features(timestamp)

        df = pd.DataFrame(bars)
        for col in ("open", "high", "low", "close", "vwap"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].astype(float)

        ret_1m = compute_returns(close, 1)
        ret_5m = compute_returns(close, 5)
        ret_15m = compute_returns(close, 15)

        vol_20 = rolling_volatility(ret_1m, 20)
        vol_60 = rolling_volatility(ret_1m, 60)

        vwap_col = df.get("vwap")
        if vwap_col is not None and vwap_col.notna().any():
            vwap_series = pd.to_numeric(vwap_col, errors="coerce").fillna(close)
        else:
            vwap_series = compute_vwap(high, low, close, volume)

        dist_vwap = distance_from_vwap(close, vwap_series)
        rel_vol = relative_volume(volume, window=20)

        orb_h, orb_l = opening_range(high, low, orb_bars=15)
        brk_up, brk_down = orb_breakout(close, orb_h, orb_l)

        mom_5 = momentum(close, 5)
        mom_15 = momentum(close, 15)

        zs_close = rolling_zscore(close, 20)
        zs_vol = rolling_zscore(volume, 20)

        ts = timestamp or datetime.now()
        mins = minutes_since_open(ts)
        sess_frac = session_fraction(ts)
        t_bucket = time_bucket(mins, bucket_size=30)

        last_idx = len(df) - 1
        features = {
            "return_1m": float(ret_1m.iloc[last_idx]),
            "return_5m": float(ret_5m.iloc[last_idx]),
            "return_15m": float(ret_15m.iloc[last_idx]),
            "rolling_volatility_20": float(vol_20.iloc[last_idx]),
            "rolling_volatility_60": float(vol_60.iloc[last_idx]),
            "vwap": float(vwap_series.iloc[last_idx]),
            "distance_from_vwap": float(dist_vwap.iloc[last_idx]),
            "relative_volume": float(rel_vol.iloc[last_idx]),
            "orb_high": float(orb_h.iloc[last_idx]),
            "orb_low": float(orb_l.iloc[last_idx]),
            "orb_breakout_up": float(brk_up.iloc[last_idx]),
            "orb_breakout_down": float(brk_down.iloc[last_idx]),
            "momentum_5m": float(mom_5.iloc[last_idx]),
            "momentum_15m": float(mom_15.iloc[last_idx]),
            "zscore_close_20": float(zs_close.iloc[last_idx]),
            "zscore_volume_20": float(zs_vol.iloc[last_idx]),
            "spread_bps": 0.0,
            "session_fraction": sess_frac,
            "minutes_since_open": mins,
            "time_bucket": float(t_bucket),
        }

        for k, v in features.items():
            if np.isnan(v) or np.isinf(v):
                features[k] = 0.0

        return features

    def get_bar_count(self, symbol: str) -> int:
        return len(self._bars.get(symbol, []))

    def get_latest_bar(self, symbol: str) -> dict | None:
        bars = self._bars.get(symbol, [])
        return bars[-1] if bars else None

    def clear(self, symbol: str | None = None) -> None:
        if symbol:
            self._bars.pop(symbol, None)
        else:
            self._bars.clear()

    def _empty_features(self, timestamp: datetime | None = None) -> dict:
        ts = timestamp or datetime.now()
        return {
            "return_1m": 0.0,
            "return_5m": 0.0,
            "return_15m": 0.0,
            "rolling_volatility_20": 0.0,
            "rolling_volatility_60": 0.0,
            "vwap": 0.0,
            "distance_from_vwap": 0.0,
            "relative_volume": 1.0,
            "orb_high": 0.0,
            "orb_low": 0.0,
            "orb_breakout_up": 0.0,
            "orb_breakout_down": 0.0,
            "momentum_5m": 0.0,
            "momentum_15m": 0.0,
            "zscore_close_20": 0.0,
            "zscore_volume_20": 0.0,
            "spread_bps": 0.0,
            "session_fraction": session_fraction(ts),
            "minutes_since_open": minutes_since_open(ts),
            "time_bucket": float(time_bucket(minutes_since_open(ts))),
        }

    @property
    def feature_version(self) -> str:
        return FEATURE_VERSION
