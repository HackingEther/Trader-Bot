"""Individual indicator calculation functions operating on pandas Series/DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(close: pd.Series, periods: int) -> pd.Series:
    """Compute log returns over N periods."""
    return np.log(close / close.shift(periods)).fillna(0.0)


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Compute rolling standard deviation of returns."""
    return returns.rolling(window=window, min_periods=max(1, window // 2)).std().fillna(0.0)


def compute_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute cumulative VWAP for the session."""
    typical_price = (high + low + close) / 3.0
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap.fillna(close)


def distance_from_vwap(close: pd.Series, vwap: pd.Series) -> pd.Series:
    """Compute distance from VWAP in basis points."""
    return ((close - vwap) / vwap * 10000).fillna(0.0)


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Compute relative volume vs rolling average."""
    avg_vol = volume.rolling(window=window, min_periods=1).mean()
    return (volume / avg_vol.replace(0, np.nan)).fillna(1.0)


def opening_range(high: pd.Series, low: pd.Series, orb_bars: int = 15) -> tuple[pd.Series, pd.Series]:
    """Compute opening range high and low (first N bars)."""
    orb_high = high.iloc[:orb_bars].max() if len(high) >= orb_bars else high.max()
    orb_low = low.iloc[:orb_bars].min() if len(low) >= orb_bars else low.min()
    orb_h = pd.Series(orb_high, index=high.index)
    orb_l = pd.Series(orb_low, index=low.index)
    return orb_h, orb_l


def orb_breakout(close: pd.Series, orb_high: pd.Series, orb_low: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Compute ORB breakout flags (1 if broken, 0 otherwise)."""
    breakout_up = (close > orb_high).astype(float)
    breakout_down = (close < orb_low).astype(float)
    return breakout_up, breakout_down


def momentum(close: pd.Series, periods: int) -> pd.Series:
    """Compute price momentum (percentage change over N periods)."""
    return close.pct_change(periods=periods).fillna(0.0)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score."""
    mean = series.rolling(window=window, min_periods=max(1, window // 2)).mean()
    std = series.rolling(window=window, min_periods=max(1, window // 2)).std()
    zscore = (series - mean) / std.replace(0, np.nan)
    return zscore.fillna(0.0)


def spread_bps(bid: pd.Series | None, ask: pd.Series | None, mid: pd.Series | None = None) -> pd.Series:
    """Compute bid-ask spread in basis points."""
    if bid is None or ask is None:
        if mid is not None:
            return pd.Series(0.0, index=mid.index)
        return pd.Series(dtype=float)
    midpoint = (bid + ask) / 2.0
    spread = (ask - bid) / midpoint.replace(0, np.nan) * 10000
    return spread.fillna(0.0)


def time_bucket(minutes_since_open: float, bucket_size: int = 30) -> int:
    """Encode time of day into discrete buckets."""
    return int(minutes_since_open // bucket_size)
