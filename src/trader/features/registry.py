"""Feature registry for naming and versioning feature columns."""

from __future__ import annotations

FEATURE_VERSION = "v1"

FEATURE_COLUMNS: list[str] = [
    "return_1m",
    "return_5m",
    "return_15m",
    "rolling_volatility_20",
    "rolling_volatility_60",
    "vwap",
    "distance_from_vwap",
    "relative_volume",
    "orb_high",
    "orb_low",
    "orb_breakout_up",
    "orb_breakout_down",
    "momentum_5m",
    "momentum_15m",
    "zscore_close_20",
    "zscore_volume_20",
    "spread_bps",
    "session_fraction",
    "minutes_since_open",
    "time_bucket",
]


def get_feature_names() -> list[str]:
    """Return the list of feature column names."""
    return FEATURE_COLUMNS.copy()
