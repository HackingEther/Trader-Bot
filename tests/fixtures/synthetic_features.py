"""Synthetic feature data generators for testing."""

from __future__ import annotations

import random


def generate_synthetic_features(seed: int = 42) -> dict[str, float]:
    """Generate a deterministic feature vector matching the feature registry."""
    rng = random.Random(seed)
    return {
        "return_1m": rng.gauss(0, 0.001),
        "return_5m": rng.gauss(0, 0.003),
        "return_15m": rng.gauss(0, 0.005),
        "rolling_volatility_20": abs(rng.gauss(0.01, 0.005)),
        "rolling_volatility_60": abs(rng.gauss(0.012, 0.005)),
        "vwap": 150.0 + rng.gauss(0, 2),
        "distance_from_vwap": rng.gauss(0, 30),
        "relative_volume": max(0.1, rng.gauss(1.0, 0.3)),
        "orb_high": 152.0 + rng.gauss(0, 1),
        "orb_low": 148.0 + rng.gauss(0, 1),
        "orb_breakout_up": float(rng.random() > 0.7),
        "orb_breakout_down": float(rng.random() > 0.7),
        "momentum_5m": rng.gauss(0, 0.003),
        "momentum_15m": rng.gauss(0, 0.005),
        "zscore_close_20": rng.gauss(0, 1),
        "zscore_volume_20": rng.gauss(0, 1),
        "spread_bps": abs(rng.gauss(5, 3)),
        "session_fraction": rng.random(),
        "minutes_since_open": rng.uniform(0, 390),
        "time_bucket": float(rng.randint(0, 12)),
    }
