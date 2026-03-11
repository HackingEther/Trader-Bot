"""Build real training datasets from historical market bars."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from trader.core.events import BarEvent
from trader.features.engine import FeatureEngine
from trader.features.registry import get_feature_names


@dataclass
class TrainingDataset:
    features: np.ndarray
    direction_labels: np.ndarray
    magnitude_labels: np.ndarray
    filter_labels: np.ndarray
    regime_labels: np.ndarray
    timestamps: list


def build_momentum_dataset(
    bars_by_symbol: dict[str, list[dict]],
    *,
    lookahead_bars: int = 15,
    min_history: int = 60,
    net_move_threshold_bps: float = 12.0,
    cost_estimate_bps: float = 4.0,
) -> TrainingDataset:
    """Create a narrow momentum/continuation dataset from real bars."""
    feature_names = get_feature_names()
    feature_rows: list[list[float]] = []
    direction_labels: list[str] = []
    magnitude_labels: list[float] = []
    filter_labels: list[int] = []
    regime_labels: list[str] = []
    timestamps: list = []

    engine = FeatureEngine(max_bars=max(min_history + lookahead_bars + 5, 200))
    net_threshold = net_move_threshold_bps + cost_estimate_bps

    for symbol, bars in bars_by_symbol.items():
        engine.clear(symbol)
        sorted_bars = sorted(bars, key=lambda b: b["timestamp"])
        events = [BarEvent(**bar) if not isinstance(bar, BarEvent) else bar for bar in sorted_bars]
        for index, event in enumerate(events):
            engine.add_bar(event)
            if index + 1 < min_history:
                continue
            future_index = index + lookahead_bars
            if future_index >= len(events):
                break

            features = engine.compute_features(symbol, event.timestamp)
            current_price = Decimal(str(event.close))
            future_price = Decimal(str(events[future_index].close))
            move_bps = float((future_price - current_price) / current_price * Decimal("10000"))

            momentum_ok_long = features["momentum_5m"] > 0 and features["distance_from_vwap"] >= 0
            momentum_ok_short = features["momentum_5m"] < 0 and features["distance_from_vwap"] <= 0
            if move_bps >= net_threshold and momentum_ok_long:
                direction = "long"
            elif move_bps <= -net_threshold and momentum_ok_short:
                direction = "short"
            else:
                direction = "no_trade"

            feature_rows.append([float(features.get(name, 0.0)) for name in feature_names])
            direction_labels.append(direction)
            magnitude_labels.append(abs(move_bps))
            filter_labels.append(1 if direction == "no_trade" else 0)
            regime_labels.append(_infer_regime(features))
            timestamps.append(event.timestamp)

    return TrainingDataset(
        features=np.asarray(feature_rows, dtype=np.float64),
        direction_labels=np.asarray(direction_labels, dtype=object),
        magnitude_labels=np.asarray(magnitude_labels, dtype=np.float64),
        filter_labels=np.asarray(filter_labels, dtype=np.int64),
        regime_labels=np.asarray(regime_labels, dtype=object),
        timestamps=timestamps,
    )


def _infer_regime(features: dict[str, float]) -> str:
    if features["rolling_volatility_20"] > 0.02:
        return "high_volatility"
    if features["momentum_15m"] > 0.003:
        return "trending_up"
    if features["momentum_15m"] < -0.003:
        return "trending_down"
    if features["rolling_volatility_20"] < 0.005:
        return "low_volatility"
    return "mean_reverting"
