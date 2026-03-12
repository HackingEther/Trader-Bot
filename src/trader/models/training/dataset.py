"""Build real training datasets from historical market bars."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from trader.core.events import BarEvent
from trader.features.engine import FeatureEngine
from trader.features.registry import get_feature_names
from trader.models.training.labels import (
    CostParams,
    TradeOutcome,
    compute_net_pnl_bps,
    compute_stop_target,
    simulate_trade_outcome,
)


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
            regime_labels.append(infer_regime(features))
            timestamps.append(event.timestamp)

    return TrainingDataset(
        features=np.asarray(feature_rows, dtype=np.float64),
        direction_labels=np.asarray(direction_labels, dtype=object),
        magnitude_labels=np.asarray(magnitude_labels, dtype=np.float64),
        filter_labels=np.asarray(filter_labels, dtype=np.int64),
        regime_labels=np.asarray(regime_labels, dtype=object),
        timestamps=timestamps,
    )


@dataclass
class TradableDataset:
    """Dataset with cost-aware tradable outcome labels."""

    features: np.ndarray
    direction_labels: np.ndarray
    magnitude_labels: np.ndarray
    filter_labels: np.ndarray
    regime_labels: np.ndarray
    timestamps: list


def build_tradable_dataset(
    bars_by_symbol: dict[str, list[dict]],
    *,
    min_history: int = 60,
    max_hold_bars: int = 60,
    costs: CostParams | None = None,
    atr_multiple: float = 1.5,
    rr_ratio: float = 2.0,
) -> TradableDataset:
    """Create dataset with tradable outcome labels (post-cost, target/stop simulation).

    For each bar, simulates long and short trades with entry at next-bar open,
    volatility-based stop/target, and max hold. Labels reflect whether the trade
    would have been profitable after costs.
    """
    costs = costs or CostParams()
    feature_names = get_feature_names()
    feature_rows: list[list[float]] = []
    direction_labels: list[str] = []
    magnitude_labels: list[float] = []
    filter_labels: list[int] = []
    regime_labels: list[str] = []
    timestamps: list = []

    engine = FeatureEngine(max_bars=max(min_history + max_hold_bars + 10, 200))

    for symbol, bars in bars_by_symbol.items():
        engine.clear(symbol)
        sorted_bars = sorted(bars, key=lambda b: b["timestamp"])
        events = [BarEvent(**bar) if not isinstance(bar, BarEvent) else bar for bar in sorted_bars]
        for index, event in enumerate(events):
            engine.add_bar(event)
            if index + 1 < min_history:
                continue
            if index + 2 + max_hold_bars >= len(events):
                break

            features = engine.compute_features(symbol, event.timestamp)
            current_price = Decimal(str(event.close))
            vol = max(0.005, float(features.get("rolling_volatility_20", 0.01)))
            entry_bar = events[index + 1]
            entry_price = Decimal(str(entry_bar.open))

            direction = "no_trade"
            magnitude_bps = 0.0
            long_net = 0.0
            short_net = 0.0

            for side in ("buy", "sell"):
                stop, target = compute_stop_target(
                    entry_price, side, vol, atr_multiple, rr_ratio
                )
                outcome, exit_price = simulate_trade_outcome(
                    bars=events,
                    decision_bar_index=index,
                    side=side,
                    entry_price=entry_price,
                    stop_loss=stop,
                    take_profit=target,
                    max_hold_bars=max_hold_bars,
                )
                net_bps = compute_net_pnl_bps(side, entry_price, exit_price, costs)
                if outcome == TradeOutcome.WIN and net_bps > 0:
                    if side == "buy":
                        long_net = net_bps
                    else:
                        short_net = net_bps

            if long_net > 0 or short_net > 0:
                if long_net >= short_net:
                    direction = "long"
                    magnitude_bps = long_net
                else:
                    direction = "short"
                    magnitude_bps = short_net

            feature_rows.append([float(features.get(name, 0.0)) for name in feature_names])
            direction_labels.append(direction)
            magnitude_labels.append(magnitude_bps)
            filter_labels.append(1 if direction == "no_trade" else 0)
            regime_labels.append(infer_regime(features))
            timestamps.append(event.timestamp)

    return TradableDataset(
        features=np.asarray(feature_rows, dtype=np.float64),
        direction_labels=np.asarray(direction_labels, dtype=object),
        magnitude_labels=np.asarray(magnitude_labels, dtype=np.float64),
        filter_labels=np.asarray(filter_labels, dtype=np.int64),
        regime_labels=np.asarray(regime_labels, dtype=object),
        timestamps=timestamps,
    )


def infer_regime(features: dict[str, float]) -> str:
    vol = features.get("rolling_volatility_20", 0.01)
    mom = features.get("momentum_15m", 0.0)
    if vol > 0.02:
        return "high_volatility"
    if mom > 0.003:
        return "trending_up"
    if mom < -0.003:
        return "trending_down"
    if vol < 0.005:
        return "low_volatility"
    return "mean_reverting"
