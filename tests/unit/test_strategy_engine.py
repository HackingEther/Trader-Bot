"""Unit tests for strategy engine."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from trader.core.events import ModelPrediction
from trader.strategy.engine import StrategyEngine
from trader.strategy.sizing import PositionSizer
from trader.strategy.universe import SymbolUniverse


def _make_prediction(
    symbol: str = "AAPL",
    direction: str = "long",
    confidence: float = 0.8,
    expected_move_bps: float = 25.0,
    no_trade_score: float = 0.2,
    regime: str = "trending_up",
) -> ModelPrediction:
    return ModelPrediction(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        direction=direction,
        confidence=confidence,
        expected_move_bps=expected_move_bps,
        expected_holding_minutes=30.0,
        no_trade_score=no_trade_score,
        regime=regime,
    )


def _make_features(**overrides: float) -> dict[str, float]:
    features = {
        "relative_volume": 1.2,
        "spread_bps": 10.0,
        "minutes_since_open": 45.0,
        "momentum_5m": 0.01,
        "momentum_15m": 0.02,
        "distance_from_vwap": 15.0,
        "zscore_close_20": 0.0,
        "orb_breakout_up": 1.0,
        "orb_breakout_down": 0.0,
    }
    features.update(overrides)
    return features


class TestStrategyEngine:
    def setup_method(self) -> None:
        self.universe = SymbolUniverse(["AAPL", "MSFT", "GOOGL"])
        self.sizer = PositionSizer()
        self.engine = StrategyEngine(
            universe=self.universe,
            sizer=self.sizer,
            min_confidence=0.6,
            min_expected_move_bps=15.0,
            min_relative_volume=0.8,
            max_spread_bps=50.0,
        )

    def test_generates_intent_for_good_signal(self) -> None:
        pred = _make_prediction(confidence=0.8, expected_move_bps=30.0)
        intent = self.engine.evaluate(pred, Decimal("150.00"), features=_make_features())
        assert intent is not None
        assert intent.symbol == "AAPL"
        assert intent.side == "buy"
        assert intent.qty > 0
        assert intent.entry_order_type == "limit"
        assert intent.strategy_tag in {"orb_continuation", "vwap_continuation"}

    def test_skips_no_trade_direction(self) -> None:
        pred = _make_prediction(direction="no_trade")
        intent = self.engine.evaluate(pred, Decimal("150.00"), features=_make_features())
        assert intent is None

    def test_skips_low_confidence(self) -> None:
        pred = _make_prediction(confidence=0.3)
        intent = self.engine.evaluate(pred, Decimal("150.00"), features=_make_features())
        assert intent is None

    def test_skips_low_expected_move(self) -> None:
        pred = _make_prediction(expected_move_bps=5.0)
        intent = self.engine.evaluate(pred, Decimal("150.00"), features=_make_features())
        assert intent is None

    def test_skips_high_no_trade_score(self) -> None:
        pred = _make_prediction(no_trade_score=0.8)
        intent = self.engine.evaluate(pred, Decimal("150.00"), features=_make_features())
        assert intent is None

    def test_skips_disabled_symbol(self) -> None:
        self.universe.disable("AAPL")
        pred = _make_prediction(symbol="AAPL")
        intent = self.engine.evaluate(pred, Decimal("150.00"), features=_make_features())
        assert intent is None

    def test_skips_existing_position(self) -> None:
        pred = _make_prediction()
        intent = self.engine.evaluate(pred, Decimal("150.00"), has_open_position=True, features=_make_features())
        assert intent is None

    def test_short_direction(self) -> None:
        pred = _make_prediction(direction="short", regime="trending_down")
        intent = self.engine.evaluate(
            pred,
            Decimal("150.00"),
            features=_make_features(
                momentum_5m=-0.01,
                momentum_15m=-0.02,
                distance_from_vwap=-15.0,
                orb_breakout_up=0.0,
                orb_breakout_down=1.0,
            ),
        )
        assert intent is not None
        assert intent.side == "sell"

    def test_intent_has_stops(self) -> None:
        pred = _make_prediction()
        intent = self.engine.evaluate(pred, Decimal("150.00"), features=_make_features())
        assert intent is not None
        assert intent.stop_loss is not None
        assert intent.take_profit is not None
        assert intent.stop_loss < Decimal("150.00")
        assert intent.take_profit > Decimal("150.00")

    def test_skips_low_relative_volume(self) -> None:
        pred = _make_prediction()
        intent = self.engine.evaluate(
            pred,
            Decimal("150.00"),
            features=_make_features(relative_volume=0.4),
        )
        assert intent is None

    def test_skips_wide_spread(self) -> None:
        pred = _make_prediction()
        intent = self.engine.evaluate(
            pred,
            Decimal("150.00"),
            features=_make_features(spread_bps=75.0),
            current_spread_bps=75.0,
        )
        assert intent is None

    def test_generates_reversion_playbook(self) -> None:
        pred = _make_prediction(direction="long", regime="mean_reverting")
        intent = self.engine.evaluate(
            pred,
            Decimal("150.00"),
            features=_make_features(
                minutes_since_open=90.0,
                momentum_5m=0.001,
                momentum_15m=0.0,
                distance_from_vwap=-35.0,
                zscore_close_20=-2.0,
                orb_breakout_up=0.0,
            ),
        )
        assert intent is not None
        assert intent.strategy_tag == "vwap_reversion"
