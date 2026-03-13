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

    def test_short_with_negative_expected_move_uses_abs_for_threshold(self) -> None:
        """Negative expected_move_bps for shorts should pass when magnitude meets threshold."""
        pred = _make_prediction(
            direction="short",
            expected_move_bps=-25.0,
            regime="trending_down",
        )
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

    def test_skips_high_no_trade_score(self) -> None:
        pred = _make_prediction(no_trade_score=0.9)
        intent = self.engine.evaluate(pred, Decimal("150.00"), features=_make_features())
        assert intent is None

    def test_no_trade_score_soft_penalty_not_hard_block(self) -> None:
        """no_trade_score 0.6 passes when hard_veto=0.85 (soft penalty only)."""
        pred = _make_prediction(no_trade_score=0.6)
        intent = self.engine.evaluate(pred, Decimal("150.00"), features=_make_features())
        assert intent is not None

    def test_no_trade_score_hard_veto_extreme(self) -> None:
        """no_trade_score 0.9 still blocked when hard_veto=0.85."""
        pred = _make_prediction(no_trade_score=0.9)
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

    def test_build_limit_price_applies_buy_and_sell_buffers(self) -> None:
        buy_price = self.engine._build_limit_price(Decimal("100.00"), "buy", 20.0)
        sell_price = self.engine._build_limit_price(Decimal("100.00"), "sell", 20.0)

        assert buy_price == Decimal("100.15")
        assert sell_price == Decimal("99.85")

    def test_build_limit_price_returns_none_for_non_positive_price(self) -> None:
        assert self.engine._build_limit_price(Decimal("0"), "buy", 10.0) is None
        assert self.engine._build_limit_price(Decimal("-1"), "sell", 10.0) is None

    def test_select_playbook_orb_cutoff_falls_back_after_first_hour(self) -> None:
        playbook, _, _ = self.engine._select_playbook(
            prediction=_make_prediction(direction="long", regime="trending_up"),
            features=_make_features(minutes_since_open=61.0),
            spread_bps=10.0,
        )

        assert playbook is not None
        assert playbook["name"] == "vwap_continuation"

    def test_select_playbook_rejects_trend_setup_when_regime_and_momentum_mismatch(self) -> None:
        """No playbook when regime is mean_reverting and momentum does not align (e.g. below vwap for long)."""
        playbook, _, _ = self.engine._select_playbook(
            prediction=_make_prediction(direction="long", regime="mean_reverting"),
            features=_make_features(
                minutes_since_open=61.0,
                distance_from_vwap=-35.0,
                momentum_5m=0.01,
                momentum_15m=0.02,
            ),
            spread_bps=10.0,
        )

        assert playbook is None

    def test_select_playbook_requires_orb_volume_threshold(self) -> None:
        playbook, _, _ = self.engine._select_playbook(
            prediction=_make_prediction(direction="long", regime="trending_up"),
            features=_make_features(relative_volume=1.19),
            spread_bps=10.0,
        )

        assert playbook is not None
        assert playbook["name"] == "vwap_continuation"

    def test_playbook_fit_scoring_selects_best(self) -> None:
        """Best fit above threshold is selected."""
        playbook, fit, _ = self.engine._select_playbook(
            prediction=_make_prediction(direction="long", regime="trending_up"),
            features=_make_features(minutes_since_open=45.0),
            spread_bps=10.0,
        )
        assert playbook is not None
        assert playbook["name"] == "orb_continuation"
        assert fit >= 0.4

    def test_playbook_fit_below_threshold_returns_none(self) -> None:
        """All fits < min_playbook_fit -> no_playbook."""
        engine = StrategyEngine(
            universe=self.universe,
            sizer=self.sizer,
            min_confidence=0.6,
            min_expected_move_bps=15.0,
            min_relative_volume=0.0,
            min_playbook_fit=0.9,
        )
        playbook, fit, _ = engine._select_playbook(
            prediction=_make_prediction(direction="long", regime="mean_reverting"),
            features=_make_features(
                minutes_since_open=90.0,
                zscore_close_20=0.0,
                distance_from_vwap=15.0,
            ),
            spread_bps=10.0,
        )
        assert playbook is None
        assert fit < 0.9

    def test_block_stats_tracked_when_enabled(self) -> None:
        engine = StrategyEngine(
            universe=self.universe,
            sizer=self.sizer,
            min_confidence=0.6,
            min_expected_move_bps=15.0,
            min_relative_volume=0.0,
            track_block_reasons=True,
        )
        assert engine.get_block_stats() == {}
        engine.evaluate(_make_prediction(no_trade_score=0.9), Decimal("150.00"), features=_make_features())
        engine.evaluate(_make_prediction(no_trade_score=0.9), Decimal("150.00"), features=_make_features())
        engine.evaluate(_make_prediction(expected_move_bps=3.0), Decimal("150.00"), features=_make_features())
        stats = engine.get_block_stats()
        assert stats["filter"] == 2
        assert stats["low_move"] == 1
        engine.reset_block_stats()
        assert engine.get_block_stats() == {}

    def test_net_edge_threshold_used_when_enabled(self) -> None:
        """When magnitude_is_net_edge=True, use min_expected_net_edge_bps for move check."""
        engine_net = StrategyEngine(
            universe=self.universe,
            sizer=self.sizer,
            min_confidence=0.6,
            min_expected_move_bps=15.0,
            min_expected_net_edge_bps=8.0,
            magnitude_is_net_edge=True,
            min_relative_volume=0.0,
        )
        pred_10_bps = _make_prediction(expected_move_bps=10.0)
        intent = engine_net.evaluate(pred_10_bps, Decimal("150.00"), features=_make_features())
        assert intent is not None

        engine_raw = StrategyEngine(
            universe=self.universe,
            sizer=self.sizer,
            min_confidence=0.6,
            min_expected_move_bps=15.0,
            magnitude_is_net_edge=False,
            min_relative_volume=0.0,
        )
        intent_raw = engine_raw.evaluate(pred_10_bps, Decimal("150.00"), features=_make_features())
        assert intent_raw is None

    def test_playbook_partial_credit_low_vol_momentum(self) -> None:
        """vwap_continuation gets partial score (0.5) in low_vol/mean_rev when momentum aligns."""
        engine = StrategyEngine(
            universe=self.universe,
            sizer=self.sizer,
            min_confidence=0.6,
            min_expected_move_bps=15.0,
            min_relative_volume=0.0,
        )
        playbook, fit, _ = engine._select_playbook(
            prediction=_make_prediction(direction="long", regime="mean_reverting"),
            features=_make_features(
                minutes_since_open=90.0,
                momentum_5m=0.01,
                momentum_15m=0.02,
                distance_from_vwap=15.0,
                relative_volume=1.2,
                orb_breakout_up=0.0,
                orb_breakout_down=0.0,
            ),
            spread_bps=10.0,
        )
        assert playbook is not None
        assert playbook["name"] == "vwap_continuation"
        assert fit >= 0.4
