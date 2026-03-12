"""Unit tests for tradable outcome label construction."""

from __future__ import annotations

from decimal import Decimal

from trader.core.events import BarEvent
from trader.models.training.labels import (
    CostParams,
    TradeOutcome,
    compute_net_pnl_bps,
    compute_stop_target,
    simulate_trade_outcome,
)


def _bar(open_p: float, high: float, low: float, close: float, ts_offset: int = 0) -> BarEvent:
    from datetime import datetime, timezone, timedelta
    base = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    return BarEvent(
        symbol="TEST",
        timestamp=base + timedelta(minutes=ts_offset),
        open=Decimal(str(open_p)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=1000,
        vwap=Decimal(str((high + low + close) / 3)),
    )


class TestSimulateTradeOutcome:
    def test_long_target_hit(self) -> None:
        stop, target = compute_stop_target(Decimal("100"), "buy", 0.01, 1.5, 2.0)
        bars = [
            _bar(100, 100, 100, 100, 0),
            _bar(100, 100, 100, 100, 1),
            _bar(100, float(target) + 1, 99, float(target) + 0.5, 2),
        ]
        outcome, exit_price = simulate_trade_outcome(
            bars, 0, "buy", Decimal("100"), stop, target, max_hold_bars=5
        )
        assert outcome == TradeOutcome.WIN
        assert exit_price == target

    def test_long_stop_hit(self) -> None:
        stop, target = compute_stop_target(Decimal("100"), "buy", 0.01, 1.5, 2.0)
        bars = [
            _bar(100, 100, 100, 100, 0),
            _bar(100, 100, 100, 100, 1),
            _bar(100, 101, float(stop) - 0.5, 99, 2),
        ]
        outcome, exit_price = simulate_trade_outcome(
            bars, 0, "buy", Decimal("100"), stop, target, max_hold_bars=5
        )
        assert outcome == TradeOutcome.LOSS
        assert exit_price == stop

    def test_short_target_hit(self) -> None:
        stop, target = compute_stop_target(Decimal("100"), "sell", 0.01, 1.5, 2.0)
        bars = [
            _bar(100, 100, 100, 100, 0),
            _bar(100, 100, 100, 100, 1),
            _bar(100, 101, float(target) - 0.5, float(target), 2),
        ]
        outcome, exit_price = simulate_trade_outcome(
            bars, 0, "sell", Decimal("100"), stop, target, max_hold_bars=5
        )
        assert outcome == TradeOutcome.WIN
        assert exit_price == target

    def test_timeout_exits_at_close(self) -> None:
        stop, target = compute_stop_target(Decimal("100"), "buy", 0.0001, 1.5, 2.0)
        bars = [
            _bar(100, 100, 100, 100, 0),
            _bar(100, 100, 100, 100, 1),
            _bar(100, 100.01, 99.99, 100, 2),
        ]
        outcome, exit_price = simulate_trade_outcome(
            bars, 0, "buy", Decimal("100"), stop, target, max_hold_bars=1
        )
        assert outcome == TradeOutcome.TIMEOUT
        assert exit_price == bars[2].close


class TestComputeNetPnlBps:
    def test_long_win_after_costs(self) -> None:
        costs = CostParams(spread_bps=5, slippage_entry_bps=5, slippage_exit_bps=5)
        net = compute_net_pnl_bps("buy", Decimal("100"), Decimal("100.5"), costs)
        assert net > 0

    def test_short_win_after_costs(self) -> None:
        costs = CostParams(spread_bps=5, slippage_entry_bps=5, slippage_exit_bps=5)
        net = compute_net_pnl_bps("sell", Decimal("100"), Decimal("99.5"), costs)
        assert net > 0

    def test_cost_subtraction(self) -> None:
        costs = CostParams(spread_bps=0, slippage_entry_bps=0, slippage_exit_bps=0)
        net_zero = compute_net_pnl_bps("buy", Decimal("100"), Decimal("100.1"), costs)
        costs_with = CostParams(spread_bps=5, slippage_entry_bps=5, slippage_exit_bps=5)
        net_with = compute_net_pnl_bps("buy", Decimal("100"), Decimal("100.1"), costs_with)
        assert net_with < net_zero
        assert net_with == net_zero - costs_with.total_round_trip_bps


class TestComputeStopTarget:
    def test_buy_stop_below_entry(self) -> None:
        stop, target = compute_stop_target(Decimal("100"), "buy", 0.01, 1.5, 2.0)
        assert stop < Decimal("100")
        assert target > Decimal("100")

    def test_sell_stop_above_entry(self) -> None:
        stop, target = compute_stop_target(Decimal("100"), "sell", 0.01, 1.5, 2.0)
        assert stop > Decimal("100")
        assert target < Decimal("100")

    def test_zero_volatility_clamped(self) -> None:
        stop, target = compute_stop_target(Decimal("100"), "buy", 0.0, 1.5, 2.0)
        assert stop < Decimal("100")
        assert target > Decimal("100")
