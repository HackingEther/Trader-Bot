"""Unit tests for backtest metrics and compute_metrics_from_trades."""

from __future__ import annotations

from decimal import Decimal

import numpy as np

from trader.backtest.metrics import (
    BacktestMetrics,
    compute_metrics,
    compute_metrics_from_trades,
)


class _MockTrade:
    def __init__(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        hold_minutes: float = 10.0,
    ) -> None:
        self.side = side
        self.entry_price = Decimal(str(entry_price))
        self.exit_price = Decimal(str(exit_price))
        self.pnl = pnl
        self._hold_minutes = hold_minutes

    @property
    def hold_minutes(self) -> float:
        return self._hold_minutes


def test_compute_metrics_from_trades_empty() -> None:
    result = compute_metrics_from_trades([], [100000.0], 100000.0)
    assert result.total_trades == 0
    assert result.long_trade_count == 0
    assert result.short_trade_count == 0
    assert result.loss_rate == 0.0
    assert result.average_net_pnl_bps == 0.0
    assert result.median_net_pnl_bps == 0.0


def test_compute_metrics_from_trades_long_short_counts() -> None:
    trades = [
        _MockTrade("buy", 100.0, 100.5, 50.0),
        _MockTrade("sell", 100.0, 99.5, 50.0),
        _MockTrade("buy", 100.0, 99.0, -100.0),
    ]
    result = compute_metrics_from_trades(trades, [100000.0, 100100.0, 100050.0, 100000.0], 100000.0)
    assert result.long_trade_count == 2
    assert result.short_trade_count == 1
    assert result.total_trades == 3
    assert result.loss_rate == 1 / 3


def test_compute_metrics_from_trades_pnl_bps() -> None:
    trades = [
        _MockTrade("buy", 100.0, 101.0, 100.0),
        _MockTrade("sell", 100.0, 99.0, 100.0),
    ]
    result = compute_metrics_from_trades(trades, [100000.0, 100100.0, 100200.0], 100000.0)
    assert result.average_net_pnl_bps == 100.0
    assert result.median_net_pnl_bps == 100.0


def test_compute_metrics_zero_trades() -> None:
    result = compute_metrics([], [], [], 100000.0)
    assert result.total_trades == 0
    assert result.win_rate == 0.0
    assert result.loss_rate == 0.0
    assert result.expectancy == 0.0
    assert result.profit_factor == 0.0
