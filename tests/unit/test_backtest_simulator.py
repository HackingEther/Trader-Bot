"""Unit tests for the backtest simulator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trader.backtest.simulator import BacktestSimulator
from trader.backtest.slippage import CommissionModel, SlippageModel
from trader.core.events import ModelPrediction


class _SequencedEnsemble:
    def __init__(self, predictions: list[ModelPrediction]) -> None:
        self._predictions = predictions
        self._index = 0

    def predict(self, symbol: str, features: dict, timestamp: datetime | None = None) -> ModelPrediction:
        if self._index >= len(self._predictions):
            return ModelPrediction(
                symbol=symbol,
                timestamp=timestamp or datetime.now(timezone.utc),
                direction="no_trade",
                confidence=1.0,
                expected_move_bps=0.0,
                expected_holding_minutes=1.0,
                no_trade_score=1.0,
                regime="low_volatility",
            )
        prediction = self._predictions[self._index]
        self._index += 1
        return prediction.model_copy(update={"symbol": symbol, "timestamp": timestamp})


def _bars(start_price: float, moves: list[float]) -> list[dict]:
    start = datetime(2025, 1, 15, 9, 30, tzinfo=timezone.utc)
    bars: list[dict] = []
    price = start_price
    for idx, move in enumerate(moves):
        price += move
        bars.append(
            {
                "symbol": "AAPL",
                "timestamp": start + timedelta(minutes=idx),
                "open": Decimal(str(price - 0.2)),
                "high": Decimal(str(price + 0.3)),
                "low": Decimal(str(price - 0.3)),
                "close": Decimal(str(price)),
                "volume": 100_000,
                "vwap": Decimal(str(price)),
                "trade_count": 100,
                "interval": "1m",
            }
        )
    return bars


def _prediction(direction: str, regime: str) -> ModelPrediction:
    return ModelPrediction(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        direction=direction,
        confidence=0.9,
        expected_move_bps=40.0,
        expected_holding_minutes=1.0,
        no_trade_score=0.1,
        regime=regime,
    )


def test_backtest_waits_for_minimum_history() -> None:
    simulator = BacktestSimulator(
        symbols=["AAPL"],
        strategy_config={
            "min_history_bars": 5,
            "min_confidence": 0.0,
            "min_expected_move_bps": 0.0,
            "min_relative_volume": 0.0,
            "max_position_value": 101.0,
            "risk_per_trade_pct": 1.0,
        },
        slippage=SlippageModel(fixed_bps=0.0),
        commission=CommissionModel(),
        initial_capital=1000.0,
        ensemble=_SequencedEnsemble([_prediction("long", "trending_up")]),
    )

    results = simulator.run({"AAPL": _bars(100.0, [0.5, 0.5, 0.5, 0.5])})

    assert results["total_trades"] == 0
    assert simulator._positions == []


def test_backtest_long_cash_flow_matches_realized_pnl() -> None:
    long_bars = _bars(100.0, [0.2] * 15 + [1.0] * 11)
    simulator = BacktestSimulator(
        symbols=["AAPL"],
        strategy_config={
            "min_history_bars": 16,
            "min_confidence": 0.0,
            "min_expected_move_bps": 0.0,
            "min_relative_volume": 0.0,
            "max_position_value": 101.0,
            "risk_per_trade_pct": 1.0,
        },
        slippage=SlippageModel(fixed_bps=0.0),
        commission=CommissionModel(),
        initial_capital=1000.0,
        ensemble=_SequencedEnsemble(
            [
                _prediction("long", "trending_up"),
                _prediction("no_trade", "low_volatility"),
            ]
        ),
    )

    results = simulator.run({"AAPL": long_bars})

    assert results["total_trades"] == 1
    assert results["final_equity"] == pytest.approx(1000.0 + results["total_pnl"], abs=1e-6)
    assert simulator._cash == pytest.approx(results["final_equity"], abs=1e-6)


def test_backtest_short_cash_flow_matches_realized_pnl() -> None:
    short_bars = _bars(100.0, [-0.2] * 15 + [-1.0] * 11)
    simulator = BacktestSimulator(
        symbols=["AAPL"],
        strategy_config={
            "min_history_bars": 16,
            "min_confidence": 0.0,
            "min_expected_move_bps": 0.0,
            "min_relative_volume": 0.0,
            "max_position_value": 101.0,
            "risk_per_trade_pct": 1.0,
        },
        slippage=SlippageModel(fixed_bps=0.0),
        commission=CommissionModel(),
        initial_capital=1000.0,
        ensemble=_SequencedEnsemble(
            [
                _prediction("short", "trending_down"),
                _prediction("no_trade", "low_volatility"),
            ]
        ),
    )

    results = simulator.run({"AAPL": short_bars})

    assert results["total_trades"] == 1
    assert results["final_equity"] == pytest.approx(1000.0 + results["total_pnl"], abs=1e-6)
    assert simulator._cash == pytest.approx(results["final_equity"], abs=1e-6)


def test_backtest_enters_on_next_bar_open_not_signal_bar() -> None:
    bars = _bars(100.0, [0.2] * 16 + [5.0, 0.0])
    simulator = BacktestSimulator(
        symbols=["AAPL"],
        strategy_config={
            "min_history_bars": 16,
            "min_confidence": 0.0,
            "min_expected_move_bps": 0.0,
            "min_relative_volume": 0.0,
            "max_position_value": 101.0,
            "risk_per_trade_pct": 1.0,
        },
        slippage=SlippageModel(fixed_bps=0.0),
        commission=CommissionModel(),
        initial_capital=1000.0,
        ensemble=_SequencedEnsemble(
            [
                _prediction("long", "trending_up"),
                _prediction("no_trade", "low_volatility"),
            ]
        ),
    )

    simulator.run({"AAPL": bars})

    closed_trade = simulator._closed_trades[0]
    assert closed_trade.entry_price == bars[16]["open"]
    assert closed_trade.entry_price != bars[15]["close"]


def test_backtest_force_closes_open_position_at_last_bar() -> None:
    bars = _bars(100.0, [0.2] * 16 + [0.1, 0.0])
    simulator = BacktestSimulator(
        symbols=["AAPL"],
        strategy_config={
            "min_history_bars": 16,
            "min_confidence": 0.0,
            "min_expected_move_bps": 0.0,
            "min_relative_volume": 0.0,
            "max_position_value": 101.0,
            "risk_per_trade_pct": 1.0,
        },
        slippage=SlippageModel(fixed_bps=0.0),
        commission=CommissionModel(),
        initial_capital=1000.0,
        ensemble=_SequencedEnsemble(
            [
                _prediction("long", "trending_up"),
                _prediction("no_trade", "low_volatility"),
            ]
        ),
    )

    results = simulator.run({"AAPL": bars})

    assert results["total_trades"] == 1
    assert simulator._positions == []
    assert simulator._closed_trades[0].exit_time == bars[-1]["timestamp"]
    assert results["final_equity"] == pytest.approx(1000.0 + results["total_pnl"], abs=1e-6)
