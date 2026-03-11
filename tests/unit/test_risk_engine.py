"""Unit tests for risk engine and rules."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from trader.risk.engine import RiskContext, RiskEngine
from trader.risk.rules.circuit_breaker import CircuitBreakerRule
from trader.risk.rules.daily_loss import DailyLossRule
from trader.risk.rules.concurrent_positions import ConcurrentPositionsRule
from trader.risk.rules.cooldown import CooldownRule
from trader.risk.rules.spread import SpreadRule
from trader.risk.rules.stale_data import StaleDataRule
from trader.strategy.engine import TradeIntentParams


def _make_intent(symbol: str = "AAPL", side: str = "buy", qty: int = 10) -> TradeIntentParams:
    return TradeIntentParams(
        symbol=symbol,
        side=side,
        qty=qty,
        stop_loss=Decimal("145.00"),
        take_profit=Decimal("160.00"),
    )


class TestDailyLossRule:
    def test_passes_when_within_limit(self) -> None:
        rule = DailyLossRule(1000.0)
        passed, _ = rule.check(daily_realized_pnl=Decimal("-500"))
        assert passed

    def test_fails_when_exceeded(self) -> None:
        rule = DailyLossRule(1000.0)
        passed, reason = rule.check(daily_realized_pnl=Decimal("-1500"))
        assert not passed
        assert "exceeds" in reason.lower()


class TestConcurrentPositionsRule:
    def test_passes_under_limit(self) -> None:
        rule = ConcurrentPositionsRule(10)
        passed, _ = rule.check(open_position_count=5)
        assert passed

    def test_fails_at_limit(self) -> None:
        rule = ConcurrentPositionsRule(10)
        passed, _ = rule.check(open_position_count=10)
        assert not passed


class TestCooldownRule:
    def test_passes_no_losses(self) -> None:
        rule = CooldownRule(3)
        passed, _ = rule.check(consecutive_losses=0)
        assert passed

    def test_fails_on_cooldown(self) -> None:
        rule = CooldownRule(3)
        passed, _ = rule.check(consecutive_losses=3)
        assert not passed


class TestSpreadRule:
    def test_passes_narrow_spread(self) -> None:
        rule = SpreadRule(50.0)
        passed, _ = rule.check(spread_bps=10.0)
        assert passed

    def test_fails_wide_spread(self) -> None:
        rule = SpreadRule(50.0)
        passed, _ = rule.check(spread_bps=100.0)
        assert not passed


class TestStaleDataRule:
    def test_passes_fresh_data(self) -> None:
        rule = StaleDataRule(30.0)
        now = datetime.now(timezone.utc)
        passed, _ = rule.check(last_data_time=now - timedelta(seconds=5))
        assert passed

    def test_fails_stale_data(self) -> None:
        rule = StaleDataRule(30.0)
        old = datetime.now(timezone.utc) - timedelta(seconds=60)
        passed, _ = rule.check(last_data_time=old)
        assert not passed

    def test_fails_no_data(self) -> None:
        rule = StaleDataRule(30.0)
        passed, _ = rule.check(last_data_time=None)
        assert not passed


class TestCircuitBreaker:
    def test_inactive_by_default(self) -> None:
        cb = CircuitBreakerRule()
        passed, _ = cb.check()
        assert passed

    def test_blocks_when_active(self) -> None:
        cb = CircuitBreakerRule()
        cb.activate()
        passed, reason = cb.check()
        assert not passed
        assert "kill switch" in reason.lower()

    def test_resumes_after_deactivate(self) -> None:
        cb = CircuitBreakerRule()
        cb.activate()
        cb.deactivate()
        passed, _ = cb.check()
        assert passed


class TestRiskEngine:
    def test_approves_valid_trade(self) -> None:
        engine = RiskEngine(max_daily_loss=10000, max_positions=20)
        intent = _make_intent()
        context = RiskContext(
            daily_realized_pnl=Decimal("0"),
            open_position_count=0,
            entry_price=Decimal("150.00"),
            last_data_time=datetime.now(timezone.utc),
        )
        decision = engine.evaluate(intent, context)
        assert decision.approved or "market_hours" in str(decision.reasons)

    def test_rejects_on_kill_switch(self) -> None:
        engine = RiskEngine()
        engine.kill_switch.activate()
        intent = _make_intent()
        context = RiskContext(entry_price=Decimal("150.00"))
        decision = engine.evaluate(intent, context)
        assert not decision.approved
        assert any("kill switch" in r.lower() for r in decision.reasons)

    def test_rejects_over_daily_loss(self) -> None:
        engine = RiskEngine(max_daily_loss=500)
        intent = _make_intent()
        context = RiskContext(
            daily_realized_pnl=Decimal("-600"),
            entry_price=Decimal("150.00"),
            last_data_time=datetime.now(timezone.utc),
        )
        decision = engine.evaluate(intent, context)
        assert not decision.approved
