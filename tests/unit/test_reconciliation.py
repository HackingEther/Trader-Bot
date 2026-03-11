"""Unit tests for position reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from trader.execution.reconciliation import PositionReconciler, ReconciliationResult, _prices_match


def test_prices_match_exact() -> None:
    assert _prices_match(Decimal("100.00"), Decimal("100.00")) is True


def test_prices_match_within_tolerance() -> None:
    assert _prices_match(Decimal("100.00"), Decimal("100.001")) is True


def test_prices_match_none_b() -> None:
    assert _prices_match(Decimal("100.00"), None) is True


def test_prices_match_outside_tolerance() -> None:
    assert _prices_match(Decimal("100.00"), Decimal("100.50")) is False


@pytest.mark.asyncio
async def test_reconciliation_price_tolerance() -> None:
    """Reconciliation tolerates minor price differences from floating-point/broker rounding."""
    broker_pos = SimpleNamespace(
        symbol="AAPL",
        qty=10,
        side="buy",
        avg_entry_price=Decimal("100.0001"),
        current_price=Decimal("105.0002"),
        market_value=Decimal("1050.002"),
        unrealized_pnl=Decimal("50.001"),
    )
    local_pos = SimpleNamespace(
        symbol="AAPL",
        qty=10,
        side="buy",
        avg_entry_price=Decimal("100.0002"),
        current_price=Decimal("105.0003"),
        market_value=Decimal("1050.003"),
        unrealized_pnl=Decimal("50.002"),
    )

    class _FakeBroker:
        async def get_positions(self):
            return [broker_pos]

    class _FakeRepo:
        async def get_open_positions(self):
            return [local_pos]

    reconciler = PositionReconciler.__new__(PositionReconciler)
    reconciler._broker = _FakeBroker()
    reconciler._session = None
    reconciler._repo = _FakeRepo()

    result = await reconciler.reconcile(auto_fix=False)

    assert "AAPL" in result.matched
    assert result.is_clean
