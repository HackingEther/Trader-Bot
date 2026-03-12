"""Unit tests for marketable limit pricing."""

from __future__ import annotations

from decimal import Decimal

import pytest

from trader.strategy.engine import StrategyEngine
from trader.strategy.sizing import PositionSizer
from trader.strategy.universe import SymbolUniverse


def test_marketable_limit_price_buy_above_ask() -> None:
    """Marketable limit for buy should be above ask."""
    engine = StrategyEngine(
        universe=SymbolUniverse(["AAPL"]),
        sizer=PositionSizer(),
        use_marketable_limits=True,
        marketable_limit_buffer_bps=5.0,
    )
    quote = {"bid": 150.00, "ask": 150.05, "mid": 150.025, "spread_bps": 3.33}
    price = engine._build_limit_price_from_quote(quote, "buy", 5.0)
    assert price is not None
    assert price >= Decimal("150.05")
    expected = (Decimal("150.05") * (Decimal("1") + Decimal("5") / Decimal("10000"))).quantize(Decimal("0.01"))
    assert price == expected


def test_marketable_limit_price_sell_below_bid() -> None:
    """Marketable limit for sell should be below bid."""
    engine = StrategyEngine(
        universe=SymbolUniverse(["AAPL"]),
        sizer=PositionSizer(),
        use_marketable_limits=True,
        marketable_limit_buffer_bps=5.0,
    )
    quote = {"bid": 150.00, "ask": 150.05, "mid": 150.025, "spread_bps": 3.33}
    price = engine._build_limit_price_from_quote(quote, "sell", 5.0)
    assert price is not None
    assert price <= Decimal("150.00")
    expected = (Decimal("150.00") * (Decimal("1") - Decimal("5") / Decimal("10000"))).quantize(Decimal("0.01"))
    assert price == expected


def test_passive_limit_uses_spread_buffer_when_not_marketable() -> None:
    """When use_marketable_limits=False, uses passive buffer."""
    engine = StrategyEngine(
        universe=SymbolUniverse(["AAPL"]),
        sizer=PositionSizer(),
        use_marketable_limits=False,
        limit_entry_buffer_bps=10.0,
    )
    quote = {"bid": 150.00, "ask": 150.05, "mid": 150.025, "spread_bps": 5.0}
    price = engine._build_limit_price_from_quote(quote, "buy", 5.0)
    assert price is not None
    buffer_bps = max(10.0, 5.0 * 0.75)
    expected = Decimal("150.05") * (Decimal("1") + Decimal(str(buffer_bps)) / Decimal("10000"))
    assert price == expected.quantize(Decimal("0.01"))
