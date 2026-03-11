"""Unit tests for order translation and sizing."""

from __future__ import annotations

from decimal import Decimal

import pytest

from trader.strategy.sizing import PositionSizer


class TestPositionSizer:
    def setup_method(self) -> None:
        self.sizer = PositionSizer(
            max_position_value=10000.0,
            risk_per_trade_pct=0.01,
            max_shares=1000,
        )

    def test_compute_qty_basic(self) -> None:
        qty = self.sizer.compute_qty(Decimal("100.00"))
        assert qty >= 1
        assert qty <= 1000

    def test_compute_qty_respects_max_value(self) -> None:
        qty = self.sizer.compute_qty(Decimal("10000.00"))
        assert qty == 1

    def test_compute_qty_zero_price(self) -> None:
        qty = self.sizer.compute_qty(Decimal("0"))
        assert qty == 0

    def test_stop_loss_buy(self) -> None:
        sl = self.sizer.compute_stop_loss(Decimal("100.00"), "buy")
        assert sl < Decimal("100.00")

    def test_stop_loss_sell(self) -> None:
        sl = self.sizer.compute_stop_loss(Decimal("100.00"), "sell")
        assert sl > Decimal("100.00")

    def test_take_profit_buy(self) -> None:
        tp = self.sizer.compute_take_profit(Decimal("100.00"), "buy")
        assert tp > Decimal("100.00")

    def test_take_profit_sell(self) -> None:
        tp = self.sizer.compute_take_profit(Decimal("100.00"), "sell")
        assert tp < Decimal("100.00")
