"""Position sizing logic."""

from __future__ import annotations

from decimal import Decimal

import structlog

logger = structlog.get_logger(__name__)


class PositionSizer:
    """Computes position sizes based on risk and account parameters."""

    def __init__(
        self,
        max_position_value: float = 10000.0,
        risk_per_trade_pct: float = 0.01,
        max_shares: int = 1000,
    ) -> None:
        self._max_position_value = max_position_value
        self._risk_per_trade_pct = risk_per_trade_pct
        self._max_shares = max_shares

    def compute_qty(
        self,
        price: Decimal,
        stop_distance_pct: float = 0.01,
        account_equity: Decimal = Decimal("100000"),
    ) -> int:
        """Compute the number of shares to trade.

        Args:
            price: Current price per share.
            stop_distance_pct: Distance to stop loss as fraction of price.
            account_equity: Total account equity.
        """
        if price <= 0:
            return 0

        risk_amount = float(account_equity) * self._risk_per_trade_pct
        price_f = float(price)
        stop_distance = price_f * max(stop_distance_pct, 0.001)
        qty_from_risk = int(risk_amount / stop_distance) if stop_distance > 0 else 0
        qty_from_value = int(self._max_position_value / price_f) if price_f > 0 else 0
        qty = min(qty_from_risk, qty_from_value, self._max_shares)

        return max(1, qty)

    def compute_stop_loss(self, price: Decimal, side: str, atr_multiple: float = 1.5, volatility: float = 0.01) -> Decimal:
        """Compute stop loss price."""
        stop_distance = float(price) * volatility * atr_multiple
        if side == "buy":
            return Decimal(str(round(float(price) - stop_distance, 2)))
        return Decimal(str(round(float(price) + stop_distance, 2)))

    def compute_take_profit(self, price: Decimal, side: str, rr_ratio: float = 2.0, volatility: float = 0.01, atr_multiple: float = 1.5) -> Decimal:
        """Compute take profit price based on risk-reward ratio."""
        stop_distance = float(price) * volatility * atr_multiple
        tp_distance = stop_distance * rr_ratio
        if side == "buy":
            return Decimal(str(round(float(price) + tp_distance, 2)))
        return Decimal(str(round(float(price) - tp_distance, 2)))
