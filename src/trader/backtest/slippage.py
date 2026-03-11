"""Slippage and commission modeling for backtests."""

from __future__ import annotations

from decimal import Decimal


class SlippageModel:
    """Models execution slippage for backtest simulations."""

    def __init__(
        self,
        fixed_bps: float = 5.0,
        volume_impact_bps: float = 0.0,
    ) -> None:
        self._fixed_bps = fixed_bps
        self._volume_impact_bps = volume_impact_bps

    def apply(self, price: Decimal, side: str, qty: int = 0, avg_volume: int = 0) -> Decimal:
        """Apply slippage to a fill price.

        For buys, slippage increases the price. For sells, it decreases.
        """
        bps = self._fixed_bps
        if avg_volume > 0 and qty > 0:
            participation = qty / avg_volume
            bps += self._volume_impact_bps * participation

        slip_pct = Decimal(str(bps)) / Decimal("10000")
        if side == "buy":
            return price * (1 + slip_pct)
        return price * (1 - slip_pct)


class CommissionModel:
    """Models trading commissions for backtests."""

    def __init__(
        self,
        per_share: float = 0.0,
        per_order: float = 0.0,
        min_per_order: float = 0.0,
    ) -> None:
        self._per_share = Decimal(str(per_share))
        self._per_order = Decimal(str(per_order))
        self._min = Decimal(str(min_per_order))

    def calculate(self, qty: int) -> Decimal:
        """Calculate commission for a trade."""
        commission = self._per_order + self._per_share * qty
        return max(commission, self._min)
