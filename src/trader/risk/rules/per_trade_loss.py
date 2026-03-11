"""Per-trade loss limit rule."""

from __future__ import annotations

from decimal import Decimal


class PerTradeLossRule:
    """Rejects trades where potential loss exceeds per-trade limit."""

    def __init__(self, max_loss_per_trade: float) -> None:
        self._max = Decimal(str(max_loss_per_trade))

    def check(self, qty: int, entry_price: Decimal, stop_loss: Decimal | None, side: str, **kwargs: object) -> tuple[bool, str]:
        if stop_loss is None:
            return True, ""
        if side == "buy":
            potential_loss = (entry_price - stop_loss) * qty
        else:
            potential_loss = (stop_loss - entry_price) * qty
        if potential_loss > self._max:
            return False, f"Potential loss {potential_loss} exceeds max per-trade {self._max}"
        return True, ""
