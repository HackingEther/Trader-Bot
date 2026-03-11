"""Daily loss limit rule."""

from __future__ import annotations

from decimal import Decimal


class DailyLossRule:
    """Rejects trades if daily realized loss exceeds threshold."""

    def __init__(self, max_daily_loss: float) -> None:
        self._max = Decimal(str(max_daily_loss))

    def check(self, daily_realized_pnl: Decimal, **kwargs: object) -> tuple[bool, str]:
        if daily_realized_pnl < -self._max:
            return False, f"Daily loss {daily_realized_pnl} exceeds max {-self._max}"
        return True, ""
