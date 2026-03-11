"""Notional exposure limit rule."""

from __future__ import annotations

from decimal import Decimal


class NotionalExposureRule:
    """Rejects trades that would exceed maximum notional exposure."""

    def __init__(self, max_notional: float) -> None:
        self._max = Decimal(str(max_notional))

    def check(self, current_exposure: Decimal, new_notional: Decimal, **kwargs: object) -> tuple[bool, str]:
        projected = current_exposure + new_notional
        if projected > self._max:
            return False, f"Projected exposure {projected} exceeds max {self._max}"
        return True, ""
