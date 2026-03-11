"""Per-symbol exposure limit rule."""

from __future__ import annotations

from decimal import Decimal


class SymbolExposureRule:
    """Rejects trades that would exceed per-symbol exposure limit."""

    def __init__(self, max_per_symbol: float) -> None:
        self._max = Decimal(str(max_per_symbol))

    def check(self, symbol_exposure: Decimal, new_notional: Decimal, **kwargs: object) -> tuple[bool, str]:
        projected = symbol_exposure + new_notional
        if projected > self._max:
            return False, f"Symbol exposure {projected} exceeds max {self._max}"
        return True, ""
