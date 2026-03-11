"""Spread threshold rule."""

from __future__ import annotations


class SpreadRule:
    """Rejects trades when bid-ask spread is too wide."""

    def __init__(self, max_spread_bps: float) -> None:
        self._max = max_spread_bps

    def check(self, spread_bps: float, **kwargs: object) -> tuple[bool, str]:
        if spread_bps > self._max:
            return False, f"Spread {spread_bps:.1f} bps exceeds max {self._max:.1f} bps"
        return True, ""
