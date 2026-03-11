"""Cooldown rule after consecutive losses."""

from __future__ import annotations


class CooldownRule:
    """Rejects trades after N consecutive losses."""

    def __init__(self, max_consecutive_losses: int) -> None:
        self._max = max_consecutive_losses

    def check(self, consecutive_losses: int, **kwargs: object) -> tuple[bool, str]:
        if consecutive_losses >= self._max:
            return False, f"Consecutive losses {consecutive_losses} >= cooldown threshold {self._max}"
        return True, ""
