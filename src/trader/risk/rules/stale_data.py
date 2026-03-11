"""Stale data rejection rule."""

from __future__ import annotations

from datetime import datetime, timezone


class StaleDataRule:
    """Rejects trades when market data is stale."""

    def __init__(self, max_age_seconds: float = 30.0) -> None:
        self._max_age = max_age_seconds

    def check(self, last_data_time: datetime | None, **kwargs: object) -> tuple[bool, str]:
        if last_data_time is None:
            return False, "No market data timestamp available"
        now = datetime.now(timezone.utc)
        age = (now - last_data_time).total_seconds()
        if age > self._max_age:
            return False, f"Data age {age:.1f}s exceeds max {self._max_age:.1f}s"
        return True, ""
