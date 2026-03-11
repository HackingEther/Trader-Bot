"""Concurrent positions limit rule."""

from __future__ import annotations


class ConcurrentPositionsRule:
    """Rejects trades if maximum concurrent positions would be exceeded."""

    def __init__(self, max_positions: int) -> None:
        self._max = max_positions

    def check(self, open_position_count: int, **kwargs: object) -> tuple[bool, str]:
        if open_position_count >= self._max:
            return False, f"Open positions {open_position_count} >= max {self._max}"
        return True, ""
