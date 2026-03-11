"""Stale market data feed detection."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from trader.core.exceptions import StaleDataError

logger = structlog.get_logger(__name__)


class StalenessDetector:
    """Detects when market data feed becomes stale."""

    def __init__(self, threshold_seconds: float = 30.0) -> None:
        self._threshold = threshold_seconds
        self._last_update: dict[str, datetime] = {}
        self._alerted: set[str] = set()

    def record_update(self, symbol: str, timestamp: datetime | None = None) -> None:
        """Record that data was received for a symbol."""
        self._last_update[symbol] = timestamp or datetime.now(timezone.utc)
        self._alerted.discard(symbol)

    def check(self, symbol: str, now: datetime | None = None) -> bool:
        """Check if data for a symbol is stale. Returns True if stale."""
        now = now or datetime.now(timezone.utc)
        last = self._last_update.get(symbol)
        if last is None:
            return False
        elapsed = (now - last).total_seconds()
        return elapsed > self._threshold

    def check_and_alert(self, symbol: str, now: datetime | None = None) -> bool:
        """Check staleness and log alert on first detection. Returns True if stale."""
        is_stale = self.check(symbol, now)
        if is_stale and symbol not in self._alerted:
            last = self._last_update.get(symbol)
            now = now or datetime.now(timezone.utc)
            elapsed = (now - last).total_seconds() if last else 0
            logger.warning(
                "stale_data_detected",
                symbol=symbol,
                seconds_since_update=round(elapsed, 1),
                threshold=self._threshold,
            )
            self._alerted.add(symbol)
        return is_stale

    def check_all(self, now: datetime | None = None) -> list[str]:
        """Check all tracked symbols for staleness. Returns list of stale symbols."""
        now = now or datetime.now(timezone.utc)
        return [sym for sym in self._last_update if self.check_and_alert(sym, now)]

    def get_last_update(self, symbol: str) -> datetime | None:
        return self._last_update.get(symbol)

    def reset(self, symbol: str | None = None) -> None:
        if symbol:
            self._last_update.pop(symbol, None)
            self._alerted.discard(symbol)
        else:
            self._last_update.clear()
            self._alerted.clear()
