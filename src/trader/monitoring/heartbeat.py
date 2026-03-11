"""Heartbeat monitoring for system components."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import structlog

from trader.monitoring.metrics import HEARTBEAT_TIMESTAMP

logger = structlog.get_logger(__name__)


class HeartbeatMonitor:
    """Tracks heartbeats for various system components."""

    def __init__(self) -> None:
        self._last_beats: dict[str, datetime] = {}

    def beat(self, component: str) -> None:
        """Record a heartbeat for a component."""
        now = datetime.now(timezone.utc)
        self._last_beats[component] = now
        HEARTBEAT_TIMESTAMP.labels(component=component).set(time.time())

    def get_last_beat(self, component: str) -> datetime | None:
        return self._last_beats.get(component)

    def is_healthy(self, component: str, max_age_seconds: float = 120.0) -> bool:
        """Check if component has sent a heartbeat within threshold."""
        last = self._last_beats.get(component)
        if last is None:
            return False
        age = (datetime.now(timezone.utc) - last).total_seconds()
        return age <= max_age_seconds

    def get_status(self) -> dict[str, dict]:
        """Get health status for all tracked components."""
        now = datetime.now(timezone.utc)
        status = {}
        for component, last_beat in self._last_beats.items():
            age = (now - last_beat).total_seconds()
            status[component] = {
                "last_beat": last_beat.isoformat(),
                "age_seconds": round(age, 1),
                "healthy": age <= 120.0,
            }
        return status
