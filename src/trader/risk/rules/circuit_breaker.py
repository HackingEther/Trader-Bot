"""Circuit breaker / kill switch rule."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


class CircuitBreakerRule:
    """Hard kill switch that blocks all trading when active."""

    def __init__(self) -> None:
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        self._active = True
        logger.critical("kill_switch_activated")

    def deactivate(self) -> None:
        self._active = False
        logger.warning("kill_switch_deactivated")

    def check(self, **kwargs: object) -> tuple[bool, str]:
        if self._active:
            return False, "Kill switch is active - all trading halted"
        return True, ""
