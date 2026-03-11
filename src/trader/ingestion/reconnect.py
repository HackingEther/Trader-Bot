"""Exponential backoff reconnection logic for market data feeds."""

from __future__ import annotations

import asyncio
import random

import structlog

logger = structlog.get_logger(__name__)


class ReconnectPolicy:
    """Manages exponential backoff reconnection attempts."""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_retries: int = 50,
        jitter: float = 0.5,
    ) -> None:
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._max_retries = max_retries
        self._jitter = jitter
        self._attempt = 0

    @property
    def attempt(self) -> int:
        return self._attempt

    def reset(self) -> None:
        self._attempt = 0

    async def wait(self) -> bool:
        """Wait with exponential backoff. Returns False if max retries exceeded."""
        if self._attempt >= self._max_retries:
            logger.error(
                "reconnect_max_retries_exceeded",
                max_retries=self._max_retries,
                attempt=self._attempt,
            )
            return False

        delay = min(self._base_delay * (2 ** self._attempt), self._max_delay)
        jitter = random.uniform(0, self._jitter * delay)
        total_delay = delay + jitter
        self._attempt += 1

        logger.info(
            "reconnect_waiting",
            attempt=self._attempt,
            delay_seconds=round(total_delay, 2),
        )
        await asyncio.sleep(total_delay)
        return True
