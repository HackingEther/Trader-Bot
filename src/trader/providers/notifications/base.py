"""Abstract base class for notification providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class NotificationProvider(ABC):
    """Abstract notification interface."""

    @abstractmethod
    async def send(self, subject: str, message: str, level: str = "info") -> bool:
        """Send a notification. Returns True on success."""

    @abstractmethod
    async def send_alert(self, title: str, message: str, details: dict | None = None) -> bool:
        """Send a high-priority alert."""
