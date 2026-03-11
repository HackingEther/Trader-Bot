"""Email notification provider stub."""

from __future__ import annotations

import structlog

from trader.providers.notifications.base import NotificationProvider

logger = structlog.get_logger(__name__)


class EmailNotificationProvider(NotificationProvider):
    """SMTP email notification provider - stub implementation."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        to_address: str,
    ) -> None:
        self._host = smtp_host
        self._port = smtp_port
        self._user = smtp_user
        self._password = smtp_password
        self._to = to_address

    async def send(self, subject: str, message: str, level: str = "info") -> bool:
        if not self._host or not self._to:
            logger.debug("email_notification_skipped", reason="SMTP not configured")
            return False
        logger.info(
            "email_notification_stub",
            subject=subject,
            to=self._to,
            message="Email sending not yet implemented - log only",
        )
        return False

    async def send_alert(self, title: str, message: str, details: dict | None = None) -> bool:
        return await self.send(f"[ALERT] {title}", message, level="critical")
