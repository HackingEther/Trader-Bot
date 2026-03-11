"""Slack notification provider stub."""

from __future__ import annotations

import structlog
import httpx

from trader.providers.notifications.base import NotificationProvider

logger = structlog.get_logger(__name__)


class SlackNotificationProvider(NotificationProvider):
    """Slack webhook notification provider."""

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    async def send(self, subject: str, message: str, level: str = "info") -> bool:
        if not self._webhook_url:
            logger.debug("slack_notification_skipped", reason="no webhook URL configured")
            return False
        emoji = {"info": ":information_source:", "warning": ":warning:", "error": ":x:", "critical": ":rotating_light:"}.get(level, ":speech_balloon:")
        payload = {"text": f"{emoji} *{subject}*\n{message}"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._webhook_url, json=payload)
                resp.raise_for_status()
            logger.info("slack_notification_sent", subject=subject)
            return True
        except Exception as e:
            logger.error("slack_notification_failed", error=str(e))
            return False

    async def send_alert(self, title: str, message: str, details: dict | None = None) -> bool:
        detail_str = ""
        if details:
            detail_str = "\n".join(f"- *{k}*: {v}" for k, v in details.items())
            message = f"{message}\n{detail_str}"
        return await self.send(title, message, level="critical")
