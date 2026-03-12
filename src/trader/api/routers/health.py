"""Health check endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from trader.config import get_settings
from trader.ingestion.trade_updates import TRADE_UPDATES_HEALTH_KEY

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    settings = get_settings()
    result: dict = {
        "status": "ok",
        "version": "0.1.0",
        "environment": settings.app_env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "live_trading": settings.is_live,
    }
    try:
        from trader.core.redis_client import get_cached

        last_event = await get_cached(TRADE_UPDATES_HEALTH_KEY)
        if last_event:
            try:
                last_ts = datetime.fromisoformat(last_event)
                age = (datetime.now(timezone.utc) - last_ts).total_seconds()
                result["trade_updates_last_event_age_seconds"] = round(age, 1)
            except (ValueError, TypeError):
                pass
    except Exception:
        pass
    return result
