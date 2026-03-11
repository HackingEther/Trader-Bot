"""Health check endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from trader.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": "0.1.0",
        "environment": settings.app_env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "live_trading": settings.is_live,
    }
