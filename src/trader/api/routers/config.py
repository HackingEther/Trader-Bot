"""Configuration read endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from trader.config import get_settings

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("/")
async def get_config() -> dict:
    settings = get_settings()
    return {
        "app_env": settings.app_env,
        "live_trading": settings.live_trading,
        "is_live": settings.is_live,
        "is_paper": settings.is_paper,
        "symbol_universe": settings.symbol_universe,
        "max_daily_loss_usd": settings.max_daily_loss_usd,
        "max_loss_per_trade_usd": settings.max_loss_per_trade_usd,
        "max_notional_exposure_usd": settings.max_notional_exposure_usd,
        "max_concurrent_positions": settings.max_concurrent_positions,
        "max_exposure_per_symbol_usd": settings.max_exposure_per_symbol_usd,
        "min_confidence": settings.min_confidence,
        "min_expected_move_bps": settings.min_expected_move_bps,
    }
