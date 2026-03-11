"""Durable system state helpers for kill switch and market freshness."""

from __future__ import annotations

from datetime import datetime

import structlog

from trader.config import get_settings
from trader.core.redis_client import get_cached, init_redis, set_cached

logger = structlog.get_logger(__name__)

_fallback_state: dict[str, str] = {}

KILL_SWITCH_KEY = "trader:control:kill_switch"
LAST_BAR_TS_PREFIX = "trader:market:last_bar_ts:"
LAST_SPREAD_BPS_PREFIX = "trader:market:last_spread_bps:"


class SystemStateStore:
    """Read and write durable trading state backed by Redis when available."""

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_url = redis_url or get_settings().redis_url

    async def _get(self, key: str) -> str | None:
        try:
            return await get_cached(key)
        except RuntimeError:
            try:
                await init_redis(self._redis_url)
                return await get_cached(key)
            except Exception as exc:
                logger.warning("system_state_get_fallback", key=key, error=str(exc))
                return _fallback_state.get(key)
        except Exception as exc:
            logger.warning("system_state_get_fallback", key=key, error=str(exc))
            return _fallback_state.get(key)

    async def _set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        try:
            await set_cached(key, value, ttl_seconds=ttl_seconds or 86400)
        except RuntimeError:
            try:
                await init_redis(self._redis_url)
                await set_cached(key, value, ttl_seconds=ttl_seconds or 86400)
                return
            except Exception as exc:
                logger.warning("system_state_set_fallback", key=key, error=str(exc))
                _fallback_state[key] = value
                return
        except Exception as exc:
            logger.warning("system_state_set_fallback", key=key, error=str(exc))
            _fallback_state[key] = value

    async def is_kill_switch_active(self) -> bool:
        value = await self._get(KILL_SWITCH_KEY)
        return value == "1"

    async def set_kill_switch(self, active: bool) -> bool:
        await self._set(KILL_SWITCH_KEY, "1" if active else "0", ttl_seconds=30 * 24 * 3600)
        return active

    async def record_bar_timestamp(self, symbol: str, timestamp: datetime) -> None:
        await self._set(f"{LAST_BAR_TS_PREFIX}{symbol}", timestamp.isoformat(), ttl_seconds=12 * 3600)

    async def get_last_bar_timestamp(self, symbol: str) -> datetime | None:
        value = await self._get(f"{LAST_BAR_TS_PREFIX}{symbol}")
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            logger.warning("system_state_invalid_timestamp", symbol=symbol, value=value)
            return None

    async def record_spread_bps(self, symbol: str, spread_bps: float) -> None:
        await self._set(f"{LAST_SPREAD_BPS_PREFIX}{symbol}", str(spread_bps), ttl_seconds=12 * 3600)

    async def get_spread_bps(self, symbol: str) -> float | None:
        value = await self._get(f"{LAST_SPREAD_BPS_PREFIX}{symbol}")
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            logger.warning("system_state_invalid_spread", symbol=symbol, value=value)
            return None
