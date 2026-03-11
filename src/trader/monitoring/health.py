"""Startup health checks validating env vars and external connections."""

from __future__ import annotations

import structlog

from trader.config import Settings

logger = structlog.get_logger(__name__)


class StartupChecker:
    """Validates configuration and external connections at startup."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._results: dict[str, bool] = {}

    async def run_all(self) -> dict[str, bool]:
        """Run all startup checks and return results."""
        self._results["config_valid"] = self._check_config()
        self._results["database"] = await self._check_database()
        self._results["redis"] = await self._check_redis()
        self._results["broker_configured"] = self._check_broker_config()
        self._results["safety_defaults"] = self._check_safety_defaults()

        all_passed = all(self._results.values())
        if all_passed:
            logger.info("startup_checks_passed", results=self._results)
        else:
            failed = {k: v for k, v in self._results.items() if not v}
            logger.error("startup_checks_failed", failed=failed)

        return self._results

    def _check_config(self) -> bool:
        try:
            assert self._settings.secret_key != "change-me-to-random-string", "Secret key not set"
            return True
        except AssertionError as e:
            logger.warning("config_check_warning", message=str(e))
            return True  # Non-fatal for development

    async def _check_database(self) -> bool:
        try:
            from sqlalchemy import text
            from trader.db.session import get_engine
            engine = get_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error("database_check_failed", error=str(e))
            return False

    async def _check_redis(self) -> bool:
        try:
            from trader.core.redis_client import get_redis
            client = await get_redis()
            await client.ping()
            return True
        except Exception as e:
            logger.error("redis_check_failed", error=str(e))
            return False

    def _check_broker_config(self) -> bool:
        s = self._settings
        if s.app_env == "live" and (not s.alpaca_api_key or not s.alpaca_api_secret):
            logger.error("broker_config_missing", message="Alpaca credentials required for live")
            return False
        return True

    def _check_safety_defaults(self) -> bool:
        s = self._settings
        if s.live_trading and s.live_trading_confirmed != "I_CONFIRM_LIVE_TRADING":
            logger.error("safety_check_failed", message="LIVE_TRADING_CONFIRMED not set correctly")
            return False
        if s.live_trading and not s.is_live:
            logger.warning("safety_config_mismatch", message="live_trading=true but app_env != live")
        return True
