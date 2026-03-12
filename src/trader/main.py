"""FastAPI application factory."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trader.api.middleware import AdminTokenMiddleware
from trader.api.routers import (
    config,
    fills,
    health,
    killswitch,
    orders,
    pnl,
    positions,
    predictions,
    risk_decisions,
    symbols,
    trade_intents,
)
from trader.config import get_settings
from trader.core.redis_client import close_redis, init_redis
from trader.db.session import close_engine, init_engine
from trader.ingestion.manager import IngestionManager
from trader.ingestion.trade_updates import TradeUpdateStream
from trader.logging import setup_logging
from trader.providers.market_data.factory import create_market_data_provider

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    init_engine(settings.database_url)
    ingestion_manager: IngestionManager | None = None
    ingestion_task = None
    trade_updates_stream: TradeUpdateStream | None = None
    trade_updates_task = None
    try:
        await init_redis(settings.redis_url)
    except Exception as e:
        logger.warning("redis_init_failed", error=str(e))

    if settings.sentry_dsn:
        try:
            import sentry_sdk

            sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.1)
        except Exception as e:
            logger.warning("sentry_init_failed", error=str(e))

    logger.info(
        "app_started",
        env=settings.app_env,
        live=settings.is_live,
        paper=settings.is_paper,
    )

    if settings.market_data_autostart:
        try:
            provider = create_market_data_provider(settings)
            ingestion_manager = IngestionManager(
                provider=provider,
                symbols=settings.symbol_universe,
                staleness_threshold=settings.market_data_staleness_threshold,
                subscribe_bars=settings.market_data_subscribe_bars,
                subscribe_quotes=settings.market_data_subscribe_quotes,
                subscribe_trades=settings.market_data_subscribe_trades,
            )
            ingestion_task = asyncio.create_task(ingestion_manager.start())
            logger.info(
                "market_data_autostart_enabled",
                provider=settings.market_data_provider,
                symbols=settings.symbol_universe,
            )
        except Exception as e:
            logger.warning(
                "market_data_autostart_failed",
                provider=settings.market_data_provider,
                error=str(e),
            )

    if getattr(settings, "trade_updates_stream_enabled", True) and settings.alpaca_api_key and settings.alpaca_api_secret:
        try:
            from trader.providers.broker.factory import create_broker_provider

            broker = create_broker_provider(settings)
            trade_updates_stream = TradeUpdateStream(
                api_key=settings.alpaca_api_key,
                api_secret=settings.alpaca_api_secret,
                paper=settings.alpaca_paper,
                broker=broker,
            )
            trade_updates_task = asyncio.create_task(trade_updates_stream.start())
            logger.info("trade_updates_stream_enabled")
        except Exception as e:
            logger.warning("trade_updates_stream_start_failed", error=str(e))

    yield

    if trade_updates_stream is not None:
        try:
            await trade_updates_stream.stop()
        except Exception as e:
            logger.warning("trade_updates_stream_shutdown_failed", error=str(e))
    if trade_updates_task is not None:
        try:
            await trade_updates_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    if ingestion_manager is not None:
        try:
            await ingestion_manager.stop()
        except Exception as e:
            logger.warning("market_data_shutdown_failed", error=str(e))
    if ingestion_task is not None:
        try:
            await ingestion_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    try:
        await close_redis()
    except Exception as e:
        logger.warning("redis_close_failed", error=str(e))
    await close_engine()
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Claude Trader",
        description="Institutional-style autonomous intraday trading platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(AdminTokenMiddleware, admin_token=settings.admin_api_token)

    app.include_router(health.router)
    app.include_router(symbols.router)
    app.include_router(positions.router)
    app.include_router(orders.router)
    app.include_router(fills.router)
    app.include_router(pnl.router)
    app.include_router(predictions.router)
    app.include_router(trade_intents.router)
    app.include_router(risk_decisions.router)
    app.include_router(killswitch.router)
    app.include_router(config.router)

    return app


app = create_app()
