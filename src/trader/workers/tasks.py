"""Celery tasks for the trading platform."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog

from trader.celery_app import celery
from trader.config import get_settings
from trader.core.events import BarEvent
from trader.core.redis_client import init_redis
from trader.db.session import get_session_factory, init_engine
from trader.features.engine import FeatureEngine
from trader.providers.broker.factory import create_broker_provider
from trader.services.model_loader import ChampionModelLoader
from trader.services.system_state import SystemStateStore
from trader.services.trading_cycle import TradingCycleService

logger = structlog.get_logger(__name__)

_broker = None
_model_loader = ChampionModelLoader()


def _bootstrap() -> tuple:
    settings = get_settings()
    try:
        factory = get_session_factory()
    except RuntimeError:
        init_engine(settings.database_url)
        factory = get_session_factory()

    global _broker
    if _broker is None:
        _broker = create_broker_provider(settings)

    return settings, factory, _broker


async def _ensure_redis() -> None:
    settings = get_settings()
    try:
        await init_redis(settings.redis_url)
    except Exception:
        logger.warning("redis_bootstrap_failed", url=settings.redis_url)


@celery.task(name="trader.workers.tasks.heartbeat_system")
def heartbeat_system() -> dict:
    """System heartbeat task - runs every 30 seconds."""
    ts = datetime.now(timezone.utc).isoformat()
    logger.info("heartbeat", timestamp=ts)
    return {"status": "alive", "timestamp": ts}


@celery.task(name="trader.workers.tasks.compute_features")
def compute_features(symbol: str) -> dict:
    """Compute features for a symbol using the most recent persisted bars."""

    async def _run() -> dict:
        settings, factory, _ = _bootstrap()
        await _ensure_redis()
        async with factory() as session:
            from trader.db.repositories.market_bars import MarketBarRepository

            repo = MarketBarRepository(session)
            bars = await repo.get_recent(symbol, limit=400)
            engine = FeatureEngine(max_bars=400)
            for bar in bars:
                engine.add_bar(
                    BarEvent(
                        symbol=bar.symbol,
                        timestamp=bar.timestamp,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                        vwap=bar.vwap,
                        trade_count=bar.trade_count,
                        interval=bar.interval,
                    )
                )
            features = engine.compute_features(symbol, bars[-1].timestamp if bars else None)
            logger.info(
                "features_computed",
                symbol=symbol,
                feature_count=len(features),
                bar_count=len(bars),
                env=settings.app_env,
            )
            return {"symbol": symbol, "features": features, "bar_count": len(bars)}

    return asyncio.run(_run())


@celery.task(name="trader.workers.tasks.run_prediction")
def run_prediction(symbol: str, features: dict) -> dict:
    """Run model ensemble prediction for a symbol."""

    async def _run() -> dict:
        _, factory, _ = _bootstrap()
        await _ensure_redis()
        async with factory() as session:
            pipeline = await _model_loader.load_ensemble(session=session)
            prediction = pipeline.predict(symbol, features, datetime.now(timezone.utc))
            return prediction.model_dump(mode="json")

    return asyncio.run(_run())


@celery.task(name="trader.workers.tasks.execute_trading_cycle")
def execute_trading_cycle() -> dict:
    """Run the full periodic trading cycle."""

    async def _run() -> dict:
        settings, factory, broker = _bootstrap()
        await _ensure_redis()
        async with factory() as session:
            service = TradingCycleService(
                settings=settings,
                session=session,
                broker=broker,
                model_loader=_model_loader,
                state_store=SystemStateStore(),
            )
            results = await service.run_cycle()
            logger.info("trading_cycle_complete", **results)
            return results

    return asyncio.run(_run())


@celery.task(name="trader.workers.tasks.reconcile_positions")
def reconcile_positions() -> dict:
    """Reconcile broker positions with local state."""

    async def _run() -> dict:
        settings, factory, broker = _bootstrap()
        await _ensure_redis()
        async with factory() as session:
            service = TradingCycleService(
                settings=settings,
                session=session,
                broker=broker,
                model_loader=_model_loader,
                state_store=SystemStateStore(),
            )
            result = await service.reconcile_positions(auto_fix=True, halt_on_discrepancy=True)
            logger.info("reconciliation_task_completed", **result)
            return result

    return asyncio.run(_run())


@celery.task(name="trader.workers.tasks.snapshot_pnl")
def snapshot_pnl() -> dict:
    """Take a P&L snapshot."""

    async def _run() -> dict:
        settings, factory, broker = _bootstrap()
        await _ensure_redis()
        async with factory() as session:
            service = TradingCycleService(
                settings=settings,
                session=session,
                broker=broker,
                model_loader=_model_loader,
                state_store=SystemStateStore(),
            )
            snapshot = await service.snapshot_pnl()
            logger.info("pnl_snapshot_taken", **snapshot)
            return snapshot

    return asyncio.run(_run())


@celery.task(name="trader.workers.tasks.check_stale_feeds")
def check_stale_feeds() -> dict:
    """Check for stale market data feeds."""

    async def _run() -> dict:
        settings = get_settings()
        await _ensure_redis()
        state_store = SystemStateStore()
        stale_symbols: list[str] = []
        now = datetime.now(timezone.utc)
        max_age_seconds = max(settings.market_data_staleness_threshold, 30.0) * 2
        for symbol in settings.symbol_universe:
            last_bar = await state_store.get_last_bar_timestamp(symbol)
            if last_bar is None or (now - last_bar).total_seconds() > max_age_seconds:
                stale_symbols.append(symbol)
        logger.info("stale_feed_check", stale_symbols=stale_symbols)
        return {"stale_symbols": stale_symbols}

    return asyncio.run(_run())
