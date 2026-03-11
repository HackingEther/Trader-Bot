"""Celery tasks for the trading platform."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from trader.celery_app import celery
from trader.config import get_settings
from trader.core.events import BarEvent
from trader.core.redis_client import init_redis
from trader.db.session import get_session_factory, init_engine
from trader.features.engine import FeatureEngine
from trader.providers.broker.factory import create_broker_provider
from trader.db.repositories.fills import FillRepository
from trader.db.repositories.orders import OrderRepository
from trader.execution.lifecycle import OrderLifecycleTracker
from trader.execution.position_ledger import PositionLedger
from trader.services.model_loader import ChampionModelLoader
from trader.services.system_state import SystemStateStore
from trader.services.trading_cycle import TradingCycleService
from trader.strategy.engine import TradeIntentParams

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
            state_store = SystemStateStore()
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
            spread_bps = await state_store.get_spread_bps(symbol)
            features = engine.compute_features(
                symbol,
                bars[-1].timestamp if bars else None,
                spread_bps=spread_bps,
            )
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


@celery.task(name="trader.workers.tasks.manage_open_orders")
def manage_open_orders() -> dict:
    """Synchronize open orders with the broker and cancel stale orders."""

    async def _run() -> dict:
        settings, factory, broker = _bootstrap()
        await _ensure_redis()
        async with factory() as session:
            orders = OrderRepository(session)
            fills = FillRepository(session)
            lifecycle = OrderLifecycleTracker(session)
            ledger = PositionLedger(session)

            open_orders = await orders.get_open_orders()
            synced = 0
            cancelled = 0
            filled = 0
            fill_recorded = 0
            now = datetime.now(timezone.utc)

            for order in open_orders:
                if not order.broker_order_id:
                    continue

                broker_order = await broker.get_order(order.broker_order_id)
                await lifecycle.update_status(
                    order_id=order.id,
                    new_status=broker_order.status,
                    broker_order_id=broker_order.broker_order_id,
                    filled_qty=broker_order.filled_qty,
                    filled_avg_price=float(broker_order.filled_avg_price)
                    if broker_order.filled_avg_price is not None
                    else None,
                )
                synced += 1

                if broker_order.status == "filled" and broker_order.filled_avg_price is not None:
                    existing_fill = await fills.get_latest_by_order_id(order.id)
                    if existing_fill is None:
                        await lifecycle.record_fill(
                            order_id=order.id,
                            broker_order_id=broker_order.broker_order_id,
                            symbol=order.symbol,
                            side=order.side,
                            qty=broker_order.filled_qty,
                            price=float(broker_order.filled_avg_price),
                        )
                        synthetic_intent = TradeIntentParams(
                            symbol=order.symbol,
                            side=order.side,
                            qty=broker_order.filled_qty,
                            entry_order_type=order.order_type,
                            limit_price=order.limit_price,
                            strategy_tag=order.strategy_tag,
                            rationale=order.rationale,
                        )
                        await ledger.apply_fill(
                            order=order,
                            intent=synthetic_intent,
                            fill_price=broker_order.filled_avg_price,
                            fill_qty=broker_order.filled_qty,
                            commission=Decimal("0"),
                            timestamp=order.filled_at or now,
                        )
                        fill_recorded += 1
                    filled += 1
                    continue

                order_timestamp = order.submitted_at or order.created_at
                if order_timestamp is None:
                    continue
                age_seconds = (now - order_timestamp).total_seconds()
                if age_seconds <= settings.open_order_stale_seconds:
                    continue
                if broker_order.status not in {"pending", "submitted", "accepted", "partially_filled"}:
                    continue

                cancelled_order = await broker.cancel_order(order.broker_order_id)
                await lifecycle.update_status(
                    order_id=order.id,
                    new_status=cancelled_order.status,
                    broker_order_id=cancelled_order.broker_order_id,
                )
                cancelled += 1

            await session.commit()
            result = {
                "open_orders_seen": len(open_orders),
                "synced": synced,
                "filled": filled,
                "fills_recorded": fill_recorded,
                "cancelled": cancelled,
            }
            logger.info("open_orders_managed", **result)
            return result

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
