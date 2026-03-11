"""Celery tasks for the trading platform."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import structlog

from trader.celery_app import celery
from trader.config import get_settings
from trader.core.events import BarEvent
from trader.core.redis_client import init_redis
from trader.db.session import get_session_factory, init_engine, reset_engine_for_celery
from trader.features.engine import FeatureEngine
from trader.providers.broker.factory import create_broker_provider
from trader.db.repositories.fills import FillRepository
from trader.db.repositories.orders import OrderRepository
from trader.db.repositories.trade_intents import TradeIntentRepository
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


def _tracked_fill_qty(existing_fills: list) -> int:
    """Sum recorded fill quantity for an order."""
    return sum(int(fill.qty) for fill in existing_fills)


def _tracked_fill_notional(existing_fills: list) -> Decimal:
    """Sum recorded fill notional for an order."""
    return sum((Decimal(fill.price) * int(fill.qty) for fill in existing_fills), Decimal("0"))


def _incremental_fill_price(
    *,
    cumulative_qty: int,
    cumulative_avg_price: Decimal,
    recorded_qty: int,
    recorded_notional: Decimal,
) -> Decimal:
    """Infer delta fill price from cumulative broker state when no execution details exist."""
    total_notional = cumulative_avg_price * cumulative_qty
    delta_qty = cumulative_qty - recorded_qty
    if delta_qty <= 0:
        return cumulative_avg_price
    delta_notional = total_notional - recorded_notional
    return delta_notional / delta_qty


def _coerce_utc(value: datetime) -> datetime:
    """Normalize timestamps loaded from different backends."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _fill_timestamp_from_raw(raw: dict | None, fallback: datetime) -> datetime:
    if not raw:
        return fallback
    for key in ("filled_at", "updated_at", "submitted_at"):
        value = raw.get(key)
        if not value:
            continue
        try:
            return _coerce_utc(datetime.fromisoformat(str(value)))
        except ValueError:
            continue
    return fallback


def _execution_key(broker_order_id: str | None, cumulative_qty: int) -> str | None:
    if not broker_order_id or cumulative_qty <= 0:
        return None
    return f"{broker_order_id}:{cumulative_qty}"


def _embedded_leg_orders(snapshot) -> list[SimpleNamespace]:
    legs = []
    raw = getattr(snapshot, "raw", None) or {}
    for leg in raw.get("legs", []) or []:
        legs.append(
            SimpleNamespace(
                broker_order_id=str(leg.get("alpaca_id") or ""),
                symbol=str(leg.get("symbol") or getattr(snapshot, "symbol", "")),
                side=str(leg.get("side") or ""),
                order_type=str(leg.get("order_type") or "market"),
                qty=int(leg.get("qty") or getattr(snapshot, "qty", 0)),
                limit_price=Decimal(str(leg["limit_price"])) if leg.get("limit_price") not in (None, "") else None,
                stop_price=Decimal(str(leg["stop_price"])) if leg.get("stop_price") not in (None, "") else None,
                filled_qty=int(leg.get("filled_qty") or 0),
                filled_avg_price=Decimal(str(leg["filled_avg_price"]))
                if leg.get("filled_avg_price") not in (None, "")
                else None,
                status=str(leg.get("status") or "pending"),
                order_class="simple",
                time_in_force=str(leg.get("time_in_force") or "day"),
                raw=leg,
            )
        )
    return [leg for leg in legs if leg.broker_order_id]


@celery.task(name="trader.workers.tasks.heartbeat_system")
def heartbeat_system() -> dict:
    """System heartbeat task - runs every 30 seconds."""
    ts = datetime.now(timezone.utc).isoformat()
    logger.info("heartbeat", timestamp=ts)
    return {"status": "alive", "timestamp": ts}


@celery.task(name="trader.workers.tasks.compute_features")
def compute_features(symbol: str) -> dict:
    """Compute features for a symbol using the most recent persisted bars."""
    reset_engine_for_celery()

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
    reset_engine_for_celery()

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
    reset_engine_for_celery()

    async def _run() -> dict:
        settings, factory, broker = _bootstrap()
        await _ensure_redis()
        async with factory() as session:
            orders = OrderRepository(session)
            fills = FillRepository(session)
            trade_intents = TradeIntentRepository(session)
            lifecycle = OrderLifecycleTracker(session)
            ledger = PositionLedger(session)

            async def _sync_snapshot(order, broker_order) -> tuple[int, int]:
                recorded = 0
                completed = 0
                await lifecycle.update_status(
                    order_id=order.id,
                    new_status=broker_order.status,
                    broker_order_id=broker_order.broker_order_id,
                    filled_qty=broker_order.filled_qty,
                    filled_avg_price=float(broker_order.filled_avg_price)
                    if broker_order.filled_avg_price is not None
                    else None,
                )
                order.broker_metadata = getattr(broker_order, "raw", {}) or {}

                if order.trade_intent_id is not None:
                    if broker_order.status in {"partially_filled", "filled"}:
                        await trade_intents.update_by_id(order.trade_intent_id, status="executed")
                    elif broker_order.status in {"rejected", "failed"}:
                        await trade_intents.update_by_id(order.trade_intent_id, status="rejected")
                    elif broker_order.status in {"cancelled", "expired"}:
                        await trade_intents.update_by_id(order.trade_intent_id, status="cancelled")

                if (
                    broker_order.status in {"partially_filled", "filled"}
                    and broker_order.filled_avg_price is not None
                    and broker_order.filled_qty > 0
                ):
                    existing_fills = await fills.get_by_order_id(order.id)
                    recorded_qty = _tracked_fill_qty(existing_fills)
                    recorded_notional = _tracked_fill_notional(existing_fills)
                    delta_qty = int(broker_order.filled_qty) - recorded_qty
                    if delta_qty > 0:
                        delta_fill_price = _incremental_fill_price(
                            cumulative_qty=int(broker_order.filled_qty),
                            cumulative_avg_price=broker_order.filled_avg_price,
                            recorded_qty=recorded_qty,
                            recorded_notional=recorded_notional,
                        )
                        fill_timestamp = _fill_timestamp_from_raw(
                            getattr(broker_order, "raw", None),
                            order.filled_at or order.submitted_at or now,
                        )
                        await lifecycle.record_fill(
                            order_id=order.id,
                            broker_order_id=broker_order.broker_order_id,
                            symbol=order.symbol,
                            side=order.side,
                            qty=delta_qty,
                            price=float(delta_fill_price),
                            execution_key=_execution_key(broker_order.broker_order_id, int(broker_order.filled_qty)),
                            broker_execution_timestamp=fill_timestamp,
                            timestamp=fill_timestamp,
                            raw=getattr(broker_order, "raw", None),
                        )
                        synthetic_intent = TradeIntentParams(
                            symbol=order.symbol,
                            side=order.side,
                            qty=delta_qty,
                            entry_order_type=order.order_type,
                            limit_price=order.limit_price,
                            strategy_tag=order.strategy_tag,
                            rationale=order.rationale,
                        )
                        await ledger.apply_fill(
                            order=order,
                            intent=synthetic_intent,
                            fill_price=delta_fill_price,
                            fill_qty=delta_qty,
                            commission=Decimal("0"),
                            timestamp=fill_timestamp,
                        )
                        recorded += 1

                for leg_snapshot in _embedded_leg_orders(broker_order):
                    child_order = await orders.get_by_broker_order_id(leg_snapshot.broker_order_id)
                    if child_order is None:
                        child_order, _ = await orders.create_idempotent(
                            idempotency_key=f"broker-child:{leg_snapshot.broker_order_id}",
                            trade_intent_id=order.trade_intent_id,
                            broker_order_id=leg_snapshot.broker_order_id,
                            symbol=leg_snapshot.symbol or order.symbol,
                            side=leg_snapshot.side or ("sell" if order.side == "buy" else "buy"),
                            order_type=leg_snapshot.order_type,
                            order_class="simple",
                            qty=leg_snapshot.qty,
                            limit_price=leg_snapshot.limit_price,
                            stop_price=leg_snapshot.stop_price,
                            status=leg_snapshot.status,
                            strategy_tag=order.strategy_tag,
                            rationale=order.rationale,
                            broker_metadata={
                                **getattr(leg_snapshot, "raw", {}),
                                "parent_order_id": order.broker_order_id,
                            },
                        )
                    child_recorded, child_completed = await _sync_snapshot(child_order, leg_snapshot)
                    recorded += child_recorded
                    completed += child_completed

                if broker_order.status == "filled":
                    completed += 1
                return recorded, completed

            open_orders = await orders.get_open_orders()
            synced = 0
            cancelled = 0
            filled = 0
            fill_recorded = 0
            errors = 0
            now = datetime.now(timezone.utc)

            for order in open_orders:
                if not order.broker_order_id:
                    continue

                try:
                    broker_order = await broker.get_order(order.broker_order_id)
                    recorded_delta, completed_delta = await _sync_snapshot(order, broker_order)
                    synced += 1
                    fill_recorded += recorded_delta
                    filled += completed_delta
                    if broker_order.status == "filled":
                        continue

                    if broker_order.status == "partially_filled":
                        continue

                    age_anchor = _coerce_utc(order.submitted_at or order.created_at)
                    age_seconds = (now - age_anchor).total_seconds()
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
                except Exception as exc:
                    errors += 1
                    logger.exception(
                        "open_order_sync_failed",
                        order_id=order.id,
                        broker_order_id=order.broker_order_id,
                        error=str(exc),
                    )

            await session.commit()
            result = {
                "open_orders_seen": len(open_orders),
                "synced": synced,
                "filled": filled,
                "fills_recorded": fill_recorded,
                "cancelled": cancelled,
                "errors": errors,
            }
            logger.info("open_orders_managed", **result)
            return result

    return asyncio.run(_run())


@celery.task(name="trader.workers.tasks.execute_trading_cycle")
def execute_trading_cycle() -> dict:
    """Run the full periodic trading cycle."""
    reset_engine_for_celery()

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
    reset_engine_for_celery()

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
            result = await service.reconcile_positions(auto_fix=False, halt_on_discrepancy=True)
            logger.info("reconciliation_task_completed", **result)
            return result

    return asyncio.run(_run())


@celery.task(name="trader.workers.tasks.snapshot_pnl")
def snapshot_pnl() -> dict:
    """Take a P&L snapshot."""
    reset_engine_for_celery()

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
