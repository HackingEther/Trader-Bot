"""Broker-truth order/fill streaming via Alpaca TradingStream."""

from __future__ import annotations

import asyncio
import queue
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from trader.db.session import get_session_factory
from trader.execution.lifecycle import OrderLifecycleTracker
from trader.execution.position_ledger import PositionLedger
from trader.db.repositories.execution_attribution import ExecutionAttributionRepository
from trader.db.repositories.fills import FillRepository
from trader.db.repositories.orders import OrderRepository
from trader.db.repositories.quote_snapshots import QuoteSnapshotRepository
from trader.db.repositories.trade_intents import TradeIntentRepository
from trader.strategy.engine import TradeIntentParams

if TYPE_CHECKING:
    from trader.providers.broker.base import BrokerProvider

logger = structlog.get_logger(__name__)

TRADE_UPDATES_HEALTH_KEY = "trader:health:trade_updates_last_event"


def _coerce_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _execution_key(
    broker_order_id: str | None,
    execution_id: str | None,
    cumulative_qty: int,
    timestamp: datetime | None,
    delta_qty: int,
) -> str | None:
    """Unique key for idempotent fill recording. Prefer execution_id; fallback uses cumulative for uniqueness."""
    if execution_id:
        return str(execution_id)
    if broker_order_id and cumulative_qty > 0:
        return f"{broker_order_id}:{cumulative_qty}"
    if broker_order_id and timestamp is not None and delta_qty > 0:
        return f"{broker_order_id}:{timestamp.isoformat()}:{delta_qty}"
    return None


def _tracked_fill_qty(fills: list) -> int:
    return sum(int(f.qty) for f in fills)


def _tracked_fill_notional(fills: list) -> Decimal:
    return sum((Decimal(str(f.price)) * int(f.qty) for f in fills), Decimal("0"))


def _incremental_fill_price(
    cumulative_qty: int,
    cumulative_avg_price: float | None,
    recorded_qty: int,
    recorded_notional: Decimal,
) -> Decimal:
    if cumulative_avg_price is None or cumulative_qty <= 0:
        return Decimal("0")
    total_notional = Decimal(str(cumulative_avg_price)) * cumulative_qty
    delta_qty = cumulative_qty - recorded_qty
    if delta_qty <= 0:
        return Decimal(str(cumulative_avg_price))
    delta_notional = total_notional - recorded_notional
    return delta_notional / delta_qty


class TradeUpdateStream:
    """Streams Alpaca trade updates into local DB and position ledger."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        queue_maxsize: int = 1000,
        broker: BrokerProvider | None = None,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._paper = paper
        self._broker = broker
        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._running = False
        self._stream: object | None = None
        self._run_task: asyncio.Task | None = None
        self._consumer_task: asyncio.Task | None = None

    def _on_trade_update(self, update: object) -> None:
        """Callback from Alpaca stream (runs in stream thread). Enqueue for processing."""
        try:
            self._queue.put_nowait(update)
        except queue.Full:
            logger.warning("trade_update_queue_full")

    async def _process_update(self, update: object) -> None:
        """Process a trade update in the main event loop."""
        try:
            ev = getattr(update, "event", "")
            event = getattr(ev, "value", str(ev)).lower()
            order_obj = getattr(update, "order", None)
            if order_obj is None:
                return
            broker_order_id = str(getattr(order_obj, "id", ""))
            if not broker_order_id:
                return

            try:
                factory = get_session_factory()
            except RuntimeError:
                logger.debug("trade_update_no_session_factory")
                return

            async with factory() as session:
                orders = OrderRepository(session)
                fills = FillRepository(session)
                lifecycle = OrderLifecycleTracker(session)
                ledger = PositionLedger(session)
                trade_intents = TradeIntentRepository(session)
                quote_snapshots = QuoteSnapshotRepository(session)
                attribution_repo = ExecutionAttributionRepository(session)

                local_order = await orders.get_by_broker_order_id(broker_order_id)
                if local_order is None and self._broker is not None:
                    local_order = await self._resolve_bracket_leg(
                        broker_order_id=broker_order_id,
                        orders=orders,
                        order_obj=order_obj,
                    )
                if local_order is None:
                    logger.debug("trade_update_unknown_order", broker_order_id=broker_order_id)
                    return

                status = str(getattr(order_obj, "status", "")).lower()
                if event == "replaced":
                    await lifecycle.update_status(
                        order_id=local_order.id,
                        new_status=status,
                        broker_order_id=broker_order_id,
                        filled_qty=int(getattr(order_obj, "filled_qty", 0) or 0),
                        filled_avg_price=float(f) if (f := getattr(order_obj, "filled_avg_price", None)) else None,
                    )
                    await self._record_health_event()
                    await session.commit()
                    return
                filled_qty = int(getattr(order_obj, "filled_qty", 0) or 0)
                filled_avg_price = getattr(order_obj, "filled_avg_price", None)
                if filled_avg_price is not None:
                    filled_avg_price = float(filled_avg_price)

                await lifecycle.update_status(
                    order_id=local_order.id,
                    new_status=status,
                    broker_order_id=broker_order_id,
                    filled_qty=filled_qty,
                    filled_avg_price=filled_avg_price,
                )

                if local_order.trade_intent_id is not None:
                    if status in {"partially_filled", "filled"}:
                        await trade_intents.update_by_id(local_order.trade_intent_id, status="executed")
                    elif status in {"rejected", "failed"}:
                        await trade_intents.update_by_id(local_order.trade_intent_id, status="rejected")
                    elif status in {"cancelled", "expired"}:
                        await trade_intents.update_by_id(local_order.trade_intent_id, status="cancelled")

                if event in ("fill", "partial_fill") and filled_qty > 0 and filled_avg_price is not None:
                    existing_fills = await fills.get_by_order_id(local_order.id)
                    recorded_qty = _tracked_fill_qty(existing_fills)
                    recorded_notional = _tracked_fill_notional(existing_fills)
                    delta_qty = filled_qty - recorded_qty
                    if delta_qty > 0:
                        fill_price = getattr(update, "price", None)
                        fill_qty_attr = getattr(update, "qty", None)
                        if fill_price is not None and fill_qty_attr is not None:
                            delta_price = float(fill_price)
                            delta_qty_val = int(fill_qty_attr)
                        else:
                            delta_price = float(
                                _incremental_fill_price(
                                    filled_qty,
                                    filled_avg_price,
                                    recorded_qty,
                                    recorded_notional,
                                )
                            )
                            delta_qty_val = delta_qty

                        exec_id = getattr(update, "execution_id", None)
                        ts = _coerce_utc(getattr(update, "timestamp", None)) or datetime.now(timezone.utc)
                        exec_key = _execution_key(
                            broker_order_id,
                            str(exec_id) if exec_id else None,
                            filled_qty,
                            ts,
                            delta_qty_val,
                        )

                        _, is_new = await lifecycle.record_fill(
                            order_id=local_order.id,
                            broker_order_id=broker_order_id,
                            symbol=local_order.symbol,
                            side=local_order.side,
                            qty=delta_qty_val,
                            price=delta_price,
                            execution_key=exec_key,
                            broker_execution_timestamp=ts,
                            timestamp=ts,
                            raw=order_obj.model_dump() if hasattr(order_obj, "model_dump") else {"order_id": broker_order_id},
                        )
                        if is_new:
                            synthetic_intent = TradeIntentParams(
                                symbol=local_order.symbol,
                                side=local_order.side,
                                qty=delta_qty_val,
                                entry_order_type=local_order.order_type,
                                limit_price=local_order.limit_price,
                                strategy_tag=local_order.strategy_tag,
                                rationale=local_order.rationale or {},
                            )
                            await ledger.apply_fill(
                                order=local_order,
                                intent=synthetic_intent,
                                fill_price=Decimal(str(delta_price)),
                                fill_qty=delta_qty_val,
                                commission=Decimal("0"),
                                timestamp=ts,
                            )
                            logger.info(
                                "trade_update_fill_applied",
                                order_id=local_order.id,
                                symbol=local_order.symbol,
                                qty=delta_qty_val,
                                price=delta_price,
                            )
                if status == "filled":
                    order_for_attr = await orders.get_by_id(local_order.id)
                    if order_for_attr:
                        order_fills = await fills.get_by_order_id(local_order.id)
                        await attribution_repo.ensure_attribution_for_filled_order(
                            order=order_for_attr,
                            fills=order_fills,
                            quote_snapshots=quote_snapshots,
                        )
                await self._record_health_event()
                await session.commit()
        except Exception as exc:
            logger.exception("trade_update_process_failed", error=str(exc))

    async def _resolve_bracket_leg(
        self,
        *,
        broker_order_id: str,
        orders: OrderRepository,
        order_obj: object,
    ):
        """Fetch order from broker; if it is a bracket leg, create child and return it."""
        try:
            broker_order = await self._broker.get_order(broker_order_id)
        except Exception as exc:
            logger.debug("trade_update_broker_fetch_failed", broker_order_id=broker_order_id, error=str(exc))
            return None
        raw = getattr(broker_order, "raw", {}) or {}
        parent_id = raw.get("parent_order_id")
        if not parent_id:
            return None
        parent_order = await orders.get_by_broker_order_id(str(parent_id))
        if parent_order is None:
            return None
        leg_symbol = broker_order.symbol or parent_order.symbol
        leg_side = broker_order.side or ("sell" if parent_order.side == "buy" else "buy")
        leg_qty = broker_order.qty or parent_order.qty
        leg_status = broker_order.status or "pending"
        leg_type = broker_order.order_type or "market"
        child_order, _ = await orders.create_idempotent(
            idempotency_key=f"broker-child:{broker_order_id}",
            trade_intent_id=parent_order.trade_intent_id,
            broker_order_id=broker_order_id,
            symbol=leg_symbol,
            side=leg_side,
            order_type=leg_type,
            order_class="simple",
            qty=leg_qty,
            limit_price=broker_order.limit_price,
            stop_price=broker_order.stop_price,
            status=leg_status,
            strategy_tag=parent_order.strategy_tag,
            rationale=parent_order.rationale or {},
            broker_metadata={**raw, "parent_order_id": str(parent_id)},
        )
        logger.info(
            "trade_update_bracket_leg_created",
            broker_order_id=broker_order_id,
            parent_broker_id=str(parent_id),
        )
        return child_order

    async def _record_health_event(self) -> None:
        """Record last processed event timestamp for health checks."""
        try:
            from trader.core.redis_client import set_cached

            await set_cached(
                TRADE_UPDATES_HEALTH_KEY,
                datetime.now(timezone.utc).isoformat(),
                ttl_seconds=600,
            )
        except Exception as exc:
            logger.debug("trade_update_health_record_failed", error=str(exc))

    async def _consumer(self) -> None:
        """Consume trade updates from queue."""
        try:
            while self._running:
                try:
                    update = self._queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue
                try:
                    await self._process_update(update)
                except Exception as exc:
                    logger.exception("trade_update_consumer_error", error=str(exc))
        except asyncio.CancelledError:
            pass

    async def start(self) -> None:
        """Start the trade update stream and consumer."""
        if not self._api_key or not self._api_secret:
            logger.warning("trade_updates_stream_skipped", reason="no_credentials")
            return
        try:
            from alpaca.trading.stream import TradingStream

            self._stream = TradingStream(
                api_key=self._api_key,
                secret_key=self._api_secret,
                paper=self._paper,
            )
            self._stream.subscribe_trade_updates(self._on_trade_update)
            self._running = True
            self._consumer_task = asyncio.create_task(self._consumer())
            self._run_task = asyncio.create_task(asyncio.to_thread(self._stream.run))
            logger.info("trade_updates_stream_started", paper=self._paper)
        except ImportError as e:
            logger.warning("trade_updates_stream_import_failed", error=str(e))
        except Exception as e:
            logger.exception("trade_updates_stream_start_failed", error=str(e))

    async def stop(self) -> None:
        """Stop the stream and consumer."""
        self._running = False
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
        if self._stream is not None:
            try:
                if hasattr(self._stream, "stop"):
                    self._stream.stop()
                elif hasattr(self._stream, "stop_ws"):
                    await self._stream.stop_ws()
            except Exception as e:
                logger.warning("trade_updates_stream_stop_failed", error=str(e))
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
        logger.info("trade_updates_stream_stopped")
