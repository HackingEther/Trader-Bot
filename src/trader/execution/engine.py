"""Execution engine - submit orders through broker provider with safety controls."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from trader.core.enums import OrderStatus
from trader.core.exceptions import DuplicateOrderError, OrderSubmissionError
from trader.db.models.order import Order
from trader.db.repositories.orders import OrderRepository
from trader.db.repositories.quote_snapshots import QuoteSnapshotRepository
from trader.execution.idempotency import generate_idempotency_key
from trader.execution.lifecycle import OrderLifecycleTracker
from trader.execution.position_ledger import PositionLedger
from trader.providers.broker.base import BrokerProvider, OrderRequest
from trader.services.system_state import SystemStateStore
from trader.strategy.engine import TradeIntentParams

logger = structlog.get_logger(__name__)


class ExecutionEngine:
    """Submits approved trade intents to the broker with idempotency protection."""

    def __init__(
        self,
        broker: BrokerProvider,
        session: AsyncSession,
        state_store: SystemStateStore | None = None,
    ) -> None:
        self._broker = broker
        self._session = session
        self._repo = OrderRepository(session)
        self._quote_snapshots = QuoteSnapshotRepository(session)
        self._lifecycle = OrderLifecycleTracker(session)
        self._positions = PositionLedger(session)
        self._state_store = state_store or SystemStateStore()

    async def execute(self, intent: TradeIntentParams, trade_intent_id: int | None = None) -> Order:
        """Execute a trade intent by submitting an order to the broker.

        Raises DuplicateOrderError if the idempotency key already exists.
        """
        if await self._state_store.is_kill_switch_active():
            from trader.core.exceptions import KillSwitchActiveError

            raise KillSwitchActiveError("Kill switch is active")

        idem_key = generate_idempotency_key(
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            strategy_tag=intent.strategy_tag,
            trade_intent_id=trade_intent_id,
        )

        order_class = "simple"
        if intent.stop_loss and intent.take_profit:
            order_class = "bracket"

        order, created = await self._repo.create_idempotent(
            idempotency_key=idem_key,
            trade_intent_id=trade_intent_id,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.entry_order_type,
            order_class=order_class,
            qty=intent.qty,
            limit_price=intent.limit_price,
            stop_price=intent.stop_loss if intent.entry_order_type in ("stop", "stop_limit") else None,
            status=OrderStatus.PENDING,
            strategy_tag=intent.strategy_tag,
            rationale=intent.rationale,
        )
        if not created:
            logger.warning(
                "duplicate_order_blocked",
                idempotency_key=idem_key,
                existing_order_id=order.id,
                symbol=intent.symbol,
            )
            raise DuplicateOrderError(f"Order already exists with key {idem_key}")

        try:
            broker_request = OrderRequest(
                symbol=intent.symbol,
                side=intent.side,
                qty=intent.qty,
                order_type=intent.entry_order_type,
                limit_price=intent.limit_price,
                order_class=order_class,
                take_profit_price=intent.take_profit,
                stop_loss_price=intent.stop_loss,
                client_order_id=idem_key[:20],
            )

            broker_order = await self._broker.submit_order(broker_request)
            submit_ts = datetime.now(timezone.utc)
            quote = await self._state_store.get_last_quote(intent.symbol)
            if quote and quote.get("bid") is not None and quote.get("ask") is not None:
                bid = Decimal(str(quote["bid"]))
                ask = Decimal(str(quote["ask"]))
                mid = (bid + ask) / 2
                spread_bps = Decimal(str(quote.get("spread_bps", 0) or 0))
                await self._quote_snapshots.create_snapshot(
                    snapshot_type="submit",
                    symbol=intent.symbol,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    timestamp=submit_ts,
                    spread_bps=spread_bps,
                    trade_intent_id=trade_intent_id,
                    order_id=order.id,
                )
            await self._sync_child_orders(parent_order=order, broker_order=broker_order)

            await self._lifecycle.update_status(
                order_id=order.id,
                new_status=broker_order.status,
                broker_order_id=broker_order.broker_order_id,
                filled_qty=broker_order.filled_qty,
                filled_avg_price=float(broker_order.filled_avg_price) if broker_order.filled_avg_price else None,
            )

            order.broker_order_id = broker_order.broker_order_id
            order.status = broker_order.status
            order.broker_metadata = broker_order.raw

            if broker_order.status == OrderStatus.FILLED and broker_order.filled_avg_price:
                fill_timestamp = self._fill_timestamp_from_raw(broker_order.raw) or order.filled_at or datetime.now(timezone.utc)
                _, is_new = await self._lifecycle.record_fill(
                    order_id=order.id,
                    broker_order_id=broker_order.broker_order_id,
                    symbol=intent.symbol,
                    side=intent.side,
                    qty=broker_order.filled_qty,
                    price=float(broker_order.filled_avg_price),
                    execution_key=f"{broker_order.broker_order_id}:{broker_order.filled_qty}",
                    broker_execution_timestamp=fill_timestamp,
                    timestamp=fill_timestamp,
                    raw=broker_order.raw,
                )
                if is_new:
                    await self._positions.apply_fill(
                        order=order,
                        intent=intent,
                        fill_price=broker_order.filled_avg_price,
                        fill_qty=broker_order.filled_qty,
                        timestamp=fill_timestamp,
                    )

            logger.info(
                "order_executed",
                order_id=order.id,
                broker_order_id=broker_order.broker_order_id,
                status=broker_order.status,
                symbol=intent.symbol,
            )
            return order

        except OrderSubmissionError as e:
            await self._lifecycle.update_status(
                order_id=order.id,
                new_status=OrderStatus.FAILED,
                error_message=str(e),
            )
            logger.error("order_execution_failed", order_id=order.id, error=str(e))
            raise
        except Exception as e:
            await self._lifecycle.update_status(
                order_id=order.id,
                new_status=OrderStatus.FAILED,
                error_message=str(e),
            )
            logger.error("order_execution_unexpected_error", order_id=order.id, error=str(e))
            raise OrderSubmissionError(f"Unexpected error: {e}") from e

    async def cancel(self, order_id: int) -> Order | None:
        """Cancel an order by its internal ID."""
        order = await self._repo.get_by_id(order_id)
        if not order or not order.broker_order_id:
            return None

        broker_order = await self._broker.cancel_order(order.broker_order_id)
        await self._lifecycle.update_status(
            order_id=order.id,
            new_status=broker_order.status,
        )
        return order

    async def _sync_child_orders(self, *, parent_order: Order, broker_order) -> None:
        for leg in broker_order.raw.get("legs", []):
            leg_id = leg.get("alpaca_id")
            if not leg_id:
                continue
            existing = await self._repo.get_by_broker_order_id(str(leg_id))
            if existing is not None:
                continue
            leg_limit_price = self._to_decimal(leg.get("limit_price"))
            leg_stop_price = self._to_decimal(leg.get("stop_price"))
            child_side = str(leg.get("side") or ("sell" if parent_order.side == "buy" else "buy"))
            child_qty = int(leg.get("qty") or parent_order.qty)
            child_status = str(leg.get("status") or OrderStatus.PENDING)
            child_type = str(leg.get("order_type") or "market")
            child_metadata = {
                **leg,
                "parent_order_id": broker_order.broker_order_id,
                "reference_price": (parent_order.rationale or {}).get("reference_price"),
            }
            await self._repo.create_idempotent(
                idempotency_key=f"broker-child:{leg_id}",
                trade_intent_id=parent_order.trade_intent_id,
                broker_order_id=str(leg_id),
                symbol=leg.get("symbol") or parent_order.symbol,
                side=child_side,
                order_type=child_type,
                order_class="simple",
                qty=child_qty,
                limit_price=leg_limit_price,
                stop_price=leg_stop_price,
                status=child_status,
                strategy_tag=parent_order.strategy_tag,
                rationale=parent_order.rationale,
                broker_metadata=child_metadata,
            )

    @staticmethod
    def _fill_timestamp_from_raw(raw: dict) -> datetime | None:
        timestamp_value = raw.get("filled_at") or raw.get("updated_at") or raw.get("submitted_at")
        if not timestamp_value:
            return None
        try:
            return datetime.fromisoformat(str(timestamp_value))
        except ValueError:
            return None

    @staticmethod
    def _to_decimal(value: object) -> Decimal | None:
        if value in (None, ""):
            return None
        try:
            return Decimal(str(value))
        except Exception:
            return None
