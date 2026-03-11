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
from trader.execution.idempotency import generate_idempotency_key
from trader.execution.lifecycle import OrderLifecycleTracker
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
        self._lifecycle = OrderLifecycleTracker(session)
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

        existing = await self._repo.get_by_idempotency_key(idem_key)
        if existing:
            logger.warning(
                "duplicate_order_blocked",
                idempotency_key=idem_key,
                existing_order_id=existing.id,
                symbol=intent.symbol,
            )
            raise DuplicateOrderError(f"Order already exists with key {idem_key}")

        order_class = "simple"
        if intent.stop_loss and intent.take_profit:
            order_class = "bracket"

        order = await self._repo.create(
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
                await self._lifecycle.record_fill(
                    order_id=order.id,
                    broker_order_id=broker_order.broker_order_id,
                    symbol=intent.symbol,
                    side=intent.side,
                    qty=broker_order.filled_qty,
                    price=float(broker_order.filled_avg_price),
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
