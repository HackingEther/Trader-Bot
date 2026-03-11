"""Order lifecycle tracking."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from trader.core.enums import OrderStatus
from trader.db.models.order import Order
from trader.db.models.fill import Fill
from trader.db.repositories.fills import FillRepository
from trader.db.repositories.orders import OrderRepository

logger = structlog.get_logger(__name__)

_STATUS_RANK = {
    OrderStatus.PENDING: 0,
    OrderStatus.SUBMITTED: 1,
    OrderStatus.ACCEPTED: 2,
    OrderStatus.PARTIALLY_FILLED: 3,
    OrderStatus.FILLED: 4,
    OrderStatus.CANCELLED: 5,
    OrderStatus.REJECTED: 5,
    OrderStatus.EXPIRED: 5,
    OrderStatus.FAILED: 5,
}

_TERMINAL_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
    OrderStatus.EXPIRED,
    OrderStatus.FAILED,
}


class OrderLifecycleTracker:
    """Tracks order state transitions and records fills."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = OrderRepository(session)
        self._fills = FillRepository(session)

    async def update_status(
        self,
        order_id: int,
        new_status: str,
        broker_order_id: str | None = None,
        filled_qty: int | None = None,
        filled_avg_price: float | None = None,
        error_message: str | None = None,
    ) -> Order | None:
        """Update order status and related fields."""
        updates: dict = {}
        now = datetime.now(timezone.utc)
        existing = await self._repo.get_by_id(order_id)
        resolved_status = new_status

        if existing is not None and existing.status in _TERMINAL_STATUSES:
            existing_rank = _STATUS_RANK.get(existing.status, -1)
            next_rank = _STATUS_RANK.get(new_status, -1)
            if next_rank < existing_rank or (
                existing.status != OrderStatus.PARTIALLY_FILLED and existing.status != new_status
            ):
                resolved_status = existing.status

        if existing is None or resolved_status != existing.status:
            updates["status"] = resolved_status

        if broker_order_id:
            updates["broker_order_id"] = broker_order_id
        if filled_qty is not None:
            updates["filled_qty"] = filled_qty
        if filled_avg_price is not None:
            from decimal import Decimal
            updates["filled_avg_price"] = Decimal(str(filled_avg_price))
        if error_message:
            updates["error_message"] = error_message

        if (
            resolved_status in (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED)
            and existing is not None
            and existing.submitted_at is None
        ):
            updates["submitted_at"] = now
        elif resolved_status == OrderStatus.FILLED:
            if existing is not None and existing.submitted_at is None:
                updates["submitted_at"] = now
            updates["filled_at"] = now
        elif resolved_status == OrderStatus.CANCELLED:
            updates["cancelled_at"] = now

        if not updates:
            return existing

        order = await self._repo.update_by_id(order_id, **updates)
        if order:
            logger.info(
                "order_status_updated",
                order_id=order_id,
                new_status=resolved_status,
                broker_order_id=broker_order_id,
            )
        return order

    async def record_fill(
        self,
        order_id: int,
        broker_order_id: str | None,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        commission: float = 0.0,
        timestamp: datetime | None = None,
        execution_key: str | None = None,
        broker_execution_timestamp: datetime | None = None,
        raw: dict | None = None,
    ) -> tuple[Fill, bool]:
        """Record a fill event for an order. Returns (fill, is_new)."""
        from decimal import Decimal
        if execution_key is not None:
            existing = await self._fills.get_by_execution_key(execution_key)
            if existing is not None:
                return existing, False
        fill = Fill(
            order_id=order_id,
            execution_key=execution_key,
            broker_order_id=broker_order_id,
            broker_execution_timestamp=broker_execution_timestamp,
            symbol=symbol,
            side=side,
            qty=qty,
            price=Decimal(str(price)),
            commission=Decimal(str(commission)),
            timestamp=timestamp or datetime.now(timezone.utc),
            raw=raw or {},
        )
        self._session.add(fill)
        await self._session.flush()
        logger.info(
            "fill_recorded",
            order_id=order_id,
            symbol=symbol,
            qty=qty,
            price=price,
        )
        return fill, True
