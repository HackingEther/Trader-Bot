"""Order lifecycle tracking."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from trader.core.enums import OrderStatus
from trader.db.models.order import Order
from trader.db.models.fill import Fill
from trader.db.repositories.orders import OrderRepository

logger = structlog.get_logger(__name__)


class OrderLifecycleTracker:
    """Tracks order state transitions and records fills."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = OrderRepository(session)

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
        updates: dict = {"status": new_status}
        now = datetime.now(timezone.utc)

        if broker_order_id:
            updates["broker_order_id"] = broker_order_id
        if filled_qty is not None:
            updates["filled_qty"] = filled_qty
        if filled_avg_price is not None:
            from decimal import Decimal
            updates["filled_avg_price"] = Decimal(str(filled_avg_price))
        if error_message:
            updates["error_message"] = error_message

        if new_status == OrderStatus.SUBMITTED:
            updates["submitted_at"] = now
        elif new_status == OrderStatus.FILLED:
            updates["filled_at"] = now
        elif new_status == OrderStatus.CANCELLED:
            updates["cancelled_at"] = now

        order = await self._repo.update_by_id(order_id, **updates)
        if order:
            logger.info(
                "order_status_updated",
                order_id=order_id,
                new_status=new_status,
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
        raw: dict | None = None,
    ) -> Fill:
        """Record a fill event for an order."""
        from decimal import Decimal
        fill = Fill(
            order_id=order_id,
            broker_order_id=broker_order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            price=Decimal(str(price)),
            commission=Decimal(str(commission)),
            timestamp=datetime.now(timezone.utc),
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
        return fill
