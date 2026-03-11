"""Order repository."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from trader.db.models.order import Order
from trader.db.repositories.base import BaseRepository


class OrderRepository(BaseRepository[Order]):
    model = Order

    async def create_idempotent(self, **kwargs: Any) -> tuple[Order, bool]:
        """Create an order while safely handling idempotency races.

        Returns:
            Tuple of ``(order, created)`` where ``created`` indicates whether the
            row was inserted by this call or already existed.
        """
        idempotency_key = str(kwargs["idempotency_key"])
        async with self.session.begin_nested():
            instance = Order(**kwargs)
            self.session.add(instance)
            try:
                await self.session.flush()
                return instance, True
            except IntegrityError:
                pass

        existing = await self.get_by_idempotency_key(idempotency_key)
        if existing is None:
            raise RuntimeError(
                f"Idempotent order creation failed for key {idempotency_key} without existing row"
            )
        return existing, False

    async def get_by_idempotency_key(self, key: str) -> Order | None:
        stmt = select(Order).where(Order.idempotency_key == key)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_broker_order_id(self, broker_id: str) -> Order | None:
        stmt = select(Order).where(Order.broker_order_id == broker_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_symbol(self, symbol: str, limit: int = 50) -> list[Order]:
        stmt = select(Order).where(Order.symbol == symbol).order_by(Order.created_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_open_orders(self) -> list[Order]:
        open_statuses = ("pending", "submitted", "accepted", "partially_filled")
        stmt = select(Order).where(Order.status.in_(open_statuses))
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_recent(self, limit: int = 50) -> list[Order]:
        stmt = select(Order).order_by(Order.created_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
