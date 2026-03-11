"""Order repository."""

from __future__ import annotations

from sqlalchemy import select

from trader.db.models.order import Order
from trader.db.repositories.base import BaseRepository


class OrderRepository(BaseRepository[Order]):
    model = Order

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
