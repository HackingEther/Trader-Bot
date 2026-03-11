"""Fill repository."""

from __future__ import annotations

from sqlalchemy import select

from trader.db.models.fill import Fill
from trader.db.repositories.base import BaseRepository


class FillRepository(BaseRepository[Fill]):
    model = Fill

    async def get_by_execution_key(self, execution_key: str) -> Fill | None:
        stmt = select(Fill).where(Fill.execution_key == execution_key)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_order_id(self, order_id: int) -> list[Fill]:
        stmt = select(Fill).where(Fill.order_id == order_id).order_by(Fill.timestamp.asc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_latest_by_order_id(self, order_id: int) -> Fill | None:
        stmt = select(Fill).where(Fill.order_id == order_id).order_by(Fill.timestamp.desc()).limit(1)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
