"""Position repository."""

from __future__ import annotations

from sqlalchemy import select

from trader.db.models.position import Position
from trader.db.repositories.base import BaseRepository


class PositionRepository(BaseRepository[Position]):
    model = Position

    async def get_open_positions(self) -> list[Position]:
        stmt = select(Position).where(Position.status == "open")
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_symbol(self, symbol: str, status: str | None = None) -> list[Position]:
        stmt = select(Position).where(Position.symbol == symbol)
        if status:
            stmt = stmt.where(Position.status == status)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_open_by_symbol(self, symbol: str) -> Position | None:
        stmt = select(Position).where(Position.symbol == symbol, Position.status == "open")
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def count_open(self) -> int:
        from sqlalchemy import func
        stmt = select(func.count()).select_from(Position).where(Position.status == "open")
        result = await self.session.execute(stmt)
        return result.scalar_one()
