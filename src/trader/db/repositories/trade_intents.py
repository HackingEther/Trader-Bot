"""Trade intent repository."""

from __future__ import annotations

from sqlalchemy import select

from trader.db.models.trade_intent import TradeIntent
from trader.db.repositories.base import BaseRepository


class TradeIntentRepository(BaseRepository[TradeIntent]):
    model = TradeIntent

    async def get_pending(self) -> list[TradeIntent]:
        stmt = select(TradeIntent).where(TradeIntent.status == "pending").order_by(TradeIntent.timestamp.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_symbol(self, symbol: str, limit: int = 50) -> list[TradeIntent]:
        stmt = select(TradeIntent).where(TradeIntent.symbol == symbol).order_by(TradeIntent.timestamp.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_recent(self, limit: int = 50) -> list[TradeIntent]:
        stmt = select(TradeIntent).order_by(TradeIntent.timestamp.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
