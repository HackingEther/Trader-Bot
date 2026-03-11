"""Market bar repository."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from trader.db.models.market_bar import MarketBar
from trader.db.repositories.base import BaseRepository


class MarketBarRepository(BaseRepository[MarketBar]):
    model = MarketBar

    async def get_recent(
        self,
        symbol: str,
        limit: int = 400,
        interval: str = "1m",
    ) -> list[MarketBar]:
        stmt = (
            select(MarketBar)
            .where(MarketBar.symbol == symbol, MarketBar.interval == interval)
            .order_by(MarketBar.timestamp.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(reversed(list(result.scalars().all())))

    async def get_range(
        self,
        symbols: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str = "1m",
    ) -> list[MarketBar]:
        stmt = select(MarketBar).where(MarketBar.interval == interval)
        if symbols:
            stmt = stmt.where(MarketBar.symbol.in_(symbols))
        if start is not None:
            stmt = stmt.where(MarketBar.timestamp >= start)
        if end is not None:
            stmt = stmt.where(MarketBar.timestamp <= end)
        stmt = stmt.order_by(MarketBar.symbol.asc(), MarketBar.timestamp.asc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_existing_keys(
        self,
        symbols: list[str],
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str = "1m",
    ) -> set[tuple[str, datetime]]:
        stmt = select(MarketBar.symbol, MarketBar.timestamp).where(MarketBar.interval == interval)
        if symbols:
            stmt = stmt.where(MarketBar.symbol.in_(symbols))
        if start is not None:
            stmt = stmt.where(MarketBar.timestamp >= start)
        if end is not None:
            stmt = stmt.where(MarketBar.timestamp <= end)
        result = await self.session.execute(stmt)
        return {(symbol, timestamp) for symbol, timestamp in result.all()}

    async def get_latest(self, symbol: str, interval: str = "1m") -> MarketBar | None:
        stmt = (
            select(MarketBar)
            .where(MarketBar.symbol == symbol, MarketBar.interval == interval)
            .order_by(MarketBar.timestamp.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
