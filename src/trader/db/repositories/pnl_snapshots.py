"""PnL snapshot repository."""

from __future__ import annotations

from sqlalchemy import select

from trader.db.models.pnl_snapshot import PnlSnapshot
from trader.db.repositories.base import BaseRepository


class PnlSnapshotRepository(BaseRepository[PnlSnapshot]):
    model = PnlSnapshot

    async def get_latest(self) -> PnlSnapshot | None:
        stmt = select(PnlSnapshot).order_by(PnlSnapshot.timestamp.desc()).limit(1)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_date(self, date_str: str, limit: int = 100) -> list[PnlSnapshot]:
        stmt = (
            select(PnlSnapshot)
            .where(PnlSnapshot.date_str == date_str)
            .order_by(PnlSnapshot.timestamp.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
