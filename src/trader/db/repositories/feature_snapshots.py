"""Feature snapshot repository."""

from __future__ import annotations

from sqlalchemy import select

from trader.db.models.feature_snapshot import FeatureSnapshot
from trader.db.repositories.base import BaseRepository


class FeatureSnapshotRepository(BaseRepository[FeatureSnapshot]):
    model = FeatureSnapshot

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> list[FeatureSnapshot]:
        stmt = (
            select(FeatureSnapshot)
            .where(FeatureSnapshot.symbol == symbol)
            .order_by(FeatureSnapshot.timestamp.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
