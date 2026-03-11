"""Model prediction repository."""

from __future__ import annotations

from sqlalchemy import select

from trader.db.models.model_prediction import ModelPredictionRecord
from trader.db.repositories.base import BaseRepository


class ModelPredictionRepository(BaseRepository[ModelPredictionRecord]):
    model = ModelPredictionRecord

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> list[ModelPredictionRecord]:
        stmt = (
            select(ModelPredictionRecord)
            .where(ModelPredictionRecord.symbol == symbol)
            .order_by(ModelPredictionRecord.timestamp.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
