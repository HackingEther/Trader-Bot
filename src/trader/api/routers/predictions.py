"""Model prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from trader.api.deps import get_db_session
from trader.api.schemas.common import PredictionResponse
from trader.db.models.model_prediction import ModelPredictionRecord

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.get("/", response_model=list[PredictionResponse])
async def list_predictions(
    limit: int = 50,
    symbol: str | None = None,
    session: AsyncSession = Depends(get_db_session),
) -> list:
    stmt = select(ModelPredictionRecord).order_by(ModelPredictionRecord.timestamp.desc())
    if symbol:
        stmt = stmt.where(ModelPredictionRecord.symbol == symbol)
    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())
