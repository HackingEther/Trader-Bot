"""Position endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from trader.api.deps import get_db_session
from trader.api.schemas.common import PositionResponse
from trader.db.repositories.positions import PositionRepository

router = APIRouter(prefix="/api/positions", tags=["positions"])


@router.get("/", response_model=list[PositionResponse])
async def list_positions(
    status: str = "open",
    session: AsyncSession = Depends(get_db_session),
) -> list:
    repo = PositionRepository(session)
    if status == "open":
        return await repo.get_open_positions()
    return await repo.get_all(limit=100)
