"""Fill endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from trader.api.deps import get_db_session
from trader.api.schemas.common import FillResponse
from trader.db.models.fill import Fill

router = APIRouter(prefix="/api/fills", tags=["fills"])


@router.get("/", response_model=list[FillResponse])
async def list_fills(
    limit: int = 50,
    session: AsyncSession = Depends(get_db_session),
) -> list:
    stmt = select(Fill).order_by(Fill.timestamp.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())
