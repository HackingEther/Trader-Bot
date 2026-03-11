"""P&L endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from trader.api.deps import get_db_session
from trader.api.schemas.common import PnlResponse
from trader.db.models.pnl_snapshot import PnlSnapshot

router = APIRouter(prefix="/api/pnl", tags=["pnl"])


@router.get("/", response_model=list[PnlResponse])
async def list_pnl_snapshots(
    limit: int = 50,
    session: AsyncSession = Depends(get_db_session),
) -> list:
    stmt = select(PnlSnapshot).order_by(PnlSnapshot.timestamp.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())
