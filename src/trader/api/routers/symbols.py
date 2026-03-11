"""Symbol endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from trader.api.deps import get_db_session
from trader.api.schemas.common import SymbolResponse
from trader.db.models.symbol import Symbol

router = APIRouter(prefix="/api/symbols", tags=["symbols"])


@router.get("/", response_model=list[SymbolResponse])
async def list_symbols(
    active_only: bool = True,
    session: AsyncSession = Depends(get_db_session),
) -> list[Symbol]:
    stmt = select(Symbol)
    if active_only:
        stmt = stmt.where(Symbol.is_active == True)  # noqa: E712
    result = await session.execute(stmt)
    return list(result.scalars().all())
