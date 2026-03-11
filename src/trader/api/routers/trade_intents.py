"""Trade intent endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from trader.api.deps import get_db_session
from trader.api.schemas.common import TradeIntentResponse
from trader.db.repositories.trade_intents import TradeIntentRepository

router = APIRouter(prefix="/api/trade-intents", tags=["trade_intents"])


@router.get("/", response_model=list[TradeIntentResponse])
async def list_trade_intents(
    limit: int = 50,
    session: AsyncSession = Depends(get_db_session),
) -> list:
    repo = TradeIntentRepository(session)
    return await repo.get_recent(limit=limit)
