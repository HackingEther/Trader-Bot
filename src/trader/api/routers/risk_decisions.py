"""Risk decision endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from trader.api.deps import get_db_session
from trader.api.schemas.common import RiskDecisionResponse
from trader.db.repositories.risk_decisions import RiskDecisionRepository

router = APIRouter(prefix="/api/risk-decisions", tags=["risk_decisions"])


@router.get("/", response_model=list[RiskDecisionResponse])
async def list_risk_decisions(
    limit: int = 50,
    session: AsyncSession = Depends(get_db_session),
) -> list:
    repo = RiskDecisionRepository(session)
    return await repo.get_recent(limit=limit)


@router.get("/rejections", response_model=list[RiskDecisionResponse])
async def list_rejections(
    limit: int = 50,
    session: AsyncSession = Depends(get_db_session),
) -> list:
    repo = RiskDecisionRepository(session)
    return await repo.get_recent_rejections(limit=limit)
