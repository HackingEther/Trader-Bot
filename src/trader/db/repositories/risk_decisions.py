"""Risk decision repository."""

from __future__ import annotations

from sqlalchemy import select

from trader.db.models.risk_decision import RiskDecisionRecord
from trader.db.repositories.base import BaseRepository


class RiskDecisionRepository(BaseRepository[RiskDecisionRecord]):
    model = RiskDecisionRecord

    async def get_by_trade_intent(self, trade_intent_id: int) -> RiskDecisionRecord | None:
        stmt = select(RiskDecisionRecord).where(RiskDecisionRecord.trade_intent_id == trade_intent_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_recent_rejections(self, limit: int = 50) -> list[RiskDecisionRecord]:
        stmt = (
            select(RiskDecisionRecord)
            .where(RiskDecisionRecord.decision == "rejected")
            .order_by(RiskDecisionRecord.timestamp.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_recent(self, limit: int = 50) -> list[RiskDecisionRecord]:
        stmt = select(RiskDecisionRecord).order_by(RiskDecisionRecord.timestamp.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
