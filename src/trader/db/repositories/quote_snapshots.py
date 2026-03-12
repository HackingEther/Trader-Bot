"""Quote snapshot repository."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import select

from trader.db.models.quote_snapshot import QuoteSnapshot
from trader.db.repositories.base import BaseRepository


class QuoteSnapshotRepository(BaseRepository[QuoteSnapshot]):
    model = QuoteSnapshot

    async def create_snapshot(
        self,
        *,
        snapshot_type: str,
        symbol: str,
        bid: Decimal,
        ask: Decimal,
        mid: Decimal,
        timestamp: datetime,
        spread_bps: Decimal | None = None,
        trade_intent_id: int | None = None,
        order_id: int | None = None,
    ) -> QuoteSnapshot:
        return await self.create(
            snapshot_type=snapshot_type,
            symbol=symbol,
            bid=bid,
            ask=ask,
            mid=mid,
            timestamp=timestamp,
            spread_bps=spread_bps,
            trade_intent_id=trade_intent_id,
            order_id=order_id,
        )

    async def get_by_trade_intent(self, trade_intent_id: int, snapshot_type: str | None = None) -> list[QuoteSnapshot]:
        stmt = select(QuoteSnapshot).where(QuoteSnapshot.trade_intent_id == trade_intent_id)
        if snapshot_type:
            stmt = stmt.where(QuoteSnapshot.snapshot_type == snapshot_type)
        stmt = stmt.order_by(QuoteSnapshot.timestamp.asc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_decision_for_trade_intent(self, trade_intent_id: int) -> QuoteSnapshot | None:
        stmt = (
            select(QuoteSnapshot)
            .where(
                QuoteSnapshot.trade_intent_id == trade_intent_id,
                QuoteSnapshot.snapshot_type == "decision",
            )
            .order_by(QuoteSnapshot.timestamp.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_submit_for_order(self, order_id: int) -> QuoteSnapshot | None:
        stmt = (
            select(QuoteSnapshot)
            .where(
                QuoteSnapshot.order_id == order_id,
                QuoteSnapshot.snapshot_type == "submit",
            )
            .order_by(QuoteSnapshot.timestamp.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
