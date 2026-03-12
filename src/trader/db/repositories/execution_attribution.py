"""Execution attribution repository."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import select

from trader.db.models.execution_attribution import ExecutionAttribution
from trader.db.models.order import Order
from trader.db.repositories.base import BaseRepository

if TYPE_CHECKING:
    from trader.db.repositories.quote_snapshots import QuoteSnapshotRepository


def _coerce_utc(dt: datetime | None) -> datetime | None:
    """Ensure datetime has UTC timezone for subtraction."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _compute_attribution_metrics(
    *,
    side: str,
    filled_avg_price: Decimal,
    decision_bid: Decimal | None,
    decision_ask: Decimal | None,
    decision_mid: Decimal | None,
    submitted_at: datetime | None,
    last_fill_at: datetime | None,
) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
    """Compute realized_spread_bps, slippage_bps, time_to_fill_seconds."""
    realized_spread_bps = None
    slippage_bps = None
    time_to_fill_seconds = None
    if decision_mid and decision_mid > 0:
        if side == "buy":
            realized_spread_bps = (filled_avg_price - decision_mid) / decision_mid * Decimal("10000")
            if decision_ask and decision_ask > 0:
                slippage_bps = (filled_avg_price - decision_ask) / decision_ask * Decimal("10000")
        else:
            realized_spread_bps = (decision_mid - filled_avg_price) / decision_mid * Decimal("10000")
            if decision_bid and decision_bid > 0:
                slippage_bps = (decision_bid - filled_avg_price) / decision_bid * Decimal("10000")
    if submitted_at and last_fill_at:
        sub_utc = _coerce_utc(submitted_at)
        last_utc = _coerce_utc(last_fill_at)
        if sub_utc and last_utc:
            time_to_fill_seconds = Decimal(str((last_utc - sub_utc).total_seconds()))
    return realized_spread_bps, slippage_bps, time_to_fill_seconds


class ExecutionAttributionRepository(BaseRepository[ExecutionAttribution]):
    model = ExecutionAttribution

    async def ensure_attribution_for_filled_order(
        self,
        order: Order,
        fills: list,
        quote_snapshots: QuoteSnapshotRepository,
    ) -> ExecutionAttribution | None:
        """Create attribution if order is filled and not already attributed."""
        if order.status != "filled" or not order.filled_qty or not order.filled_avg_price:
            return None
        existing = await self.get_by_order_id(order.id)
        if existing is not None:
            return existing
        decision_snapshot = None
        submit_snapshot = None
        if order.trade_intent_id:
            decision_snapshot = await quote_snapshots.get_decision_for_trade_intent(order.trade_intent_id)
        submit_snapshot = await quote_snapshots.get_submit_for_order(order.id)
        first_fill_at = fills[0].timestamp if fills else None
        last_fill_at = fills[-1].timestamp if fills else None
        decision_bid = decision_snapshot.bid if decision_snapshot else None
        decision_ask = decision_snapshot.ask if decision_snapshot else None
        decision_mid = decision_snapshot.mid if decision_snapshot else None
        realized_spread_bps, slippage_bps, time_to_fill_seconds = _compute_attribution_metrics(
            side=order.side,
            filled_avg_price=order.filled_avg_price,
            decision_bid=decision_bid,
            decision_ask=decision_ask,
            decision_mid=decision_mid,
            submitted_at=order.submitted_at,
            last_fill_at=last_fill_at,
        )
        return await self.create_attribution(
            trade_intent_id=order.trade_intent_id,
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            filled_qty=order.filled_qty,
            filled_avg_price=order.filled_avg_price,
            decision_quote_snapshot_id=decision_snapshot.id if decision_snapshot else None,
            submit_quote_snapshot_id=submit_snapshot.id if submit_snapshot else None,
            realized_spread_bps=realized_spread_bps,
            slippage_bps=slippage_bps,
            time_to_fill_seconds=time_to_fill_seconds,
            first_fill_at=first_fill_at,
            last_fill_at=last_fill_at,
        )

    async def create_attribution(
        self,
        *,
        trade_intent_id: int | None,
        order_id: int | None,
        symbol: str,
        side: str,
        filled_qty: int,
        filled_avg_price: Decimal,
        decision_quote_snapshot_id: int | None = None,
        submit_quote_snapshot_id: int | None = None,
        realized_spread_bps: Decimal | None = None,
        slippage_bps: Decimal | None = None,
        time_to_fill_seconds: Decimal | None = None,
        first_fill_at: datetime | None = None,
        last_fill_at: datetime | None = None,
    ) -> ExecutionAttribution:
        return await self.create(
            trade_intent_id=trade_intent_id,
            order_id=order_id,
            symbol=symbol,
            side=side,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            decision_quote_snapshot_id=decision_quote_snapshot_id,
            submit_quote_snapshot_id=submit_quote_snapshot_id,
            realized_spread_bps=realized_spread_bps,
            slippage_bps=slippage_bps,
            time_to_fill_seconds=time_to_fill_seconds,
            first_fill_at=first_fill_at,
            last_fill_at=last_fill_at,
        )

    async def get_by_order_id(self, order_id: int) -> ExecutionAttribution | None:
        stmt = select(ExecutionAttribution).where(ExecutionAttribution.order_id == order_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_trade_intent(self, trade_intent_id: int) -> ExecutionAttribution | None:
        stmt = select(ExecutionAttribution).where(ExecutionAttribution.trade_intent_id == trade_intent_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
