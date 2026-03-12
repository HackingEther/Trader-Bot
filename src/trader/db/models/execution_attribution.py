"""Execution attribution model for per-trade metrics."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base, TimestampMixin


class ExecutionAttribution(Base, TimestampMixin):
    __tablename__ = "execution_attribution"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trade_intent_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("trade_intents.id", ondelete="SET NULL"), nullable=True, index=True
    )
    order_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("orders.id", ondelete="SET NULL"), nullable=True, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    filled_qty: Mapped[int] = mapped_column(Integer, nullable=False)
    filled_avg_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    decision_quote_snapshot_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("quote_snapshots.id", ondelete="SET NULL"), nullable=True
    )
    submit_quote_snapshot_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("quote_snapshots.id", ondelete="SET NULL"), nullable=True
    )
    realized_spread_bps: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    slippage_bps: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    time_to_fill_seconds: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    first_fill_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_fill_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_execution_attribution_order", "order_id"),
        Index("ix_execution_attribution_trade_intent", "trade_intent_id"),
    )

    def __repr__(self) -> str:
        return f"<ExecutionAttribution({self.symbol} {self.side} qty={self.filled_qty})>"
