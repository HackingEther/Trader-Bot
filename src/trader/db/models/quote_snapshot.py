"""Quote snapshot model for decision/submit/replace attribution."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base, TimestampMixin


class QuoteSnapshot(Base, TimestampMixin):
    __tablename__ = "quote_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trade_intent_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("trade_intents.id", ondelete="SET NULL"), nullable=True, index=True
    )
    order_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("orders.id", ondelete="SET NULL"), nullable=True, index=True
    )
    snapshot_type: Mapped[str] = mapped_column(String(20), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    bid: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    ask: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    mid: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    spread_bps: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_quote_snapshots_trade_intent", "trade_intent_id"),
        Index("ix_quote_snapshots_order", "order_id"),
    )

    def __repr__(self) -> str:
        return f"<QuoteSnapshot({self.symbol} {self.snapshot_type} @ {self.timestamp})>"
