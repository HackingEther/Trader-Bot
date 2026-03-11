"""Position model for tracking open and closed positions."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Index, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base, TimestampMixin
from trader.db.types import JSONVariant


class Position(Base, TimestampMixin):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_entry_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    current_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    market_value: Mapped[Decimal | None] = mapped_column(Numeric(14, 4), nullable=True)
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False, default=Decimal("0"))
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False, default=Decimal("0"))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open", index=True)
    strategy_tag: Mapped[str] = mapped_column(String(50), nullable=False, default="default")
    trade_intent_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONVariant, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_positions_symbol_status", "symbol", "status"),
    )

    def __repr__(self) -> str:
        return f"<Position({self.symbol} {self.side} qty={self.qty} {self.status})>"
