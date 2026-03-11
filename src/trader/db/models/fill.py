"""Fill model for order executions."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Index, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base, TimestampMixin
from trader.db.types import JSONVariant


class Fill(Base, TimestampMixin):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    execution_key: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True, index=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    broker_execution_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    commission: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False, default=Decimal("0"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw: Mapped[dict] = mapped_column(JSONVariant, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_fills_order_timestamp", "order_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<Fill({self.symbol} {self.side} qty={self.qty} @ {self.price})>"
