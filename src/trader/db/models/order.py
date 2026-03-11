"""Order model with full lifecycle tracking."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Index, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base, TimestampMixin
from trader.db.types import JSONVariant


class Order(Base, TimestampMixin):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    trade_intent_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)
    order_class: Mapped[str] = mapped_column(String(20), nullable=False, default="simple")
    time_in_force: Mapped[str] = mapped_column(String(10), nullable=False, default="day")
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    stop_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    filled_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    filled_avg_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending", index=True)
    strategy_tag: Mapped[str] = mapped_column(String(50), nullable=False, default="default")
    rationale: Mapped[dict] = mapped_column(JSONVariant, nullable=False, default=dict)
    broker_metadata: Mapped[dict] = mapped_column(JSONVariant, nullable=False, default=dict)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)

    __table_args__ = (
        Index("ix_orders_symbol_status", "symbol", "status"),
    )

    def __repr__(self) -> str:
        return f"<Order({self.symbol} {self.side} qty={self.qty} {self.status})>"
