"""Trade intent model - proposed trades before risk check."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Float, Index, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base, TimestampMixin
from trader.db.types import JSONVariant


class TradeIntent(Base, TimestampMixin):
    __tablename__ = "trade_intents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_order_type: Mapped[str] = mapped_column(String(20), nullable=False, default="market")
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    stop_loss: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    take_profit: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    max_hold_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    strategy_tag: Mapped[str] = mapped_column(String(50), nullable=False, default="default")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    model_prediction_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_move_bps: Mapped[float | None] = mapped_column(Float, nullable=True)
    rationale: Mapped[dict] = mapped_column(JSONVariant, nullable=False, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_trade_intents_symbol_ts", "symbol", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<TradeIntent({self.symbol} {self.side} qty={self.qty} {self.status})>"
