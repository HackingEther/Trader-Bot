"""Market bar (OHLCV) model."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Index, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base, TimestampMixin


class MarketBar(Base, TimestampMixin):
    __tablename__ = "market_bars"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    interval: Mapped[str] = mapped_column(String(10), nullable=False, default="1m")
    open: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    vwap: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    trade_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_market_bars_symbol_ts", "symbol", "timestamp"),
        Index("ix_market_bars_symbol_interval_ts", "symbol", "interval", "timestamp"),
        UniqueConstraint("symbol", "interval", "timestamp", name="uq_market_bars_symbol_interval_ts"),
    )

    def __repr__(self) -> str:
        return f"<MarketBar({self.symbol} {self.timestamp} {self.interval})>"
