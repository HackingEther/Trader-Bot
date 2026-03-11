"""Feature snapshot model for storing computed feature vectors."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base, TimestampMixin
from trader.db.types import JSONVariant


class FeatureSnapshot(Base, TimestampMixin):
    __tablename__ = "feature_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    features: Mapped[dict] = mapped_column(JSONVariant, nullable=False)
    feature_version: Mapped[str] = mapped_column(String(20), nullable=False, default="v1")
    bar_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_feature_snapshots_symbol_ts", "symbol", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<FeatureSnapshot({self.symbol} {self.timestamp})>"
