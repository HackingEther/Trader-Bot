"""Model prediction record for audit trail."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base, TimestampMixin


class ModelPredictionRecord(Base, TimestampMixin):
    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    direction: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    expected_move_bps: Mapped[float] = mapped_column(Float, nullable=False)
    expected_holding_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    no_trade_score: Mapped[float] = mapped_column(Float, nullable=False)
    regime: Mapped[str] = mapped_column(String(30), nullable=False)
    feature_snapshot_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_versions_used: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    def __repr__(self) -> str:
        return f"<ModelPredictionRecord({self.symbol} {self.direction} conf={self.confidence:.2f})>"
