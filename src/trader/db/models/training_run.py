"""Training run model for ML model training audit."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base, TimestampMixin
from trader.db.types import JSONVariant


class TrainingRun(Base, TimestampMixin):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    hyperparameters: Mapped[dict] = mapped_column(JSONVariant, nullable=False, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONVariant, nullable=False, default=dict)
    training_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validation_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    best_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    def __repr__(self) -> str:
        return f"<TrainingRun({self.model_type} {self.status})>"
