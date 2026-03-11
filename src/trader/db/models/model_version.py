"""Model version registry."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base, TimestampMixin
from trader.db.types import JSONVariant


class ModelVersion(Base, TimestampMixin):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False, default="lightgbm")
    hyperparameters: Mapped[dict] = mapped_column(JSONVariant, nullable=False, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONVariant, nullable=False, default=dict)
    training_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_champion: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<ModelVersion({self.model_type} {self.version})>"
