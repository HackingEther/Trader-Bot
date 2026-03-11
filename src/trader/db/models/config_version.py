"""Configuration version snapshot for audit."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base
from trader.db.types import JSONVariant


class ConfigVersion(Base):
    __tablename__ = "config_versions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    config_snapshot: Mapped[dict] = mapped_column(JSONVariant, nullable=False)
    changed_by: Mapped[str] = mapped_column(String(100), nullable=False, default="system")
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    def __repr__(self) -> str:
        return f"<ConfigVersion({self.timestamp} by {self.changed_by})>"
