"""Risk decision record for audit trail."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base, TimestampMixin
from trader.db.types import JSONVariant


class RiskDecisionRecord(Base, TimestampMixin):
    __tablename__ = "risk_decisions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trade_intent_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    reasons: Mapped[list] = mapped_column(JSONVariant, nullable=False, default=list)
    rule_results: Mapped[dict] = mapped_column(JSONVariant, nullable=False, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_risk_decisions_intent_ts", "trade_intent_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<RiskDecisionRecord(intent={self.trade_intent_id} {self.decision})>"
