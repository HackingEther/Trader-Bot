"""Symbol / instrument model."""

from __future__ import annotations

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base, TimestampMixin


class Symbol(Base, TimestampMixin):
    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    exchange: Mapped[str] = mapped_column(String(20), nullable=False, default="")
    asset_class: Mapped[str] = mapped_column(String(20), nullable=False, default="us_equity")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_tradable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_shortable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        return f"<Symbol(ticker={self.ticker!r})>"
