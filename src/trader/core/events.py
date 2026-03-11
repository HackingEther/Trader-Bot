"""Normalized event types used across the trading platform."""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class BarEvent(BaseModel):
    """Normalized OHLCV bar event."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Decimal | None = None
    trade_count: int | None = None
    interval: str = "1m"

    model_config = {"frozen": True}


class QuoteEvent(BaseModel):
    """Normalized quote (BBO) event."""

    symbol: str
    timestamp: datetime
    bid_price: Decimal
    bid_size: int
    ask_price: Decimal
    ask_size: int
    spread: Decimal = Field(default=Decimal("0"))

    def model_post_init(self, __context: object) -> None:
        if self.spread == Decimal("0") and self.ask_price > 0 and self.bid_price > 0:
            object.__setattr__(self, "spread", self.ask_price - self.bid_price)

    model_config = {"frozen": True}


class TradeEvent(BaseModel):
    """Normalized trade/last-sale event."""

    symbol: str
    timestamp: datetime
    price: Decimal
    size: int
    conditions: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}


class SessionEvent(BaseModel):
    """Market session state change event."""

    timestamp: datetime
    session: str
    message: str = ""

    model_config = {"frozen": True}


class OrderUpdateEvent(BaseModel):
    """Order status update from broker."""

    broker_order_id: str
    symbol: str
    status: str
    filled_qty: Decimal = Decimal("0")
    filled_avg_price: Decimal | None = None
    timestamp: datetime
    raw: dict = Field(default_factory=dict)

    model_config = {"frozen": True}


class ModelPrediction(BaseModel):
    """Output from the model ensemble inference pipeline."""

    symbol: str
    timestamp: datetime
    direction: str
    confidence: float = Field(ge=0.0, le=1.0)
    expected_move_bps: float
    expected_holding_minutes: float
    no_trade_score: float = Field(ge=0.0, le=1.0)
    regime: str
    feature_snapshot_id: int | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)

    model_config = {"frozen": True}
