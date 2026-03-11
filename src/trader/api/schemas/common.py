"""Common API response schemas."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    environment: str = "paper"
    timestamp: datetime
    checks: dict[str, bool] = Field(default_factory=dict)


class PaginatedResponse(BaseModel):
    items: list = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


class SymbolResponse(BaseModel):
    id: int
    ticker: str
    name: str
    exchange: str
    is_active: bool
    is_tradable: bool

    model_config = {"from_attributes": True}


class PositionResponse(BaseModel):
    id: int
    symbol: str
    side: str
    qty: int
    avg_entry_price: Decimal
    current_price: Decimal | None
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    status: str
    opened_at: datetime
    closed_at: datetime | None

    model_config = {"from_attributes": True}


class OrderResponse(BaseModel):
    id: int
    idempotency_key: str
    broker_order_id: str | None
    symbol: str
    side: str
    order_type: str
    qty: int
    filled_qty: int
    filled_avg_price: Decimal | None
    status: str
    strategy_tag: str
    submitted_at: datetime | None
    filled_at: datetime | None
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class FillResponse(BaseModel):
    id: int
    order_id: int
    symbol: str
    side: str
    qty: int
    price: Decimal
    commission: Decimal
    timestamp: datetime

    model_config = {"from_attributes": True}


class PnlResponse(BaseModel):
    id: int
    date_str: str
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    open_positions: int
    trade_count: int
    timestamp: datetime

    model_config = {"from_attributes": True}


class PredictionResponse(BaseModel):
    id: int
    symbol: str
    direction: str
    confidence: float
    expected_move_bps: float
    expected_holding_minutes: float
    no_trade_score: float
    regime: str
    timestamp: datetime

    model_config = {"from_attributes": True}


class TradeIntentResponse(BaseModel):
    id: int
    symbol: str
    side: str
    qty: int
    entry_order_type: str
    stop_loss: Decimal | None
    take_profit: Decimal | None
    max_hold_minutes: int
    strategy_tag: str
    status: str
    confidence: float | None
    timestamp: datetime

    model_config = {"from_attributes": True}


class RiskDecisionResponse(BaseModel):
    id: int
    trade_intent_id: int
    decision: str
    reasons: list
    rule_results: dict
    timestamp: datetime

    model_config = {"from_attributes": True}


class KillSwitchResponse(BaseModel):
    active: bool
    message: str = ""
