"""Alpaca broker provider implementation using alpaca-py SDK."""

from __future__ import annotations

from decimal import Decimal

import structlog
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType as AlpacaOrderType, TimeInForce as AlpacaTIF, OrderClass as AlpacaOrderClass
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
    ReplaceOrderRequest,
)

from trader.core.exceptions import BrokerConnectionError, OrderCancellationError, OrderSubmissionError
from trader.providers.broker.base import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
    BrokerProvider,
    OrderRequest,
)

logger = structlog.get_logger(__name__)

_SIDE_MAP = {"buy": OrderSide.BUY, "sell": OrderSide.SELL}
_TIF_MAP = {
    "day": AlpacaTIF.DAY,
    "gtc": AlpacaTIF.GTC,
    "ioc": AlpacaTIF.IOC,
    "fok": AlpacaTIF.FOK,
}


def _maybe_iso(value: object) -> str | None:
    return value.isoformat() if value is not None and hasattr(value, "isoformat") else None


def _serialize_order_entity(raw_order: object, *, include_legs: bool = True) -> dict:
    o = raw_order  # type: ignore[assignment]
    raw: dict = {
        "alpaca_id": str(o.id),
        "client_order_id": getattr(o, "client_order_id", None),
        "parent_order_id": str(getattr(o, "parent_order_id", "")) if getattr(o, "parent_order_id", None) else None,
        "submitted_at": _maybe_iso(getattr(o, "submitted_at", None)),
        "updated_at": _maybe_iso(getattr(o, "updated_at", None)),
        "filled_at": _maybe_iso(getattr(o, "filled_at", None)),
        "canceled_at": _maybe_iso(getattr(o, "canceled_at", None)),
        "replaced_at": _maybe_iso(getattr(o, "replaced_at", None)),
    }
    if include_legs:
        raw["legs"] = [
            _serialize_order_entity(leg, include_legs=False)
            for leg in list(getattr(o, "legs", []) or [])
        ]
    return raw


def _to_broker_order(raw_order: object) -> BrokerOrder:
    """Convert Alpaca order object to standardized BrokerOrder."""
    o = raw_order  # type: ignore[assignment]
    raw = _serialize_order_entity(o)
    return BrokerOrder(
        broker_order_id=str(o.id),
        symbol=o.symbol,
        side=str(o.side.value) if o.side else "buy",
        order_type=str(o.order_type.value) if o.order_type else "market",
        qty=int(o.qty) if o.qty else 0,
        limit_price=Decimal(str(o.limit_price)) if o.limit_price else None,
        stop_price=Decimal(str(o.stop_price)) if o.stop_price else None,
        filled_qty=int(o.filled_qty) if o.filled_qty else 0,
        filled_avg_price=Decimal(str(o.filled_avg_price)) if o.filled_avg_price else None,
        status=str(o.status.value) if o.status else "unknown",
        order_class=str(o.order_class.value) if o.order_class else "simple",
        time_in_force=str(o.time_in_force.value) if o.time_in_force else "day",
        raw=raw,
    )


def _to_broker_position(raw_pos: object) -> BrokerPosition:
    """Convert Alpaca position object to standardized BrokerPosition."""
    p = raw_pos  # type: ignore[assignment]
    return BrokerPosition(
        symbol=p.symbol,
        qty=abs(int(p.qty)),
        side="buy" if int(p.qty) > 0 else "sell",
        avg_entry_price=Decimal(str(p.avg_entry_price)),
        current_price=Decimal(str(p.current_price)),
        market_value=Decimal(str(p.market_value)),
        unrealized_pnl=Decimal(str(p.unrealized_pl)),
        raw={"asset_id": str(p.asset_id)},
    )


class AlpacaProvider(BrokerProvider):
    """Alpaca Markets broker implementation."""

    def __init__(self, api_key: str, api_secret: str, paper: bool = True) -> None:
        if not api_key or not api_secret:
            raise BrokerConnectionError("Alpaca API key and secret are required")
        self._client = TradingClient(api_key, api_secret, paper=paper)
        self._paper = paper
        logger.info("alpaca_provider_initialized", paper=paper)

    def _build_order_request(self, req: OrderRequest) -> object:
        """Build Alpaca SDK order request from our OrderRequest."""
        side = _SIDE_MAP.get(req.side, OrderSide.BUY)
        tif = _TIF_MAP.get(req.time_in_force, AlpacaTIF.DAY)

        common: dict = {
            "symbol": req.symbol,
            "qty": req.qty,
            "side": side,
            "time_in_force": tif,
        }
        if req.client_order_id:
            common["client_order_id"] = req.client_order_id

        if req.order_class == "bracket" and req.take_profit_price and req.stop_loss_price:
            common["order_class"] = AlpacaOrderClass.BRACKET
            common["take_profit"] = {"limit_price": float(req.take_profit_price)}
            sl: dict = {"stop_price": float(req.stop_loss_price)}
            if req.stop_loss_limit_price:
                sl["limit_price"] = float(req.stop_loss_limit_price)
            common["stop_loss"] = sl

        if req.order_type == "limit":
            if req.limit_price is None:
                raise OrderSubmissionError("Limit price required for limit orders")
            common["limit_price"] = float(req.limit_price)
            return LimitOrderRequest(**common)
        elif req.order_type == "stop":
            if req.stop_price is None:
                raise OrderSubmissionError("Stop price required for stop orders")
            common["stop_price"] = float(req.stop_price)
            return StopOrderRequest(**common)
        elif req.order_type == "stop_limit":
            if req.limit_price is None or req.stop_price is None:
                raise OrderSubmissionError("Both limit and stop prices required for stop-limit orders")
            common["limit_price"] = float(req.limit_price)
            common["stop_price"] = float(req.stop_price)
            return StopLimitOrderRequest(**common)
        else:
            return MarketOrderRequest(**common)

    async def submit_order(self, request: OrderRequest) -> BrokerOrder:
        try:
            alpaca_req = self._build_order_request(request)
            result = self._client.submit_order(order_data=alpaca_req)
            order = _to_broker_order(result)
            logger.info(
                "alpaca_order_submitted",
                broker_order_id=order.broker_order_id,
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
            )
            return order
        except Exception as e:
            logger.error("alpaca_order_submission_failed", error=str(e), symbol=request.symbol)
            raise OrderSubmissionError(f"Alpaca order submission failed: {e}") from e

    async def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        try:
            self._client.cancel_order_by_id(broker_order_id)
            result = self._client.get_order_by_id(broker_order_id)
            order = _to_broker_order(result)
            logger.info("alpaca_order_cancelled", broker_order_id=broker_order_id)
            return order
        except Exception as e:
            logger.error("alpaca_order_cancel_failed", error=str(e), broker_order_id=broker_order_id)
            raise OrderCancellationError(f"Alpaca order cancellation failed: {e}") from e

    async def replace_order(
        self, broker_order_id: str, qty: int | None = None, limit_price: Decimal | None = None
    ) -> BrokerOrder:
        try:
            replace_data: dict = {}
            if qty is not None:
                replace_data["qty"] = qty
            if limit_price is not None:
                replace_data["limit_price"] = float(limit_price)
            req = ReplaceOrderRequest(**replace_data)
            result = self._client.replace_order_by_id(broker_order_id, req)
            return _to_broker_order(result)
        except Exception as e:
            logger.error("alpaca_order_replace_failed", error=str(e))
            raise OrderSubmissionError(f"Alpaca order replace failed: {e}") from e

    async def get_order(self, broker_order_id: str) -> BrokerOrder:
        result = self._client.get_order_by_id(broker_order_id)
        return _to_broker_order(result)

    async def get_open_orders(self) -> list[BrokerOrder]:
        req = GetOrdersRequest(status="open")
        results = self._client.get_orders(filter=req)
        return [_to_broker_order(o) for o in results]

    async def get_positions(self) -> list[BrokerPosition]:
        results = self._client.get_all_positions()
        return [_to_broker_position(p) for p in results]

    async def get_position(self, symbol: str) -> BrokerPosition | None:
        try:
            result = self._client.get_open_position(symbol)
            return _to_broker_position(result)
        except Exception:
            return None

    async def get_account(self) -> BrokerAccount:
        acct = self._client.get_account()
        return BrokerAccount(
            account_id=str(acct.id),
            equity=Decimal(str(acct.equity)),
            cash=Decimal(str(acct.cash)),
            buying_power=Decimal(str(acct.buying_power)),
            portfolio_value=Decimal(str(acct.portfolio_value)),
            raw={"status": str(acct.status)},
        )

    async def close_position(self, symbol: str) -> BrokerOrder:
        try:
            result = self._client.close_position(symbol)
            return _to_broker_order(result)
        except Exception as e:
            raise OrderSubmissionError(f"Failed to close position for {symbol}: {e}") from e

    async def close_all_positions(self) -> list[BrokerOrder]:
        try:
            results = self._client.close_all_positions(cancel_orders=True)
            return [_to_broker_order(r) for r in results if hasattr(r, "id")]
        except Exception as e:
            raise OrderSubmissionError(f"Failed to close all positions: {e}") from e
