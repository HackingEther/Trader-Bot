"""In-memory paper trading broker provider for simulation."""

from __future__ import annotations

import uuid
from decimal import Decimal

import structlog

from trader.core.exceptions import OrderCancellationError, OrderSubmissionError
from trader.providers.broker.base import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
    BrokerProvider,
    OrderRequest,
)

logger = structlog.get_logger(__name__)


class PaperBrokerProvider(BrokerProvider):
    """Paper trading broker that simulates order fills in memory."""

    def __init__(self, initial_cash: Decimal = Decimal("100000")) -> None:
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._orders: dict[str, BrokerOrder] = {}
        self._positions: dict[str, BrokerPosition] = {}
        logger.info("paper_broker_initialized", initial_cash=str(initial_cash))

    async def submit_order(self, request: OrderRequest) -> BrokerOrder:
        broker_id = str(uuid.uuid4())
        client_id = request.client_order_id or str(uuid.uuid4())

        if request.order_type in ("limit", "stop", "stop_limit"):
            order = BrokerOrder(
                broker_order_id=broker_id,
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                qty=request.qty,
                limit_price=request.limit_price,
                stop_price=request.stop_price,
                status="accepted",
                raw={"client_order_id": client_id, "paper": True},
            )
            self._orders[broker_id] = order
            logger.info("paper_order_accepted", broker_order_id=broker_id, symbol=request.symbol)
            return order

        fill_price = request.limit_price or Decimal("100.00")
        notional = fill_price * request.qty

        if request.side == "buy" and notional > self._cash:
            raise OrderSubmissionError(f"Insufficient paper cash: need {notional}, have {self._cash}")

        order = BrokerOrder(
            broker_order_id=broker_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            qty=request.qty,
            filled_qty=request.qty,
            filled_avg_price=fill_price,
            status="filled",
            raw={"client_order_id": client_id, "paper": True},
        )
        self._orders[broker_id] = order

        self._update_position(request.symbol, request.side, request.qty, fill_price)
        logger.info(
            "paper_order_filled",
            broker_order_id=broker_id,
            symbol=request.symbol,
            side=request.side,
            qty=request.qty,
            price=str(fill_price),
        )
        return order

    def _update_position(self, symbol: str, side: str, qty: int, price: Decimal) -> None:
        existing = self._positions.get(symbol)
        if existing is None:
            mv = price * qty
            self._positions[symbol] = BrokerPosition(
                symbol=symbol,
                qty=qty,
                side=side,
                avg_entry_price=price,
                current_price=price,
                market_value=mv,
                unrealized_pnl=Decimal("0"),
            )
            if side == "buy":
                self._cash -= mv
            else:
                self._cash += mv
        else:
            if existing.side == side:
                total_qty = existing.qty + qty
                total_cost = existing.avg_entry_price * existing.qty + price * qty
                new_avg = total_cost / total_qty if total_qty > 0 else price
                mv = new_avg * total_qty
                self._positions[symbol] = BrokerPosition(
                    symbol=symbol, qty=total_qty, side=side,
                    avg_entry_price=new_avg, current_price=price,
                    market_value=mv, unrealized_pnl=Decimal("0"),
                )
                if side == "buy":
                    self._cash -= price * qty
                else:
                    self._cash += price * qty
            else:
                remaining = existing.qty - qty
                pnl = (price - existing.avg_entry_price) * qty
                if existing.side == "sell":
                    pnl = -pnl
                self._cash += pnl + existing.avg_entry_price * min(qty, existing.qty)
                if remaining > 0:
                    mv = existing.avg_entry_price * remaining
                    self._positions[symbol] = BrokerPosition(
                        symbol=symbol, qty=remaining, side=existing.side,
                        avg_entry_price=existing.avg_entry_price, current_price=price,
                        market_value=mv, unrealized_pnl=Decimal("0"),
                    )
                elif remaining == 0:
                    del self._positions[symbol]
                else:
                    new_qty = abs(remaining)
                    mv = price * new_qty
                    self._positions[symbol] = BrokerPosition(
                        symbol=symbol, qty=new_qty, side=side,
                        avg_entry_price=price, current_price=price,
                        market_value=mv, unrealized_pnl=Decimal("0"),
                    )

    async def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        order = self._orders.get(broker_order_id)
        if not order:
            raise OrderCancellationError(f"Order {broker_order_id} not found")
        if order.status in ("filled", "cancelled"):
            raise OrderCancellationError(f"Order {broker_order_id} cannot be cancelled (status={order.status})")
        cancelled = BrokerOrder(
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            qty=order.qty,
            status="cancelled",
            raw=order.raw,
        )
        self._orders[broker_order_id] = cancelled
        return cancelled

    async def replace_order(
        self, broker_order_id: str, qty: int | None = None, limit_price: Decimal | None = None
    ) -> BrokerOrder:
        order = self._orders.get(broker_order_id)
        if not order:
            raise OrderSubmissionError(f"Order {broker_order_id} not found")
        replaced = BrokerOrder(
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            qty=qty or order.qty,
            limit_price=limit_price or order.limit_price,
            status=order.status,
            raw=order.raw,
        )
        self._orders[broker_order_id] = replaced
        return replaced

    async def get_order(self, broker_order_id: str) -> BrokerOrder:
        order = self._orders.get(broker_order_id)
        if not order:
            raise OrderSubmissionError(f"Order {broker_order_id} not found")
        return order

    async def get_open_orders(self) -> list[BrokerOrder]:
        return [o for o in self._orders.values() if o.status in ("pending", "submitted", "accepted")]

    async def get_positions(self) -> list[BrokerPosition]:
        return list(self._positions.values())

    async def get_position(self, symbol: str) -> BrokerPosition | None:
        return self._positions.get(symbol)

    async def get_account(self) -> BrokerAccount:
        equity = self._cash + sum(p.market_value for p in self._positions.values())
        return BrokerAccount(
            account_id="paper-account",
            equity=equity,
            cash=self._cash,
            buying_power=self._cash,
            portfolio_value=equity,
            raw={"paper": True},
        )

    async def close_position(self, symbol: str) -> BrokerOrder:
        pos = self._positions.get(symbol)
        if not pos:
            raise OrderSubmissionError(f"No position for {symbol}")
        close_side = "sell" if pos.side == "buy" else "buy"
        return await self.submit_order(OrderRequest(
            symbol=symbol, side=close_side, qty=pos.qty, order_type="market",
        ))

    async def close_all_positions(self) -> list[BrokerOrder]:
        symbols = list(self._positions.keys())
        results = []
        for sym in symbols:
            results.append(await self.close_position(sym))
        return results
