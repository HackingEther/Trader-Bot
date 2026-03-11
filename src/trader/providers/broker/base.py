"""Abstract base class for broker integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True)
class BrokerOrder:
    """Standardized order representation from broker."""

    broker_order_id: str
    symbol: str
    side: str
    order_type: str
    qty: int
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    filled_qty: int = 0
    filled_avg_price: Decimal | None = None
    status: str = "pending"
    order_class: str = "simple"
    time_in_force: str = "day"
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerPosition:
    """Standardized position representation from broker."""

    symbol: str
    qty: int
    side: str
    avg_entry_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BrokerAccount:
    """Standardized account representation from broker."""

    account_id: str
    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    portfolio_value: Decimal
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OrderRequest:
    """Order submission request."""

    symbol: str
    side: str
    qty: int
    order_type: str = "market"
    time_in_force: str = "day"
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    order_class: str = "simple"
    take_profit_price: Decimal | None = None
    stop_loss_price: Decimal | None = None
    stop_loss_limit_price: Decimal | None = None
    client_order_id: str | None = None


class BrokerProvider(ABC):
    """Abstract broker interface for order management and account queries."""

    @abstractmethod
    async def submit_order(self, request: OrderRequest) -> BrokerOrder:
        """Submit an order to the broker."""

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        """Cancel an existing order."""

    @abstractmethod
    async def replace_order(
        self, broker_order_id: str, qty: int | None = None, limit_price: Decimal | None = None
    ) -> BrokerOrder:
        """Replace/modify an existing order."""

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> BrokerOrder:
        """Get order details by broker order ID."""

    @abstractmethod
    async def get_open_orders(self) -> list[BrokerOrder]:
        """Get all open orders."""

    @abstractmethod
    async def get_positions(self) -> list[BrokerPosition]:
        """Get all current positions."""

    @abstractmethod
    async def get_position(self, symbol: str) -> BrokerPosition | None:
        """Get position for a specific symbol."""

    @abstractmethod
    async def get_account(self) -> BrokerAccount:
        """Get account details."""

    @abstractmethod
    async def close_position(self, symbol: str) -> BrokerOrder:
        """Close an entire position for a symbol."""

    @abstractmethod
    async def close_all_positions(self) -> list[BrokerOrder]:
        """Close all open positions."""
