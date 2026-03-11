"""Abstract base class for market data provider integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Awaitable
from typing import Any

from trader.core.events import BarEvent, QuoteEvent, TradeEvent, SessionEvent


OnBarCallback = Callable[[BarEvent], Awaitable[None]]
OnQuoteCallback = Callable[[QuoteEvent], Awaitable[None]]
OnTradeCallback = Callable[[TradeEvent], Awaitable[None]]
OnSessionCallback = Callable[[SessionEvent], Awaitable[None]]
OnErrorCallback = Callable[[Exception], Awaitable[None]]


class MarketDataProvider(ABC):
    """Abstract market data provider interface."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to market data source."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from market data source."""

    @abstractmethod
    async def subscribe(
        self,
        symbols: list[str],
        bars: bool = True,
        quotes: bool = False,
        trades: bool = False,
    ) -> None:
        """Subscribe to market data for given symbols."""

    @abstractmethod
    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from market data for given symbols."""

    @abstractmethod
    def on_bar(self, callback: OnBarCallback) -> None:
        """Register callback for bar events."""

    @abstractmethod
    def on_quote(self, callback: OnQuoteCallback) -> None:
        """Register callback for quote events."""

    @abstractmethod
    def on_trade(self, callback: OnTradeCallback) -> None:
        """Register callback for trade events."""

    @abstractmethod
    def on_session(self, callback: OnSessionCallback) -> None:
        """Register callback for session events."""

    @abstractmethod
    def on_error(self, callback: OnErrorCallback) -> None:
        """Register callback for error events."""

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if the provider is currently connected."""
