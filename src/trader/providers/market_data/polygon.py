"""Polygon.io market data provider stub."""

from __future__ import annotations

import structlog

from trader.core.exceptions import MarketDataConnectionError
from trader.providers.market_data.base import (
    MarketDataProvider,
    OnBarCallback,
    OnQuoteCallback,
    OnTradeCallback,
    OnSessionCallback,
    OnErrorCallback,
)

logger = structlog.get_logger(__name__)


class PolygonProvider(MarketDataProvider):
    """Polygon.io market data provider - stub implementation.

    Placeholder for future Polygon integration. All methods raise
    NotImplementedError to clearly indicate this provider is not yet functional.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        logger.warning("polygon_provider_stub", message="PolygonProvider is a stub")

    async def connect(self) -> None:
        raise NotImplementedError("PolygonProvider is not yet implemented")

    async def disconnect(self) -> None:
        pass

    async def subscribe(
        self,
        symbols: list[str],
        bars: bool = True,
        quotes: bool = False,
        trades: bool = False,
    ) -> None:
        raise NotImplementedError("PolygonProvider is not yet implemented")

    async def unsubscribe(self, symbols: list[str]) -> None:
        raise NotImplementedError("PolygonProvider is not yet implemented")

    def on_bar(self, callback: OnBarCallback) -> None:
        pass

    def on_quote(self, callback: OnQuoteCallback) -> None:
        pass

    def on_trade(self, callback: OnTradeCallback) -> None:
        pass

    def on_session(self, callback: OnSessionCallback) -> None:
        pass

    def on_error(self, callback: OnErrorCallback) -> None:
        pass

    async def is_connected(self) -> bool:
        return False
