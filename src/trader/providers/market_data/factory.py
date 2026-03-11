"""Factory for selecting the configured market data provider."""

from __future__ import annotations

from trader.config import Settings
from trader.core.exceptions import MarketDataConnectionError
from trader.providers.market_data.alpaca import AlpacaProvider
from trader.providers.market_data.base import MarketDataProvider
from trader.providers.market_data.databento import DatabentoProvider


def create_market_data_provider(settings: Settings) -> MarketDataProvider:
    """Create a market data provider based on application settings."""
    provider = settings.market_data_provider
    if provider == "alpaca":
        return AlpacaProvider(
            api_key=settings.alpaca_api_key,
            api_secret=settings.alpaca_api_secret,
            feed=settings.alpaca_data_feed,
        )
    if provider == "databento":
        return DatabentoProvider(
            api_key=settings.databento_api_key,
            dataset=settings.databento_dataset,
        )
    raise MarketDataConnectionError(f"Unsupported market data provider: {provider}")
