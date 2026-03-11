"""Integration tests for market data providers."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from trader.config import Settings
from trader.providers.market_data.alpaca import AlpacaProvider
from trader.providers.market_data.factory import create_market_data_provider
from trader.providers.market_data.polygon import PolygonProvider


class TestPolygonProviderStub:
    def test_init(self) -> None:
        provider = PolygonProvider(api_key="test")
        assert provider is not None

    @pytest.mark.asyncio
    async def test_connect_raises(self) -> None:
        provider = PolygonProvider(api_key="test")
        with pytest.raises(NotImplementedError):
            await provider.connect()

    @pytest.mark.asyncio
    async def test_is_not_connected(self) -> None:
        provider = PolygonProvider(api_key="test")
        assert not await provider.is_connected()


class TestAlpacaProvider:
    def test_init(self) -> None:
        provider = AlpacaProvider(api_key="key", api_secret="secret", feed="iex")
        assert provider is not None

    def test_parse_bar(self) -> None:
        provider = AlpacaProvider(api_key="key", api_secret="secret")
        bar = provider._parse_bar(
            SimpleNamespace(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open=150.0,
                high=151.0,
                low=149.5,
                close=150.5,
                volume=1000,
                vwap=150.2,
                trade_count=42,
            )
        )
        assert bar is not None
        assert bar.symbol == "AAPL"
        assert bar.volume == 1000

    def test_parse_quote(self) -> None:
        provider = AlpacaProvider(api_key="key", api_secret="secret")
        quote = provider._parse_quote(
            SimpleNamespace(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                bid_price=150.0,
                bid_size=10,
                ask_price=150.1,
                ask_size=12,
            )
        )
        assert quote is not None
        assert quote.symbol == "AAPL"
        assert quote.ask_price > quote.bid_price

    def test_parse_trade(self) -> None:
        provider = AlpacaProvider(api_key="key", api_secret="secret")
        trade = provider._parse_trade(
            SimpleNamespace(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                price=150.05,
                size=25,
                conditions=["@"],
            )
        )
        assert trade is not None
        assert trade.symbol == "AAPL"
        assert trade.size == 25


class TestMarketDataFactory:
    def test_creates_databento_provider(self) -> None:
        settings = Settings(
            databento_api_key="db-key",
            market_data_provider="databento",
        )
        provider = create_market_data_provider(settings)
        from trader.providers.market_data.databento import DatabentoProvider

        assert isinstance(provider, DatabentoProvider)

    def test_creates_alpaca_provider(self) -> None:
        settings = Settings(
            alpaca_api_key="alpaca-key",
            alpaca_api_secret="alpaca-secret",
            market_data_provider="alpaca",
            alpaca_data_feed="iex",
        )
        provider = create_market_data_provider(settings)
        assert isinstance(provider, AlpacaProvider)
