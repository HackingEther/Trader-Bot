"""Broker provider factory."""

from __future__ import annotations

from decimal import Decimal

from trader.config import Settings
from trader.providers.broker.alpaca import AlpacaProvider
from trader.providers.broker.base import BrokerProvider
from trader.providers.broker.paper import PaperBrokerProvider

_cached_broker: BrokerProvider | None = None


def create_broker_provider(settings: Settings) -> BrokerProvider:
    """Create the broker provider for the current environment."""
    global _cached_broker
    if _cached_broker is not None:
        return _cached_broker

    if settings.is_live:
        _cached_broker = AlpacaProvider(
            api_key=settings.alpaca_api_key,
            api_secret=settings.alpaca_api_secret,
            paper=False,
        )
        return _cached_broker
    if settings.alpaca_api_key and settings.alpaca_api_secret:
        _cached_broker = AlpacaProvider(
            api_key=settings.alpaca_api_key,
            api_secret=settings.alpaca_api_secret,
            paper=settings.alpaca_paper,
        )
        return _cached_broker
    _cached_broker = PaperBrokerProvider(initial_cash=Decimal(str(settings.paper_initial_cash)))
    return _cached_broker
