"""Alpaca live market data provider implementation."""

from __future__ import annotations

import asyncio
from datetime import timezone
from decimal import Decimal

import structlog

from trader.core.events import BarEvent, QuoteEvent, TradeEvent
from trader.core.exceptions import MarketDataConnectionError
from trader.providers.market_data.base import (
    MarketDataProvider,
    OnBarCallback,
    OnErrorCallback,
    OnQuoteCallback,
    OnSessionCallback,
    OnTradeCallback,
)

logger = structlog.get_logger(__name__)


class AlpacaProvider(MarketDataProvider):
    """Alpaca live market data provider using alpaca-py streams."""

    def __init__(self, api_key: str, api_secret: str, feed: str = "iex") -> None:
        if not api_key or not api_secret:
            raise MarketDataConnectionError("Alpaca API key and secret are required for live data")
        self._api_key = api_key
        self._api_secret = api_secret
        self._feed = feed.lower()
        self._stream: object | None = None
        self._connected = False
        self._run_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._bar_callbacks: list[OnBarCallback] = []
        self._quote_callbacks: list[OnQuoteCallback] = []
        self._trade_callbacks: list[OnTradeCallback] = []
        self._session_callbacks: list[OnSessionCallback] = []
        self._error_callbacks: list[OnErrorCallback] = []

    async def connect(self) -> None:
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.live.stock import StockDataStream

            feed = DataFeed.IEX if self._feed == "iex" else DataFeed.SIP
            self._stream = StockDataStream(
                api_key=self._api_key,
                secret_key=self._api_secret,
                feed=feed,
                raw_data=False,
            )
            self._connected = True
            logger.info("alpaca_market_data_connected", feed=self._feed)
        except Exception as e:
            self._connected = False
            logger.error("alpaca_market_data_connect_failed", error=str(e), feed=self._feed)
            raise MarketDataConnectionError(f"Alpaca market data connection failed: {e}") from e

    async def disconnect(self) -> None:
        if self._stream is not None:
            try:
                if hasattr(self._stream, "stop_ws"):
                    await self._stream.stop_ws()  # type: ignore[union-attr]
                elif hasattr(self._stream, "stop"):
                    self._stream.stop()  # type: ignore[union-attr]
            except Exception as e:
                logger.warning("alpaca_market_data_stop_failed", error=str(e))
            try:
                if hasattr(self._stream, "close"):
                    await self._stream.close()  # type: ignore[union-attr]
            except Exception as e:
                logger.warning("alpaca_market_data_close_failed", error=str(e))
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
        self._connected = False
        logger.info("alpaca_market_data_disconnected")

    async def subscribe(
        self,
        symbols: list[str],
        bars: bool = True,
        quotes: bool = False,
        trades: bool = False,
    ) -> None:
        if self._stream is None:
            raise MarketDataConnectionError("Not connected to Alpaca market data")

        try:
            if bars:
                self._stream.subscribe_bars(self._handle_bar, *symbols)  # type: ignore[union-attr]
            if quotes:
                self._stream.subscribe_quotes(self._handle_quote, *symbols)  # type: ignore[union-attr]
            if trades:
                self._stream.subscribe_trades(self._handle_trade, *symbols)  # type: ignore[union-attr]
        except Exception as e:
            logger.error("alpaca_market_data_subscribe_failed", symbols=symbols, error=str(e))
            raise MarketDataConnectionError(f"Alpaca subscribe failed: {e}") from e

        if self._run_task is None or self._run_task.done():
            self._run_task = asyncio.create_task(asyncio.to_thread(self._stream.run))  # type: ignore[union-attr]
        logger.info(
            "alpaca_market_data_subscribed",
            feed=self._feed,
            symbols=symbols,
            bars=bars,
            quotes=quotes,
            trades=trades,
        )

    async def unsubscribe(self, symbols: list[str]) -> None:
        logger.info("alpaca_market_data_unsubscribe", symbols=symbols)

    def on_bar(self, callback: OnBarCallback) -> None:
        if callback not in self._bar_callbacks:
            self._bar_callbacks.append(callback)

    def on_quote(self, callback: OnQuoteCallback) -> None:
        if callback not in self._quote_callbacks:
            self._quote_callbacks.append(callback)

    def on_trade(self, callback: OnTradeCallback) -> None:
        if callback not in self._trade_callbacks:
            self._trade_callbacks.append(callback)

    def on_session(self, callback: OnSessionCallback) -> None:
        if callback not in self._session_callbacks:
            self._session_callbacks.append(callback)

    def on_error(self, callback: OnErrorCallback) -> None:
        if callback not in self._error_callbacks:
            self._error_callbacks.append(callback)

    async def is_connected(self) -> bool:
        return self._connected and (self._run_task is None or not self._run_task.done())

    async def _handle_bar(self, bar: object) -> None:
        parsed = self._parse_bar(bar)
        if parsed is None:
            return
        for callback in self._bar_callbacks:
            await callback(parsed)

    async def _handle_quote(self, quote: object) -> None:
        parsed = self._parse_quote(quote)
        if parsed is None:
            return
        for callback in self._quote_callbacks:
            await callback(parsed)

    async def _handle_trade(self, trade: object) -> None:
        parsed = self._parse_trade(trade)
        if parsed is None:
            return
        for callback in self._trade_callbacks:
            await callback(parsed)

    def _parse_bar(self, bar: object) -> BarEvent | None:
        try:
            b = bar  # type: ignore[assignment]
            ts = getattr(b, "timestamp")
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return BarEvent(
                symbol=str(getattr(b, "symbol")),
                timestamp=ts,
                open=Decimal(str(getattr(b, "open"))),
                high=Decimal(str(getattr(b, "high"))),
                low=Decimal(str(getattr(b, "low"))),
                close=Decimal(str(getattr(b, "close"))),
                volume=int(getattr(b, "volume")),
                vwap=Decimal(str(getattr(b, "vwap"))) if getattr(b, "vwap", None) is not None else None,
                trade_count=int(getattr(b, "trade_count")) if getattr(b, "trade_count", None) is not None else None,
                interval="1m",
            )
        except Exception as e:
            logger.warning("alpaca_market_data_bar_parse_failed", error=str(e))
            return None

    def _parse_quote(self, quote: object) -> QuoteEvent | None:
        try:
            q = quote  # type: ignore[assignment]
            ts = getattr(q, "timestamp")
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return QuoteEvent(
                symbol=str(getattr(q, "symbol")),
                timestamp=ts,
                bid_price=Decimal(str(getattr(q, "bid_price"))),
                bid_size=int(getattr(q, "bid_size")),
                ask_price=Decimal(str(getattr(q, "ask_price"))),
                ask_size=int(getattr(q, "ask_size")),
            )
        except Exception as e:
            logger.warning("alpaca_market_data_quote_parse_failed", error=str(e))
            return None

    def _parse_trade(self, trade: object) -> TradeEvent | None:
        try:
            t = trade  # type: ignore[assignment]
            ts = getattr(t, "timestamp")
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            conditions = getattr(t, "conditions", None) or []
            return TradeEvent(
                symbol=str(getattr(t, "symbol")),
                timestamp=ts,
                price=Decimal(str(getattr(t, "price"))),
                size=int(getattr(t, "size")),
                conditions=[str(condition) for condition in conditions],
            )
        except Exception as e:
            logger.warning("alpaca_market_data_trade_parse_failed", error=str(e))
            return None
