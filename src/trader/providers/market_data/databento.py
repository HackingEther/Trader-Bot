"""Databento market data provider implementation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from trader.core.events import BarEvent, QuoteEvent, TradeEvent, SessionEvent
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


class DatabentoProvider(MarketDataProvider):
    """Databento live market data provider."""

    def __init__(self, api_key: str, dataset: str = "XNAS.ITCH") -> None:
        if not api_key:
            raise MarketDataConnectionError("Databento API key is required")
        self._api_key = api_key
        self._dataset = dataset
        self._client: object | None = None
        self._connected = False
        self._bar_callbacks: list[OnBarCallback] = []
        self._quote_callbacks: list[OnQuoteCallback] = []
        self._trade_callbacks: list[OnTradeCallback] = []
        self._session_callbacks: list[OnSessionCallback] = []
        self._error_callbacks: list[OnErrorCallback] = []
        self._recv_task: asyncio.Task | None = None  # type: ignore[type-arg]

    async def connect(self) -> None:
        try:
            import databento as db
            self._client = db.Live(key=self._api_key)
            self._connected = True
            logger.info("databento_connected", dataset=self._dataset)
        except Exception as e:
            self._connected = False
            logger.error("databento_connection_failed", error=str(e))
            raise MarketDataConnectionError(f"Databento connection failed: {e}") from e

    async def disconnect(self) -> None:
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
        if self._client and hasattr(self._client, "stop"):
            try:
                self._client.stop()  # type: ignore[union-attr]
            except Exception:
                pass
        self._connected = False
        logger.info("databento_disconnected")

    async def subscribe(
        self,
        symbols: list[str],
        bars: bool = True,
        quotes: bool = False,
        trades: bool = False,
    ) -> None:
        if not self._client:
            raise MarketDataConnectionError("Not connected to Databento")

        schemas: list[str] = []
        if bars:
            schemas.append("ohlcv-1m")
        if quotes:
            schemas.append("mbp-1")
        if trades:
            schemas.append("trades")

        for schema in schemas:
            try:
                self._client.subscribe(  # type: ignore[union-attr]
                    dataset=self._dataset,
                    schema=schema,
                    symbols=symbols,
                    stype_in="raw_symbol",
                )
                logger.info("databento_subscribed", schema=schema, symbols=symbols)
            except Exception as e:
                logger.error("databento_subscribe_failed", schema=schema, error=str(e))
                raise MarketDataConnectionError(f"Subscribe failed: {e}") from e

        self._recv_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        """Background loop receiving data from Databento."""
        if not self._client:
            return
        try:
            for record in self._client:  # type: ignore[union-attr]
                try:
                    await self._dispatch_record(record)
                except Exception as e:
                    for cb in self._error_callbacks:
                        await cb(e)
        except asyncio.CancelledError:
            return
        except Exception as e:
            self._connected = False
            logger.error("databento_receive_error", error=str(e))
            for cb in self._error_callbacks:
                await cb(e)

    async def _dispatch_record(self, record: object) -> None:
        """Dispatch a Databento record to the appropriate callbacks."""
        rtype = type(record).__name__

        if rtype == "OhlcvMsg":
            bar = self._parse_ohlcv(record)
            if bar:
                for cb in self._bar_callbacks:
                    await cb(bar)
        elif rtype == "Mbp1Msg":
            quote = self._parse_quote(record)
            if quote:
                for cb in self._quote_callbacks:
                    await cb(quote)
        elif rtype == "TradeMsg":
            trade = self._parse_trade(record)
            if trade:
                for cb in self._trade_callbacks:
                    await cb(trade)

    def _parse_ohlcv(self, record: object) -> BarEvent | None:
        try:
            r = record  # type: ignore[assignment]
            FIXED_PRICE_SCALE = Decimal("1e-9")
            symbol = getattr(r, "symbol", "") or str(getattr(r, "instrument_id", ""))
            return BarEvent(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(r.ts_event / 1e9, tz=timezone.utc),
                open=Decimal(str(r.open)) * FIXED_PRICE_SCALE,
                high=Decimal(str(r.high)) * FIXED_PRICE_SCALE,
                low=Decimal(str(r.low)) * FIXED_PRICE_SCALE,
                close=Decimal(str(r.close)) * FIXED_PRICE_SCALE,
                volume=int(r.volume),
                interval="1m",
            )
        except Exception as e:
            logger.warning("databento_ohlcv_parse_error", error=str(e))
            return None

    def _parse_quote(self, record: object) -> QuoteEvent | None:
        try:
            r = record  # type: ignore[assignment]
            FIXED_PRICE_SCALE = Decimal("1e-9")
            symbol = getattr(r, "symbol", "") or str(getattr(r, "instrument_id", ""))
            bid = Decimal(str(r.bid_px_00)) * FIXED_PRICE_SCALE
            ask = Decimal(str(r.ask_px_00)) * FIXED_PRICE_SCALE
            return QuoteEvent(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(r.ts_event / 1e9, tz=timezone.utc),
                bid_price=bid,
                bid_size=int(r.bid_sz_00),
                ask_price=ask,
                ask_size=int(r.ask_sz_00),
            )
        except Exception as e:
            logger.warning("databento_quote_parse_error", error=str(e))
            return None

    def _parse_trade(self, record: object) -> TradeEvent | None:
        try:
            r = record  # type: ignore[assignment]
            FIXED_PRICE_SCALE = Decimal("1e-9")
            symbol = getattr(r, "symbol", "") or str(getattr(r, "instrument_id", ""))
            return TradeEvent(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(r.ts_event / 1e9, tz=timezone.utc),
                price=Decimal(str(r.price)) * FIXED_PRICE_SCALE,
                size=int(r.size),
            )
        except Exception as e:
            logger.warning("databento_trade_parse_error", error=str(e))
            return None

    async def unsubscribe(self, symbols: list[str]) -> None:
        logger.info("databento_unsubscribe", symbols=symbols)

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
        return self._connected
