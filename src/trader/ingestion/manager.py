"""Ingestion orchestrator managing market data feeds."""

from __future__ import annotations

import asyncio
import uuid

import structlog

from trader.core.exceptions import MarketDataConnectionError
from trader.db.session import get_session_factory
from trader.ingestion.handlers import BarHandler, QuoteHandler, TradeHandler
from trader.ingestion.reconnect import ReconnectPolicy
from trader.ingestion.staleness import StalenessDetector
from trader.providers.market_data.base import MarketDataProvider
from trader.services.system_state import SystemStateStore

logger = structlog.get_logger(__name__)


class IngestionManager:
    """Orchestrates market data ingestion with reconnection and staleness detection."""

    def __init__(
        self,
        provider: MarketDataProvider,
        symbols: list[str],
        staleness_threshold: float = 30.0,
        subscribe_bars: bool = True,
        subscribe_quotes: bool = False,
        subscribe_trades: bool = False,
    ) -> None:
        self._provider = provider
        self._symbols = symbols
        self._subscribe_bars = subscribe_bars
        self._subscribe_quotes = subscribe_quotes
        self._subscribe_trades = subscribe_trades

        self._staleness = StalenessDetector(threshold_seconds=staleness_threshold)
        self._reconnect = ReconnectPolicy()
        self._state_store = SystemStateStore()
        self._bar_handler = BarHandler(self._staleness, state_store=self._state_store)
        self._quote_handler = QuoteHandler(self._staleness, state_store=self._state_store)
        self._trade_handler = TradeHandler(self._staleness, state_store=self._state_store)
        self._lease_name = f"ingestion:{type(provider).__name__}:{','.join(sorted(symbols))}"
        self._lease_owner = str(uuid.uuid4())

        self._running = False
        self._staleness_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._lease_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._callbacks_registered = False

    @property
    def staleness_detector(self) -> StalenessDetector:
        return self._staleness

    @property
    def bar_handler(self) -> BarHandler:
        return self._bar_handler

    async def start(self) -> None:
        """Start the ingestion pipeline with auto-reconnect."""
        lease_acquired = await self._state_store.acquire_lease(self._lease_name, self._lease_owner, ttl_seconds=15)
        if not lease_acquired:
            logger.warning("ingestion_lock_unavailable", lease=self._lease_name)
            return
        self._running = True
        self._lease_task = asyncio.create_task(self._lease_keepalive())
        logger.info("ingestion_starting", symbols=self._symbols)

        while self._running:
            try:
                await self._connect_and_subscribe()
                self._reconnect.reset()
                self._staleness_task = asyncio.create_task(self._staleness_monitor())

                while self._running and await self._provider.is_connected():
                    await asyncio.sleep(1.0)

                if self._running:
                    logger.warning("ingestion_connection_lost")

            except MarketDataConnectionError as e:
                logger.error("ingestion_connection_error", error=str(e))
            except Exception as e:
                logger.error("ingestion_unexpected_error", error=str(e))

            if self._staleness_task and not self._staleness_task.done():
                self._staleness_task.cancel()

            if self._running:
                should_retry = await self._reconnect.wait()
                if not should_retry:
                    logger.critical("ingestion_giving_up", message="Max reconnection attempts exceeded")
                    self._running = False

    async def stop(self) -> None:
        """Stop the ingestion pipeline."""
        self._running = False
        if self._staleness_task and not self._staleness_task.done():
            self._staleness_task.cancel()
        if self._lease_task and not self._lease_task.done():
            self._lease_task.cancel()
        try:
            await self._provider.disconnect()
        except Exception as e:
            logger.error("ingestion_disconnect_error", error=str(e))
        logger.info("ingestion_stopped")

    async def _connect_and_subscribe(self) -> None:
        """Connect to provider and subscribe to data feeds."""
        await self._provider.connect()

        if not self._callbacks_registered:
            self._provider.on_bar(self._on_bar)
            self._provider.on_quote(self._on_quote)
            self._provider.on_trade(self._on_trade)
            self._provider.on_error(self._on_error)
            self._callbacks_registered = True

        await self._provider.subscribe(
            symbols=self._symbols,
            bars=self._subscribe_bars,
            quotes=self._subscribe_quotes,
            trades=self._subscribe_trades,
        )
        logger.info("ingestion_subscribed", symbols=self._symbols)

    async def _on_bar(self, event: object) -> None:
        from trader.core.events import BarEvent
        if isinstance(event, BarEvent):
            try:
                factory = get_session_factory()
            except RuntimeError:
                factory = None

            if factory is None:
                await self._bar_handler.handle(event)
            else:
                async with factory() as session:
                    await self._bar_handler.handle(event, session=session)
                    await session.commit()
            await self._state_store.record_bar_timestamp(event.symbol, event.timestamp)

    async def _on_quote(self, event: object) -> None:
        from trader.core.events import QuoteEvent
        if isinstance(event, QuoteEvent):
            await self._quote_handler.handle(event)
            if event.bid_price > 0:
                spread_bps = float(event.spread / event.bid_price * 10000)
                await self._state_store.record_spread_bps(event.symbol, spread_bps)

    async def _on_trade(self, event: object) -> None:
        from trader.core.events import TradeEvent
        if isinstance(event, TradeEvent):
            await self._trade_handler.handle(event)

    async def _on_error(self, error: Exception) -> None:
        logger.error("ingestion_provider_error", error=str(error))

    async def _staleness_monitor(self) -> None:
        """Periodically check for stale data feeds."""
        try:
            while self._running:
                await asyncio.sleep(10.0)
                stale = self._staleness.check_all()
                if stale:
                    logger.warning("ingestion_stale_symbols", symbols=stale)
        except asyncio.CancelledError:
            pass

    async def _lease_keepalive(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(5.0)
                renewed = await self._state_store.renew_lease(self._lease_name, self._lease_owner, ttl_seconds=15)
                if not renewed:
                    logger.warning("ingestion_lock_lost", lease=self._lease_name)
                    self._running = False
                    try:
                        await self._provider.disconnect()
                    except Exception as exc:
                        logger.error("ingestion_disconnect_error", error=str(exc))
                    return
        except asyncio.CancelledError:
            pass
