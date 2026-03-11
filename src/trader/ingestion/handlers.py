"""Event handlers for normalized market data events."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Hashable

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from trader.core.events import BarEvent, QuoteEvent, TradeEvent
from trader.core.redis_client import publish_event, xadd_event
from trader.db.models.market_bar import MarketBar
from trader.ingestion.staleness import StalenessDetector
from trader.services.system_state import SystemStateStore

logger = structlog.get_logger(__name__)


class _DeduplicatingHandler:
    """Small in-memory dedupe window for reconnect/replay duplicates."""

    def __init__(self, state_store: SystemStateStore | None = None, max_recent_keys: int = 10000) -> None:
        self._recent_keys: set[Hashable] = set()
        self._recent_order: deque[Hashable] = deque()
        self._max_recent_keys = max_recent_keys
        self._state_store = state_store

    def _remember(self, key: Hashable) -> bool:
        if key in self._recent_keys:
            return False
        self._recent_keys.add(key)
        self._recent_order.append(key)
        while len(self._recent_order) > self._max_recent_keys:
            expired = self._recent_order.popleft()
            self._recent_keys.discard(expired)
        return True

    async def _remember_distributed(self, namespace: str, key: Hashable) -> bool:
        if not self._remember(key):
            return False
        if self._state_store is None:
            return True
        serialized = repr(key)
        return await self._state_store.remember_once(namespace, serialized, ttl_seconds=300)


class BarHandler(_DeduplicatingHandler):
    """Handles incoming bar events: persist to DB and publish to Redis."""

    def __init__(
        self,
        staleness: StalenessDetector,
        persist: bool = True,
        state_store: SystemStateStore | None = None,
    ) -> None:
        super().__init__(state_store=state_store)
        self._staleness = staleness
        self._persist = persist
        self._count = 0

    async def handle(self, event: BarEvent, session: AsyncSession | None = None) -> None:
        event_key = (
            "bar",
            event.symbol,
            event.interval,
            event.timestamp,
            event.open,
            event.high,
            event.low,
            event.close,
            event.volume,
        )
        if not await self._remember_distributed("bars", event_key):
            logger.debug("duplicate_bar_ignored", symbol=event.symbol, timestamp=event.timestamp.isoformat())
            return
        self._staleness.record_update(event.symbol, event.timestamp)
        self._count += 1

        bar_data = event.model_dump(mode="json")

        await publish_event(f"bars:{event.symbol}", bar_data)
        await xadd_event("stream:bars", {"symbol": event.symbol, "data": json.dumps(bar_data, default=str)})

        if self._persist and session is not None:
            values = {
                "symbol": event.symbol,
                "timestamp": event.timestamp,
                "interval": event.interval,
                "open": event.open,
                "high": event.high,
                "low": event.low,
                "close": event.close,
                "volume": event.volume,
                "vwap": event.vwap,
                "trade_count": event.trade_count,
            }
            dialect_name = session.get_bind().dialect.name
            if dialect_name == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as dialect_insert

                statement = dialect_insert(MarketBar).values(**values).on_conflict_do_nothing(
                    index_elements=["symbol", "interval", "timestamp"]
                )
                await session.execute(statement)
            elif dialect_name == "sqlite":
                from sqlalchemy.dialects.sqlite import insert as dialect_insert

                statement = dialect_insert(MarketBar).values(**values).on_conflict_do_nothing(
                    index_elements=["symbol", "interval", "timestamp"]
                )
                await session.execute(statement)
            else:
                existing = await session.scalar(
                    select(MarketBar.id).where(
                        MarketBar.symbol == event.symbol,
                        MarketBar.interval == event.interval,
                        MarketBar.timestamp == event.timestamp,
                    )
                )
                if existing is None:
                    session.add(MarketBar(**values))

        if self._count % 100 == 0:
            logger.debug("bar_handler_progress", count=self._count, latest_symbol=event.symbol)

    @property
    def count(self) -> int:
        return self._count


class QuoteHandler(_DeduplicatingHandler):
    """Handles incoming quote events: publish to Redis."""

    def __init__(self, staleness: StalenessDetector, state_store: SystemStateStore | None = None) -> None:
        super().__init__(state_store=state_store)
        self._staleness = staleness

    async def handle(self, event: QuoteEvent) -> None:
        event_key = (
            "quote",
            event.symbol,
            event.timestamp,
            event.bid_price,
            event.bid_size,
            event.ask_price,
            event.ask_size,
        )
        if not await self._remember_distributed("quotes", event_key):
            logger.debug(
                "duplicate_quote_ignored",
                symbol=event.symbol,
                timestamp=event.timestamp.isoformat(),
            )
            return
        self._staleness.record_update(event.symbol, event.timestamp)
        quote_data = event.model_dump(mode="json")
        await publish_event(f"quotes:{event.symbol}", quote_data)
        await xadd_event("stream:quotes", {"symbol": event.symbol, "data": json.dumps(quote_data, default=str)})


class TradeHandler(_DeduplicatingHandler):
    """Handles incoming trade events: publish to Redis."""

    def __init__(self, staleness: StalenessDetector, state_store: SystemStateStore | None = None) -> None:
        super().__init__(state_store=state_store)
        self._staleness = staleness

    async def handle(self, event: TradeEvent) -> None:
        event_key = (
            "trade",
            event.symbol,
            event.timestamp,
            event.price,
            event.size,
            tuple(event.conditions),
        )
        if not await self._remember_distributed("trades", event_key):
            logger.debug(
                "duplicate_trade_ignored",
                symbol=event.symbol,
                timestamp=event.timestamp.isoformat(),
            )
            return
        self._staleness.record_update(event.symbol, event.timestamp)
        trade_data = event.model_dump(mode="json")
        await publish_event(f"trades:{event.symbol}", trade_data)
        await xadd_event("stream:trades", {"symbol": event.symbol, "data": json.dumps(trade_data, default=str)})
