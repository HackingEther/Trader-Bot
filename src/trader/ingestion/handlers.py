"""Event handlers for normalized market data events."""

from __future__ import annotations

import json

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from trader.core.events import BarEvent, QuoteEvent, TradeEvent
from trader.core.redis_client import publish_event, xadd_event
from trader.db.models.market_bar import MarketBar
from trader.ingestion.staleness import StalenessDetector

logger = structlog.get_logger(__name__)


class BarHandler:
    """Handles incoming bar events: persist to DB and publish to Redis."""

    def __init__(self, staleness: StalenessDetector, persist: bool = True) -> None:
        self._staleness = staleness
        self._persist = persist
        self._count = 0

    async def handle(self, event: BarEvent, session: AsyncSession | None = None) -> None:
        self._staleness.record_update(event.symbol, event.timestamp)
        self._count += 1

        bar_data = event.model_dump(mode="json")

        await publish_event(f"bars:{event.symbol}", bar_data)
        await xadd_event("stream:bars", {"symbol": event.symbol, "data": json.dumps(bar_data, default=str)})

        if self._persist and session is not None:
            bar = MarketBar(
                symbol=event.symbol,
                timestamp=event.timestamp,
                interval=event.interval,
                open=event.open,
                high=event.high,
                low=event.low,
                close=event.close,
                volume=event.volume,
                vwap=event.vwap,
                trade_count=event.trade_count,
            )
            session.add(bar)

        if self._count % 100 == 0:
            logger.debug("bar_handler_progress", count=self._count, latest_symbol=event.symbol)

    @property
    def count(self) -> int:
        return self._count


class QuoteHandler:
    """Handles incoming quote events: publish to Redis."""

    def __init__(self, staleness: StalenessDetector) -> None:
        self._staleness = staleness

    async def handle(self, event: QuoteEvent) -> None:
        self._staleness.record_update(event.symbol, event.timestamp)
        quote_data = event.model_dump(mode="json")
        await publish_event(f"quotes:{event.symbol}", quote_data)
        await xadd_event("stream:quotes", {"symbol": event.symbol, "data": json.dumps(quote_data, default=str)})


class TradeHandler:
    """Handles incoming trade events: publish to Redis."""

    def __init__(self, staleness: StalenessDetector) -> None:
        self._staleness = staleness

    async def handle(self, event: TradeEvent) -> None:
        self._staleness.record_update(event.symbol, event.timestamp)
        trade_data = event.model_dump(mode="json")
        await publish_event(f"trades:{event.symbol}", trade_data)
