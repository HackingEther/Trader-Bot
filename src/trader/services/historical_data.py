"""Historical market-data utilities for backfills and backtests."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import structlog
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from trader.config import get_settings
from trader.db.models.market_bar import MarketBar
from trader.db.repositories.market_bars import MarketBarRepository

logger = structlog.get_logger(__name__)


class HistoricalDataService:
    """Load and backfill historical bars for research and backtests."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = MarketBarRepository(session)
        self._settings = get_settings()

    async def load_bars_by_symbol(
        self,
        symbols: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str = "1m",
    ) -> dict[str, list[dict]]:
        bars = await self._repo.get_range(symbols=symbols, start=start, end=end, interval=interval)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for bar in bars:
            grouped[bar.symbol].append(self._to_bar_dict(bar))
        return dict(grouped)

    async def backfill_from_alpaca(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1Min",
        adjustment: str = "raw",
        feed: str = "iex",
    ) -> int:
        """Backfill bars from Alpaca's market-data API into `market_bars`."""
        if not self._settings.alpaca_api_key or not self._settings.alpaca_api_secret:
            raise ValueError("Alpaca credentials are required for historical backfills")

        headers = {
            "APCA-API-KEY-ID": self._settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": self._settings.alpaca_api_secret,
        }
        params: dict[str, str | int] = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": start.astimezone(UTC).isoformat(),
            "end": end.astimezone(UTC).isoformat(),
            "adjustment": adjustment,
            "feed": feed,
            "limit": 10000,
            "sort": "asc",
        }

        inserted = 0
        interval = "1m" if timeframe == "1Min" else timeframe.lower()
        existing = await self._repo.get_existing_keys(
            symbols,
            start=start.astimezone(UTC),
            end=end.astimezone(UTC),
            interval=interval,
        )
        dialect_name = self._session.get_bind().dialect.name
        use_on_conflict = dialect_name in ("postgresql", "sqlite")

        async with httpx.AsyncClient(base_url="https://data.alpaca.markets", timeout=30.0) as client:
            next_page_token: str | None = None
            while True:
                request_params = dict(params)
                if next_page_token:
                    request_params["page_token"] = next_page_token
                response = await client.get("/v2/stocks/bars", headers=headers, params=request_params)
                response.raise_for_status()
                payload = response.json()

                batch_values: list[dict] = []
                for symbol, bars in payload.get("bars", {}).items():
                    for raw_bar in bars:
                        timestamp = datetime.fromisoformat(raw_bar["t"].replace("Z", "+00:00"))
                        key = (symbol, timestamp)
                        if key in existing:
                            continue
                        existing.add(key)
                        batch_values.append({
                            "symbol": symbol,
                            "timestamp": timestamp,
                            "interval": interval,
                            "open": Decimal(str(raw_bar["o"])),
                            "high": Decimal(str(raw_bar["h"])),
                            "low": Decimal(str(raw_bar["l"])),
                            "close": Decimal(str(raw_bar["c"])),
                            "volume": int(raw_bar["v"]),
                            "vwap": Decimal(str(raw_bar["vw"])) if raw_bar.get("vw") is not None else None,
                            "trade_count": int(raw_bar["n"]) if raw_bar.get("n") is not None else None,
                        })

                if batch_values:
                    if use_on_conflict:
                        if dialect_name == "postgresql":
                            from sqlalchemy.dialects.postgresql import insert as dialect_insert
                        else:
                            from sqlalchemy.dialects.sqlite import insert as dialect_insert
                        stmt = dialect_insert(MarketBar).values(batch_values).on_conflict_do_nothing(
                            index_elements=["symbol", "interval", "timestamp"]
                        )
                        await self._session.execute(stmt)
                        inserted += len(batch_values)
                    else:
                        for values in batch_values:
                            try:
                                async with self._session.begin_nested():
                                    self._session.add(MarketBar(**values))
                                    await self._session.flush()
                                inserted += 1
                            except IntegrityError:
                                pass

                await self._session.flush()
                next_page_token = payload.get("next_page_token")
                if not next_page_token:
                    break

        logger.info(
            "historical_bars_backfilled",
            symbols=symbols,
            inserted=inserted,
            timeframe=timeframe,
            feed=feed,
        )
        return inserted

    @staticmethod
    def _to_bar_dict(bar: MarketBar) -> dict:
        return {
            "symbol": bar.symbol,
            "timestamp": bar.timestamp,
            "interval": bar.interval,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "vwap": bar.vwap,
            "trade_count": bar.trade_count,
        }
