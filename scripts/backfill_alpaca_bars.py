"""Backfill historical minute bars from Alpaca into the local database."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trader.config import get_settings
from trader.logging import setup_logging
from trader.services.historical_data import HistoricalDataService


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill bars from Alpaca")
    parser.add_argument("--symbols", default="SPY,QQQ,AAPL,MSFT,NVDA")
    parser.add_argument("--start", required=True, help="ISO timestamp")
    parser.add_argument("--end", required=True, help="ISO timestamp")
    parser.add_argument("--feed", default="iex")
    args = parser.parse_args()

    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    start = _parse_dt(args.start)
    end = _parse_dt(args.end)
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]

    async with factory() as session:
        history = HistoricalDataService(session)
        inserted = await history.backfill_from_alpaca(
            symbols=symbols,
            start=start,
            end=end,
            timeframe="1Min",
            feed=args.feed,
        )
        await session.commit()

    await engine.dispose()
    print(f"Inserted {inserted} bars for {', '.join(symbols)}")


if __name__ == "__main__":
    setup_logging(log_level="INFO", json_output=False)
    asyncio.run(main())
