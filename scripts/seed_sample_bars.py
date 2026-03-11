"""Seed script for populating sample bar data."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from tests.fixtures.synthetic_bars import generate_synthetic_bars
from trader.config import get_settings
from trader.db.models.market_bar import MarketBar


async def seed() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    symbols = ["AAPL", "MSFT", "GOOGL", "SPY"]

    async with factory() as session:
        for i, symbol in enumerate(symbols):
            bars = generate_synthetic_bars(symbol=symbol, count=390, seed=42 + i)
            for bar_data in bars:
                bar = MarketBar(
                    symbol=bar_data["symbol"],
                    timestamp=bar_data["timestamp"],
                    interval=bar_data["interval"],
                    open=bar_data["open"],
                    high=bar_data["high"],
                    low=bar_data["low"],
                    close=bar_data["close"],
                    volume=bar_data["volume"],
                    vwap=bar_data.get("vwap"),
                    trade_count=bar_data.get("trade_count"),
                )
                session.add(bar)
            print(f"  Added {len(bars)} bars for {symbol}")
        await session.commit()
    await engine.dispose()
    print("Sample bars seeded.")


if __name__ == "__main__":
    asyncio.run(seed())
