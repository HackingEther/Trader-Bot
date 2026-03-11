"""Seed script for populating the symbols table."""

from __future__ import annotations

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from trader.config import get_settings
from trader.db.models.symbol import Symbol

SEED_SYMBOLS = [
    {"ticker": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ"},
    {"ticker": "META", "name": "Meta Platforms Inc.", "exchange": "NASDAQ"},
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ"},
    {"ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "NYSE"},
    {"ticker": "QQQ", "name": "Invesco QQQ Trust", "exchange": "NASDAQ"},
    {"ticker": "AMD", "name": "Advanced Micro Devices", "exchange": "NASDAQ"},
    {"ticker": "NFLX", "name": "Netflix Inc.", "exchange": "NASDAQ"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "exchange": "NYSE"},
    {"ticker": "V", "name": "Visa Inc.", "exchange": "NYSE"},
    {"ticker": "BA", "name": "The Boeing Company", "exchange": "NYSE"},
    {"ticker": "DIS", "name": "The Walt Disney Company", "exchange": "NYSE"},
]


async def seed() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        for sym_data in SEED_SYMBOLS:
            stmt = select(Symbol).where(Symbol.ticker == sym_data["ticker"])
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            if not existing:
                session.add(Symbol(**sym_data))
                print(f"  Added {sym_data['ticker']}")
            else:
                print(f"  Skipped {sym_data['ticker']} (exists)")
        await session.commit()
    await engine.dispose()
    print("Seed complete.")


if __name__ == "__main__":
    asyncio.run(seed())
