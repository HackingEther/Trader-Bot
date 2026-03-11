"""Synthetic bar data generators for deterministic testing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import random


def generate_synthetic_bars(
    symbol: str = "AAPL",
    count: int = 100,
    start_price: float = 150.0,
    volatility: float = 0.001,
    start_time: datetime | None = None,
    seed: int = 42,
) -> list[dict]:
    """Generate deterministic synthetic 1-minute OHLCV bars.

    Args:
        symbol: Ticker symbol.
        count: Number of bars to generate.
        start_price: Starting close price.
        volatility: Per-bar volatility (std dev of returns).
        start_time: Starting timestamp (defaults to market open today).
        seed: Random seed for reproducibility.
    """
    rng = random.Random(seed)
    if start_time is None:
        today = datetime.now(timezone.utc).date()
        start_time = datetime(today.year, today.month, today.day, 14, 30, tzinfo=timezone.utc)

    bars: list[dict] = []
    price = start_price

    for i in range(count):
        ret = rng.gauss(0, volatility)
        close = round(price * (1 + ret), 4)
        high = round(max(price, close) * (1 + abs(rng.gauss(0, volatility * 0.5))), 4)
        low = round(min(price, close) * (1 - abs(rng.gauss(0, volatility * 0.5))), 4)
        volume = max(100, int(rng.gauss(50000, 15000)))

        bar = {
            "symbol": symbol,
            "timestamp": start_time + timedelta(minutes=i),
            "interval": "1m",
            "open": Decimal(str(price)),
            "high": Decimal(str(high)),
            "low": Decimal(str(low)),
            "close": Decimal(str(close)),
            "volume": volume,
            "vwap": Decimal(str(round((high + low + close) / 3, 4))),
            "trade_count": max(10, int(rng.gauss(500, 100))),
        }
        bars.append(bar)
        price = close

    return bars


def generate_multi_symbol_bars(
    symbols: list[str] | None = None,
    count: int = 100,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Generate bars for multiple symbols."""
    symbols = symbols or ["AAPL", "MSFT", "GOOGL"]
    result: dict[str, list[dict]] = {}
    for i, sym in enumerate(symbols):
        result[sym] = generate_synthetic_bars(
            symbol=sym,
            count=count,
            start_price=150.0 + i * 50,
            seed=seed + i,
        )
    return result
