"""Run a historical backtest from stored market bars."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tests.fixtures.synthetic_bars import generate_multi_symbol_bars
from trader.backtest.engine import BacktestEngine
from trader.config import get_settings
from trader.logging import setup_logging
from trader.services.historical_data import HistoricalDataService


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backtest from historical market bars")
    parser.add_argument("--symbols", default="SPY,QQQ,AAPL,MSFT,NVDA")
    parser.add_argument("--start", help="ISO timestamp")
    parser.add_argument("--end", help="ISO timestamp")
    parser.add_argument("--name", default="historical-backtest")
    parser.add_argument("--synthetic-fallback", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    start = _parse_dt(args.start)
    end = _parse_dt(args.end)

    async with factory() as session:
        history = HistoricalDataService(session)
        bars_by_symbol = await history.load_bars_by_symbol(symbols, start=start, end=end)
        if not any(bars_by_symbol.values()) and args.synthetic_fallback:
            bars_by_symbol = generate_multi_symbol_bars(symbols=symbols, count=390, seed=42)

        normalized = {
            sym: [
                {
                    k: str(v) if hasattr(v, "__abs__") and not isinstance(v, (int, float, str)) else v
                    for k, v in bar.items()
                }
                for bar in sym_bars
            ]
            for sym, sym_bars in bars_by_symbol.items()
        }

        bt_engine = BacktestEngine(session=session)
        results = await bt_engine.run(
            name=args.name,
            symbols=symbols,
            bars_by_symbol=normalized,
            start_date=(start.date().isoformat() if start else "unknown"),
            end_date=(end.date().isoformat() if end else "unknown"),
            strategy_config={
                "min_confidence": settings.min_confidence,
                "min_expected_move_bps": settings.min_expected_move_bps,
            },
            risk_config={
                "max_daily_loss": settings.max_daily_loss_usd,
                "max_positions": settings.max_concurrent_positions,
                "max_notional": settings.max_notional_exposure_usd,
            },
            use_champion_models=True,
        )
        await session.commit()

    await engine.dispose()

    print("\n=== Backtest Results ===")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    setup_logging(log_level="INFO", json_output=False)
    asyncio.run(main())
