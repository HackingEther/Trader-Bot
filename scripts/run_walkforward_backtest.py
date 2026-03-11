"""Train on one historical window and backtest on the following window."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trader.backtest.engine import BacktestEngine
from trader.config import get_settings
from trader.logging import setup_logging
from trader.models.training.dataset import build_momentum_dataset
from trader.models.training.pipeline import TrainingPipeline
from trader.services.historical_data import HistoricalDataService


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a walk-forward train/backtest cycle")
    parser.add_argument("--symbols", default="SPY,QQQ,AAPL,MSFT,NVDA")
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--lookahead-bars", type=int, default=15)
    parser.add_argument("--version-prefix", default="walkforward")
    args = parser.parse_args()

    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    train_start = _parse_dt(args.train_start)
    train_end = _parse_dt(args.train_end)
    test_start = _parse_dt(args.test_start)
    test_end = _parse_dt(args.test_end)
    version = f"{args.version_prefix}-{train_start.date().isoformat()}-{train_end.date().isoformat()}"

    async with factory() as session:
        history = HistoricalDataService(session)
        train_bars = await history.load_bars_by_symbol(symbols, start=train_start, end=train_end)
        dataset = build_momentum_dataset(train_bars, lookahead_bars=args.lookahead_bars)
        if len(dataset.features) == 0:
            raise RuntimeError("No training samples built from the selected training window")

        trainer = TrainingPipeline(session)
        for model_type, labels in (
            ("regime", dataset.regime_labels),
            ("direction", dataset.direction_labels),
            ("magnitude", dataset.magnitude_labels),
            ("filter", dataset.filter_labels),
        ):
            await trainer.run(
                model_type=model_type,
                features=dataset.features,
                labels=labels,
                version=version,
                promote=True,
                notes=(
                    f"Walk-forward momentum continuation train_window="
                    f"{train_start.date().isoformat()}:{train_end.date().isoformat()}"
                ),
            )

        test_bars = await history.load_bars_by_symbol(symbols, start=test_start, end=test_end)
        normalized = {
            sym: [
                {
                    k: str(v) if hasattr(v, "__abs__") and not isinstance(v, (int, float, str)) else v
                    for k, v in bar.items()
                }
                for bar in bars
            ]
            for sym, bars in test_bars.items()
        }
        bt = BacktestEngine(session=session)
        results = await bt.run(
            name=f"walkforward-{version}",
            symbols=symbols,
            bars_by_symbol=normalized,
            start_date=test_start.date().isoformat(),
            end_date=test_end.date().isoformat(),
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
    print(results)


if __name__ == "__main__":
    setup_logging(log_level="INFO", json_output=False)
    asyncio.run(main())
