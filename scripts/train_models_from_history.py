"""Train and promote ensemble models from historical bars stored in the DB."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

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
    parser = argparse.ArgumentParser(description="Train trader models from historical bars")
    parser.add_argument("--symbols", default="SPY,QQQ,AAPL,MSFT,NVDA")
    parser.add_argument("--start", required=True, help="ISO timestamp")
    parser.add_argument("--end", required=True, help="ISO timestamp")
    parser.add_argument("--lookahead-bars", type=int, default=15)
    parser.add_argument("--version-prefix", default="momentum")
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
        dataset = build_momentum_dataset(
            bars_by_symbol,
            lookahead_bars=args.lookahead_bars,
        )
        if len(dataset.features) == 0:
            raise RuntimeError("No trainable samples were built from the selected history")

        trainer = TrainingPipeline(session)
        training_window = f"{start.date().isoformat()}_{end.date().isoformat()}"

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
                version=f"{args.version_prefix}-{training_window}",
                promote=True,
                notes=(
                    f"Momentum continuation training window={training_window} "
                    f"symbols={','.join(symbols)} lookahead_bars={args.lookahead_bars}"
                ),
            )

        await session.commit()

    await engine.dispose()


if __name__ == "__main__":
    setup_logging(log_level="INFO", json_output=False)
    asyncio.run(main())
