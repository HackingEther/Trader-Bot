"""Train and promote ensemble models from historical bars stored in the DB."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trader.config import get_settings
from trader.logging import setup_logging
from trader.models.training.dataset import (
    TrainingDataset,
    TradableDataset,
    build_momentum_dataset,
    build_tradable_dataset,
)
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
    parser.add_argument(
        "--labels",
        choices=["legacy", "tradable"],
        default="legacy",
        help="Label construction: legacy (momentum) or tradable (post-cost)",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["simple", "walkforward"],
        default="simple",
        help="Validation: simple 80/20 split or walk-forward with purge/embargo",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--purge-bars", type=int, default=75)
    parser.add_argument("--embargo-bars", type=int, default=5)
    parser.add_argument("--train-regime-legacy", action="store_true", help="Train regime model (legacy) when using tradable labels")
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

        if args.labels == "tradable":
            dataset: TrainingDataset | TradableDataset = build_tradable_dataset(bars_by_symbol)
        else:
            dataset = build_momentum_dataset(
                bars_by_symbol,
                lookahead_bars=args.lookahead_bars,
            )

        if len(dataset.features) == 0:
            raise RuntimeError("No trainable samples were built from the selected history")

        trainer = TrainingPipeline(session)
        training_window = f"{start.date().isoformat()}_{end.date().isoformat()}"
        version = f"{args.version_prefix}-{training_window}"
        notes = f"Training window={training_window} symbols={','.join(symbols)} labels={args.labels} val={args.validation_mode}"

        model_configs: list[tuple[str, object]] = []
        if args.labels == "legacy" or args.train_regime_legacy:
            model_configs.append(("regime", dataset.regime_labels))
        model_configs.extend([
            ("direction", dataset.direction_labels),
            ("magnitude", dataset.magnitude_labels),
            ("filter", dataset.filter_labels),
        ])

        for model_type, labels in model_configs:
            if args.validation_mode == "walkforward":
                await trainer.run_walkforward(
                    model_type=model_type,
                    features=dataset.features,
                    labels=labels,
                    timestamps=dataset.timestamps,
                    n_folds=args.n_folds,
                    purge_bars=args.purge_bars,
                    embargo_bars=args.embargo_bars,
                    version=version,
                    promote=True,
                    notes=notes,
                    magnitude_labels=dataset.magnitude_labels if model_type == "magnitude" else None,
                )
            else:
                await trainer.run(
                    model_type=model_type,
                    features=dataset.features,
                    labels=labels,
                    version=version,
                    promote=True,
                    notes=notes,
                )

        await session.commit()

    await engine.dispose()


if __name__ == "__main__":
    setup_logging(log_level="INFO", json_output=False)
    asyncio.run(main())
