"""Train on one historical window and backtest on the following window."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trader.backtest.engine import BacktestEngine
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
    parser = argparse.ArgumentParser(description="Run a walk-forward train/backtest cycle")
    parser.add_argument("--symbols", default="SPY,QQQ,AAPL,MSFT,NVDA")
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--lookahead-bars", type=int, default=15)
    parser.add_argument("--version-prefix", default="walkforward")
    parser.add_argument(
        "--labels",
        choices=["legacy", "tradable"],
        default="legacy",
        help="Label construction: legacy or tradable",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["simple", "walkforward"],
        default="walkforward",
        help="Validation: simple or walk-forward",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--purge-bars", type=int, default=75)
    parser.add_argument("--embargo-bars", type=int, default=5)
    parser.add_argument("--train-regime-legacy", action="store_true")
    parser.add_argument("--output-json", help="Write backtest + training metrics to JSON file")
    parser.add_argument("--output-csv", help="Write backtest + training metrics to CSV file")
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

        if args.labels == "tradable":
            dataset: TrainingDataset | TradableDataset = build_tradable_dataset(train_bars)
        else:
            dataset = build_momentum_dataset(train_bars, lookahead_bars=args.lookahead_bars)

        if len(dataset.features) == 0:
            raise RuntimeError("No training samples built from the selected training window")

        trainer = TrainingPipeline(session)
        notes = f"Walk-forward train_window={train_start.date().isoformat()}:{train_end.date().isoformat()} labels={args.labels} val={args.validation_mode}"

        model_configs: list[tuple[str, object]] = []
        if args.labels == "legacy" or args.train_regime_legacy:
            model_configs.append(("regime", dataset.regime_labels))
        model_configs.extend([
            ("direction", dataset.direction_labels),
            ("magnitude", dataset.magnitude_labels),
            ("filter", dataset.filter_labels),
        ])

        last_direction_metrics: dict = {}
        for model_type, labels in model_configs:
            if args.validation_mode == "walkforward":
                run = await trainer.run_walkforward(
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
                run = await trainer.run(
                    model_type=model_type,
                    features=dataset.features,
                    labels=labels,
                    version=version,
                    promote=True,
                    notes=notes,
                )
            if model_type == "direction":
                last_direction_metrics = run.metrics or {}

        test_bars = await history.load_bars_by_symbol(symbols, start=test_start, end=test_end)
        normalized = {
            sym: [
                {
                    k: str(v) if hasattr(v, "__abs__") and not isinstance(v, (int, float, str)) else v
                    for k, v in bar.items()
                }
                for bar in sym_bars
            ]
            for sym, sym_bars in test_bars.items()
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

    if args.output_json or args.output_csv:
        from pathlib import Path

        report = {
            "config": {
                "labels": args.labels,
                "validation_mode": args.validation_mode,
                "train_window": f"{train_start.date().isoformat()}-{train_end.date().isoformat()}",
                "test_window": f"{test_start.date().isoformat()}-{test_end.date().isoformat()}",
            },
            "training_metrics": last_direction_metrics,
            "backtest_metrics": results,
        }
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Wrote {args.output_json}")
        if args.output_csv:
            import csv

            Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
            flat = {"labels": args.labels, "validation_mode": args.validation_mode}
            for k, v in results.items():
                if k != "by_symbol" and not isinstance(v, (dict, list)):
                    flat[f"backtest_{k}"] = v
            with open(args.output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(flat.keys()), extrasaction="ignore")
                writer.writeheader()
                writer.writerow(flat)
            print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    setup_logging(log_level="INFO", json_output=False)
    asyncio.run(main())
