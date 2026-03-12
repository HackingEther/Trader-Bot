"""Compare legacy vs tradable and simple vs walk-forward on the same time windows."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trader.backtest.engine import BacktestEngine
from trader.config import get_settings
from trader.logging import setup_logging
from trader.models.training.comparison import (
    ComparisonReport,
    format_comparison_console,
    report_filename,
    write_comparison_csv,
    write_comparison_json,
)
from trader.models.training.dataset import (
    TrainingDataset,
    TradableDataset,
    build_momentum_dataset,
    build_tradable_dataset,
)
from trader.models.training.pipeline import TrainingPipeline
from trader.services.historical_data import HistoricalDataService


HEARTBEAT_INTERVAL_SEC = 300  # 5 minutes


async def _heartbeat() -> None:
    """Print a status line every 5 minutes to show the script is still running."""
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
        ts = datetime.now(UTC).strftime("%H:%M:%S UTC")
        print(f"  ✓ Still running... ({ts})", flush=True)


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


async def _run_config(
    session: AsyncSession,
    labels: str,
    validation_mode: str,
    train_bars: dict,
    test_bars: dict,
    symbols: list[str],
    train_start: datetime,
    train_end: datetime,
    test_start: datetime,
    test_end: datetime,
    lookahead_bars: int,
    n_folds: int,
    purge_bars: int,
    embargo_bars: int,
    train_regime_legacy: bool,
    diagnostic_mode: bool = False,
) -> tuple[dict, dict, list[dict] | None, dict | None]:
    """Train models and run backtest. Returns (training_metrics, backtest_metrics, fold_metrics, aggregate)."""
    if labels == "tradable":
        dataset: TrainingDataset | TradableDataset = build_tradable_dataset(train_bars)
    else:
        dataset = build_momentum_dataset(train_bars, lookahead_bars=lookahead_bars)

    if len(dataset.features) == 0:
        raise RuntimeError(f"No samples for {labels}+{validation_mode}")

    trainer = TrainingPipeline(session)
    version = f"compare-{labels}-{validation_mode}-{train_start.date().isoformat()}-{train_end.date().isoformat()}"
    notes = f"Comparison {labels} {validation_mode}"

    model_configs: list[tuple[str, object]] = []
    if labels == "legacy" or train_regime_legacy:
        model_configs.append(("regime", dataset.regime_labels))
    model_configs.extend([
        ("direction", dataset.direction_labels),
        ("magnitude", dataset.magnitude_labels),
        ("filter", dataset.filter_labels),
    ])

    last_direction_metrics: dict = {}
    fold_metrics: list[dict] | None = None
    aggregate: dict | None = None

    for model_type, lbls in model_configs:
        if validation_mode == "walkforward":
            run = await trainer.run_walkforward(
                model_type=model_type,
                features=dataset.features,
                labels=lbls,
                timestamps=dataset.timestamps,
                n_folds=n_folds,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
                version=version,
                promote=True,
                notes=notes,
                magnitude_labels=dataset.magnitude_labels if model_type == "magnitude" else None,
            )
        else:
            run = await trainer.run(
                model_type=model_type,
                features=dataset.features,
                labels=lbls,
                version=version,
                promote=True,
                notes=notes,
            )
        if model_type == "direction":
            last_direction_metrics = run.metrics or {}
            if isinstance(last_direction_metrics, dict):
                fold_metrics = last_direction_metrics.get("fold_metrics")
                aggregate = last_direction_metrics.get("aggregate")

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

    settings = get_settings()
    if diagnostic_mode:
        strategy_config = {
            "min_confidence": 0.5,
            "min_expected_move_bps": 5.0,
            "max_no_trade_score": 0.85,
            "track_block_reasons": True,
        }
    else:
        strategy_config = {
            "min_confidence": settings.min_confidence,
            "min_expected_move_bps": settings.min_expected_move_bps,
        }
    bt = BacktestEngine(session=session)
    backtest_results = await bt.run(
        name=f"compare-{labels}-{validation_mode}",
        symbols=symbols,
        bars_by_symbol=normalized,
        start_date=test_start.date().isoformat(),
        end_date=test_end.date().isoformat(),
        strategy_config=strategy_config,
        risk_config={
            "max_daily_loss": settings.max_daily_loss_usd,
            "max_positions": settings.max_concurrent_positions,
            "max_notional": settings.max_notional_exposure_usd,
        },
        use_champion_models=True,
    )

    return last_direction_metrics, backtest_results, fold_metrics, aggregate


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare legacy vs tradable and simple vs walk-forward on same windows"
    )
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--symbols", default="SPY,QQQ,AAPL,MSFT,NVDA")
    parser.add_argument("--output-dir", default="./reports")
    parser.add_argument("--output-format", choices=["json", "csv", "both"], default="both")
    parser.add_argument("--lookahead-bars", type=int, default=15)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--purge-bars", type=int, default=75)
    parser.add_argument("--embargo-bars", type=int, default=5)
    parser.add_argument("--train-regime-legacy", action="store_true")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Use relaxed thresholds and enable block-reason diagnostics for debugging zero-trade runs",
    )
    args = parser.parse_args()

    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    train_start = _parse_dt(args.train_start)
    train_end = _parse_dt(args.train_end)
    test_start = _parse_dt(args.test_start)
    test_end = _parse_dt(args.test_end)

    reports: list[ComparisonReport] = []

    async with factory() as session:
        history = HistoricalDataService(session)
        train_bars = await history.load_bars_by_symbol(symbols, start=train_start, end=train_end)
        test_bars = await history.load_bars_by_symbol(symbols, start=test_start, end=test_end)

        configs = [
            ("legacy_simple", "legacy", "simple"),
            ("legacy_walkforward", "legacy", "walkforward"),
            ("tradable_simple", "tradable", "simple"),
            ("tradable_walkforward", "tradable", "walkforward"),
        ]

        heartbeat_task = asyncio.create_task(_heartbeat())
        try:
            for config_name, labels, validation_mode in configs:
                train_m, bt_m, fold_m, agg = await _run_config(
                    session=session,
                    labels=labels,
                    validation_mode=validation_mode,
                    train_bars=train_bars,
                    test_bars=test_bars,
                    symbols=symbols,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    lookahead_bars=args.lookahead_bars,
                    n_folds=args.n_folds,
                    purge_bars=args.purge_bars,
                    embargo_bars=args.embargo_bars,
                    train_regime_legacy=args.train_regime_legacy,
                    diagnostic_mode=args.diagnostic,
                )
                await session.commit()

                reports.append(
                    ComparisonReport(
                        config_name=config_name,
                        labels=labels,
                        validation_mode=validation_mode,
                        training_metrics=train_m,
                        backtest_metrics=bt_m,
                        fold_metrics=fold_m,
                        aggregate=agg,
                    )
                )
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    await engine.dispose()

    print(format_comparison_console(reports))

    out_dir = args.output_dir
    base = report_filename()
    if args.output_format in ("json", "both"):
        write_comparison_json(reports, f"{out_dir}/{base}.json")
        print(f"\nWrote {out_dir}/{base}.json")
    if args.output_format in ("csv", "both"):
        write_comparison_csv(reports, f"{out_dir}/{base}.csv")
        print(f"Wrote {out_dir}/{base}.csv")


if __name__ == "__main__":
    setup_logging(log_level="INFO", json_output=False)
    asyncio.run(main())
