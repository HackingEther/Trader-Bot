"""Run confidence-testing backtest: multiple windows, stability metrics, pass/fail checks."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trader.backtest.engine import BacktestEngine
from trader.config import get_settings
from trader.logging import setup_logging
from trader.models.training.confidence_report import (
    compute_stability_summary,
    format_confidence_console,
    overall_confidence,
    report_filename,
    run_confidence_checks,
    write_confidence_csv,
    write_confidence_json,
)
from trader.models.training.dataset import build_tradable_dataset
from trader.models.training.pipeline import TrainingPipeline
from trader.models.training.window_scheduler import generate_windows
from trader.services.historical_data import HistoricalDataService


HEARTBEAT_INTERVAL_SEC = 300


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


async def _run_window(
    session: AsyncSession,
    train_bars: dict,
    test_bars: dict,
    symbols: list[str],
    train_start: datetime,
    train_end: datetime,
    test_start: datetime,
    test_end: datetime,
    n_folds: int,
    purge_bars: int,
    embargo_bars: int,
    initial_capital: float,
) -> dict:
    """Train tradable+walkforward models and run backtest for one window."""
    dataset = build_tradable_dataset(train_bars)
    if len(dataset.features) == 0:
        raise RuntimeError(
            f"No training samples for window train={train_start.date()}..{train_end.date()}"
        )

    settings = get_settings()
    trainer = TrainingPipeline(session)
    version = f"confidence-tradable-wf-{train_start.date().isoformat()}-{train_end.date().isoformat()}"
    notes = f"Confidence test tradable walkforward"

    model_configs = [
        ("direction", dataset.direction_labels),
        ("magnitude", dataset.magnitude_labels),
        ("filter", dataset.filter_labels),
    ]

    for model_type, lbls in model_configs:
        await trainer.run_walkforward(
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

    strategy_config = {
        "min_confidence": settings.min_confidence,
        "min_expected_move_bps": settings.min_expected_move_bps,
        "track_block_reasons": True,
        "funnel_audit": True,
        "framework": "tradable_walkforward",
        "magnitude_is_net_edge": True,
        "max_position_value": settings.max_exposure_per_symbol_usd,
        "risk_per_trade_pct": 0.01,
    }

    bt = BacktestEngine(session=session)
    backtest_results = await bt.run(
        name=f"confidence-{train_start.date()}-{test_end.date()}",
        symbols=symbols,
        bars_by_symbol=normalized,
        start_date=test_start.date().isoformat(),
        end_date=test_end.date().isoformat(),
        strategy_config=strategy_config,
        risk_config={
            "max_daily_loss": settings.max_daily_loss_usd,
            "max_loss_per_trade": settings.max_loss_per_trade_usd,
            "max_positions": settings.max_concurrent_positions,
            "max_notional": settings.max_notional_exposure_usd,
            "max_per_symbol": settings.max_exposure_per_symbol_usd,
            "cooldown_losses": settings.cooldown_after_losses,
        },
        initial_capital=initial_capital,
        use_champion_models=True,
    )

    return {
        "window_id": 0,
        "train_start": train_start.date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "test_end": test_end.date().isoformat(),
        "backtest_metrics": backtest_results,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run confidence-testing backtest across multiple rolling windows"
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--train-weeks", type=int, default=8)
    parser.add_argument("--test-weeks", type=int, default=2)
    parser.add_argument("--step-weeks", type=int, default=2)
    parser.add_argument("--symbols", default="SPY,QQQ,AAPL,MSFT,NVDA")
    parser.add_argument("--output-dir", default="./reports")
    parser.add_argument("--output-format", choices=["json", "csv", "both"], default="both")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--purge-bars", type=int, default=75)
    parser.add_argument("--embargo-bars", type=int, default=5)
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=None,
        help="Backtest starting capital. Defaults to paper_initial_cash.",
    )
    args = parser.parse_args()

    settings = get_settings()
    initial_capital = args.initial_capital or settings.paper_initial_cash
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    start = _parse_dt(args.start + "T00:00:00")
    end = _parse_dt(args.end + "T23:59:59")

    windows = generate_windows(
        start=start,
        end=end,
        train_weeks=args.train_weeks,
        test_weeks=args.test_weeks,
        step_weeks=args.step_weeks,
    )

    if not windows:
        print("No windows generated. Check --start, --end, --train-weeks, --test-weeks.")
        return

    print(f"Running confidence backtest: {len(windows)} windows")
    print(f"  train_weeks={args.train_weeks} test_weeks={args.test_weeks} step_weeks={args.step_weeks}")
    print("")

    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    window_results: list[dict] = []

    async with factory() as session:
        history = HistoricalDataService(session)
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"  Window {i + 1}/{len(windows)}: train {train_start.date()}..{train_end.date()} "
                  f"test {test_start.date()}..{test_end.date()}")
            train_bars = await history.load_bars_by_symbol(
                symbols, start=train_start, end=train_end
            )
            test_bars = await history.load_bars_by_symbol(
                symbols, start=test_start, end=test_end
            )
            result = await _run_window(
                session=session,
                train_bars=train_bars,
                test_bars=test_bars,
                symbols=symbols,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_folds=args.n_folds,
                purge_bars=args.purge_bars,
                embargo_bars=args.embargo_bars,
                initial_capital=initial_capital,
            )
            result["window_id"] = i
            window_results.append(result)
            await session.commit()

    await engine.dispose()

    stability = compute_stability_summary(window_results)
    checks = run_confidence_checks(stability, window_results)
    overall = overall_confidence(checks)

    config = {
        "labels": "tradable",
        "validation_mode": "walkforward",
        "symbols": symbols,
        "train_weeks": args.train_weeks,
        "test_weeks": args.test_weeks,
        "step_weeks": args.step_weeks,
        "window_count": len(window_results),
    }

    report = {
        "config": config,
        "windows": window_results,
        "stability": stability,
        "confidence_checks": checks,
        "overall_confidence": overall,
    }

    print(format_confidence_console(window_results, stability, checks, overall, config))

    out_dir = args.output_dir
    base = report_filename("confidence_report")
    if args.output_format in ("json", "both"):
        write_confidence_json(report, f"{out_dir}/{base}.json")
        print(f"Wrote {out_dir}/{base}.json")
    if args.output_format in ("csv", "both"):
        write_confidence_csv(window_results, stability, checks, f"{out_dir}/{base}.csv")
        print(f"Wrote {out_dir}/{base}.csv")


if __name__ == "__main__":
    setup_logging(log_level="INFO", json_output=False)
    asyncio.run(main())
