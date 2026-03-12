"""Framework comparison reporting for legacy vs tradable, simple vs walk-forward."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ComparisonReport:
    """Single config run: training + backtest metrics."""

    config_name: str
    labels: str
    validation_mode: str
    training_metrics: dict = field(default_factory=dict)
    backtest_metrics: dict = field(default_factory=dict)
    fold_metrics: list[dict] | None = None
    aggregate: dict | None = None


def format_comparison_console(reports: list[ComparisonReport]) -> str:
    """Format side-by-side comparison for console output."""
    lines: list[str] = []
    lines.append("\n=== Framework Comparison ===")
    lines.append("")

    economic_keys = [
        "total_trades",
        "long_trade_count",
        "short_trade_count",
        "win_rate",
        "loss_rate",
        "expectancy",
        "average_net_pnl_bps",
        "median_net_pnl_bps",
        "profit_factor",
        "avg_hold_minutes",
        "total_pnl",
    ]

    for report in reports:
        lines.append(f"--- {report.config_name} ({report.labels} + {report.validation_mode}) ---")
        bt = report.backtest_metrics
        for k in economic_keys:
            if k in bt:
                v = bt[k]
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")
        if report.aggregate:
            lines.append("  [walk-forward aggregate]")
            for k in ("val_score_mean", "val_score_std", "accuracy_mean", "accuracy_std"):
                if k in report.aggregate:
                    lines.append(f"    {k}: {report.aggregate[k]:.4f}")
        lines.append("")

    if len(reports) >= 2:
        lines.append("--- Side-by-side (economic) ---")
        best_pnl = max((r.backtest_metrics.get("total_pnl", 0) for r in reports), default=0)
        best_exp = max((r.backtest_metrics.get("expectancy", 0) for r in reports), default=0)
        for r in reports:
            pnl = r.backtest_metrics.get("total_pnl", 0)
            exp = r.backtest_metrics.get("expectancy", 0)
            pnl_mark = " *" if pnl == best_pnl and best_pnl != 0 else ""
            exp_mark = " *" if exp == best_exp and best_exp != 0 else ""
            lines.append(f"  {r.config_name}: total_pnl={pnl:.2f}{pnl_mark} expectancy={exp:.2f}{exp_mark}")
        lines.append("  (* = best in column)")
        lines.append("")

    return "\n".join(lines)


def write_comparison_json(reports: list[ComparisonReport], path: str | Path) -> None:
    """Write comparison report to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "config_name": r.config_name,
            "labels": r.labels,
            "validation_mode": r.validation_mode,
            "training_metrics": r.training_metrics,
            "backtest_metrics": r.backtest_metrics,
            "fold_metrics": r.fold_metrics,
            "aggregate": r.aggregate,
        }
        for r in reports
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_comparison_csv(reports: list[ComparisonReport], path: str | Path) -> None:
    """Write comparison report to CSV (one row per config, flat metrics)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    all_keys: set[str] = set()
    for r in reports:
        all_keys.add("config_name")
        all_keys.add("labels")
        all_keys.add("validation_mode")
        for k in r.backtest_metrics:
            all_keys.add(f"backtest_{k}")
        for k in r.training_metrics:
            all_keys.add(f"training_{k}")
        if r.aggregate:
            for k in r.aggregate:
                all_keys.add(f"aggregate_{k}")

    base_cols = ["config_name", "labels", "validation_mode"]
    metric_cols = sorted(k for k in all_keys if k not in base_cols)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + metric_cols, extrasaction="ignore")
        writer.writeheader()
        for r in reports:
            row: dict = {
                "config_name": r.config_name,
                "labels": r.labels,
                "validation_mode": r.validation_mode,
            }
            for k, v in r.backtest_metrics.items():
                row[f"backtest_{k}"] = v
            for k, v in r.training_metrics.items():
                row[f"training_{k}"] = v
            if r.aggregate:
                for k, v in r.aggregate.items():
                    row[f"aggregate_{k}"] = v
            writer.writerow(row)


def report_filename(prefix: str = "comparison_report") -> str:
    """Generate timestamped report filename."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"
