"""Confidence metrics and reporting for multi-window backtest robustness."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


METRIC_KEYS = [
    "total_trades",
    "long_trade_count",
    "short_trade_count",
    "win_rate",
    "loss_rate",
    "expectancy",
    "average_net_pnl_bps",
    "median_net_pnl_bps",
    "profit_factor",
    "max_drawdown",
    "avg_hold_minutes",
    "turnover",
]


def compute_symbol_concentration(by_symbol: dict[str, Any], total_trades: int) -> float:
    """Max fraction of trades from any single symbol. 0=diversified, 1=all one symbol."""
    if total_trades <= 0:
        return 0.0
    max_trades = max(
        (s.get("total_trades", 0) for s in by_symbol.values() if isinstance(s, dict)),
        default=0,
    )
    return float(max_trades / total_trades)


def compute_side_balance(long_count: int, short_count: int) -> float:
    """min(long_pct, short_pct). 0=all one side, 0.5=perfect balance."""
    total = long_count + short_count
    if total <= 0:
        return 0.0
    return float(min(long_count, short_count) / total)


def compute_blocker_dominance(block_reasons: dict[str, int] | None) -> float:
    """Max fraction of blocks from any single reason. 0=spread, 1=one blocker dominates."""
    if not block_reasons:
        return 0.0
    total = sum(block_reasons.values())
    if total <= 0:
        return 0.0
    return float(max(block_reasons.values()) / total)


def _extract_per_window_metrics(
    window_results: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """Extract numeric metrics and derived concentration metrics per window."""
    out: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    out["symbol_concentration"] = []
    out["side_balance"] = []
    out["blocker_dominance"] = []

    for w in window_results:
        bt = w.get("backtest_metrics", w)
        for k in METRIC_KEYS:
            v = bt.get(k)
            if isinstance(v, (int, float)):
                out[k].append(float(v))
            else:
                out[k].append(0.0 if "count" in k or "trades" in k else float("nan"))

        total = bt.get("total_trades", 0) or 0
        long_c = bt.get("long_trade_count", 0) or 0
        short_c = bt.get("short_trade_count", 0) or 0
        by_sym = bt.get("by_symbol", {})
        block_reasons = bt.get("strategy_block_reasons") or bt.get("decision_funnel", {}).get(
            "total_by_reason", {}
        )

        out["symbol_concentration"].append(compute_symbol_concentration(by_sym, total))
        out["side_balance"].append(compute_side_balance(long_c, short_c))
        out["blocker_dominance"].append(compute_blocker_dominance(block_reasons))

    return out


def compute_stability_summary(
    window_results: list[dict[str, Any]],
    metric_keys: list[str] | None = None,
) -> dict[str, float]:
    """Compute mean, std, min, max for each metric across windows."""
    keys = metric_keys or METRIC_KEYS
    extracted = _extract_per_window_metrics(window_results)
    all_keys = list(extracted.keys())
    stability: dict[str, float] = {}

    for k in all_keys:
        vals = [v for v in extracted[k] if not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        stability[f"{k}_mean"] = float(np.mean(arr))
        stability[f"{k}_std"] = float(np.std(arr)) if len(arr) > 1 else 0.0
        stability[f"{k}_min"] = float(np.min(arr))
        stability[f"{k}_max"] = float(np.max(arr))
        mean_v = np.mean(arr)
        if abs(mean_v) > 1e-9:
            stability[f"{k}_cv"] = float(np.std(arr) / abs(mean_v))
        else:
            stability[f"{k}_cv"] = float("inf") if np.std(arr) > 0 else 0.0

    return stability


def run_confidence_checks(
    stability: dict[str, float],
    window_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run pass/fail confidence checks. Returns list of {check, passed, value, threshold}."""
    checks: list[dict[str, Any]] = []

    mean_trades = stability.get("total_trades_mean", 0)
    checks.append({
        "check": "min_trades",
        "passed": mean_trades >= 5,
        "value": mean_trades,
        "threshold": 5,
    })

    cv_trades = stability.get("total_trades_cv", 0)
    checks.append({
        "check": "trade_count_stability",
        "passed": cv_trades < 2.0,
        "value": cv_trades,
        "threshold": 2.0,
    })

    exp_mean = stability.get("expectancy_mean", 0)
    exp_std = stability.get("expectancy_std", 0)
    exp_cv = exp_std / (abs(exp_mean) + 1e-6)
    checks.append({
        "check": "expectancy_stability",
        "passed": exp_cv < 3.0,
        "value": exp_cv,
        "threshold": 3.0,
    })

    pf_min = stability.get("profit_factor_min", 0)
    checks.append({
        "check": "profit_factor_floor",
        "passed": pf_min > 0.3,
        "value": pf_min,
        "threshold": 0.3,
    })

    sym_conc_mean = stability.get("symbol_concentration_mean", 0)
    checks.append({
        "check": "symbol_concentration",
        "passed": sym_conc_mean < 0.85,
        "value": sym_conc_mean,
        "threshold": 0.85,
    })

    side_bal_mean = stability.get("side_balance_mean", 0)
    checks.append({
        "check": "side_balance",
        "passed": side_bal_mean > 0.1,
        "value": side_bal_mean,
        "threshold": 0.1,
    })

    block_dom_mean = stability.get("blocker_dominance_mean", 0)
    checks.append({
        "check": "blocker_dominance",
        "passed": block_dom_mean < 0.95,
        "value": block_dom_mean,
        "threshold": 0.95,
    })

    checks.append({
        "check": "positive_expectancy",
        "passed": exp_mean > 0,
        "value": exp_mean,
        "threshold": 0,
    })

    return checks


def overall_confidence(checks: list[dict[str, Any]]) -> str:
    """STRONG if all pass, MODERATE if 1-2 fail, WEAK if 3+ fail."""
    failed = sum(1 for c in checks if not c.get("passed", True))
    if failed == 0:
        return "STRONG"
    if failed <= 2:
        return "MODERATE"
    return "WEAK"


def format_confidence_console(
    window_results: list[dict[str, Any]],
    stability: dict[str, float],
    checks: list[dict[str, Any]],
    overall: str,
    config: dict[str, Any],
) -> str:
    """Human-readable console summary."""
    lines: list[str] = []
    lines.append("\n=== Confidence-Testing Backtest ===")
    lines.append("")
    lines.append(f"Config: {config.get('labels', 'tradable')} + {config.get('validation_mode', 'walkforward')}")
    lines.append(f"Windows: {len(window_results)}")
    lines.append("")

    lines.append("--- Per-Window Summary ---")
    for i, w in enumerate(window_results):
        bt = w.get("backtest_metrics", w)
        train = w.get("train_start", ""), w.get("train_end", "")
        test = w.get("test_start", ""), w.get("test_end", "")
        lines.append(f"  Window {i}: train {train[0]}..{train[1]} test {test[0]}..{test[1]}")
        lines.append(f"    trades={bt.get('total_trades', 0)} long={bt.get('long_trade_count', 0)} short={bt.get('short_trade_count', 0)} "
                    f"expectancy={bt.get('expectancy', 0):.4f} pf={bt.get('profit_factor', 0):.4f}")
    lines.append("")

    lines.append("--- Stability (across windows) ---")
    for k in ["total_trades", "expectancy", "profit_factor", "symbol_concentration", "side_balance"]:
        mean_v = stability.get(f"{k}_mean")
        std_v = stability.get(f"{k}_std")
        if mean_v is not None:
            lines.append(f"  {k}: mean={mean_v:.4f} std={std_v:.4f}" if std_v is not None else f"  {k}: mean={mean_v:.4f}")
    lines.append("")

    lines.append("--- Confidence Checks ---")
    for c in checks:
        status = "PASS" if c.get("passed") else "FAIL"
        lines.append(f"  {c['check']}: {status} (value={c.get('value', 0):.4f} threshold={c.get('threshold', 0)})")
    lines.append("")

    lines.append(f"--- Overall Confidence: {overall} ---")
    failed = [c["check"] for c in checks if not c.get("passed")]
    if failed:
        lines.append("  Red flags: " + ", ".join(failed))
    lines.append("")

    return "\n".join(lines)


def write_confidence_json(report: dict[str, Any], path: str | Path) -> None:
    """Write full confidence report to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def write_confidence_csv(
    window_results: list[dict[str, Any]],
    stability: dict[str, float],
    checks: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Write CSV: one row per window, then stability row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    all_keys: set[str] = set()
    for w in window_results:
        bt = w.get("backtest_metrics", w)
        for k in bt:
            if k != "by_symbol" and k != "decision_funnel" and not isinstance(bt[k], (dict, list)):
                all_keys.add(k)
        for k in ["window_id", "train_start", "train_end", "test_start", "test_end"]:
            all_keys.add(k)
    for k in stability:
        all_keys.add(k)

    base_cols = ["window_id", "train_start", "train_end", "test_start", "test_end"]
    metric_cols = sorted(k for k in all_keys if k not in base_cols)
    fieldnames = base_cols + metric_cols

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, w in enumerate(window_results):
            row: dict[str, Any] = {
                "window_id": i,
                "train_start": w.get("train_start", ""),
                "train_end": w.get("train_end", ""),
                "test_start": w.get("test_start", ""),
                "test_end": w.get("test_end", ""),
            }
            bt = w.get("backtest_metrics", w)
            for k, v in bt.items():
                if k not in ("by_symbol", "decision_funnel") and not isinstance(v, (dict, list)):
                    row[k] = v
            writer.writerow(row)

        stability_row: dict[str, Any] = {
            "window_id": "stability",
            "train_start": "",
            "train_end": "",
            "test_start": "",
            "test_end": "",
        }
        for k, v in stability.items():
            stability_row[k] = v
        writer.writerow(stability_row)


def report_filename(prefix: str = "confidence_report") -> str:
    """Generate timestamped report filename."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"
