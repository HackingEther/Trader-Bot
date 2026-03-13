"""Unit tests for confidence-testing backtest components."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trader.models.training.confidence_report import (
    compute_blocker_dominance,
    compute_side_balance,
    compute_stability_summary,
    compute_symbol_concentration,
    overall_confidence,
    run_confidence_checks,
)
from trader.models.training.window_scheduler import generate_windows


def test_window_scheduler_generates_non_overlapping() -> None:
    """Windows have correct train/test bounds and test windows do not overlap."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 4, 15, tzinfo=timezone.utc)
    windows = generate_windows(
        start=start,
        end=end,
        train_weeks=8,
        test_weeks=2,
        step_weeks=2,
    )
    assert len(windows) >= 2
    for train_start, train_end, test_start, test_end in windows:
        assert train_start < train_end
        assert train_end < test_start
        assert test_start < test_end
        assert (test_start - train_end).days >= 1


def test_window_scheduler_respects_step() -> None:
    """Step weeks controls spacing between windows."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 5, 1, tzinfo=timezone.utc)
    windows_step1 = generate_windows(
        start=start, end=end, train_weeks=4, test_weeks=1, step_weeks=1
    )
    windows_step2 = generate_windows(
        start=start, end=end, train_weeks=4, test_weeks=1, step_weeks=2
    )
    assert len(windows_step1) > len(windows_step2)


def test_concentration_max_symbol_share() -> None:
    """Concentration = max symbol trades / total."""
    by_symbol = {
        "AAPL": {"total_trades": 5},
        "NVDA": {"total_trades": 3},
        "MSFT": {"total_trades": 2},
    }
    assert compute_symbol_concentration(by_symbol, 10) == 0.5
    by_symbol_single = {"AAPL": {"total_trades": 10}}
    assert compute_symbol_concentration(by_symbol_single, 10) == 1.0


def test_side_balance() -> None:
    """side_balance = min(long_pct, short_pct)."""
    assert compute_side_balance(5, 5) == 0.5
    assert compute_side_balance(8, 2) == 0.2
    assert compute_side_balance(10, 0) == 0.0


def test_blocker_dominance() -> None:
    """blocker_dominance = max reason count / total."""
    reasons = {"low_move": 80, "no_playbook": 15, "filter": 5}
    assert compute_blocker_dominance(reasons) == 0.8


def test_confidence_checks_pass_fail() -> None:
    """Checks return correct pass/fail given mock data."""
    window_results = [
        {
            "backtest_metrics": {
                "total_trades": 10,
                "long_trade_count": 5,
                "short_trade_count": 5,
                "expectancy": 0.2,
                "profit_factor": 1.5,
                "by_symbol": {
                    "AAPL": {"total_trades": 3},
                    "NVDA": {"total_trades": 4},
                    "MSFT": {"total_trades": 3},
                },
                "strategy_block_reasons": {"low_move": 50, "no_playbook": 50},
            }
        },
        {
            "backtest_metrics": {
                "total_trades": 12,
                "long_trade_count": 6,
                "short_trade_count": 6,
                "expectancy": 0.25,
                "profit_factor": 1.8,
                "by_symbol": {
                    "AAPL": {"total_trades": 4},
                    "NVDA": {"total_trades": 4},
                    "MSFT": {"total_trades": 4},
                },
                "strategy_block_reasons": {"low_move": 40, "no_playbook": 60},
            }
        },
    ]
    stability = compute_stability_summary(window_results)
    checks = run_confidence_checks(stability, window_results)
    min_trades = next(c for c in checks if c["check"] == "min_trades")
    assert min_trades["passed"] is True
    assert min_trades["value"] == 11.0


def test_overall_confidence_strong_when_all_pass() -> None:
    """Overall is STRONG when all checks pass."""
    checks = [{"check": "a", "passed": True}, {"check": "b", "passed": True}]
    assert overall_confidence(checks) == "STRONG"


def test_overall_confidence_moderate_when_1_2_fail() -> None:
    """Overall is MODERATE when 1-2 checks fail."""
    checks = [
        {"check": "a", "passed": True},
        {"check": "b", "passed": False},
        {"check": "c", "passed": True},
    ]
    assert overall_confidence(checks) == "MODERATE"


def test_overall_confidence_weak_when_3_plus_fail() -> None:
    """Overall is WEAK when 3+ checks fail."""
    checks = [
        {"check": "a", "passed": False},
        {"check": "b", "passed": False},
        {"check": "c", "passed": False},
    ]
    assert overall_confidence(checks) == "WEAK"


def test_confidence_backtest_script_imports() -> None:
    """Confidence backtest dependencies are importable."""
    from trader.models.training.confidence_report import (
        format_confidence_console,
        report_filename,
        write_confidence_csv,
        write_confidence_json,
    )
    from trader.models.training.window_scheduler import generate_windows

    assert callable(generate_windows)
    assert callable(format_confidence_console)
    assert callable(report_filename)
