"""Unit tests for framework comparison reporting."""

from __future__ import annotations

import tempfile
from pathlib import Path

from trader.models.training.comparison import (
    ComparisonReport,
    format_comparison_console,
    report_filename,
    write_comparison_csv,
    write_comparison_json,
)


def test_comparison_report_structure() -> None:
    r = ComparisonReport(
        config_name="legacy_simple",
        labels="legacy",
        validation_mode="simple",
        training_metrics={"val_score": 0.8, "train_score": 0.9},
        backtest_metrics={
            "total_trades": 10,
            "win_rate": 0.6,
            "expectancy": 5.0,
            "total_pnl": 50.0,
        },
    )
    assert r.config_name == "legacy_simple"
    assert r.labels == "legacy"
    assert r.backtest_metrics["total_trades"] == 10


def test_format_comparison_console() -> None:
    reports = [
        ComparisonReport(
            config_name="legacy_simple",
            labels="legacy",
            validation_mode="simple",
            backtest_metrics={"total_pnl": 100.0, "expectancy": 10.0},
        ),
        ComparisonReport(
            config_name="tradable_wf",
            labels="tradable",
            validation_mode="walkforward",
            backtest_metrics={"total_pnl": 80.0, "expectancy": 8.0},
        ),
    ]
    out = format_comparison_console(reports)
    assert "legacy_simple" in out
    assert "tradable_wf" in out
    assert "100.00" in out
    assert "Side-by-side" in out


def test_write_comparison_json_roundtrip() -> None:
    reports = [
        ComparisonReport(
            config_name="test",
            labels="legacy",
            validation_mode="simple",
            backtest_metrics={"total_trades": 5},
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.json"
        write_comparison_json(reports, path)
        assert path.exists()
        data = path.read_text()
        assert "test" in data
        assert "total_trades" in data


def test_write_comparison_csv() -> None:
    reports = [
        ComparisonReport(
            config_name="test",
            labels="legacy",
            validation_mode="simple",
            backtest_metrics={"total_trades": 5, "win_rate": 0.6},
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.csv"
        write_comparison_csv(reports, path)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) >= 2
        assert "config_name" in lines[0]
        assert "backtest_total_trades" in lines[0]


def test_report_filename() -> None:
    name = report_filename()
    assert name.startswith("comparison_report_")
    assert len(name) > 20


def test_compare_frameworks_script_imports() -> None:
    """Verify compare_frameworks module components are importable."""
    from trader.models.training.comparison import (
        ComparisonReport,
        format_comparison_console,
        report_filename,
        write_comparison_csv,
        write_comparison_json,
    )

    assert ComparisonReport is not None
    assert callable(format_comparison_console)
    assert callable(report_filename)
