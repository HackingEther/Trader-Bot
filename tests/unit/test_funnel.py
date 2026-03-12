"""Unit tests for DecisionFunnelTracker."""

from __future__ import annotations

from trader.strategy.funnel import DecisionFunnelTracker


def test_funnel_tracks_block_reasons() -> None:
    tracker = DecisionFunnelTracker(framework="test_fw")
    tracker.record("AAPL", "long", "low_confidence")
    tracker.record("AAPL", "long", "filter")
    tracker.record("MSFT", "short", "no_playbook")

    summary = tracker.get_summary()
    assert summary["event_count"] == 3
    assert summary["total_by_reason"]["low_confidence"] == 1
    assert summary["total_by_reason"]["filter"] == 1
    assert summary["total_by_reason"]["no_playbook"] == 1

    by_key = summary["by_framework_symbol_side"]
    assert "test_fw|AAPL|long" in by_key
    assert by_key["test_fw|AAPL|long"]["low_confidence"] == 1
    assert by_key["test_fw|AAPL|long"]["filter"] == 1
    assert "test_fw|MSFT|short" in by_key
    assert by_key["test_fw|MSFT|short"]["no_playbook"] == 1
