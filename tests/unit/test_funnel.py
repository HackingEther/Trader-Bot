"""Unit tests for DecisionFunnelTracker."""

from __future__ import annotations

from trader.strategy.funnel import DecisionFunnelTracker


def test_funnel_distribution_summary() -> None:
    """get_distribution_summary returns percentiles for confidence, expected_move_bps, best_playbook_fit."""
    tracker = DecisionFunnelTracker(framework="test_fw")
    tracker.record("AAPL", "long", "low_confidence", confidence=0.45)
    tracker.record("AAPL", "long", "low_confidence", confidence=0.55)
    tracker.record("AAPL", "long", "low_confidence", confidence=0.65)
    tracker.record("MSFT", "short", "low_move", expected_move_bps=8.0)
    tracker.record("MSFT", "short", "low_move", expected_move_bps=12.0)
    tracker.record("MSFT", "short", "low_move", expected_move_bps=14.0)
    tracker.record("GOOGL", "long", "no_playbook", best_playbook_fit=0.25)
    tracker.record("GOOGL", "long", "no_playbook", best_playbook_fit=0.35)
    tracker.record("GOOGL", "long", "no_playbook", best_playbook_fit=0.45)

    dist = tracker.get_distribution_summary()
    assert "block_distributions" in dist
    by_key = dist["block_distributions"]

    assert "test_fw|AAPL|long" in by_key
    assert "low_confidence" in by_key["test_fw|AAPL|long"]
    assert "confidence" in by_key["test_fw|AAPL|long"]["low_confidence"]
    conf = by_key["test_fw|AAPL|long"]["low_confidence"]["confidence"]
    assert "p10" in conf
    assert "p50" in conf
    assert "p90" in conf

    assert "test_fw|MSFT|short" in by_key
    assert "low_move" in by_key["test_fw|MSFT|short"]
    assert "expected_move_bps" in by_key["test_fw|MSFT|short"]["low_move"]

    assert "test_fw|GOOGL|long" in by_key
    assert "no_playbook" in by_key["test_fw|GOOGL|long"]
    assert "best_playbook_fit" in by_key["test_fw|GOOGL|long"]["no_playbook"]


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
