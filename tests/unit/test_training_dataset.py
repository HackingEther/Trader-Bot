"""Unit tests for historical training dataset generation."""

from __future__ import annotations

from tests.fixtures.synthetic_bars import generate_multi_symbol_bars
from trader.models.training.dataset import build_momentum_dataset


class TestMomentumDataset:
    def test_builds_samples_from_bars(self) -> None:
        bars = generate_multi_symbol_bars(symbols=["AAPL", "MSFT"], count=180, seed=42)
        dataset = build_momentum_dataset(bars, lookahead_bars=10, min_history=30)

        assert len(dataset.features) > 0
        assert len(dataset.features) == len(dataset.direction_labels)
        assert len(dataset.features) == len(dataset.magnitude_labels)
        assert len(dataset.features) == len(dataset.filter_labels)
        assert len(dataset.features) == len(dataset.regime_labels)
        assert set(dataset.direction_labels.tolist()).issubset({"long", "short", "no_trade"})

    def test_filter_label_matches_no_trade(self) -> None:
        bars = generate_multi_symbol_bars(symbols=["SPY"], count=180, seed=7)
        dataset = build_momentum_dataset(bars, lookahead_bars=5, min_history=20)

        for direction, label in zip(dataset.direction_labels.tolist(), dataset.filter_labels.tolist(), strict=False):
            assert label in (0, 1)
            if direction == "no_trade":
                assert label == 1
