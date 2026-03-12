"""Unit tests for walk-forward validation splitter."""

from __future__ import annotations

import numpy as np

from trader.models.training.walkforward import WalkForwardMetrics, WalkForwardSplitter


class TestWalkForwardSplitter:
    def test_generates_folds(self) -> None:
        splitter = WalkForwardSplitter(n_folds=4, purge_bars=10, embargo_bars=5)
        folds = splitter.get_folds(n_samples=500)
        assert len(folds) >= 1
        for fold in folds:
            assert len(fold.train_indices) > 0
            assert len(fold.val_indices) > 0

    def test_no_train_val_overlap(self) -> None:
        splitter = WalkForwardSplitter(n_folds=3, purge_bars=20, embargo_bars=5)
        folds = splitter.get_folds(n_samples=400)
        for fold in folds:
            train_set = set(fold.train_indices.tolist())
            val_set = set(fold.val_indices.tolist())
            assert len(train_set & val_set) == 0

    def test_val_after_train_with_gap(self) -> None:
        splitter = WalkForwardSplitter(n_folds=2, purge_bars=10, embargo_bars=5)
        folds = splitter.get_folds(n_samples=300)
        gap = 15
        for fold in folds:
            max_train = int(np.max(fold.train_indices))
            min_val = int(np.min(fold.val_indices))
            assert min_val >= max_train + gap

    def test_expanding_train(self) -> None:
        splitter = WalkForwardSplitter(n_folds=3, purge_bars=5, embargo_bars=2)
        folds = splitter.get_folds(n_samples=250)
        if len(folds) >= 2:
            assert len(folds[1].train_indices) >= len(folds[0].train_indices)

    def test_small_sample_returns_empty(self) -> None:
        splitter = WalkForwardSplitter(n_folds=5, purge_bars=50, embargo_bars=10)
        folds = splitter.get_folds(n_samples=20)
        assert len(folds) == 0


class TestWalkForwardMetrics:
    def test_aggregate_computes_mean_std(self) -> None:
        wf = WalkForwardMetrics()
        wf.add_fold(0, {"val_score": 0.8, "accuracy": 0.75})
        wf.add_fold(1, {"val_score": 0.7, "accuracy": 0.65})
        wf.compute_aggregate()
        assert "val_score_mean" in wf.aggregate
        assert "val_score_std" in wf.aggregate
        assert wf.aggregate["val_score_mean"] == 0.75
        assert wf.aggregate["accuracy_mean"] == 0.7

    def test_empty_folds(self) -> None:
        wf = WalkForwardMetrics()
        wf.compute_aggregate()
        assert wf.aggregate == {}


def test_compute_trading_metrics_long_short_no_trade() -> None:
    from trader.models.training.walkforward import compute_trading_metrics

    y_true = np.array(["long", "short", "no_trade", "long", "no_trade"])
    y_pred = np.array(["long", "no_trade", "no_trade", "short", "long"])
    m = compute_trading_metrics(y_true, y_pred)
    assert m["long_trade_count"] == 2
    assert m["short_trade_count"] == 1
    assert m["no_trade_count"] == 2
    assert m["trade_count"] == 3
    assert "accuracy" in m
    assert "loss_rate" in m
