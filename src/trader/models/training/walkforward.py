"""Walk-forward validation with purge and embargo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass
class WalkForwardFold:
    """Single fold with train and validation indices."""

    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray


@dataclass
class WalkForwardMetrics:
    """Per-fold and aggregate metrics for walk-forward validation."""

    fold_metrics: list[dict] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)

    def add_fold(self, fold_id: int, metrics: dict) -> None:
        """Record metrics for a fold."""
        self.fold_metrics.append({"fold_id": fold_id, **metrics})

    def compute_aggregate(self) -> None:
        """Compute mean, std, min across folds for numeric metrics."""
        if not self.fold_metrics:
            return
        numeric_keys = set()
        for m in self.fold_metrics:
            for k, v in m.items():
                if k != "fold_id" and isinstance(v, (int, float)):
                    numeric_keys.add(k)
        agg: dict = {}
        for k in numeric_keys:
            vals = [m[k] for m in self.fold_metrics if k in m and isinstance(m[k], (int, float))]
            if vals:
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
                agg[f"{k}_min"] = float(np.min(vals))
                agg[f"{k}_max"] = float(np.max(vals))
        self.aggregate = agg


class WalkForwardSplitter:
    """Generates walk-forward folds with purge and embargo to prevent leakage."""

    def __init__(
        self,
        n_folds: int = 5,
        min_train_pct: float = 0.5,
        purge_bars: int = 75,
        embargo_bars: int = 5,
    ) -> None:
        """Initialize splitter.

        Args:
            n_folds: Number of validation folds.
            min_train_pct: Minimum fraction of data used for training (expanding window).
            purge_bars: Gap between train end and val start (e.g. lookahead + max_hold).
            embargo_bars: Extra bars after purge before val start.
        """
        self.n_folds = n_folds
        self.min_train_pct = min_train_pct
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars

    def split(
        self,
        n_samples: int,
        timestamps: list | np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, val_indices) for each fold.

        Uses expanding train window. Val windows are non-overlapping.
        Purge + embargo ensure no temporal leakage between train and val.

        Args:
            n_samples: Total number of samples (temporal order assumed).
            timestamps: Optional; if provided, used to compute gaps (not yet used).
        """
        if n_samples < 10:
            return
        gap = self.purge_bars + self.embargo_bars
        min_train_size = max(10, int(n_samples * self.min_train_pct))
        val_start = min_train_size + gap
        val_size = max(1, (n_samples - val_start - gap) // self.n_folds)
        for fold in range(self.n_folds):
            train_end = val_start + fold * val_size
            val_start_idx = train_end + gap
            val_end_idx = min(val_start_idx + val_size, n_samples)
            if train_end >= n_samples or val_start_idx >= n_samples or val_end_idx <= val_start_idx:
                continue
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start_idx, val_end_idx)
            if len(val_idx) < 1:
                continue
            yield train_idx, val_idx

    def get_folds(
        self,
        n_samples: int,
        timestamps: list | np.ndarray | None = None,
    ) -> list[WalkForwardFold]:
        """Return list of WalkForwardFold with train/val indices."""
        folds: list[WalkForwardFold] = []
        for fold_id, (train_idx, val_idx) in enumerate(
            self.split(n_samples, timestamps)
        ):
            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_indices=train_idx,
                    val_indices=val_idx,
                )
            )
        return folds


def compute_trading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    magnitude_true: np.ndarray | None = None,
    magnitude_pred: np.ndarray | None = None,
) -> dict:
    """Compute trading-relevant metrics from predictions.

    Args:
        y_true: True direction labels (long/short/no_trade).
        y_pred: Predicted direction labels.
        magnitude_true: Optional true magnitude in bps.
        magnitude_pred: Optional predicted magnitude in bps.
    """
    metrics: dict = {}
    pred_list = y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)
    true_list = y_true.tolist() if hasattr(y_true, "tolist") else list(y_true)

    metrics["long_trade_count"] = sum(1 for p in pred_list if p == "long")
    metrics["short_trade_count"] = sum(1 for p in pred_list if p == "short")
    metrics["no_trade_count"] = sum(1 for p in pred_list if p == "no_trade")
    metrics["trade_count"] = metrics["long_trade_count"] + metrics["short_trade_count"]

    if len(true_list) > 0:
        match = sum(1 for t, p in zip(true_list, pred_list, strict=False) if t == p)
        metrics["accuracy"] = float(match / len(true_list))
        metrics["loss_rate"] = 1.0 - metrics["accuracy"]
    else:
        metrics["accuracy"] = 0.0
        metrics["loss_rate"] = 0.0

    if magnitude_true is not None and magnitude_pred is not None and len(magnitude_true) > 0:
        from sklearn.metrics import r2_score
        try:
            metrics["magnitude_r2"] = float(r2_score(magnitude_true, magnitude_pred))
        except Exception:
            metrics["magnitude_r2"] = 0.0

    return metrics
