"""Default direction classifier using LightGBM/sklearn."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import structlog

from trader.models.interfaces import DirectionClassifier

logger = structlog.get_logger(__name__)


class DefaultDirectionClassifier(DirectionClassifier):
    """Default direction classifier with heuristic fallback."""

    def __init__(self) -> None:
        self._model: object | None = None
        self._version = "default-v1"

    def predict(self, features: np.ndarray, regime: str) -> tuple[str, float]:
        if self._model is not None:
            try:
                proba = self._model.predict_proba(features.reshape(1, -1))[0]  # type: ignore[union-attr]
                classes = ["long", "short", "no_trade"]
                idx = int(np.argmax(proba))
                return classes[idx], float(proba[idx])
            except Exception as e:
                logger.warning("direction_model_predict_error", error=str(e))

        return self._heuristic(features, regime)

    def _heuristic(self, features: np.ndarray, regime: str) -> tuple[str, float]:
        momentum_5m = features[12] if len(features) > 12 else 0.0
        momentum_15m = features[13] if len(features) > 13 else 0.0
        zscore = features[14] if len(features) > 14 else 0.0

        if regime in ("trending_up",) and momentum_5m > 0.001 and momentum_15m > 0:
            return "long", 0.55
        if regime in ("trending_down",) and momentum_5m < -0.001 and momentum_15m < 0:
            return "short", 0.55
        if regime == "mean_reverting":
            if zscore < -1.5:
                return "long", 0.52
            if zscore > 1.5:
                return "short", 0.52
        return "no_trade", 0.6

    def load(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            with open(p, "rb") as f:
                self._model = pickle.load(f)
            logger.info("direction_model_loaded", path=path)
        else:
            logger.warning("direction_model_not_found", path=path)

    @property
    def version(self) -> str:
        return self._version
