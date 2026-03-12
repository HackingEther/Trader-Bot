"""Default trade filter model."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import structlog

from trader.models.interfaces import TradeFilterModel

logger = structlog.get_logger(__name__)


class DefaultTradeFilter(TradeFilterModel):
    """Default trade filter with heuristic fallback."""

    def __init__(self) -> None:
        self._model: object | None = None
        self._version = "default-v1"

    def predict(self, features: np.ndarray, direction: str, confidence: float) -> float:
        if self._model is not None:
            try:
                pred = self._model.predict_proba(features.reshape(1, -1))[0]  # type: ignore[union-attr]
                return float(pred[1]) if len(pred) > 1 else float(pred[0])
            except Exception as e:
                logger.warning("filter_model_predict_error", error=str(e))

        return self._heuristic(features, direction, confidence)

    def _heuristic(self, features: np.ndarray, direction: str, confidence: float) -> float:
        if direction == "no_trade":
            return 0.9
        if confidence < 0.5:
            return 0.7
        vol_20 = features[3] if len(features) > 3 else 0.0
        rel_vol = features[7] if len(features) > 7 else 1.0
        no_trade = 0.3
        if vol_20 > 0.03:
            no_trade += 0.2
        if rel_vol < 0.3:
            no_trade += 0.2
        return min(1.0, max(0.0, no_trade))

    def load(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            with open(p, "rb") as f:
                self._model = pickle.load(f)
            logger.info("filter_model_loaded", path=path)
        else:
            logger.warning("filter_model_not_found", path=path)

    @property
    def version(self) -> str:
        return self._version
