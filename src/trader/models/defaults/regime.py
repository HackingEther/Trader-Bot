"""Default regime classifier using LightGBM/sklearn."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import structlog

from trader.models.interfaces import RegimeClassifier

logger = structlog.get_logger(__name__)

REGIMES = ["trending_up", "trending_down", "mean_reverting", "high_volatility", "low_volatility"]


class DefaultRegimeClassifier(RegimeClassifier):
    """Default regime classifier with heuristic fallback."""

    def __init__(self) -> None:
        self._model: object | None = None
        self._version = "default-v1"

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        if self._model is not None:
            try:
                proba = self._model.predict_proba(features.reshape(1, -1))[0]  # type: ignore[union-attr]
                idx = int(np.argmax(proba))
                return REGIMES[idx], float(proba[idx])
            except Exception as e:
                logger.warning("regime_model_predict_error", error=str(e))

        return self._heuristic(features)

    def _heuristic(self, features: np.ndarray) -> tuple[str, float]:
        """Simple heuristic based on volatility and momentum features."""
        if len(features) < 5:
            return "low_volatility", 0.5

        vol_20 = features[3] if len(features) > 3 else 0.0
        ret_15m = features[2] if len(features) > 2 else 0.0

        if vol_20 > 0.02:
            return "high_volatility", 0.6
        if ret_15m > 0.005:
            return "trending_up", 0.55
        if ret_15m < -0.005:
            return "trending_down", 0.55
        if vol_20 < 0.005:
            return "low_volatility", 0.6
        return "mean_reverting", 0.5

    def load(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            with open(p, "rb") as f:
                self._model = pickle.load(f)
            logger.info("regime_model_loaded", path=path)
        else:
            logger.warning("regime_model_not_found", path=path)

    @property
    def version(self) -> str:
        return self._version
