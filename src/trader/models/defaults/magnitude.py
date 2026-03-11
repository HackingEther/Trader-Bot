"""Default move magnitude regressor."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import structlog

from trader.models.interfaces import MoveMagnitudeRegressor

logger = structlog.get_logger(__name__)


class DefaultMagnitudeRegressor(MoveMagnitudeRegressor):
    """Default magnitude regressor with heuristic fallback."""

    def __init__(self) -> None:
        self._model: object | None = None
        self._version = "default-v1"

    def predict(self, features: np.ndarray, direction: str) -> tuple[float, float]:
        if self._model is not None:
            try:
                pred = self._model.predict(features.reshape(1, -1))[0]  # type: ignore[union-attr]
                if isinstance(pred, (list, np.ndarray)) and len(pred) >= 2:
                    return float(pred[0]), float(pred[1])
                return float(pred), 30.0
            except Exception as e:
                logger.warning("magnitude_model_predict_error", error=str(e))

        return self._heuristic(features, direction)

    def _heuristic(self, features: np.ndarray, direction: str) -> tuple[float, float]:
        vol_20 = features[3] if len(features) > 3 else 0.01
        base_move = max(10.0, abs(vol_20) * 10000 * 2)
        hold_time = 30.0 if vol_20 > 0.01 else 45.0
        return min(base_move, 100.0), hold_time

    def load(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            with open(p, "rb") as f:
                self._model = pickle.load(f)
            logger.info("magnitude_model_loaded", path=path)
        else:
            logger.warning("magnitude_model_not_found", path=path)

    @property
    def version(self) -> str:
        return self._version
