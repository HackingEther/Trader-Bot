"""Default regime classifier using deterministic logic."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import structlog

from trader.features.registry import get_feature_names
from trader.models.interfaces import RegimeClassifier
from trader.models.training.dataset import infer_regime

logger = structlog.get_logger(__name__)

REGIMES = ["trending_up", "trending_down", "mean_reverting", "high_volatility", "low_volatility"]


class DefaultRegimeClassifier(RegimeClassifier):
    """Deterministic regime classifier. Uses infer_regime(features) - no ML model.

    Regime is a rule over rolling_volatility_20 and momentum_15m. Keeps load()
    for backward compatibility with existing champion artifacts.
    """

    def __init__(self) -> None:
        self._model: object | None = None
        self._version = "deterministic-v1"

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        if self._model is not None:
            try:
                proba = self._model.predict_proba(features.reshape(1, -1))[0]  # type: ignore[union-attr]
                idx = int(np.argmax(proba))
                return REGIMES[idx], float(proba[idx])
            except Exception as e:
                logger.warning("regime_model_predict_error", error=str(e))

        features_dict = dict(zip(get_feature_names(), features.flatten(), strict=False))
        regime = infer_regime(features_dict)
        return regime, 0.9

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
