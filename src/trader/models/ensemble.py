"""Ensemble inference pipeline combining all model components."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import structlog

from trader.core.events import ModelPrediction
from trader.features.registry import get_feature_names
from trader.models.interfaces import (
    DirectionClassifier,
    MoveMagnitudeRegressor,
    RegimeClassifier,
    TradeFilterModel,
)

logger = structlog.get_logger(__name__)


class EnsemblePipeline:
    """Runs the full model ensemble to produce a structured prediction."""

    def __init__(
        self,
        regime: RegimeClassifier,
        direction: DirectionClassifier,
        magnitude: MoveMagnitudeRegressor,
        trade_filter: TradeFilterModel,
    ) -> None:
        self._regime = regime
        self._direction = direction
        self._magnitude = magnitude
        self._filter = trade_filter

    def predict(self, symbol: str, features: dict, timestamp: datetime | None = None) -> ModelPrediction:
        """Run full ensemble prediction pipeline.

        Args:
            symbol: Ticker symbol.
            features: Dict of feature_name -> float from FeatureEngine.
            timestamp: Prediction timestamp.

        Returns:
            Structured ModelPrediction.
        """
        ts = timestamp or datetime.now(timezone.utc)
        feature_names = get_feature_names()
        feature_array = np.array([features.get(name, 0.0) for name in feature_names], dtype=np.float64)

        for i, v in enumerate(feature_array):
            if np.isnan(v) or np.isinf(v):
                feature_array[i] = 0.0

        regime_label, regime_conf = self._regime.predict(feature_array)
        direction, dir_confidence = self._direction.predict(feature_array, regime_label)
        expected_move_bps, expected_hold = self._magnitude.predict(feature_array, direction)
        no_trade_score = self._filter.predict(feature_array, direction, dir_confidence)

        prediction = ModelPrediction(
            symbol=symbol,
            timestamp=ts,
            direction=direction,
            confidence=dir_confidence,
            expected_move_bps=expected_move_bps,
            expected_holding_minutes=expected_hold,
            no_trade_score=no_trade_score,
            regime=regime_label,
            model_versions={
                "regime": self._regime.version,
                "direction": self._direction.version,
                "magnitude": self._magnitude.version,
                "filter": self._filter.version,
            },
        )

        logger.info(
            "ensemble_prediction",
            symbol=symbol,
            direction=direction,
            confidence=round(dir_confidence, 3),
            expected_move_bps=round(expected_move_bps, 1),
            no_trade_score=round(no_trade_score, 3),
            regime=regime_label,
        )

        return prediction

    @classmethod
    def create_default(cls) -> EnsemblePipeline:
        """Factory method creating pipeline with default model implementations."""
        from trader.models.defaults.direction import DefaultDirectionClassifier
        from trader.models.defaults.filter import DefaultTradeFilter
        from trader.models.defaults.magnitude import DefaultMagnitudeRegressor
        from trader.models.defaults.regime import DefaultRegimeClassifier

        return cls(
            regime=DefaultRegimeClassifier(),
            direction=DefaultDirectionClassifier(),
            magnitude=DefaultMagnitudeRegressor(),
            trade_filter=DefaultTradeFilter(),
        )
