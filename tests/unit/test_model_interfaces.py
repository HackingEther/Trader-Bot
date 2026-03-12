"""Unit tests for model interfaces and ensemble."""

from __future__ import annotations

import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.synthetic_features import generate_synthetic_features
from trader.features.registry import get_feature_names
from trader.models.defaults.direction import DefaultDirectionClassifier
from trader.models.defaults.filter import DefaultTradeFilter
from trader.models.defaults.magnitude import DefaultMagnitudeRegressor
from trader.models.defaults.regime import DefaultRegimeClassifier
from trader.models.ensemble import EnsemblePipeline


class TestDefaultModels:
    def test_regime_classifier(self) -> None:
        model = DefaultRegimeClassifier()
        features = np.random.randn(20)
        regime, conf = model.predict(features)
        assert regime in ("trending_up", "trending_down", "mean_reverting", "high_volatility", "low_volatility")
        assert 0 <= conf <= 1

    def test_direction_classifier(self) -> None:
        model = DefaultDirectionClassifier()
        features = np.random.randn(20)
        direction, conf = model.predict(features, "trending_up")
        assert direction in ("long", "short", "no_trade")
        assert 0 <= conf <= 1

    def test_magnitude_regressor(self) -> None:
        model = DefaultMagnitudeRegressor()
        features = np.random.randn(20)
        move_bps, hold = model.predict(features, "long")
        assert move_bps > 0
        assert hold > 0

    def test_trade_filter(self) -> None:
        model = DefaultTradeFilter()
        features = np.random.randn(20)
        score = model.predict(features, "long", 0.7)
        assert 0 <= score <= 1

    def test_filter_feature_count_matches_training(self) -> None:
        """Filter must receive same feature count at inference as used during training."""
        n_features = len(get_feature_names())
        assert n_features == 20, "Registry defines 20 features; filter trained with this count"

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, n_features))
        y = (rng.random(100) > 0.5).astype(int)

        try:
            import lightgbm as lgb

            lgb_model = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
            lgb_model.fit(X, y)
        except ImportError:
            pytest.skip("lightgbm required for filter feature count test")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            Path(f.name).write_bytes(pickle.dumps(lgb_model))

        try:
            model = DefaultTradeFilter()
            model.load(f.name)
            features_20 = np.random.randn(n_features).astype(np.float64)
            score = model.predict(features_20, "long", 0.7)
            assert 0 <= score <= 1
        finally:
            Path(f.name).unlink(missing_ok=True)

    def test_model_versions(self) -> None:
        assert DefaultRegimeClassifier().version == "deterministic-v1"
        assert DefaultDirectionClassifier().version == "default-v1"
        assert DefaultMagnitudeRegressor().version == "default-v1"
        assert DefaultTradeFilter().version == "default-v1"


class TestEnsemblePipeline:
    def test_create_default(self) -> None:
        pipeline = EnsemblePipeline.create_default()
        assert pipeline is not None

    def test_predict(self) -> None:
        pipeline = EnsemblePipeline.create_default()
        features = generate_synthetic_features(seed=42)
        prediction = pipeline.predict("AAPL", features, datetime.now(timezone.utc))
        assert prediction.symbol == "AAPL"
        assert prediction.direction in ("long", "short", "no_trade")
        assert 0 <= prediction.confidence <= 1
        assert prediction.expected_move_bps > 0
        assert prediction.expected_holding_minutes > 0
        assert 0 <= prediction.no_trade_score <= 1
        assert prediction.regime in ("trending_up", "trending_down", "mean_reverting", "high_volatility", "low_volatility")
        assert "regime" in prediction.model_versions
