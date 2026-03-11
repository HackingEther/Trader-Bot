"""Unit tests for feature engine and indicators."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic_bars import generate_synthetic_bars
from trader.core.events import BarEvent
from trader.features.engine import FeatureEngine
from trader.features.indicators import (
    compute_returns,
    compute_vwap,
    distance_from_vwap,
    momentum,
    relative_volume,
    rolling_volatility,
    rolling_zscore,
)
from trader.features.registry import FEATURE_COLUMNS, get_feature_names


class TestIndicators:
    def test_compute_returns(self) -> None:
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        returns = compute_returns(close, 1)
        assert len(returns) == 5
        assert returns.iloc[0] == 0.0
        assert returns.iloc[1] > 0

    def test_rolling_volatility(self) -> None:
        returns = pd.Series([0.01, -0.005, 0.008, -0.003, 0.012, -0.007, 0.009, -0.002, 0.006, -0.004])
        vol = rolling_volatility(returns, window=5)
        assert len(vol) == 10
        assert vol.iloc[-1] > 0

    def test_compute_vwap(self) -> None:
        high = pd.Series([101.0, 102.0, 103.0])
        low = pd.Series([99.0, 100.0, 101.0])
        close = pd.Series([100.0, 101.0, 102.0])
        volume = pd.Series([1000, 2000, 1500])
        vwap = compute_vwap(high, low, close, volume)
        assert len(vwap) == 3
        assert vwap.iloc[-1] > 0

    def test_distance_from_vwap(self) -> None:
        close = pd.Series([100.0, 101.0, 99.0])
        vwap = pd.Series([100.0, 100.5, 100.0])
        dist = distance_from_vwap(close, vwap)
        assert dist.iloc[0] == 0.0
        assert dist.iloc[1] > 0
        assert dist.iloc[2] < 0

    def test_relative_volume(self) -> None:
        volume = pd.Series([1000, 2000, 1500, 3000, 1000])
        rel = relative_volume(volume, window=3)
        assert len(rel) == 5

    def test_momentum(self) -> None:
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        mom = momentum(close, 2)
        assert len(mom) == 5
        assert mom.iloc[-1] > 0

    def test_rolling_zscore(self) -> None:
        series = pd.Series([100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0])
        zs = rolling_zscore(series, 5)
        assert len(zs) == 7


class TestFeatureEngine:
    def test_compute_features_empty(self) -> None:
        engine = FeatureEngine()
        features = engine.compute_features("AAPL")
        assert isinstance(features, dict)
        assert "return_1m" in features

    def test_compute_features_with_bars(self) -> None:
        engine = FeatureEngine()
        bars = generate_synthetic_bars("AAPL", count=50, seed=42)
        for bar_data in bars:
            event = BarEvent(**bar_data)
            engine.add_bar(event)

        features = engine.compute_features("AAPL")
        assert len(features) == len(FEATURE_COLUMNS)
        for col in FEATURE_COLUMNS:
            assert col in features
            assert not np.isnan(features[col])
            assert not np.isinf(features[col])

    def test_bar_count(self) -> None:
        engine = FeatureEngine()
        assert engine.get_bar_count("AAPL") == 0
        bars = generate_synthetic_bars("AAPL", count=10, seed=42)
        for b in bars:
            engine.add_bar(BarEvent(**b))
        assert engine.get_bar_count("AAPL") == 10

    def test_clear(self) -> None:
        engine = FeatureEngine()
        bars = generate_synthetic_bars("AAPL", count=10, seed=42)
        for b in bars:
            engine.add_bar(BarEvent(**b))
        engine.clear("AAPL")
        assert engine.get_bar_count("AAPL") == 0

    def test_feature_names_registry(self) -> None:
        names = get_feature_names()
        assert len(names) == len(FEATURE_COLUMNS)
        assert "return_1m" in names
        assert "session_fraction" in names
