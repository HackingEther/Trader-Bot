"""Abstract model interfaces for the ensemble prediction pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RegimeClassifier(ABC):
    """Classifies current market regime."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Predict regime and confidence.

        Returns:
            Tuple of (regime_label, confidence).
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from artifact path."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Model version identifier."""


class DirectionClassifier(ABC):
    """Predicts trade direction (long/short/no_trade)."""

    @abstractmethod
    def predict(self, features: np.ndarray, regime: str) -> tuple[str, float]:
        """Predict direction and confidence.

        Returns:
            Tuple of (direction, confidence).
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from artifact path."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Model version identifier."""


class MoveMagnitudeRegressor(ABC):
    """Predicts expected price move magnitude."""

    @abstractmethod
    def predict(self, features: np.ndarray, direction: str) -> tuple[float, float]:
        """Predict expected move in bps and expected holding time in minutes.

        Returns:
            Tuple of (expected_move_bps, expected_holding_minutes).
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from artifact path."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Model version identifier."""


class TradeFilterModel(ABC):
    """Filters out low-quality trade signals."""

    @abstractmethod
    def predict(self, features: np.ndarray, direction: str, confidence: float) -> float:
        """Compute a no-trade score (0 = trade, 1 = definitely skip).

        Returns:
            no_trade_score between 0 and 1.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from artifact path."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Model version identifier."""
