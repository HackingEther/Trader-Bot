"""Offline training script for default models using historical feature data."""

from __future__ import annotations

import asyncio

import numpy as np
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trader.config import get_settings
from trader.models.training.pipeline import TrainingPipeline

logger = structlog.get_logger(__name__)


def generate_synthetic_training_data(
    n_samples: int = 5000, n_features: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic training data for demonstration."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features))
    direction_labels = rng.choice(["long", "short", "no_trade"], size=n_samples, p=[0.35, 0.35, 0.3])
    magnitude_labels = np.abs(rng.standard_normal(n_samples) * 20 + 15)
    return X, direction_labels, magnitude_labels


def train_and_save() -> None:
    """Train default models on synthetic data and register them as champions."""

    X, dir_labels, mag_labels = generate_synthetic_training_data()

    regime_labels = np.where(
        X[:, 3] > 0.5, "high_volatility",
        np.where(X[:, 3] < -0.5, "low_volatility",
        np.where(X[:, 2] > 0.3, "trending_up",
        np.where(X[:, 2] < -0.3, "trending_down", "mean_reverting")))
    )

    settings = get_settings()

    async def _train() -> None:
        engine = create_async_engine(settings.database_url)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            trainer = TrainingPipeline(session)
            synthetic_filter = (dir_labels == "no_trade").astype(int)
            for model_type, labels in (
                ("regime", regime_labels),
                ("direction", dir_labels),
                ("magnitude", mag_labels),
                ("filter", synthetic_filter),
            ):
                await trainer.run(
                    model_type=model_type,
                    features=X,
                    labels=labels,
                    version="default-v1",
                    promote=True,
                    notes="Synthetic fallback training run",
                )
            await session.commit()
        await engine.dispose()

    asyncio.run(_train())
    logger.info("training_complete", version="default-v1")


if __name__ == "__main__":
    from trader.logging import setup_logging
    setup_logging(log_level="INFO", json_output=False)
    train_and_save()
