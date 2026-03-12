"""Unit tests for pipeline walk-forward validation."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import trader.db.models  # noqa: F401
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trader.db.base import Base
from trader.models.training.pipeline import TrainingPipeline
from trader.models.training.walkforward import WalkForwardSplitter


@pytest.fixture
async def pipeline_session(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'train.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_run_walkforward_produces_fold_metrics(pipeline_session: AsyncSession, tmp_path) -> None:
    from trader.config import get_settings
    from trader.providers.artifacts.local import LocalArtifactStore

    settings = get_settings()
    artifact_store = LocalArtifactStore(tmp_path / "artifacts")
    pipeline = TrainingPipeline(pipeline_session, artifact_store=artifact_store)

    np.random.seed(42)
    n = 300
    features = np.random.randn(n, 20).astype(np.float64)
    labels = np.random.choice(["long", "short", "no_trade"], size=n)
    timestamps = list(range(n))

    run = await pipeline.run_walkforward(
        model_type="direction",
        features=features,
        labels=labels,
        timestamps=timestamps,
        n_folds=3,
        purge_bars=10,
        embargo_bars=5,
        version="test-wf",
        promote=False,
    )

    assert run.status == "completed"
    assert run.metrics is not None
    assert "fold_metrics" in run.metrics
    assert "aggregate" in run.metrics
    assert len(run.metrics["fold_metrics"]) >= 1


@pytest.mark.asyncio
async def test_run_simple_backward_compat(pipeline_session: AsyncSession, tmp_path) -> None:
    from trader.providers.artifacts.local import LocalArtifactStore

    artifact_store = LocalArtifactStore(tmp_path / "artifacts")
    pipeline = TrainingPipeline(pipeline_session, artifact_store=artifact_store)

    np.random.seed(42)
    features = np.random.randn(100, 20).astype(np.float64)
    labels = np.random.choice(["long", "short", "no_trade"], size=100)

    run = await pipeline.run(
        model_type="direction",
        features=features,
        labels=labels,
        version="test-simple",
        promote=False,
    )

    assert run.status == "completed"
    assert run.metrics.get("val_score") is not None
