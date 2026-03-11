"""Model registry for managing model versions in the database."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from trader.db.models.model_version import ModelVersion

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """Database-backed model version registry."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def register(
        self,
        model_type: str,
        version: str,
        artifact_path: str,
        algorithm: str = "lightgbm",
        hyperparameters: dict | None = None,
        metrics: dict | None = None,
        training_run_id: int | None = None,
        score: float | None = None,
    ) -> ModelVersion:
        mv = ModelVersion(
            model_type=model_type,
            version=version,
            artifact_path=artifact_path,
            algorithm=algorithm,
            hyperparameters=hyperparameters or {},
            metrics=metrics or {},
            training_run_id=training_run_id,
            score=score,
            trained_at=datetime.now(timezone.utc),
        )
        self._session.add(mv)
        await self._session.flush()
        logger.info("model_registered", model_type=model_type, version=version)
        return mv

    async def get_champion(self, model_type: str) -> ModelVersion | None:
        stmt = select(ModelVersion).where(
            ModelVersion.model_type == model_type,
            ModelVersion.is_champion == True,  # noqa: E712
            ModelVersion.is_active == True,  # noqa: E712
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active(self, model_type: str) -> list[ModelVersion]:
        stmt = select(ModelVersion).where(
            ModelVersion.model_type == model_type,
            ModelVersion.is_active == True,  # noqa: E712
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def promote_to_champion(self, model_id: int) -> None:
        mv = await self._session.get(ModelVersion, model_id)
        if not mv:
            raise ValueError(f"Model version {model_id} not found")
        old_champs = await self.get_active(mv.model_type)
        for old in old_champs:
            if old.is_champion:
                old.is_champion = False
        mv.is_champion = True
        mv.is_active = True
        await self._session.flush()
        logger.info("model_promoted", model_type=mv.model_type, version=mv.version)

    async def get_all_types(self) -> list[str]:
        from sqlalchemy import distinct
        stmt = select(distinct(ModelVersion.model_type))
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
