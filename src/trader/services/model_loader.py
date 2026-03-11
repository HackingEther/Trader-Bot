"""Registry-backed model loading for workers and backtests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from trader.config import get_settings
from trader.core.exceptions import ModelNotFoundError
from trader.db.models.model_version import ModelVersion
from trader.features.registry import FEATURE_VERSION
from trader.models.ensemble import EnsemblePipeline
from trader.models.interfaces import (
    DirectionClassifier,
    MoveMagnitudeRegressor,
    RegimeClassifier,
    TradeFilterModel,
)
from trader.models.registry import ModelRegistry

logger = structlog.get_logger(__name__)

MODEL_TYPES = ("regime", "direction", "magnitude", "filter")


class ChampionModelLoader:
    """Load champion models from the registry with cached fallbacks."""

    def __init__(self, require_champions: bool | None = None) -> None:
        settings = get_settings()
        self._require_champions = settings.is_live if require_champions is None else require_champions
        self._cached_versions: tuple[str, ...] | None = None
        self._cached_pipeline: EnsemblePipeline | None = None

    async def load_ensemble(self, session: AsyncSession | None = None, reload: bool = False) -> EnsemblePipeline:
        if session is None:
            logger.info("model_loader_default_fallback", reason="no_session")
            return EnsemblePipeline.create_default()

        registry = ModelRegistry(session)
        champions: dict[str, ModelVersion] = {}
        for model_type in MODEL_TYPES:
            champion = await registry.get_champion(model_type)
            if champion is None:
                if self._require_champions:
                    raise ModelNotFoundError(f"No champion model registered for {model_type}")
                logger.warning("model_loader_missing_champion", model_type=model_type)
                return EnsemblePipeline.create_default()
            champions[model_type] = champion

        cache_key = tuple(champions[m].version for m in MODEL_TYPES)
        if not reload and self._cached_pipeline is not None and self._cached_versions == cache_key:
            return self._cached_pipeline

        pipeline = EnsemblePipeline(
            regime=self._load_component(champions["regime"], "regime"),
            direction=self._load_component(champions["direction"], "direction"),
            magnitude=self._load_component(champions["magnitude"], "magnitude"),
            trade_filter=self._load_component(champions["filter"], "filter"),
        )
        self._cached_versions = cache_key
        self._cached_pipeline = pipeline
        logger.info(
            "champion_models_loaded",
            versions={name: champions[name].version for name in MODEL_TYPES},
        )
        return pipeline

    async def get_loaded_versions(self, session: AsyncSession | None = None) -> Mapping[str, str]:
        pipeline = await self.load_ensemble(session=session)
        return {
            "regime": pipeline._regime.version,  # type: ignore[attr-defined]
            "direction": pipeline._direction.version,  # type: ignore[attr-defined]
            "magnitude": pipeline._magnitude.version,  # type: ignore[attr-defined]
            "filter": pipeline._filter.version,  # type: ignore[attr-defined]
        }

    def _load_component(
        self,
        champion: ModelVersion,
        model_type: str,
    ) -> RegimeClassifier | DirectionClassifier | MoveMagnitudeRegressor | TradeFilterModel:
        if model_type == "regime":
            from trader.models.defaults.regime import DefaultRegimeClassifier

            component: RegimeClassifier = DefaultRegimeClassifier()
        elif model_type == "direction":
            from trader.models.defaults.direction import DefaultDirectionClassifier

            component = DefaultDirectionClassifier()
        elif model_type == "magnitude":
            from trader.models.defaults.magnitude import DefaultMagnitudeRegressor

            component = DefaultMagnitudeRegressor()
        else:
            from trader.models.defaults.filter import DefaultTradeFilter

            component = DefaultTradeFilter()

        artifact_path = Path(champion.artifact_path)
        if not artifact_path.is_absolute() and not artifact_path.exists():
            settings = get_settings()
            artifact_path = Path(settings.artifacts_dir) / artifact_path
        if not artifact_path.exists():
            raise ModelNotFoundError(
                f"Champion artifact for {model_type} not found at {artifact_path}"
            )

        component.load(str(artifact_path))
        if hasattr(component, "_version"):
            setattr(component, "_version", champion.version)
        return component


def build_model_metadata(*, model_type: str, version: str, training_run_id: int | None = None) -> dict:
    """Consistent metadata payload stored in the registry and logs."""
    return {
        "model_type": model_type,
        "version": version,
        "feature_version": FEATURE_VERSION,
        "training_run_id": training_run_id,
    }
