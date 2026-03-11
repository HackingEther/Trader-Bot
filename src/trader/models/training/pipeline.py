"""Training pipeline structure for model retraining jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import pickle

import numpy as np
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from trader.db.models.training_run import TrainingRun
from trader.config import get_settings
from trader.models.registry import ModelRegistry
from trader.providers.artifacts.local import LocalArtifactStore
from trader.services.model_loader import build_model_metadata

logger = structlog.get_logger(__name__)


class TrainingPipeline:
    """Orchestrates model training with logging and artifact management."""

    def __init__(
        self,
        session: AsyncSession,
        artifact_store: LocalArtifactStore | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._session = session
        self._artifact_store = artifact_store or LocalArtifactStore(get_settings().artifacts_dir)
        self._registry = registry or ModelRegistry(session)

    async def run(
        self,
        model_type: str,
        features: np.ndarray,
        labels: np.ndarray,
        hyperparameters: dict | None = None,
        version: str | None = None,
        promote: bool = False,
        notes: str = "",
    ) -> TrainingRun:
        """Execute a training run.

        Args:
            model_type: Type of model to train (regime, direction, magnitude, filter).
            features: Feature matrix (n_samples, n_features).
            labels: Label array (n_samples,).
            hyperparameters: Optional hyperparameter overrides.

        Returns:
            TrainingRun record.
        """
        params = hyperparameters or {}
        now = datetime.now(timezone.utc)
        resolved_version = version or f"{model_type}-{now.strftime('%Y%m%d%H%M%S')}"

        run = TrainingRun(
            model_type=model_type,
            status="running",
            started_at=now,
            hyperparameters=params,
            training_samples=len(features),
            notes=notes,
        )
        self._session.add(run)
        await self._session.flush()

        logger.info(
            "training_started",
            run_id=run.id,
            model_type=model_type,
            samples=len(features),
        )

        try:
            split_idx = int(len(features) * 0.8)
            x_train, x_val = features[:split_idx], features[split_idx:]
            y_train, y_val = labels[:split_idx], labels[split_idx:]

            run.validation_samples = len(x_val)

            if model_type in ("regime", "direction", "filter"):
                model, metrics, algorithm = self._train_classifier(x_train, y_train, x_val, y_val, params)
            else:
                model, metrics, algorithm = self._train_regressor(x_train, y_train, x_val, y_val, params)

            artifact_key = f"models/{resolved_version}/{model_type}.pkl"
            artifact_path = await self._artifact_store.save(artifact_key, pickle.dumps(model))

            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            run.metrics = metrics
            run.best_score = metrics.get("val_score", 0.0)
            run.artifact_path = artifact_path

            model_version = await self._registry.register(
                model_type=model_type,
                version=resolved_version,
                artifact_path=artifact_path,
                algorithm=algorithm,
                hyperparameters={
                    **params,
                    **build_model_metadata(
                        model_type=model_type,
                        version=resolved_version,
                        training_run_id=run.id,
                    ),
                },
                metrics=metrics,
                training_run_id=run.id,
                score=metrics.get("val_score", 0.0),
            )
            if promote:
                await self._registry.promote_to_champion(model_version.id)

            logger.info(
                "training_completed",
                run_id=run.id,
                model_type=model_type,
                version=resolved_version,
                artifact_path=artifact_path,
                metrics=metrics,
                promoted=promote,
            )

        except Exception as e:
            run.status = "failed"
            run.completed_at = datetime.now(timezone.utc)
            run.error_message = str(e)
            logger.error("training_failed", run_id=run.id, error=str(e))

        await self._session.flush()
        return run

    def _train_classifier(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        params: dict,
    ) -> tuple[object, dict, str]:
        try:
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.05),
                num_leaves=params.get("num_leaves", 31),
                verbose=-1,
            )
            model.fit(x_train, y_train)
            val_score = float(model.score(x_val, y_val))
            train_score = float(model.score(x_train, y_train))
            algorithm = "lightgbm_classifier"
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.1),
            )
            model.fit(x_train, y_train)
            val_score = float(model.score(x_val, y_val))
            train_score = float(model.score(x_train, y_train))
            algorithm = "gradient_boosting_classifier"

        return model, {"val_score": val_score, "train_score": train_score}, algorithm

    def _train_regressor(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        params: dict,
    ) -> tuple[object, dict, str]:
        try:
            import lightgbm as lgb

            model = lgb.LGBMRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.05),
                num_leaves=params.get("num_leaves", 31),
                verbose=-1,
            )
            model.fit(x_train, y_train)
            val_score = float(model.score(x_val, y_val))
            train_score = float(model.score(x_train, y_train))
            algorithm = "lightgbm_regressor"
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor

            model = GradientBoostingRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.1),
            )
            model.fit(x_train, y_train)
            val_score = float(model.score(x_val, y_val))
            train_score = float(model.score(x_train, y_train))
            algorithm = "gradient_boosting_regressor"

        return model, {"val_score": val_score, "train_score": train_score}, algorithm
