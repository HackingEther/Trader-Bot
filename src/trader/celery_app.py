"""Celery application factory and configuration."""

from __future__ import annotations

from celery import Celery

from trader.config import get_settings
from trader.workers.scheduler import configure_beat


def create_celery_app() -> Celery:
    """Create and configure the Celery application."""
    settings = get_settings()

    app = Celery(
        "trader",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
    )

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="US/Eastern",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=1000,
        broker_connection_retry_on_startup=True,
        result_expires=3600,
        task_default_queue="default",
        task_routes={
            "trader.workers.tasks.ingest_*": {"queue": "ingestion"},
            "trader.workers.tasks.compute_*": {"queue": "compute"},
            "trader.workers.tasks.execute_*": {"queue": "execution"},
            "trader.workers.tasks.reconcile_*": {"queue": "reconciliation"},
            "trader.workers.tasks.heartbeat_*": {"queue": "heartbeat"},
        },
    )

    app.autodiscover_tasks(["trader.workers"])
    configure_beat(app)

    return app


celery = create_celery_app()
