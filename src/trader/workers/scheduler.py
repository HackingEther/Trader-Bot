"""Celery Beat schedule configuration."""

from __future__ import annotations

from celery.schedules import crontab

BEAT_SCHEDULE: dict = {
    "heartbeat-30s": {
        "task": "trader.workers.tasks.heartbeat_system",
        "schedule": 30.0,
    },
    "trading-cycle-1m": {
        "task": "trader.workers.tasks.execute_trading_cycle",
        "schedule": 60.0,
    },
    "reconcile-positions-5m": {
        "task": "trader.workers.tasks.reconcile_positions",
        "schedule": 300.0,
    },
    "pnl-snapshot-5m": {
        "task": "trader.workers.tasks.snapshot_pnl",
        "schedule": 300.0,
    },
    "stale-feed-check-30s": {
        "task": "trader.workers.tasks.check_stale_feeds",
        "schedule": 30.0,
    },
}


def configure_beat(app: object) -> None:
    """Apply beat schedule to Celery app."""
    app.conf.beat_schedule = BEAT_SCHEDULE  # type: ignore[union-attr]
