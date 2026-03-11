"""Shared test fixtures and configuration."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal

import pytest

os.environ.setdefault("APP_ENV", "local")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///test.db")
os.environ.setdefault("DATABASE_SYNC_URL", "sqlite:///test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
os.environ.setdefault("LIVE_TRADING", "false")
os.environ.setdefault("ADMIN_API_TOKEN", "test-token")

from tests.fixtures.synthetic_bars import generate_synthetic_bars
from tests.fixtures.synthetic_features import generate_synthetic_features


@pytest.fixture
def synthetic_bars() -> list[dict]:
    return generate_synthetic_bars(symbol="AAPL", count=100, seed=42)


@pytest.fixture
def synthetic_features() -> dict[str, float]:
    return generate_synthetic_features(seed=42)


@pytest.fixture
def sample_timestamp() -> datetime:
    return datetime(2025, 1, 15, 15, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_price() -> Decimal:
    return Decimal("150.00")
