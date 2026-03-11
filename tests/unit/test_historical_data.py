"""Unit tests for historical data service."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import trader.db.models  # noqa: F401
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trader.db.base import Base
from trader.db.models.market_bar import MarketBar
from trader.services.historical_data import HistoricalDataService


@pytest.fixture
async def session(tmp_path) -> AsyncSession:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'bars.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as db_session:
        yield db_session

    await engine.dispose()


@pytest.mark.asyncio
async def test_backfill_on_conflict_ignores_duplicates(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backfill with ON CONFLICT does not fail when duplicate bars exist (e.g. concurrent insert)."""
    mock_settings = MagicMock()
    mock_settings.alpaca_api_key = "test-key"
    mock_settings.alpaca_api_secret = "test-secret"
    monkeypatch.setattr(
        "trader.services.historical_data.get_settings",
        lambda: mock_settings,
    )

    payload = {
        "bars": {
            "AAPL": [
                {
                    "t": "2025-01-15T14:30:00Z",
                    "o": 100.0,
                    "h": 101.0,
                    "l": 99.0,
                    "c": 100.5,
                    "v": 10000,
                    "vw": 100.25,
                    "n": 100,
                }
            ]
        },
        "next_page_token": None,
    }

    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status = MagicMock()

    start = datetime(2025, 1, 15, 14, 0, tzinfo=timezone.utc)
    end = datetime(2025, 1, 15, 15, 0, tzinfo=timezone.utc)

    service = HistoricalDataService(session)

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        inserted = await service.backfill_from_alpaca(
            symbols=["AAPL"],
            start=start,
            end=end,
        )

    assert inserted == 1
    await session.commit()

    async def empty_existing(*args, **kwargs):
        return set()

    with patch.object(
        service._repo,
        "get_existing_keys",
        side_effect=empty_existing,
    ):
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            inserted2 = await service.backfill_from_alpaca(
                symbols=["AAPL"],
                start=start,
                end=end,
            )

    assert inserted2 == 1
    await session.commit()
