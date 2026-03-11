"""Unit tests for champion model loading fallbacks."""

from __future__ import annotations

import pytest

from trader.services.model_loader import ChampionModelLoader


@pytest.mark.asyncio
async def test_model_loader_defaults_without_session() -> None:
    loader = ChampionModelLoader(require_champions=False)
    pipeline = await loader.load_ensemble(session=None)
    prediction = pipeline.predict("AAPL", {"momentum_5m": 0.01}, None)

    assert prediction.symbol == "AAPL"
    assert prediction.direction in ("long", "short", "no_trade")
