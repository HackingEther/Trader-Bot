"""Unit tests for environment-backed settings parsing."""

from __future__ import annotations

import pytest

from trader.config import Settings


def test_symbol_universe_accepts_comma_separated_env_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYMBOL_UNIVERSE", "AAPL, MSFT, SPY")
    settings = Settings()

    assert settings.symbol_universe == ["AAPL", "MSFT", "SPY"]
