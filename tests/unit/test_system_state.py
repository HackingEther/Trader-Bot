"""Unit tests for durable system state fallbacks."""

from __future__ import annotations

import pytest

from trader.services.system_state import SystemStateStore


@pytest.mark.asyncio
async def test_kill_switch_falls_back_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    async def raise_runtime_error(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("redis not initialized")

    async def raise_connect_error(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr("trader.services.system_state.get_cached", raise_runtime_error)
    monkeypatch.setattr("trader.services.system_state.set_cached", raise_runtime_error)
    monkeypatch.setattr("trader.services.system_state.init_redis", raise_connect_error)

    store = SystemStateStore(redis_url="redis://invalid")
    assert await store.is_kill_switch_active() is False
    await store.set_kill_switch(True)
    assert await store.is_kill_switch_active() is True
