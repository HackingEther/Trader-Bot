"""Unit tests for execution idempotency."""

from __future__ import annotations

from trader.execution.idempotency import generate_idempotency_key


class TestIdempotencyKey:
    def test_deterministic(self) -> None:
        key1 = generate_idempotency_key("AAPL", "buy", 10, "default", 1)
        key2 = generate_idempotency_key("AAPL", "buy", 10, "default", 1)
        assert key1 == key2

    def test_different_for_different_inputs(self) -> None:
        key1 = generate_idempotency_key("AAPL", "buy", 10, "default", 1)
        key2 = generate_idempotency_key("MSFT", "buy", 10, "default", 1)
        assert key1 != key2

    def test_different_side(self) -> None:
        key1 = generate_idempotency_key("AAPL", "buy", 10, "default", 1)
        key2 = generate_idempotency_key("AAPL", "sell", 10, "default", 1)
        assert key1 != key2

    def test_different_qty(self) -> None:
        key1 = generate_idempotency_key("AAPL", "buy", 10, "default", 1)
        key2 = generate_idempotency_key("AAPL", "buy", 20, "default", 1)
        assert key1 != key2

    def test_key_length(self) -> None:
        key = generate_idempotency_key("AAPL", "buy", 10, "default", 1)
        assert len(key) == 32

    def test_none_intent_id(self) -> None:
        key = generate_idempotency_key("AAPL", "buy", 10, "default", None)
        assert len(key) == 32
