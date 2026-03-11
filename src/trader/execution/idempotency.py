"""Idempotency key management for order submission."""

from __future__ import annotations

import hashlib

import structlog

logger = structlog.get_logger(__name__)


def generate_idempotency_key(
    symbol: str,
    side: str,
    qty: int,
    strategy_tag: str,
    trade_intent_id: int | None = None,
) -> str:
    """Generate a deterministic idempotency key for a trade intent.

    The key ensures the same trade intent cannot be submitted twice.
    """
    components = [
        symbol,
        side,
        str(qty),
        strategy_tag,
        str(trade_intent_id or "none"),
    ]
    raw = "|".join(components)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]
