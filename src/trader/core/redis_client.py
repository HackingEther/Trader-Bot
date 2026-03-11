"""Redis connection and pub/sub helpers."""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger(__name__)

_pool: aioredis.ConnectionPool | None = None
_client: aioredis.Redis | None = None


async def init_redis(url: str) -> aioredis.Redis:
    """Initialize Redis connection pool."""
    global _pool, _client
    _pool = aioredis.ConnectionPool.from_url(
        url,
        decode_responses=True,
        max_connections=20,
        socket_connect_timeout=1,
        socket_timeout=1,
        retry_on_timeout=False,
    )
    _client = aioredis.Redis(connection_pool=_pool)
    await _client.ping()
    logger.info("redis_connected", url=url.split("@")[-1])
    return _client


async def get_redis() -> aioredis.Redis:
    """Get Redis client, raising if not initialized."""
    if _client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _client


async def close_redis() -> None:
    """Close Redis connection pool."""
    global _pool, _client
    if _client:
        await _client.aclose()
    if _pool:
        await _pool.aclose()
    _client = None
    _pool = None
    logger.info("redis_disconnected")


async def publish_event(channel: str, data: dict[str, Any]) -> None:
    """Publish a JSON event to a Redis channel."""
    client = await get_redis()
    payload = json.dumps(data, default=str)
    await client.publish(channel, payload)


async def xadd_event(stream: str, data: dict[str, Any], maxlen: int = 10000) -> str:
    """Add an event to a Redis stream with automatic trimming."""
    client = await get_redis()
    serialized = {k: json.dumps(v, default=str) if not isinstance(v, str) else v for k, v in data.items()}
    msg_id: str = await client.xadd(stream, serialized, maxlen=maxlen)
    return msg_id


async def get_cached(key: str) -> str | None:
    """Get a cached value."""
    client = await get_redis()
    return await client.get(key)


async def set_cached(key: str, value: str, ttl_seconds: int = 300) -> None:
    """Set a cached value with TTL."""
    client = await get_redis()
    await client.set(key, value, ex=ttl_seconds)
