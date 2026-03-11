"""SQLAlchemy async engine, session factory, and dependency."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(database_url: str) -> AsyncEngine:
    """Create and cache the async engine."""
    global _engine, _session_factory
    _engine = create_async_engine(
        database_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return _engine


def get_engine() -> AsyncEngine:
    """Return the cached engine or raise."""
    if _engine is None:
        raise RuntimeError("Database engine not initialized. Call init_engine() first.")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the cached session factory or raise."""
    if _session_factory is None:
        raise RuntimeError("Session factory not initialized. Call init_engine() first.")
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async generator yielding a session; for use as FastAPI dependency."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_engine() -> None:
    """Dispose of the engine."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
    _engine = None
    _session_factory = None


def reset_engine_for_celery() -> None:
    """Clear cached engine so it can be recreated in the current event loop.

    Each Celery task uses asyncio.run() which creates a new event loop. The
    cached engine's connections are bound to the previous task's loop, causing
    "Future attached to a different loop" errors. Call this at the start of each
    task so a fresh engine is created for the current loop.
    """
    global _engine, _session_factory
    if _engine is None:
        return
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_engine.dispose())
    finally:
        loop.close()
    _engine = None
    _session_factory = None
