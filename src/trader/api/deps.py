"""FastAPI dependency injection."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from trader.db.session import get_session


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session for request scope."""
    async for session in get_session():
        yield session
