"""Generic async CRUD repository base class."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from trader.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Generic async CRUD repository."""

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, id_: int) -> ModelT | None:
        return await self.session.get(self.model, id_)

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[ModelT]:
        stmt = select(self.model).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, **kwargs: Any) -> ModelT:
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def update_by_id(self, id_: int, **kwargs: Any) -> ModelT | None:
        stmt = update(self.model).where(self.model.id == id_).values(**kwargs)  # type: ignore[attr-defined]
        await self.session.execute(stmt)
        await self.session.flush()
        return await self.get_by_id(id_)

    async def count(self) -> int:
        stmt = select(func.count()).select_from(self.model)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def delete_by_id(self, id_: int) -> bool:
        instance = await self.get_by_id(id_)
        if instance:
            await self.session.delete(instance)
            await self.session.flush()
            return True
        return False
