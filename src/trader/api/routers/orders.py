"""Order endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from trader.api.deps import get_db_session
from trader.api.schemas.common import OrderResponse
from trader.db.repositories.orders import OrderRepository

router = APIRouter(prefix="/api/orders", tags=["orders"])


@router.get("/", response_model=list[OrderResponse])
async def list_orders(
    limit: int = 50,
    session: AsyncSession = Depends(get_db_session),
) -> list:
    repo = OrderRepository(session)
    return await repo.get_recent(limit=limit)


@router.get("/open", response_model=list[OrderResponse])
async def list_open_orders(
    session: AsyncSession = Depends(get_db_session),
) -> list:
    repo = OrderRepository(session)
    return await repo.get_open_orders()


@router.get("/{order_id}", response_model=OrderResponse | None)
async def get_order(
    order_id: int,
    session: AsyncSession = Depends(get_db_session),
) -> object:
    repo = OrderRepository(session)
    return await repo.get_by_id(order_id)
