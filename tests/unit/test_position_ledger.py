"""Unit tests for the local position ledger."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest
import trader.db.models  # noqa: F401
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trader.db.base import Base
from trader.db.models.order import Order
from trader.db.repositories.positions import PositionRepository
from trader.execution.position_ledger import PositionLedger
from trader.strategy.engine import TradeIntentParams


@pytest.fixture
async def session(tmp_path) -> AsyncSession:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'ledger.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as db_session:
        yield db_session

    await engine.dispose()


def _intent(side: str, qty: int, limit_price: Decimal | None = None) -> TradeIntentParams:
    return TradeIntentParams(
        symbol="AAPL",
        side=side,
        qty=qty,
        entry_order_type="limit" if limit_price is not None else "market",
        limit_price=limit_price,
        strategy_tag="test_playbook",
    )


async def _order(session: AsyncSession, *, side: str, qty: int, broker_order_id: str) -> Order:
    order = Order(
        idempotency_key=f"{broker_order_id}-key",
        broker_order_id=broker_order_id,
        symbol="AAPL",
        side=side,
        order_type="market",
        order_class="simple",
        time_in_force="day",
        qty=qty,
        filled_qty=0,
        status="submitted",
        strategy_tag="test_playbook",
        rationale={},
        broker_metadata={},
    )
    session.add(order)
    await session.flush()
    return order


@pytest.mark.asyncio
async def test_opens_long_position(session: AsyncSession) -> None:
    ledger = PositionLedger(session)
    order = await _order(session, side="buy", qty=10, broker_order_id="open-long")

    position = await ledger.apply_fill(
        order=order,
        intent=_intent("buy", 10, Decimal("100.50")),
        fill_price=Decimal("100"),
        fill_qty=10,
        commission=Decimal("1.25"),
        timestamp=datetime.now(timezone.utc),
    )

    assert position.side == "buy"
    assert position.qty == 10
    assert position.avg_entry_price == Decimal("100")
    assert position.realized_pnl == Decimal("-1.25")


@pytest.mark.asyncio
async def test_adds_to_existing_long_position(session: AsyncSession) -> None:
    ledger = PositionLedger(session)
    first_order = await _order(session, side="buy", qty=10, broker_order_id="long-1")
    second_order = await _order(session, side="buy", qty=5, broker_order_id="long-2")

    await ledger.apply_fill(
        order=first_order,
        intent=_intent("buy", 10),
        fill_price=Decimal("100"),
        fill_qty=10,
        timestamp=datetime.now(timezone.utc),
    )
    position = await ledger.apply_fill(
        order=second_order,
        intent=_intent("buy", 5),
        fill_price=Decimal("110"),
        fill_qty=5,
        timestamp=datetime.now(timezone.utc),
    )

    assert position.qty == 15
    assert position.avg_entry_price.quantize(Decimal("0.0001")) == Decimal("103.3333")


@pytest.mark.asyncio
async def test_reduces_existing_long_position(session: AsyncSession) -> None:
    ledger = PositionLedger(session)
    open_order = await _order(session, side="buy", qty=10, broker_order_id="reduce-open")
    reduce_order = await _order(session, side="sell", qty=4, broker_order_id="reduce-close")

    await ledger.apply_fill(
        order=open_order,
        intent=_intent("buy", 10),
        fill_price=Decimal("100"),
        fill_qty=10,
        timestamp=datetime.now(timezone.utc),
    )
    position = await ledger.apply_fill(
        order=reduce_order,
        intent=_intent("sell", 4),
        fill_price=Decimal("110"),
        fill_qty=4,
        timestamp=datetime.now(timezone.utc),
    )

    assert position.status == "open"
    assert position.qty == 6
    assert position.realized_pnl == Decimal("40")


@pytest.mark.asyncio
async def test_closes_existing_long_position(session: AsyncSession) -> None:
    ledger = PositionLedger(session)
    open_order = await _order(session, side="buy", qty=10, broker_order_id="close-open")
    close_order = await _order(session, side="sell", qty=10, broker_order_id="close-close")

    await ledger.apply_fill(
        order=open_order,
        intent=_intent("buy", 10),
        fill_price=Decimal("100"),
        fill_qty=10,
        timestamp=datetime.now(timezone.utc),
    )
    position = await ledger.apply_fill(
        order=close_order,
        intent=_intent("sell", 10),
        fill_price=Decimal("105"),
        fill_qty=10,
        timestamp=datetime.now(timezone.utc),
    )

    assert position.status == "closed"
    assert position.qty == 0
    assert position.realized_pnl == Decimal("50")
    assert position.closed_at is not None


@pytest.mark.asyncio
async def test_flips_long_to_short(session: AsyncSession) -> None:
    ledger = PositionLedger(session)
    open_order = await _order(session, side="buy", qty=10, broker_order_id="flip-open")
    flip_order = await _order(session, side="sell", qty=15, broker_order_id="flip-close")

    await ledger.apply_fill(
        order=open_order,
        intent=_intent("buy", 10),
        fill_price=Decimal("100"),
        fill_qty=10,
        timestamp=datetime.now(timezone.utc),
    )
    new_position = await ledger.apply_fill(
        order=flip_order,
        intent=_intent("sell", 15),
        fill_price=Decimal("90"),
        fill_qty=15,
        timestamp=datetime.now(timezone.utc),
    )

    positions = await PositionRepository(session).get_by_symbol("AAPL")
    closed_position = next(position for position in positions if position.status == "closed")

    assert closed_position.realized_pnl == Decimal("-100")
    assert new_position.status == "open"
    assert new_position.side == "sell"
    assert new_position.qty == 5
    assert new_position.avg_entry_price == Decimal("90")


@pytest.mark.asyncio
async def test_opens_and_reduces_short_position(session: AsyncSession) -> None:
    ledger = PositionLedger(session)
    open_order = await _order(session, side="sell", qty=8, broker_order_id="short-open")
    cover_order = await _order(session, side="buy", qty=3, broker_order_id="short-cover")

    await ledger.apply_fill(
        order=open_order,
        intent=_intent("sell", 8),
        fill_price=Decimal("120"),
        fill_qty=8,
        timestamp=datetime.now(timezone.utc),
    )
    position = await ledger.apply_fill(
        order=cover_order,
        intent=_intent("buy", 3),
        fill_price=Decimal("110"),
        fill_qty=3,
        timestamp=datetime.now(timezone.utc),
    )

    assert position.side == "sell"
    assert position.qty == 5
    assert position.realized_pnl == Decimal("30")


@pytest.mark.asyncio
async def test_position_ledger_zero_fill_qty(session: AsyncSession) -> None:
    """Zero fill_qty in close path returns existing without division-by-zero."""
    ledger = PositionLedger(session)
    open_order = await _order(session, side="buy", qty=10, broker_order_id="open")
    close_order = await _order(session, side="sell", qty=0, broker_order_id="close")

    await ledger.apply_fill(
        order=open_order,
        intent=_intent("buy", 10),
        fill_price=Decimal("100"),
        fill_qty=10,
        timestamp=datetime.now(timezone.utc),
    )
    position = await ledger.apply_fill(
        order=close_order,
        intent=_intent("sell", 0),
        fill_price=Decimal("105"),
        fill_qty=0,
        timestamp=datetime.now(timezone.utc),
    )

    assert position.qty == 10
    assert position.status == "open"
