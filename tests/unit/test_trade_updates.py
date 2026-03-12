"""Unit tests for trade update stream idempotency and bracket legs."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader.ingestion.trade_updates import (
    TRADE_UPDATES_HEALTH_KEY,
    _execution_key,
    _incremental_fill_price,
    _tracked_fill_notional,
    _tracked_fill_qty,
)


def test_execution_key_prefers_execution_id() -> None:
    assert _execution_key("ord-1", "exec-abc", 10, None, 10) == "exec-abc"


def test_execution_key_fallback_uses_cumulative_qty() -> None:
    assert _execution_key("ord-1", None, 10, None, 10) == "ord-1:10"


def test_execution_key_fallback_uses_timestamp_when_no_cumulative() -> None:
    ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    result = _execution_key("ord-1", None, 0, ts, 5)
    assert result == "ord-1:2024-01-15T10:00:00+00:00:5"


def test_tracked_fill_qty() -> None:
    fills = [
        SimpleNamespace(qty=5, price=Decimal("100")),
        SimpleNamespace(qty=3, price=Decimal("101")),
    ]
    assert _tracked_fill_qty(fills) == 8


def test_tracked_fill_notional() -> None:
    fills = [
        SimpleNamespace(qty=5, price=Decimal("100")),
        SimpleNamespace(qty=3, price=Decimal("101")),
    ]
    assert _tracked_fill_notional(fills) == Decimal("803")


def test_incremental_fill_price() -> None:
    price = _incremental_fill_price(
        cumulative_qty=15,
        cumulative_avg_price=100.5,
        recorded_qty=10,
        recorded_notional=Decimal("1000"),
    )
    assert float(price) == pytest.approx(102.0, rel=0.01)


@pytest.mark.asyncio
async def test_trade_updates_execution_key_idempotency(tmp_path) -> None:
    """Same execution_id twice should not create duplicate fills."""
    import trader.db.models  # noqa: F401
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from trader.db.base import Base
    from trader.db.models.fill import Fill
    from trader.db.models.order import Order
    from trader.db.repositories.fills import FillRepository
    from trader.db.repositories.orders import OrderRepository
    from trader.execution.lifecycle import OrderLifecycleTracker

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        orders = OrderRepository(session)
        fills = FillRepository(session)
        lifecycle = OrderLifecycleTracker(session)
        order, _ = await orders.create_idempotent(
            idempotency_key="test-idem",
            symbol="AAPL",
            side="buy",
            order_type="limit",
            order_class="simple",
            qty=10,
            status="pending",
            strategy_tag="test",
            rationale={},
        )
        await session.flush()

        _, is_new_1 = await lifecycle.record_fill(
            order_id=order.id,
            broker_order_id="broker-1",
            symbol="AAPL",
            side="buy",
            qty=5,
            price=100.0,
            execution_key="exec-123",
            timestamp=datetime.now(timezone.utc),
        )
        _, is_new_2 = await lifecycle.record_fill(
            order_id=order.id,
            broker_order_id="broker-1",
            symbol="AAPL",
            side="buy",
            qty=5,
            price=100.0,
            execution_key="exec-123",
            timestamp=datetime.now(timezone.utc),
        )
        await session.commit()

    assert is_new_1 is True
    assert is_new_2 is False

    async with factory() as session:
        fills_repo = FillRepository(session)
        all_fills = await fills_repo.get_by_order_id(order.id)
        assert len(all_fills) == 1

    await engine.dispose()


@pytest.mark.asyncio
async def test_trade_updates_bracket_leg_creates_child_and_applies_fill(tmp_path, monkeypatch) -> None:
    """Unknown order with parent_order_id: fetch from broker, create child, apply fill."""
    import trader.db.models  # noqa: F401
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from trader.db.base import Base
    from trader.db.repositories.fills import FillRepository
    from trader.db.repositories.orders import OrderRepository
    from trader.db.repositories.positions import PositionRepository
    from trader.ingestion.trade_updates import TradeUpdateStream
    from trader.providers.broker.base import BrokerOrder

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'bracket.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    parent_broker_id = "parent-123"
    child_broker_id = "child-tp-456"

    class _MockBroker:
        async def get_order(self, broker_order_id: str) -> BrokerOrder:
            if broker_order_id == child_broker_id:
                return BrokerOrder(
                    broker_order_id=child_broker_id,
                    symbol="AAPL",
                    side="sell",
                    order_type="market",
                    qty=10,
                    status="filled",
                    filled_qty=10,
                    filled_avg_price=Decimal("151.00"),
                    raw={"parent_order_id": parent_broker_id},
                )
            raise RuntimeError("unknown order")

    async with factory() as session:
        orders = OrderRepository(session)
        await orders.create_idempotent(
            idempotency_key="parent-key",
            broker_order_id=parent_broker_id,
            symbol="AAPL",
            side="buy",
            order_type="limit",
            order_class="simple",
            qty=10,
            status="filled",
            strategy_tag="test",
            rationale={},
        )
        await session.commit()

    stream = TradeUpdateStream(api_key="x", api_secret="y", broker=_MockBroker())

    order_obj = SimpleNamespace(
        id=child_broker_id,
        status="filled",
        filled_qty=10,
        filled_avg_price=151.0,
        model_dump=lambda: {"id": child_broker_id},
    )
    ev = SimpleNamespace(value="fill")
    update = SimpleNamespace(
        event=ev,
        order=order_obj,
        price=151.0,
        qty=10,
        execution_id="exec-tp-1",
        timestamp=datetime.now(timezone.utc),
    )

    monkeypatch.setattr(
        "trader.ingestion.trade_updates.get_session_factory",
        lambda: factory,
    )

    await stream._process_update(update)

    async with factory() as session:
        child = await OrderRepository(session).get_by_broker_order_id(child_broker_id)
        fills = await FillRepository(session).get_by_order_id(child.id) if child else []
        pos = await PositionRepository(session).get_open_by_symbol("AAPL")

    assert child is not None
    assert child.side == "sell"
    assert len(fills) == 1
    assert fills[0].qty == 10
    assert fills[0].price == Decimal("151")
    assert pos is not None
    assert pos.side == "sell"
    assert pos.qty == 10

    await engine.dispose()


@pytest.mark.asyncio
async def test_trade_updates_stream_processes_partial_fill_sequence(tmp_path, monkeypatch) -> None:
    """Partial fills 0->2->5->10 are processed as incremental deltas; position accumulates correctly."""
    import trader.db.models  # noqa: F401
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from trader.db.base import Base
    from trader.db.repositories.fills import FillRepository
    from trader.db.repositories.orders import OrderRepository
    from trader.db.repositories.positions import PositionRepository
    from trader.ingestion.trade_updates import TradeUpdateStream

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'partial.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    monkeypatch.setattr(
        "trader.ingestion.trade_updates.get_session_factory",
        lambda: factory,
    )

    async with factory() as session:
        orders = OrderRepository(session)
        order, _ = await orders.create_idempotent(
            idempotency_key="partial-key",
            broker_order_id="partial-order",
            symbol="MSFT",
            side="buy",
            order_type="limit",
            order_class="simple",
            qty=10,
            status="pending",
            strategy_tag="test",
            rationale={},
        )
        await session.commit()

    stream = TradeUpdateStream(api_key="x", api_secret="y", broker=None)

    def make_update(
        cumulative_qty: int,
        delta_qty: int,
        fill_price: float,
        event: str,
        exec_id: str,
    ) -> SimpleNamespace:
        order_obj = SimpleNamespace(
            id="partial-order",
            status="partially_filled" if cumulative_qty < 10 else "filled",
            filled_qty=cumulative_qty,
            filled_avg_price=fill_price,
            model_dump=lambda: {"id": "partial-order"},
        )
        return SimpleNamespace(
            event=SimpleNamespace(value=event),
            order=order_obj,
            price=fill_price,
            qty=delta_qty,
            execution_id=exec_id,
            timestamp=datetime.now(timezone.utc),
        )

    await stream._process_update(make_update(2, 2, 100.0, "partial_fill", "exec-1"))
    await stream._process_update(make_update(5, 3, 100.0, "partial_fill", "exec-2"))
    await stream._process_update(make_update(10, 5, 100.0, "fill", "exec-3"))

    async with factory() as session:
        o = await OrderRepository(session).get_by_broker_order_id("partial-order")
        fills = await FillRepository(session).get_by_order_id(o.id)
        pos = await PositionRepository(session).get_open_by_symbol("MSFT")

    assert o.status == "filled"
    assert len(fills) == 3
    assert sum(f.qty for f in fills) == 10
    assert pos is not None
    assert pos.qty == 10

    await engine.dispose()
