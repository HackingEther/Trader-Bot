"""Unit tests for open-order synchronization task."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
import trader.db.models  # noqa: F401
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trader.db.base import Base
from trader.db.models.fill import Fill
from trader.db.models.order import Order
from trader.db.models.position import Position
from trader.db.repositories.fills import FillRepository
from trader.db.repositories.orders import OrderRepository
from trader.db.repositories.positions import PositionRepository
from trader.workers import tasks


@pytest.fixture
async def db_factory(tmp_path) -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'orders.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield factory
    await engine.dispose()


@dataclass
class _BrokerOrder:
    broker_order_id: str
    status: str
    filled_qty: int = 0
    filled_avg_price: Decimal | None = None
    raw: dict | None = None


class _FakeBroker:
    def __init__(self, orders: dict[str, _BrokerOrder], failing_ids: set[str] | None = None) -> None:
        self._orders = orders
        self._failing_ids = failing_ids or set()
        self.cancelled_ids: list[str] = []

    async def get_order(self, broker_order_id: str) -> _BrokerOrder:
        if broker_order_id in self._failing_ids:
            raise RuntimeError("broker lookup failed")
        return self._orders[broker_order_id]

    async def cancel_order(self, broker_order_id: str) -> _BrokerOrder:
        self.cancelled_ids.append(broker_order_id)
        return _BrokerOrder(broker_order_id=broker_order_id, status="cancelled")


async def _seed_orders(factory: async_sessionmaker[AsyncSession]) -> None:
    now = datetime.now(timezone.utc)
    stale_time = now - timedelta(minutes=10)
    async with factory() as session:
        session.add_all(
            [
                Order(
                    idempotency_key="stale-key",
                    broker_order_id="stale-order",
                    symbol="AAPL",
                    side="buy",
                    order_type="limit",
                    order_class="simple",
                    time_in_force="day",
                    qty=10,
                    status="accepted",
                    strategy_tag="orb_continuation",
                    rationale={},
                    broker_metadata={},
                    created_at=stale_time,
                    updated_at=stale_time,
                    submitted_at=stale_time,
                ),
                Order(
                    idempotency_key="partial-key",
                    broker_order_id="partial-order",
                    symbol="MSFT",
                    side="buy",
                    order_type="limit",
                    order_class="simple",
                    time_in_force="day",
                    qty=5,
                    filled_qty=2,
                    status="partially_filled",
                    strategy_tag="vwap_continuation",
                    rationale={},
                    broker_metadata={},
                    created_at=now,
                    updated_at=now,
                    submitted_at=now,
                ),
                Order(
                    idempotency_key="error-key",
                    broker_order_id="error-order",
                    symbol="NVDA",
                    side="buy",
                    order_type="limit",
                    order_class="simple",
                    time_in_force="day",
                    qty=5,
                    status="accepted",
                    strategy_tag="vwap_continuation",
                    rationale={},
                    broker_metadata={},
                    created_at=now,
                    updated_at=now,
                    submitted_at=now,
                ),
                Order(
                    idempotency_key="bracket-parent-key",
                    broker_order_id="bracket-parent",
                    symbol="TSLA",
                    side="buy",
                    order_type="market",
                    order_class="bracket",
                    time_in_force="day",
                    qty=3,
                    status="accepted",
                    strategy_tag="breakout",
                    rationale={"reference_price": "200"},
                    broker_metadata={},
                    created_at=now,
                    updated_at=now,
                    submitted_at=now,
                ),
            ]
        )
        await session.flush()

        partial_order = await OrderRepository(session).get_by_broker_order_id("partial-order")
        assert partial_order is not None
        session.add(
            Fill(
                order_id=partial_order.id,
                broker_order_id="partial-order",
                symbol="MSFT",
                side="buy",
                qty=2,
                price=Decimal("100"),
                commission=Decimal("0"),
                timestamp=now,
                raw={},
            )
        )
        session.add(
            Position(
                symbol="MSFT",
                side="buy",
                qty=2,
                avg_entry_price=Decimal("100"),
                current_price=Decimal("100"),
                market_value=Decimal("200"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                status="open",
                strategy_tag="vwap_continuation",
                trade_intent_id=None,
                opened_at=now,
                metadata_={},
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()


def test_manage_open_orders_cancels_stale_records_fill_delta_and_isolates_errors(
    monkeypatch: pytest.MonkeyPatch,
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    asyncio.run(_seed_orders(db_factory))
    broker = _FakeBroker(
        orders={
            "stale-order": _BrokerOrder(broker_order_id="stale-order", status="accepted"),
            "partial-order": _BrokerOrder(
                broker_order_id="partial-order",
                status="partially_filled",
                filled_qty=5,
                filled_avg_price=Decimal("101"),
            ),
            "bracket-parent": _BrokerOrder(
                broker_order_id="bracket-parent",
                status="filled",
                filled_qty=3,
                filled_avg_price=Decimal("200"),
                raw={
                    "filled_at": datetime.now(timezone.utc).isoformat(),
                    "legs": [
                        {
                            "alpaca_id": "bracket-tp",
                            "symbol": "TSLA",
                            "side": "sell",
                            "order_type": "limit",
                            "qty": 3,
                            "limit_price": "210",
                            "filled_qty": 3,
                            "filled_avg_price": "210",
                            "status": "filled",
                            "filled_at": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                },
            ),
        },
        failing_ids={"error-order"},
    )

    async def _noop() -> None:
        return None

    monkeypatch.setattr(
        tasks,
        "_bootstrap",
        lambda: (SimpleNamespace(open_order_stale_seconds=90), db_factory, broker),
    )
    monkeypatch.setattr(tasks, "_ensure_redis", _noop)

    result = tasks.manage_open_orders()

    assert result == {
        "open_orders_seen": 4,
        "synced": 3,
        "filled": 2,
        "fills_recorded": 3,
        "cancelled": 1,
        "errors": 1,
    }
    assert broker.cancelled_ids == ["stale-order"]

    async def _assert_db_state() -> None:
        async with db_factory() as session:
            stale_order = await OrderRepository(session).get_by_broker_order_id("stale-order")
            partial_order = await OrderRepository(session).get_by_broker_order_id("partial-order")
            error_order = await OrderRepository(session).get_by_broker_order_id("error-order")
            parent_order = await OrderRepository(session).get_by_broker_order_id("bracket-parent")
            child_order = await OrderRepository(session).get_by_broker_order_id("bracket-tp")
            assert partial_order is not None
            assert parent_order is not None
            assert child_order is not None
            fills = await FillRepository(session).get_by_order_id(partial_order.id)
            parent_fills = await FillRepository(session).get_by_order_id(parent_order.id)
            child_fills = await FillRepository(session).get_by_order_id(child_order.id)
            position = await PositionRepository(session).get_open_by_symbol("MSFT")
            tsla_position = await PositionRepository(session).get_open_by_symbol("TSLA")

            assert stale_order is not None and stale_order.status == "cancelled"
            assert partial_order.status == "partially_filled"
            assert error_order is not None and error_order.status == "accepted"
            assert parent_order.status == "filled"
            assert child_order.status == "filled"
            assert sum(fill.qty for fill in fills) == 5
            assert sum(fill.qty for fill in parent_fills) == 3
            assert sum(fill.qty for fill in child_fills) == 3
            assert position is not None and position.qty == 5
            assert tsla_position is None

    asyncio.run(_assert_db_state())
