"""Unit tests for quote snapshots and execution attribution."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest
import trader.db.models  # noqa: F401
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trader.db.base import Base
from trader.db.models.execution_attribution import ExecutionAttribution
from trader.db.models.order import Order
from trader.db.models.quote_snapshot import QuoteSnapshot
from trader.db.models.trade_intent import TradeIntent
from trader.db.repositories.execution_attribution import ExecutionAttributionRepository
from trader.db.repositories.quote_snapshots import QuoteSnapshotRepository


@pytest.fixture
async def db_session(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'quotes.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_quote_snapshot_persisted_at_decision_and_submit(db_session: AsyncSession) -> None:
    """Quote snapshots are created for decision and submit types."""
    repo = QuoteSnapshotRepository(db_session)
    ts = datetime.now(timezone.utc)

    snapshot = await repo.create_snapshot(
        snapshot_type="decision",
        symbol="AAPL",
        bid=Decimal("150.00"),
        ask=Decimal("150.05"),
        mid=Decimal("150.025"),
        timestamp=ts,
        spread_bps=Decimal("3.33"),
        trade_intent_id=1,
    )
    await db_session.flush()
    assert snapshot.id is not None
    assert snapshot.snapshot_type == "decision"
    assert snapshot.symbol == "AAPL"
    assert snapshot.bid == Decimal("150.00")

    snapshot2 = await repo.create_snapshot(
        snapshot_type="submit",
        symbol="AAPL",
        bid=Decimal("150.01"),
        ask=Decimal("150.06"),
        mid=Decimal("150.035"),
        timestamp=ts,
        spread_bps=Decimal("3.33"),
        trade_intent_id=1,
        order_id=1,
    )
    await db_session.flush()
    assert snapshot2.snapshot_type == "submit"
    assert snapshot2.order_id == 1


@pytest.mark.asyncio
async def test_execution_attribution_computes_slippage_and_time_to_fill(db_session: AsyncSession) -> None:
    """Execution attribution computes slippage_bps and time_to_fill_seconds."""
    from trader.db.repositories.fills import FillRepository
    from trader.db.repositories.orders import OrderRepository

    orders = OrderRepository(db_session)
    quote_repo = QuoteSnapshotRepository(db_session)
    attr_repo = ExecutionAttributionRepository(db_session)

    intent = TradeIntent(
        symbol="AAPL",
        side="buy",
        qty=10,
        entry_order_type="limit",
        status="executed",
        timestamp=datetime.now(timezone.utc),
    )
    db_session.add(intent)
    await db_session.flush()

    decision_snap = await quote_repo.create_snapshot(
        snapshot_type="decision",
        symbol="AAPL",
        bid=Decimal("150.00"),
        ask=Decimal("150.05"),
        mid=Decimal("150.025"),
        timestamp=datetime.now(timezone.utc),
        trade_intent_id=intent.id,
    )
    await db_session.flush()

    submitted_at = datetime.now(timezone.utc)
    filled_at = datetime.now(timezone.utc)
    order, _ = await orders.create_idempotent(
        idempotency_key="attr-test",
        trade_intent_id=intent.id,
        symbol="AAPL",
        side="buy",
        order_type="limit",
        order_class="simple",
        qty=10,
        status="filled",
        strategy_tag="test",
        rationale={},
        submitted_at=submitted_at,
        filled_at=filled_at,
        filled_qty=10,
        filled_avg_price=Decimal("150.06"),
    )
    await db_session.flush()

    from trader.db.models.fill import Fill

    fill = Fill(
        order_id=order.id,
        symbol="AAPL",
        side="buy",
        qty=10,
        price=Decimal("150.06"),
        timestamp=filled_at,
        execution_key="test-exec-1",
    )
    db_session.add(fill)
    await db_session.flush()

    attr = await attr_repo.ensure_attribution_for_filled_order(
        order=order,
        fills=[fill],
        quote_snapshots=quote_repo,
    )
    await db_session.flush()

    assert attr is not None
    assert attr.slippage_bps is not None
    assert float(attr.slippage_bps) == pytest.approx(0.67, rel=0.1)
    assert attr.realized_spread_bps is not None
    assert attr.time_to_fill_seconds is not None
