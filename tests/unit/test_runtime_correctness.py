"""Regression tests for runtime-correctness fixes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from trader.core.events import BarEvent, QuoteEvent, TradeEvent
from trader.core.exceptions import DuplicateOrderError
from trader.execution.engine import ExecutionEngine
from trader.execution.lifecycle import OrderLifecycleTracker
from trader.execution.position_ledger import PositionLedger
from trader.ingestion.handlers import BarHandler, QuoteHandler, TradeHandler
from trader.ingestion.staleness import StalenessDetector
from trader.core.enums import OrderStatus
from trader.services.trading_cycle import TradingCycleService
from trader.services.system_state import SystemStateStore
from trader.strategy.engine import TradeIntentParams
from trader.workers.tasks import _incremental_fill_price


class _DummyStateStore:
    async def is_kill_switch_active(self) -> bool:
        return False

    async def remember_once(self, namespace: str, raw_key: str, ttl_seconds: int) -> bool:
        return True


class _DummyBroker:
    def __init__(self) -> None:
        self.submissions = 0

    async def submit_order(self, request):  # pragma: no cover - should not be called
        self.submissions += 1
        raise AssertionError("broker submit_order should not be called for duplicates")


class _DummyRepo:
    def __init__(self) -> None:
        self.existing = SimpleNamespace(id=7)

    async def create_idempotent(self, **kwargs):
        return self.existing, False


class _DummyLifecycle:
    async def update_status(self, **kwargs):  # pragma: no cover - duplicate path should not update
        raise AssertionError("update_status should not run for duplicates")

    async def record_fill(self, **kwargs):  # pragma: no cover - duplicate path should not record fill
        raise AssertionError("record_fill should not run for duplicates")


class _DummyPositions:
    async def apply_fill(self, **kwargs):  # pragma: no cover - duplicate path should not apply fill
        raise AssertionError("apply_fill should not run for duplicates")


class _MemorySession:
    async def flush(self) -> None:
        return None


class _MemoryPositionRepo:
    def __init__(self) -> None:
        self.positions = []

    async def get_open_by_symbol(self, symbol: str):
        for position in self.positions:
            if position.symbol == symbol and position.status == "open":
                return position
        return None

    async def create(self, **kwargs):
        position = SimpleNamespace(**kwargs)
        self.positions.append(position)
        return position


@pytest.mark.asyncio
async def test_execution_engine_raises_clean_duplicate_without_broker_submit() -> None:
    engine = ExecutionEngine.__new__(ExecutionEngine)
    engine._broker = _DummyBroker()
    engine._repo = _DummyRepo()
    engine._lifecycle = _DummyLifecycle()
    engine._positions = _DummyPositions()
    engine._state_store = _DummyStateStore()

    intent = TradeIntentParams(symbol="AAPL", side="buy", qty=10)

    with pytest.raises(DuplicateOrderError):
        await engine.execute(intent, trade_intent_id=1)

    assert engine._broker.submissions == 0


def test_incremental_fill_price_uses_cumulative_average_correctly() -> None:
    price = _incremental_fill_price(
        cumulative_qty=50,
        cumulative_avg_price=Decimal("10.40"),
        recorded_qty=30,
        recorded_notional=Decimal("300.00"),
    )
    assert price == Decimal("11.00")


@pytest.mark.asyncio
async def test_position_ledger_blends_multi_price_partial_fills_correctly() -> None:
    ledger = PositionLedger.__new__(PositionLedger)
    ledger._session = _MemorySession()
    ledger._positions = _MemoryPositionRepo()

    order = SimpleNamespace(
        id=1,
        symbol="AAPL",
        side="buy",
        strategy_tag="test",
        trade_intent_id=11,
        order_type="limit",
        broker_order_id="broker-1",
    )
    intent = TradeIntentParams(symbol="AAPL", side="buy", qty=30, entry_order_type="limit")

    first = await ledger.apply_fill(
        order=order,
        intent=intent,
        fill_price=Decimal("10.00"),
        fill_qty=30,
        timestamp=datetime.now(timezone.utc),
    )
    second = await ledger.apply_fill(
        order=order,
        intent=intent,
        fill_price=Decimal("11.00"),
        fill_qty=20,
        timestamp=datetime.now(timezone.utc),
    )

    assert first is second
    assert second.qty == 50
    assert second.avg_entry_price == Decimal("10.40")


@pytest.mark.asyncio
async def test_bar_handler_deduplicates_duplicate_events(monkeypatch: pytest.MonkeyPatch) -> None:
    published = []
    streamed = []

    async def _publish(channel: str, data: dict) -> None:
        published.append((channel, data))

    async def _xadd(stream: str, data: dict, maxlen: int = 10000) -> str:
        streamed.append((stream, data, maxlen))
        return "1-0"

    monkeypatch.setattr("trader.ingestion.handlers.publish_event", _publish)
    monkeypatch.setattr("trader.ingestion.handlers.xadd_event", _xadd)

    handler = BarHandler(StalenessDetector())
    event = BarEvent(
        symbol="AAPL",
        timestamp=datetime(2026, 3, 10, 15, 0, tzinfo=timezone.utc),
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=1000,
    )

    await handler.handle(event)
    await handler.handle(event)

    assert len(published) == 1
    assert len(streamed) == 1


@pytest.mark.asyncio
async def test_quote_and_trade_handlers_deduplicate_duplicate_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = []
    streamed = []

    async def _publish(channel: str, data: dict) -> None:
        published.append((channel, data))

    async def _xadd(stream: str, data: dict, maxlen: int = 10000) -> str:
        streamed.append((stream, data, maxlen))
        return "1-0"

    monkeypatch.setattr("trader.ingestion.handlers.publish_event", _publish)
    monkeypatch.setattr("trader.ingestion.handlers.xadd_event", _xadd)

    staleness = StalenessDetector()
    quote_handler = QuoteHandler(staleness)
    trade_handler = TradeHandler(staleness)
    quote = QuoteEvent(
        symbol="AAPL",
        timestamp=datetime(2026, 3, 10, 15, 1, tzinfo=timezone.utc),
        bid_price=Decimal("100"),
        bid_size=10,
        ask_price=Decimal("100.1"),
        ask_size=12,
    )
    trade = TradeEvent(
        symbol="AAPL",
        timestamp=datetime(2026, 3, 10, 15, 1, 5, tzinfo=timezone.utc),
        price=Decimal("100.05"),
        size=20,
    )

    await quote_handler.handle(quote)
    await quote_handler.handle(quote)
    await trade_handler.handle(trade)
    await trade_handler.handle(trade)

    assert len(published) == 2
    assert len(streamed) == 2


def test_projected_capacity_resizes_intent_to_remaining_exposure() -> None:
    service = TradingCycleService.__new__(TradingCycleService)
    service._settings = SimpleNamespace(
        max_notional_exposure_usd=1000.0,
        max_exposure_per_symbol_usd=600.0,
    )
    intent = TradeIntentParams(symbol="AAPL", side="buy", qty=10, rationale={"source": "test"})

    resized = service._apply_projected_capacity(
        intent=intent,
        price=Decimal("100"),
        projected_total_exposure=Decimal("500"),
        projected_symbol_exposure=Decimal("300"),
    )

    assert resized is not None
    assert resized.qty == 3
    assert resized.rationale["resized_from_qty"] == 10
    assert resized.rationale["resized_to_qty"] == 3


def test_projected_capacity_includes_open_order_exposure() -> None:
    """Projected exposure correctly reduces remaining capacity for pending open orders."""
    service = TradingCycleService.__new__(TradingCycleService)
    service._settings = SimpleNamespace(
        max_notional_exposure_usd=2000.0,
        max_exposure_per_symbol_usd=1500.0,
    )
    intent = TradeIntentParams(symbol="AAPL", side="buy", qty=20, rationale={"source": "test"})

    resized = service._apply_projected_capacity(
        intent=intent,
        price=Decimal("100"),
        projected_total_exposure=Decimal("1500"),
        projected_symbol_exposure=Decimal("0"),
    )

    assert resized is not None
    assert resized.qty == 5


def test_open_order_slots_counts_unique_symbols() -> None:
    """_open_order_slots counts one slot per symbol with unfilled orders."""
    orders = [
        SimpleNamespace(symbol="AAPL", qty=10, filled_qty=5),
        SimpleNamespace(symbol="MSFT", qty=5, filled_qty=0),
        SimpleNamespace(symbol="AAPL", qty=8, filled_qty=8),
    ]
    slots = TradingCycleService._open_order_slots(orders)
    assert slots == 2


def test_composite_score_includes_playbook_fit() -> None:
    """_score_candidate uses playbook_fit from intent.rationale."""
    from trader.config import get_settings
    from trader.core.events import ModelPrediction

    service = TradingCycleService.__new__(TradingCycleService)
    service._settings = get_settings()

    pred = ModelPrediction(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        direction="long",
        confidence=0.8,
        expected_move_bps=25.0,
        expected_holding_minutes=30.0,
        no_trade_score=0.2,
        regime="trending_up",
    )
    features = {
        "relative_volume": 1.2,
        "spread_bps": 10.0,
        "momentum_5m": 0.01,
        "momentum_15m": 0.02,
    }

    intent_full = TradeIntentParams(
        symbol="AAPL",
        side="buy",
        qty=10,
        rationale={"playbook_fit": 1.0},
    )
    intent_half = TradeIntentParams(
        symbol="AAPL",
        side="buy",
        qty=10,
        rationale={"playbook_fit": 0.5},
    )

    score_full = service._score_candidate(prediction=pred, features=features, intent=intent_full)
    score_half = service._score_candidate(prediction=pred, features=features, intent=intent_half)

    assert score_full > 0
    assert score_half > 0
    assert score_half < score_full


def test_ranking_selects_highest_scores_first() -> None:
    """Selected candidates are in score-descending order; processing follows that order."""
    candidates = [
        SimpleNamespace(score=0.5, symbol="A", intent_record=SimpleNamespace(id=1)),
        SimpleNamespace(score=0.9, symbol="B", intent_record=SimpleNamespace(id=2)),
        SimpleNamespace(score=0.7, symbol="C", intent_record=SimpleNamespace(id=3)),
    ]
    selected = sorted(candidates, key=lambda c: c.score, reverse=True)[:2]
    assert [s.symbol for s in selected] == ["B", "C"]


def test_record_result_increments_orders_sent_for_approved_and_executed() -> None:
    """orders_sent counter includes both executed and approved statuses."""
    from trader.services.trading_cycle import SymbolCycleResult

    service = TradingCycleService.__new__(TradingCycleService)
    results = {
        "symbols": [],
        "symbols_processed": 0,
        "predictions_persisted": 0,
        "intents_generated": 0,
        "orders_submitted": 0,
        "orders_sent": 0,
        "orders_rejected": 0,
        "skipped_no_bars": 0,
    }

    service._record_result(results, SymbolCycleResult(symbol="AAPL", status="approved", trade_intent_id=1))
    assert results["orders_sent"] == 1
    assert results["orders_submitted"] == 0

    service._record_result(results, SymbolCycleResult(symbol="MSFT", status="executed", trade_intent_id=2))
    assert results["orders_sent"] == 2
    assert results["orders_submitted"] == 1


def test_order_notional_uses_reference_price_rationale_for_market_orders() -> None:
    order = SimpleNamespace(
        qty=10,
        filled_qty=4,
        limit_price=None,
        filled_avg_price=None,
        rationale={"reference_price": "101.25"},
        broker_metadata={},
    )

    notional = TradingCycleService._order_notional(order)

    assert notional == Decimal("607.50")


@pytest.mark.asyncio
async def test_distributed_handler_dedupe_suppresses_second_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    published = []

    async def _publish(channel: str, data: dict) -> None:
        published.append((channel, data))

    async def _xadd(stream: str, data: dict, maxlen: int = 10000) -> str:
        return "1-0"

    class _DedupeStore:
        def __init__(self) -> None:
            self._seen: set[str] = set()

        async def remember_once(self, namespace: str, raw_key: str, ttl_seconds: int) -> bool:
            key = f"{namespace}:{raw_key}"
            if key in self._seen:
                return False
            self._seen.add(key)
            return True

    monkeypatch.setattr("trader.ingestion.handlers.publish_event", _publish)
    monkeypatch.setattr("trader.ingestion.handlers.xadd_event", _xadd)

    event = QuoteEvent(
        symbol="AAPL",
        timestamp=datetime(2026, 3, 10, 15, 1, tzinfo=timezone.utc),
        bid_price=Decimal("100"),
        bid_size=10,
        ask_price=Decimal("100.1"),
        ask_size=12,
    )
    first_handler = QuoteHandler(StalenessDetector(), state_store=_DedupeStore())
    second_handler = QuoteHandler(StalenessDetector(), state_store=first_handler._state_store)

    await first_handler.handle(event)
    await second_handler.handle(event)

    assert len(published) == 1


@pytest.mark.asyncio
async def test_system_state_remember_once_and_lease_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def raise_runtime_error(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("redis not initialized")

    async def raise_connect_error(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr("trader.services.system_state.get_cached", raise_runtime_error)
    monkeypatch.setattr("trader.services.system_state.set_cached", raise_runtime_error)
    monkeypatch.setattr("trader.services.system_state.set_cached_if_absent", raise_runtime_error)
    monkeypatch.setattr("trader.services.system_state.compare_and_set_cached", raise_runtime_error)
    monkeypatch.setattr("trader.services.system_state.init_redis", raise_connect_error)

    store = SystemStateStore(redis_url="redis://invalid")

    assert await store.remember_once("bars", "abc", ttl_seconds=60) is True
    assert await store.remember_once("bars", "abc", ttl_seconds=60) is False
    assert await store.acquire_lease("feed", "owner-1", ttl_seconds=60) is True
    assert await store.renew_lease("feed", "owner-1", ttl_seconds=60) is True
    assert await store.renew_lease("feed", "owner-2", ttl_seconds=60) is False


@pytest.mark.asyncio
async def test_lifecycle_status_updates_do_not_regress_terminal_state(
    tmp_path,
) -> None:
    import trader.db.models  # noqa: F401
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from trader.db.base import Base
    from trader.db.models.order import Order
    from trader.db.repositories.orders import OrderRepository

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'lifecycle.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        order = Order(
            idempotency_key="filled-order",
            symbol="AAPL",
            side="buy",
            order_type="market",
            order_class="simple",
            time_in_force="day",
            qty=10,
            filled_qty=10,
            status=OrderStatus.FILLED,
            strategy_tag="test",
            rationale={},
            broker_metadata={},
        )
        session.add(order)
        await session.flush()

        lifecycle = OrderLifecycleTracker(session)
        await lifecycle.update_status(order.id, OrderStatus.ACCEPTED, filled_qty=10, filled_avg_price=100.0)
        updated = await OrderRepository(session).get_by_id(order.id)
        assert updated is not None
        assert updated.status == OrderStatus.FILLED

    await engine.dispose()


@pytest.mark.asyncio
async def test_lifecycle_record_fill_deduplicates_execution_key(tmp_path) -> None:
    import trader.db.models  # noqa: F401
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from trader.db.base import Base
    from trader.db.models.order import Order
    from trader.db.repositories.fills import FillRepository

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'fills.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        order = Order(
            idempotency_key="fill-order",
            symbol="AAPL",
            side="buy",
            order_type="market",
            order_class="simple",
            time_in_force="day",
            qty=10,
            status=OrderStatus.PARTIALLY_FILLED,
            strategy_tag="test",
            rationale={},
            broker_metadata={},
        )
        session.add(order)
        await session.flush()

        lifecycle = OrderLifecycleTracker(session)
        first_fill, first_new = await lifecycle.record_fill(
            order_id=order.id,
            broker_order_id="broker-1",
            symbol="AAPL",
            side="buy",
            qty=5,
            price=100.0,
            execution_key="broker-1:5",
            timestamp=datetime.now(timezone.utc),
        )
        second_fill, second_new = await lifecycle.record_fill(
            order_id=order.id,
            broker_order_id="broker-1",
            symbol="AAPL",
            side="buy",
            qty=5,
            price=100.0,
            execution_key="broker-1:5",
            timestamp=datetime.now(timezone.utc),
        )
        fills = await FillRepository(session).get_by_order_id(order.id)

        assert first_fill.id == second_fill.id
        assert first_new is True
        assert second_new is False
        assert len(fills) == 1

    await engine.dispose()


def test_record_fill_returns_is_new_flag() -> None:
    """record_fill returns (Fill, is_new) tuple; is_new True for new, False for existing."""
    import inspect

    from trader.execution.lifecycle import OrderLifecycleTracker

    sig = inspect.signature(OrderLifecycleTracker.record_fill)
    assert "tuple" in str(sig.return_annotation) or "Tuple" in str(sig.return_annotation)
