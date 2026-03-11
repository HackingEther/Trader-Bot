"""End-to-end integration tests for the trading cycle.

Validates full flow: bars -> features -> predictions -> strategy -> ranking -> risk
-> execution -> state updates. Uses mocked broker, model loader, and state store.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest
import trader.db.models  # noqa: F401
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from trader.config import Settings
from trader.core.events import ModelPrediction
from trader.db.base import Base
from trader.db.models.market_bar import MarketBar
from trader.db.models.order import Order
from trader.db.models.position import Position
from trader.db.repositories.orders import OrderRepository
from trader.db.repositories.positions import PositionRepository
from trader.db.repositories.trade_intents import TradeIntentRepository
from trader.providers.broker.base import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
    BrokerProvider,
    OrderRequest,
)
from trader.services.trading_cycle import TradingCycleService
from tests.fixtures.synthetic_bars import generate_synthetic_bars, generate_trending_up_bars


class _MockBrokerProvider(BrokerProvider):
    """Mock broker that returns deterministic values and simulates immediate fills."""

    def __init__(
        self,
        *,
        equity: Decimal = Decimal("100000"),
        fill_orders: bool = True,
        positions: list[BrokerPosition] | None = None,
    ) -> None:
        self._equity = equity
        self._fill_orders = fill_orders
        self._positions = {p.symbol: p for p in (positions or [])}
        self._order_counter = 0

    async def get_account(self) -> BrokerAccount:
        return BrokerAccount(
            account_id="mock-account",
            equity=self._equity,
            cash=self._equity,
            buying_power=self._equity,
            portfolio_value=self._equity,
        )

    async def get_position(self, symbol: str) -> BrokerPosition | None:
        return self._positions.get(symbol)

    async def get_positions(self) -> list[BrokerPosition]:
        return list(self._positions.values())

    async def submit_order(self, request: OrderRequest) -> BrokerOrder:
        self._order_counter += 1
        fill_price = request.limit_price or Decimal("100.00")
        status = "filled" if self._fill_orders else "accepted"
        filled_qty = request.qty if self._fill_orders else 0
        return BrokerOrder(
            broker_order_id=f"mock-{self._order_counter}",
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            qty=request.qty,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            filled_qty=filled_qty,
            filled_avg_price=fill_price if filled_qty else None,
            status=status,
            raw={"filled_at": datetime.now(timezone.utc).isoformat()},
        )

    async def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        return BrokerOrder(
            broker_order_id=broker_order_id,
            symbol="",
            side="",
            order_type="market",
            qty=0,
            status="cancelled",
        )

    async def replace_order(
        self, broker_order_id: str, qty: int | None = None, limit_price: Decimal | None = None
    ) -> BrokerOrder:
        return BrokerOrder(
            broker_order_id=broker_order_id,
            symbol="",
            side="",
            order_type="market",
            qty=qty or 0,
            limit_price=limit_price,
            status="accepted",
        )

    async def get_order(self, broker_order_id: str) -> BrokerOrder:
        return BrokerOrder(
            broker_order_id=broker_order_id,
            symbol="",
            side="",
            order_type="market",
            qty=0,
            status="filled",
        )

    async def get_open_orders(self) -> list[BrokerOrder]:
        return []

    async def close_position(self, symbol: str) -> BrokerOrder:
        return BrokerOrder(
            broker_order_id="close-1",
            symbol=symbol,
            side="sell",
            order_type="market",
            qty=0,
            status="filled",
        )

    async def close_all_positions(self) -> list[BrokerOrder]:
        return []


class _MockPipeline:
    """Pipeline that returns a fixed prediction."""

    def __init__(self, prediction: ModelPrediction) -> None:
        self._prediction = prediction

    def predict(
        self, symbol: str, features: dict, timestamp: datetime | None = None
    ) -> ModelPrediction:
        ts = timestamp or datetime.now(timezone.utc)
        return self._prediction.model_copy(update={"symbol": symbol, "timestamp": ts})


class _MockModelLoader:
    """Model loader that returns a pipeline with fixed predictions."""

    def __init__(self, prediction: ModelPrediction | None = None) -> None:
        self._prediction = prediction or ModelPrediction(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            direction="long",
            confidence=0.85,
            expected_move_bps=30.0,
            expected_holding_minutes=60.0,
            no_trade_score=0.1,
            regime="trending_up",
        )

    async def load_ensemble(self, session: AsyncSession | None = None, reload: bool = False):
        return _MockPipeline(self._prediction)


class _MockStateStore:
    """State store with configurable kill switch and spread."""

    def __init__(self, *, kill_switch: bool = False, spread_bps: float = 10.0) -> None:
        self._kill_switch = kill_switch
        self._spread_bps = spread_bps

    async def is_kill_switch_active(self) -> bool:
        return self._kill_switch

    async def get_spread_bps(self, symbol: str) -> float | None:
        return self._spread_bps

    async def get_last_bar_timestamp(self, symbol: str) -> datetime | None:
        return datetime.now(timezone.utc)


def _test_settings(**overrides) -> Settings:
    defaults = {
        "symbol_universe": ["AAPL"],
        "max_signals_per_cycle": 2,
        "min_confidence": 0.5,
        "min_expected_move_bps": 10.0,
        "min_relative_volume": 0.0,
        "spread_threshold_bps": 50.0,
        "limit_entry_buffer_bps": 5.0,
        "max_notional_exposure_usd": 50000.0,
        "max_exposure_per_symbol_usd": 10000.0,
        "max_concurrent_positions": 10,
        "max_daily_loss_usd": 1000.0,
        "max_loss_per_trade_usd": 200.0,
        "cooldown_after_losses": 3,
    }
    return Settings(**{**defaults, **overrides})


@pytest.fixture
async def db_session_with_bars(tmp_path) -> async_sessionmaker[AsyncSession]:
    """Create SQLite DB with market_bars seeded."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'trading_cycle.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with factory() as session:
        bars = generate_trending_up_bars(symbol="AAPL", count=25, seed=42)
        for bar in bars:
            session.add(
                MarketBar(
                    symbol=bar["symbol"],
                    timestamp=bar["timestamp"],
                    interval="1m",
                    open=bar["open"],
                    high=bar["high"],
                    low=bar["low"],
                    close=bar["close"],
                    volume=bar["volume"],
                    vwap=bar.get("vwap"),
                    trade_count=bar.get("trade_count"),
                )
            )
        await session.commit()

    yield factory
    await engine.dispose()


@pytest.fixture
async def db_session_two_symbols(tmp_path) -> async_sessionmaker[AsyncSession]:
    """Create SQLite DB with bars for AAPL and MSFT."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'trading_cycle_2.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with factory() as session:
        for symbol in ["AAPL", "MSFT"]:
            bars = generate_trending_up_bars(symbol=symbol, count=25, seed=42 if symbol == "AAPL" else 43)
            for bar in bars:
                session.add(
                    MarketBar(
                        symbol=bar["symbol"],
                        timestamp=bar["timestamp"],
                        interval="1m",
                        open=bar["open"],
                        high=bar["high"],
                        low=bar["low"],
                        close=bar["close"],
                        volume=bar["volume"],
                        vwap=bar.get("vwap"),
                        trade_count=bar.get("trade_count"),
                    )
                )
        await session.commit()

    yield factory
    await engine.dispose()


@pytest.mark.asyncio
@patch("trader.risk.rules.market_hours.is_market_open", return_value=True)
async def test_run_cycle_full_flow_bars_to_execution(
    _mock_market_open: object,
    db_session_with_bars: async_sessionmaker[AsyncSession],
) -> None:
    """Full flow: bars -> features -> predictions -> strategy -> risk -> execution -> state updates."""
    settings = _test_settings()
    broker = _MockBrokerProvider(fill_orders=True)
    model_loader = _MockModelLoader()
    state_store = _MockStateStore()

    async with db_session_with_bars() as session:
        service = TradingCycleService(
            settings=settings,
            session=session,
            broker=broker,
            model_loader=model_loader,
            state_store=state_store,
        )
        results = await service.run_cycle(symbols=["AAPL"])

    assert results["orders_sent"] >= 1
    assert results["kill_switch_active"] is False

    async with db_session_with_bars() as session:
        intents = await TradeIntentRepository(session).get_recent(limit=5)
        orders = await OrderRepository(session).get_all(limit=10)
        positions = await PositionRepository(session).get_open_positions()

    assert len(intents) >= 1
    executed = [i for i in intents if i.status == "executed"]
    assert len(executed) >= 1
    assert len(orders) >= 1
    assert len(positions) >= 1


@pytest.mark.asyncio
async def test_run_cycle_skips_when_insufficient_bars(
    tmp_path,
) -> None:
    """Cycle skips symbols with fewer than 20 bars."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'few_bars.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        bars = generate_synthetic_bars(symbol="AAPL", count=10, seed=42)
        for bar in bars:
            session.add(
                MarketBar(
                    symbol=bar["symbol"],
                    timestamp=bar["timestamp"],
                    interval="1m",
                    open=bar["open"],
                    high=bar["high"],
                    low=bar["low"],
                    close=bar["close"],
                    volume=bar["volume"],
                    vwap=bar.get("vwap"),
                    trade_count=bar.get("trade_count"),
                )
            )
        await session.commit()

    settings = _test_settings()
    broker = _MockBrokerProvider(fill_orders=True)
    model_loader = _MockModelLoader()
    state_store = _MockStateStore()

    async with factory() as session:
        service = TradingCycleService(
            settings=settings,
            session=session,
            broker=broker,
            model_loader=model_loader,
            state_store=state_store,
        )
        results = await service.run_cycle(symbols=["AAPL"])

    assert results["skipped_no_bars"] >= 1
    assert results["orders_sent"] == 0

    async with factory() as session:
        orders = await OrderRepository(session).get_all(limit=10)
        assert len(orders) == 0

    await engine.dispose()


@pytest.mark.asyncio
async def test_run_cycle_strategy_filtered_when_no_trade(
    db_session_with_bars: async_sessionmaker[AsyncSession],
) -> None:
    """No orders when model returns no_trade direction."""
    settings = _test_settings()
    broker = _MockBrokerProvider(fill_orders=True)
    prediction = ModelPrediction(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        direction="no_trade",
        confidence=0.5,
        expected_move_bps=5.0,
        expected_holding_minutes=30.0,
        no_trade_score=0.9,
        regime="low_volatility",
    )
    model_loader = _MockModelLoader(prediction=prediction)
    state_store = _MockStateStore()

    async with db_session_with_bars() as session:
        service = TradingCycleService(
            settings=settings,
            session=session,
            broker=broker,
            model_loader=model_loader,
            state_store=state_store,
        )
        results = await service.run_cycle(symbols=["AAPL"])

    assert results["orders_sent"] == 0
    assert any(
        s.get("status") == "predicted" and s.get("reason") == "strategy_filtered"
        for s in results["symbols"]
    )


@pytest.mark.asyncio
async def test_run_cycle_halted_when_kill_switch_active(
    db_session_with_bars: async_sessionmaker[AsyncSession],
) -> None:
    """Cycle halts when kill switch is active."""
    settings = _test_settings()
    broker = _MockBrokerProvider(fill_orders=True)
    model_loader = _MockModelLoader()
    state_store = _MockStateStore(kill_switch=True)

    async with db_session_with_bars() as session:
        service = TradingCycleService(
            settings=settings,
            session=session,
            broker=broker,
            model_loader=model_loader,
            state_store=state_store,
        )
        results = await service.run_cycle(symbols=["AAPL"])

    assert results["kill_switch_active"] is True
    assert results["orders_sent"] == 0


@pytest.mark.asyncio
@patch("trader.risk.rules.market_hours.is_market_open", return_value=True)
async def test_run_cycle_ranking_selects_top_scores(
    _mock_market_open: object,
    db_session_two_symbols: async_sessionmaker[AsyncSession],
) -> None:
    """Only top-ranked symbol gets executed when max_signals_per_cycle=1."""
    settings = _test_settings(symbol_universe=["AAPL", "MSFT"], max_signals_per_cycle=1)
    broker = _MockBrokerProvider(fill_orders=True)
    model_loader = _MockModelLoader()
    state_store = _MockStateStore()

    async with db_session_two_symbols() as session:
        service = TradingCycleService(
            settings=settings,
            session=session,
            broker=broker,
            model_loader=model_loader,
            state_store=state_store,
        )
        results = await service.run_cycle(symbols=["AAPL", "MSFT"])

    executed = [s for s in results["symbols"] if s.get("status") in ("executed", "approved")]
    ranked_out = [s for s in results["symbols"] if s.get("reason") == "ranked_out"]
    assert len(executed) == 1
    assert len(ranked_out) == 1


@pytest.mark.asyncio
@patch("trader.risk.rules.market_hours.is_market_open", return_value=True)
async def test_run_cycle_projected_exposure_includes_open_orders(
    _mock_market_open: object,
    db_session_with_bars: async_sessionmaker[AsyncSession],
) -> None:
    """Projected exposure includes open orders; capacity is reduced."""
    settings = _test_settings(max_notional_exposure_usd=5000.0, max_exposure_per_symbol_usd=3000.0)
    broker = _MockBrokerProvider(fill_orders=True)
    model_loader = _MockModelLoader()
    state_store = _MockStateStore()

    now = datetime.now(timezone.utc)
    async with db_session_with_bars() as session:
        session.add(
            Order(
                idempotency_key="existing-order",
                broker_order_id="broker-1",
                symbol="AAPL",
                side="buy",
                order_type="market",
                order_class="simple",
                time_in_force="day",
                qty=20,
                filled_qty=0,
                status="accepted",
                strategy_tag="test",
                rationale={"reference_price": "100"},
                broker_metadata={},
                submitted_at=now,
            )
        )
        await session.commit()

    async with db_session_with_bars() as session:
        service = TradingCycleService(
            settings=settings,
            session=session,
            broker=broker,
            model_loader=model_loader,
            state_store=state_store,
        )
        results = await service.run_cycle(symbols=["AAPL"])

    symbols_data = results["symbols"]
    capacity_filtered = [s for s in symbols_data if s.get("reason") == "capacity_filtered"]
    assert (
        len(capacity_filtered) >= 1 or results["orders_sent"] >= 1
    ), "Open order should reduce capacity (capacity_filtered or executed with reduced size)"
    if results["orders_sent"] >= 1:
        async with db_session_with_bars() as session:
            orders = await OrderRepository(session).get_all(limit=10)
            new_orders = [o for o in orders if o.broker_order_id != "broker-1"]
        if new_orders:
            assert new_orders[0].qty < 52, "Executed order should be resized due to open order exposure"


@pytest.mark.asyncio
@patch("trader.risk.rules.market_hours.is_market_open", return_value=True)
async def test_run_cycle_persists_correct_statuses(
    _mock_market_open: object,
    db_session_with_bars: async_sessionmaker[AsyncSession],
) -> None:
    """TradeIntent and Order statuses are persisted correctly through the flow."""
    settings = _test_settings()
    broker = _MockBrokerProvider(fill_orders=True)
    model_loader = _MockModelLoader()
    state_store = _MockStateStore()

    async with db_session_with_bars() as session:
        service = TradingCycleService(
            settings=settings,
            session=session,
            broker=broker,
            model_loader=model_loader,
            state_store=state_store,
        )
        results = await service.run_cycle(symbols=["AAPL"])

    assert results["orders_sent"] >= 1

    async with db_session_with_bars() as session:
        intents = await TradeIntentRepository(session).get_recent(limit=5)
        orders = await OrderRepository(session).get_all(limit=10)

    executed_intent = next((i for i in intents if i.status == "executed"), None)
    assert executed_intent is not None

    filled_order = next((o for o in orders if o.status in ("filled", "partially_filled")), None)
    assert filled_order is not None
