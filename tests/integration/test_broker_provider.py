"""Integration tests for broker providers (using paper broker)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from trader.providers.broker.base import OrderRequest
from trader.providers.broker.paper import PaperBrokerProvider


@pytest.fixture
def paper_broker() -> PaperBrokerProvider:
    return PaperBrokerProvider(initial_cash=Decimal("100000"))


class TestPaperBrokerProvider:
    @pytest.mark.asyncio
    async def test_submit_market_order(self, paper_broker: PaperBrokerProvider) -> None:
        request = OrderRequest(symbol="AAPL", side="buy", qty=10, order_type="market")
        order = await paper_broker.submit_order(request)
        assert order.broker_order_id is not None
        assert order.status == "filled"
        assert order.filled_qty == 10

    @pytest.mark.asyncio
    async def test_submit_limit_order(self, paper_broker: PaperBrokerProvider) -> None:
        request = OrderRequest(
            symbol="AAPL", side="buy", qty=10,
            order_type="limit", limit_price=Decimal("150.00"),
        )
        order = await paper_broker.submit_order(request)
        assert order.status == "accepted"

    @pytest.mark.asyncio
    async def test_cancel_order(self, paper_broker: PaperBrokerProvider) -> None:
        request = OrderRequest(
            symbol="AAPL", side="buy", qty=10,
            order_type="limit", limit_price=Decimal("150.00"),
        )
        order = await paper_broker.submit_order(request)
        cancelled = await paper_broker.cancel_order(order.broker_order_id)
        assert cancelled.status == "cancelled"

    @pytest.mark.asyncio
    async def test_get_positions_after_fill(self, paper_broker: PaperBrokerProvider) -> None:
        request = OrderRequest(symbol="AAPL", side="buy", qty=10, order_type="market")
        await paper_broker.submit_order(request)
        positions = await paper_broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].qty == 10

    @pytest.mark.asyncio
    async def test_get_account(self, paper_broker: PaperBrokerProvider) -> None:
        account = await paper_broker.get_account()
        assert account.account_id == "paper-account"
        assert account.equity == Decimal("100000")

    @pytest.mark.asyncio
    async def test_close_position(self, paper_broker: PaperBrokerProvider) -> None:
        await paper_broker.submit_order(
            OrderRequest(symbol="AAPL", side="buy", qty=10, order_type="market")
        )
        close_order = await paper_broker.close_position("AAPL")
        assert close_order.status == "filled"
        positions = await paper_broker.get_positions()
        assert len(positions) == 0

    @pytest.mark.asyncio
    async def test_close_all_positions(self, paper_broker: PaperBrokerProvider) -> None:
        await paper_broker.submit_order(
            OrderRequest(symbol="AAPL", side="buy", qty=10, order_type="market")
        )
        await paper_broker.submit_order(
            OrderRequest(symbol="MSFT", side="buy", qty=5, order_type="market")
        )
        orders = await paper_broker.close_all_positions()
        assert len(orders) == 2
        positions = await paper_broker.get_positions()
        assert len(positions) == 0
