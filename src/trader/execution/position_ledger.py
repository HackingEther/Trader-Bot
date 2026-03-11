"""Local position and realized PnL updates from fills."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from trader.db.models.order import Order
from trader.db.models.position import Position
from trader.db.repositories.positions import PositionRepository
from trader.strategy.engine import TradeIntentParams


class PositionLedger:
    """Materialize broker fills into local positions immediately."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._positions = PositionRepository(session)

    async def apply_fill(
        self,
        *,
        order: Order,
        intent: TradeIntentParams | None,
        fill_price: Decimal,
        fill_qty: int,
        commission: Decimal = Decimal("0"),
        timestamp: datetime,
    ) -> Position:
        existing = await self._positions.get_open_by_symbol(order.symbol)
        metadata = self._build_metadata(order=order, intent=intent, commission=commission)

        if existing is None:
            return await self._positions.create(
                symbol=order.symbol,
                side=order.side,
                qty=fill_qty,
                avg_entry_price=fill_price,
                current_price=fill_price,
                market_value=fill_price * fill_qty,
                unrealized_pnl=Decimal("0"),
                realized_pnl=-commission,
                status="open",
                strategy_tag=order.strategy_tag,
                trade_intent_id=order.trade_intent_id,
                opened_at=timestamp,
                metadata_=metadata,
            )

        if existing.side == order.side:
            total_qty = existing.qty + fill_qty
            blended_cost = existing.avg_entry_price * existing.qty + fill_price * fill_qty
            existing.avg_entry_price = blended_cost / total_qty
            existing.qty = total_qty
            existing.current_price = fill_price
            existing.market_value = fill_price * total_qty
            existing.unrealized_pnl = self._unrealized_pnl(
                side=existing.side,
                avg_entry_price=existing.avg_entry_price,
                mark_price=fill_price,
                qty=total_qty,
            )
            existing.realized_pnl -= commission
            existing.metadata_ = {**existing.metadata_, **metadata}
            await self._session.flush()
            return existing

        close_qty = min(existing.qty, fill_qty)
        remaining_qty = existing.qty - close_qty
        close_commission = commission * Decimal(close_qty) / Decimal(fill_qty)
        open_commission = commission - close_commission
        realized = self._realized_pnl(
            side=existing.side,
            avg_entry_price=existing.avg_entry_price,
            exit_price=fill_price,
            qty=close_qty,
        ) - close_commission

        existing.realized_pnl += realized
        existing.current_price = fill_price
        existing.metadata_ = {**existing.metadata_, **metadata}

        if remaining_qty > 0:
            existing.qty = remaining_qty
            existing.market_value = fill_price * remaining_qty
            existing.unrealized_pnl = self._unrealized_pnl(
                side=existing.side,
                avg_entry_price=existing.avg_entry_price,
                mark_price=fill_price,
                qty=remaining_qty,
            )
            await self._session.flush()
            return existing

        existing.qty = 0
        existing.market_value = Decimal("0")
        existing.unrealized_pnl = Decimal("0")
        existing.status = "closed"
        existing.closed_at = timestamp
        await self._session.flush()

        if fill_qty == close_qty:
            return existing

        flip_qty = fill_qty - close_qty
        return await self._positions.create(
            symbol=order.symbol,
            side=order.side,
            qty=flip_qty,
            avg_entry_price=fill_price,
            current_price=fill_price,
            market_value=fill_price * flip_qty,
            unrealized_pnl=Decimal("0"),
            realized_pnl=-open_commission,
            status="open",
            strategy_tag=order.strategy_tag,
            trade_intent_id=order.trade_intent_id,
            opened_at=timestamp,
            metadata_=metadata,
        )

    @staticmethod
    def _realized_pnl(
        *,
        side: str,
        avg_entry_price: Decimal,
        exit_price: Decimal,
        qty: int,
    ) -> Decimal:
        if side == "buy":
            return (exit_price - avg_entry_price) * qty
        return (avg_entry_price - exit_price) * qty

    @staticmethod
    def _unrealized_pnl(
        *,
        side: str,
        avg_entry_price: Decimal,
        mark_price: Decimal,
        qty: int,
    ) -> Decimal:
        if side == "buy":
            return (mark_price - avg_entry_price) * qty
        return (avg_entry_price - mark_price) * qty

    @staticmethod
    def _build_metadata(
        *,
        order: Order,
        intent: TradeIntentParams | None,
        commission: Decimal,
    ) -> dict:
        return {
            "entry_order_type": intent.entry_order_type if intent is not None else order.order_type,
            "limit_price": str(intent.limit_price) if intent is not None and intent.limit_price is not None else None,
            "stop_loss": str(intent.stop_loss) if intent is not None and intent.stop_loss is not None else None,
            "take_profit": str(intent.take_profit) if intent is not None and intent.take_profit is not None else None,
            "max_hold_minutes": intent.max_hold_minutes if intent is not None else None,
            "last_fill_order_id": order.id,
            "last_fill_broker_order_id": order.broker_order_id,
            "last_commission": str(commission),
        }
