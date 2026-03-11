"""Position reconciliation between broker and local state."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from trader.db.models.position import Position
from trader.db.repositories.positions import PositionRepository
from trader.providers.broker.base import BrokerProvider

logger = structlog.get_logger(__name__)


class ReconciliationResult:
    """Result of a reconciliation check."""

    def __init__(self) -> None:
        self.matched: list[str] = []
        self.mismatched: list[dict] = []
        self.broker_only: list[str] = []
        self.local_only: list[str] = []
        self.timestamp = datetime.now(timezone.utc)

    @property
    def is_clean(self) -> bool:
        return not self.mismatched and not self.broker_only and not self.local_only


class PositionReconciler:
    """Reconciles broker positions with local position records."""

    def __init__(self, broker: BrokerProvider, session: AsyncSession) -> None:
        self._broker = broker
        self._session = session
        self._repo = PositionRepository(session)

    async def reconcile(self, auto_fix: bool = False) -> ReconciliationResult:
        """Compare broker positions with local positions.

        Args:
            auto_fix: If True, update local positions to match broker.
        """
        result = ReconciliationResult()

        broker_positions = await self._broker.get_positions()
        local_positions = await self._repo.get_open_positions()

        broker_map = {bp.symbol: bp for bp in broker_positions}
        local_map = {lp.symbol: lp for lp in local_positions}

        all_symbols = set(broker_map.keys()) | set(local_map.keys())

        for symbol in all_symbols:
            bp = broker_map.get(symbol)
            lp = local_map.get(symbol)

            if bp and lp:
                avg_entry_matches = bp.avg_entry_price == lp.avg_entry_price
                current_price_matches = bp.current_price == lp.current_price or lp.current_price is None
                if bp.qty == lp.qty and bp.side == lp.side and avg_entry_matches and current_price_matches:
                    result.matched.append(symbol)
                    if auto_fix and bp.current_price:
                        lp.current_price = bp.current_price
                        lp.market_value = bp.market_value
                        lp.unrealized_pnl = bp.unrealized_pnl
                else:
                    result.mismatched.append({
                        "symbol": symbol,
                        "broker_qty": bp.qty,
                        "broker_side": bp.side,
                        "local_qty": lp.qty,
                        "local_side": lp.side,
                        "broker_avg_entry_price": str(bp.avg_entry_price),
                        "local_avg_entry_price": str(lp.avg_entry_price),
                        "broker_current_price": str(bp.current_price),
                        "local_current_price": str(lp.current_price) if lp.current_price is not None else None,
                    })
                    if auto_fix:
                        lp.qty = bp.qty
                        lp.side = bp.side
                        lp.avg_entry_price = bp.avg_entry_price
                        lp.current_price = bp.current_price
                        lp.market_value = bp.market_value
                        lp.unrealized_pnl = bp.unrealized_pnl
                        logger.warning("position_auto_fixed", symbol=symbol)

            elif bp and not lp:
                result.broker_only.append(symbol)
                if auto_fix:
                    new_pos = Position(
                        symbol=symbol,
                        side=bp.side,
                        qty=bp.qty,
                        avg_entry_price=bp.avg_entry_price,
                        current_price=bp.current_price,
                        market_value=bp.market_value,
                        unrealized_pnl=bp.unrealized_pnl,
                        realized_pnl=Decimal("0"),
                        status="open",
                        opened_at=datetime.now(timezone.utc),
                        metadata_={"reconciliation_source": "broker_snapshot"},
                    )
                    self._session.add(new_pos)
                    logger.warning("position_created_from_broker", symbol=symbol)

            elif lp and not bp:
                result.local_only.append(symbol)
                if auto_fix:
                    lp.status = "closed"
                    lp.closed_at = datetime.now(timezone.utc)
                    logger.warning("position_closed_locally", symbol=symbol)

        if auto_fix:
            await self._session.flush()

        if result.is_clean:
            logger.info("reconciliation_clean", matched=len(result.matched))
        else:
            logger.warning(
                "reconciliation_discrepancies",
                mismatched=len(result.mismatched),
                broker_only=result.broker_only,
                local_only=result.local_only,
            )

        return result
