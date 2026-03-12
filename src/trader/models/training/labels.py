"""Tradable outcome label construction with cost-aware simulation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from trader.core.events import BarEvent


class TradeOutcome(str, Enum):
    """Result of simulating a trade within the horizon."""

    WIN = "win"  # Target hit before stop
    LOSS = "loss"  # Stop hit before target
    TIMEOUT = "timeout"  # Max hold reached, closed at bar


@dataclass
class CostParams:
    """Cost assumptions for label construction. All in bps."""

    spread_bps: float = 5.0  # Half-spread per side (round-trip = 2 * this)
    slippage_entry_bps: float = 5.0
    slippage_exit_bps: float = 5.0

    @property
    def total_round_trip_bps(self) -> float:
        """Total cost to enter and exit a trade in bps."""
        return self.spread_bps * 2 + self.slippage_entry_bps + self.slippage_exit_bps


def simulate_trade_outcome(
    bars: list[BarEvent],
    decision_bar_index: int,
    side: str,
    entry_price: Decimal,
    stop_loss: Decimal,
    take_profit: Decimal,
    max_hold_bars: int,
) -> tuple[TradeOutcome, Decimal]:
    """Simulate a trade from decision bar, filling at next bar open.

    Aligns with backtest: signal on bar N, fill at bar N+1 open.
    Stop is checked before target (conservative, matches SimulatedPosition).

    Args:
        bars: Sorted list of bar events.
        decision_bar_index: Index of bar where decision is made.
        side: "buy" or "sell".
        entry_price: Fill price (bar N+1 open).
        stop_loss: Stop loss price.
        take_profit: Take profit price.
        max_hold_bars: Maximum bars to hold.

    Returns:
        (outcome, exit_price) - WIN/LOSS/TIMEOUT and price at exit.
    """
    entry_bar_index = decision_bar_index + 1
    if entry_bar_index >= len(bars):
        return TradeOutcome.TIMEOUT, entry_price

    entry_bar = bars[entry_bar_index]
    entry_time = entry_bar.timestamp

    for i in range(1, max_hold_bars + 1):
        bar_idx = entry_bar_index + i
        if bar_idx >= len(bars):
            break
        bar = bars[bar_idx]
        ts = bar.timestamp

        # Stop checked first (matches SimulatedPosition.maybe_exit)
        if side == "buy":
            if bar.low <= stop_loss:
                return TradeOutcome.LOSS, stop_loss
            if bar.high >= take_profit:
                return TradeOutcome.WIN, take_profit
        else:
            if bar.high >= stop_loss:
                return TradeOutcome.LOSS, stop_loss
            if bar.low <= take_profit:
                return TradeOutcome.WIN, take_profit

        if i >= max_hold_bars:
            return TradeOutcome.TIMEOUT, bar.close

    last_bar = bars[min(entry_bar_index + max_hold_bars, len(bars) - 1)]
    return TradeOutcome.TIMEOUT, last_bar.close


def compute_net_pnl_bps(
    side: str,
    entry_price: Decimal,
    exit_price: Decimal,
    costs: CostParams,
) -> float:
    """Compute net PnL in bps after round-trip costs."""
    if entry_price <= 0:
        return 0.0
    raw_bps = float((exit_price - entry_price) / entry_price * Decimal("10000"))
    if side == "sell":
        raw_bps = -raw_bps
    return raw_bps - costs.total_round_trip_bps


def compute_stop_target(
    price: Decimal,
    side: str,
    volatility: float,
    atr_multiple: float = 1.5,
    rr_ratio: float = 2.0,
) -> tuple[Decimal, Decimal]:
    """Compute stop and target prices from volatility. Matches PositionSizer logic."""
    vol = max(0.001, min(volatility, 0.5))
    stop_distance = float(price) * vol * atr_multiple
    tp_distance = stop_distance * rr_ratio
    if side == "buy":
        stop = Decimal(str(round(float(price) - stop_distance, 2)))
        target = Decimal(str(round(float(price) + tp_distance, 2)))
    else:
        stop = Decimal(str(round(float(price) + stop_distance, 2)))
        target = Decimal(str(round(float(price) - tp_distance, 2)))
    return stop, target
