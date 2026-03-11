"""Backtest result metrics calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BacktestMetrics:
    """Summary metrics for a backtest run."""

    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    avg_hold_minutes: float = 0.0
    turnover: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0


def compute_metrics(
    trade_pnls: list[float],
    hold_minutes: list[float],
    equity_curve: list[float],
    initial_capital: float = 100000.0,
) -> BacktestMetrics:
    """Compute backtest summary metrics from trade results.

    Args:
        trade_pnls: List of P&L per trade.
        hold_minutes: List of hold duration per trade in minutes.
        equity_curve: Time series of portfolio equity.
        initial_capital: Starting capital.
    """
    if not trade_pnls:
        return BacktestMetrics()

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]

    total_trades = len(trade_pnls)
    win_count = len(wins)
    loss_count = len(losses)
    total_pnl = sum(trade_pnls)
    win_rate = win_count / total_trades if total_trades > 0 else 0.0
    expectancy = total_pnl / total_trades if total_trades > 0 else 0.0

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    largest_win = max(wins) if wins else 0.0
    largest_loss = min(losses) if losses else 0.0

    avg_hold = sum(hold_minutes) / len(hold_minutes) if hold_minutes else 0.0

    max_dd = _max_drawdown(equity_curve) if equity_curve else 0.0
    sharpe = _sharpe_ratio(trade_pnls, initial_capital) if len(trade_pnls) > 1 else 0.0

    turnover = sum(abs(p) for p in trade_pnls) / initial_capital if initial_capital > 0 else 0.0

    return BacktestMetrics(
        total_trades=total_trades,
        win_count=win_count,
        loss_count=loss_count,
        total_pnl=total_pnl,
        win_rate=win_rate,
        expectancy=expectancy,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        profit_factor=profit_factor,
        avg_hold_minutes=avg_hold,
        turnover=turnover,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
    )


def _max_drawdown(equity_curve: list[float]) -> float:
    """Compute maximum drawdown from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def _sharpe_ratio(trade_pnls: list[float], capital: float, annualize_factor: float = 252.0) -> float:
    """Compute approximate Sharpe-like ratio from trade P&Ls."""
    if len(trade_pnls) < 2:
        return 0.0
    returns = [p / capital for p in trade_pnls]
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret == 0:
        return 0.0
    return float(mean_ret / std_ret * math.sqrt(annualize_factor))
