"""Strategy and risk engine simulation on historical data."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import structlog

from trader.backtest.slippage import CommissionModel, SlippageModel
from trader.core.events import BarEvent
from trader.features.engine import FeatureEngine
from trader.models.ensemble import EnsemblePipeline
from trader.risk.engine import RiskContext, RiskEngine
from trader.strategy.engine import StrategyEngine, TradeIntentParams
from trader.strategy.sizing import PositionSizer
from trader.strategy.universe import SymbolUniverse

logger = structlog.get_logger(__name__)


class SimulatedPosition:
    """A simulated open position during backtest."""

    def __init__(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: Decimal,
        entry_time: datetime,
        stop_loss: Decimal | None = None,
        take_profit: Decimal | None = None,
        max_hold_minutes: int = 60,
        entry_commission: float = 0.0,
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold_minutes = max_hold_minutes
        self.entry_commission = entry_commission
        self.exit_commission = 0.0
        self.exit_price: Decimal | None = None
        self.exit_time: datetime | None = None
        self.pnl: float = 0.0

    def maybe_exit(self, bar: BarEvent, current_time: datetime) -> Decimal | None:
        """Return the exit price if the position should be closed on this bar.

        Note: Stop/take-profit fills assume exact execution at the trigger price.
        In reality, gaps and slippage can cause worse fills.
        """
        if self.stop_loss:
            if self.side == "buy" and bar.low <= self.stop_loss:
                return self.stop_loss
            if self.side == "sell" and bar.high >= self.stop_loss:
                return self.stop_loss

        if self.take_profit:
            if self.side == "buy" and bar.high >= self.take_profit:
                return self.take_profit
            if self.side == "sell" and bar.low <= self.take_profit:
                return self.take_profit

        hold_minutes = (current_time - self.entry_time).total_seconds() / 60
        if hold_minutes >= self.max_hold_minutes:
            return bar.close

        return None

    def close(self, exit_price: Decimal, exit_time: datetime, exit_commission: float = 0.0) -> float:
        """Mark the position closed and compute realized PnL."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_commission = exit_commission
        raw_pnl = self.unrealized_pnl(exit_price)
        self.pnl = raw_pnl - self.entry_commission - self.exit_commission
        return self.pnl

    def unrealized_pnl(self, current_price: Decimal) -> float:
        if self.side == "buy":
            return float(current_price - self.entry_price) * self.qty
        return float(self.entry_price - current_price) * self.qty

    def market_exposure(self, current_price: Decimal) -> float:
        return abs(float(current_price) * self.qty)

    def equity_contribution(self, current_price: Decimal) -> float:
        signed_notional = float(current_price) * self.qty
        return signed_notional if self.side == "buy" else -signed_notional

    @property
    def hold_minutes(self) -> float:
        if self.exit_time and self.entry_time:
            return (self.exit_time - self.entry_time).total_seconds() / 60
        return 0.0


class BacktestSimulator:
    """Simulates the full trading pipeline on historical bar data."""

    def __init__(
        self,
        symbols: list[str],
        strategy_config: dict | None = None,
        risk_config: dict | None = None,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
        initial_capital: float = 100000.0,
        ensemble: EnsemblePipeline | None = None,
        spread_bps_override: float | None = 10.0,
    ) -> None:
        self._symbols = symbols
        self._slippage = slippage or SlippageModel()
        self._commission = commission or CommissionModel()
        self._initial_capital = initial_capital

        sc = strategy_config or {}
        rc = risk_config or {}

        self._min_history_bars = int(sc.get("min_history_bars", 20))
        self._universe = SymbolUniverse(symbols)
        self._sizer = PositionSizer(
            max_position_value=sc.get("max_position_value", 10000.0),
            risk_per_trade_pct=sc.get("risk_per_trade_pct", 0.01),
        )
        self._track_block_reasons = sc.get("track_block_reasons", False)
        self._strategy = StrategyEngine(
            universe=self._universe,
            sizer=self._sizer,
            min_confidence=sc.get("min_confidence", 0.55),
            min_expected_move_bps=sc.get("min_expected_move_bps", 10.0),
            min_relative_volume=sc.get("min_relative_volume", 0.0),
            max_spread_bps=sc.get("max_spread_bps"),
            max_no_trade_score=sc.get("max_no_trade_score", 0.5),
            track_block_reasons=self._track_block_reasons,
        )
        self._risk = RiskEngine(
            max_daily_loss=rc.get("max_daily_loss", 1000.0),
            max_loss_per_trade=rc.get("max_loss_per_trade", 200.0),
            max_notional=rc.get("max_notional", 50000.0),
            max_positions=rc.get("max_positions", 10),
            max_per_symbol=rc.get("max_per_symbol", 10000.0),
            cooldown_losses=rc.get("cooldown_losses", 3),
            max_spread_bps=rc.get("max_spread_bps", 50.0),
            max_data_age=rc.get("max_data_age", 30.0),
            enforce_stale_data=rc.get("enforce_stale_data", False),
            enforce_market_hours=rc.get("enforce_market_hours", False),
        )
        self._feature_engine = FeatureEngine()
        self._ensemble = ensemble or EnsemblePipeline.create_default()
        self._spread_bps_override = spread_bps_override

        self._positions: list[SimulatedPosition] = []
        self._closed_trades: list[SimulatedPosition] = []
        self._equity_curve: list[float] = [initial_capital]
        self._latest_prices: dict[str, Decimal] = {}
        self._latest_timestamps: dict[str, datetime] = {}
        self._cash = initial_capital
        self._pending_entries: dict[str, TradeIntentParams] = {}
        self._risk_rejected_count = 0

    def run(self, bars_by_symbol: dict[str, list[dict]]) -> dict:
        """Run backtest simulation on historical bars."""
        self._feature_engine.clear()
        self._positions = []
        self._closed_trades = []
        self._equity_curve = [self._initial_capital]
        self._latest_prices = {}
        self._latest_timestamps = {}
        self._cash = self._initial_capital
        self._pending_entries = {}
        self._risk_rejected_count = 0
        if self._track_block_reasons:
            self._strategy.reset_block_stats()

        all_bars = self._normalize_bars(bars_by_symbol)

        # Entry timing: signal on bar N uses bar N's data; fill is at bar N+1's open.
        # We never fill on the same bar as the decision (avoids same-bar optimism).
        for symbol, bar in all_bars:
            ts = bar.timestamp
            self._latest_prices[symbol] = bar.close
            self._latest_timestamps[symbol] = ts

            pending_entry = self._pending_entries.pop(symbol, None)
            if pending_entry is not None:
                fill_price = self._slippage.apply(bar.open, pending_entry.side, pending_entry.qty)
                entry_commission = float(self._commission.calculate(pending_entry.qty))
                pos = SimulatedPosition(
                    symbol=symbol,
                    side=pending_entry.side,
                    qty=pending_entry.qty,
                    entry_price=fill_price,
                    entry_time=ts,
                    stop_loss=pending_entry.stop_loss,
                    take_profit=pending_entry.take_profit,
                    max_hold_minutes=pending_entry.max_hold_minutes,
                    entry_commission=entry_commission,
                )
                self._positions.append(pos)
                if pending_entry.side == "buy":
                    self._cash -= float(fill_price) * pending_entry.qty + entry_commission
                else:
                    self._cash += float(fill_price) * pending_entry.qty - entry_commission

            open_for_symbol = [p for p in self._positions if p.symbol == symbol]
            for pos in open_for_symbol:
                exit_price = pos.maybe_exit(bar, ts)
                if exit_price is None:
                    continue
                exit_price = self._slippage.apply(
                    exit_price, "sell" if pos.side == "buy" else "buy", pos.qty
                )
                exit_commission = float(self._commission.calculate(pos.qty))
                pos.close(exit_price=exit_price, exit_time=ts, exit_commission=exit_commission)
                if pos.side == "buy":
                    self._cash += float(exit_price) * pos.qty - exit_commission
                else:
                    self._cash -= float(exit_price) * pos.qty + exit_commission
                self._positions.remove(pos)
                self._closed_trades.append(pos)

            self._feature_engine.add_bar(bar)
            if self._feature_engine.get_bar_count(symbol) < self._min_history_bars:
                self._equity_curve.append(self._mark_to_market_equity())
                continue

            has_position = any(p.symbol == symbol for p in self._positions)
            features = self._feature_engine.compute_features(
                symbol, ts, spread_bps=self._spread_bps_override
            )
            prediction = self._ensemble.predict(symbol, features, ts)

            intent = self._strategy.evaluate(
                prediction=prediction,
                current_price=bar.close,
                account_equity=Decimal(str(self._mark_to_market_equity())),
                has_open_position=has_position,
                features=features,
                current_spread_bps=features.get(
                    "spread_bps", float(self._spread_bps_override or 0.0)
                ),
            )

            if intent:
                context = RiskContext(
                    daily_realized_pnl=Decimal(str(self._daily_realized_pnl(ts))),
                    current_exposure=Decimal(str(self._current_exposure())),
                    symbol_exposure=Decimal(str(self._symbol_exposure(symbol))),
                    open_position_count=len(self._positions),
                    consecutive_losses=self._consecutive_losses(),
                    spread_bps=features.get(
                        "spread_bps", float(self._spread_bps_override or 0.0)
                    ),
                    entry_price=bar.close,
                    last_data_time=ts,
                )
                decision = self._risk.evaluate(intent, context)

                if decision.approved:
                    self._pending_entries[symbol] = intent
                elif self._track_block_reasons:
                    self._risk_rejected_count += 1

            self._equity_curve.append(self._mark_to_market_equity())

        self._force_close_open_positions()
        self._equity_curve.append(self._mark_to_market_equity())
        return self.get_results()

    def get_results(self) -> dict:
        """Get backtest results summary."""
        from trader.backtest.metrics import compute_metrics_from_trades

        metrics = compute_metrics_from_trades(
            self._closed_trades,
            self._equity_curve,
            self._initial_capital,
        )
        by_symbol: dict[str, dict] = {}
        for symbol in self._symbols:
            symbol_trades = [t for t in self._closed_trades if t.symbol == symbol]
            symbol_metrics = compute_metrics_from_trades(
                symbol_trades,
                self._equity_curve,
                self._initial_capital,
            )
            by_symbol[symbol] = {
                "total_trades": symbol_metrics.total_trades,
                "long_trade_count": symbol_metrics.long_trade_count,
                "short_trade_count": symbol_metrics.short_trade_count,
                "total_pnl": symbol_metrics.total_pnl,
                "win_rate": symbol_metrics.win_rate,
                "loss_rate": symbol_metrics.loss_rate,
                "expectancy": symbol_metrics.expectancy,
                "average_net_pnl_bps": symbol_metrics.average_net_pnl_bps,
                "profit_factor": symbol_metrics.profit_factor,
            }

        result: dict = {
            "total_trades": metrics.total_trades,
            "long_trade_count": metrics.long_trade_count,
            "short_trade_count": metrics.short_trade_count,
            "win_count": metrics.win_count,
            "loss_count": metrics.loss_count,
            "total_pnl": metrics.total_pnl,
            "win_rate": metrics.win_rate,
            "loss_rate": metrics.loss_rate,
            "expectancy": metrics.expectancy,
            "average_net_pnl_bps": metrics.average_net_pnl_bps,
            "median_net_pnl_bps": metrics.median_net_pnl_bps,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "profit_factor": metrics.profit_factor,
            "avg_hold_minutes": metrics.avg_hold_minutes,
            "turnover": metrics.turnover,
            "equity_curve_length": len(self._equity_curve),
            "final_equity": self._equity_curve[-1] if self._equity_curve else self._initial_capital,
            "by_symbol": by_symbol,
        }
        if self._track_block_reasons:
            result["strategy_block_reasons"] = self._strategy.get_block_stats()
            result["risk_rejected_count"] = self._risk_rejected_count
        return result

    def _normalize_bars(self, bars_by_symbol: dict[str, list[dict]]) -> list[tuple[str, BarEvent]]:
        normalized: list[tuple[str, BarEvent]] = []
        for symbol, bars in bars_by_symbol.items():
            for bar in bars:
                event = bar if isinstance(bar, BarEvent) else BarEvent(**bar)
                normalized.append((symbol, event))
        normalized.sort(key=lambda item: item[1].timestamp)
        return normalized

    def _mark_to_market_equity(self) -> float:
        equity = self._cash
        for position in self._positions:
            current_price = self._latest_prices.get(position.symbol, position.entry_price)
            equity += position.equity_contribution(current_price)
        return equity

    def _current_exposure(self) -> float:
        return sum(
            position.market_exposure(self._latest_prices.get(position.symbol, position.entry_price))
            for position in self._positions
        )

    def _symbol_exposure(self, symbol: str) -> float:
        return sum(
            position.market_exposure(self._latest_prices.get(position.symbol, position.entry_price))
            for position in self._positions
            if position.symbol == symbol
        )

    def _daily_realized_pnl(self, current_time: datetime) -> float:
        current_date = current_time.astimezone(timezone.utc).date()
        return sum(
            trade.pnl
            for trade in self._closed_trades
            if trade.exit_time and trade.exit_time.astimezone(timezone.utc).date() == current_date
        )

    def _consecutive_losses(self) -> int:
        losses = 0
        for trade in reversed(self._closed_trades):
            if trade.pnl < 0:
                losses += 1
                continue
            break
        return losses

    def _force_close_open_positions(self) -> None:
        for position in list(self._positions):
            exit_price = self._latest_prices.get(position.symbol, position.entry_price)
            exit_price = self._slippage.apply(
                exit_price, "sell" if position.side == "buy" else "buy", position.qty
            )
            exit_commission = float(self._commission.calculate(position.qty))
            exit_time = self._latest_timestamps.get(position.symbol, position.entry_time)
            position.close(exit_price=exit_price, exit_time=exit_time, exit_commission=exit_commission)
            if position.side == "buy":
                self._cash += float(exit_price) * position.qty - exit_commission
            else:
                self._cash -= float(exit_price) * position.qty + exit_commission
            self._positions.remove(position)
            self._closed_trades.append(position)
