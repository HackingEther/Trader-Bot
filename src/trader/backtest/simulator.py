"""Strategy and risk engine simulation on historical data."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
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
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold_minutes = max_hold_minutes
        self.exit_price: Decimal | None = None
        self.exit_time: datetime | None = None
        self.pnl: float = 0.0

    def check_exit(self, bar: BarEvent, current_time: datetime) -> bool:
        """Check if position should be closed based on stops, targets, or time."""
        price = bar.close

        if self.stop_loss:
            if self.side == "buy" and price <= self.stop_loss:
                self.exit_price = self.stop_loss
                self.exit_time = current_time
                return True
            if self.side == "sell" and price >= self.stop_loss:
                self.exit_price = self.stop_loss
                self.exit_time = current_time
                return True

        if self.take_profit:
            if self.side == "buy" and price >= self.take_profit:
                self.exit_price = self.take_profit
                self.exit_time = current_time
                return True
            if self.side == "sell" and price <= self.take_profit:
                self.exit_price = self.take_profit
                self.exit_time = current_time
                return True

        hold_minutes = (current_time - self.entry_time).total_seconds() / 60
        if hold_minutes >= self.max_hold_minutes:
            self.exit_price = price
            self.exit_time = current_time
            return True

        return False

    def compute_pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        if self.side == "buy":
            self.pnl = float(self.exit_price - self.entry_price) * self.qty
        else:
            self.pnl = float(self.entry_price - self.exit_price) * self.qty
        return self.pnl

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
    ) -> None:
        self._symbols = symbols
        self._slippage = slippage or SlippageModel()
        self._commission = commission or CommissionModel()
        self._initial_capital = initial_capital

        sc = strategy_config or {}
        rc = risk_config or {}

        self._universe = SymbolUniverse(symbols)
        self._sizer = PositionSizer(
            max_position_value=sc.get("max_position_value", 10000.0),
            risk_per_trade_pct=sc.get("risk_per_trade_pct", 0.01),
        )
        self._strategy = StrategyEngine(
            universe=self._universe,
            sizer=self._sizer,
            min_confidence=sc.get("min_confidence", 0.55),
            min_expected_move_bps=sc.get("min_expected_move_bps", 10.0),
        )
        self._risk = RiskEngine(
            max_daily_loss=rc.get("max_daily_loss", 1000.0),
            max_loss_per_trade=rc.get("max_loss_per_trade", 200.0),
            max_notional=rc.get("max_notional", 50000.0),
            max_positions=rc.get("max_positions", 10),
        )
        self._feature_engine = FeatureEngine()
        self._ensemble = ensemble or EnsemblePipeline.create_default()

        self._positions: list[SimulatedPosition] = []
        self._closed_trades: list[SimulatedPosition] = []
        self._equity_curve: list[float] = [initial_capital]
        self._cash = initial_capital

    def run(self, bars_by_symbol: dict[str, list[dict]]) -> dict:
        """Run backtest simulation on historical bars.

        Args:
            bars_by_symbol: Dict mapping symbol -> list of bar dicts
                            (sorted by timestamp ascending).
        """
        for symbol in self._symbols:
            symbol_bars = bars_by_symbol.get(symbol, [])
            if not symbol_bars:
                continue
            self._feature_engine.add_bars_bulk(symbol, symbol_bars)

        all_bars: list[tuple[str, dict]] = []
        for symbol, bars in bars_by_symbol.items():
            for bar in bars:
                all_bars.append((symbol, bar))

        all_bars.sort(key=lambda x: x[1].get("timestamp", ""))

        for symbol, bar_data in all_bars:
            bar = BarEvent(**bar_data) if not isinstance(bar_data, BarEvent) else bar_data
            ts = bar.timestamp
            self._feature_engine.add_bar(bar)

            open_for_symbol = [p for p in self._positions if p.symbol == symbol]
            for pos in open_for_symbol:
                if pos.check_exit(bar, ts):
                    raw_pnl = pos.compute_pnl()
                    comm = float(self._commission.calculate(pos.qty))
                    pos.pnl = raw_pnl - comm
                    self._cash += float(pos.exit_price or 0) * pos.qty + pos.pnl  # type: ignore[operator]
                    self._positions.remove(pos)
                    self._closed_trades.append(pos)

            has_position = any(p.symbol == symbol for p in self._positions)
            features = self._feature_engine.compute_features(symbol, ts)
            prediction = self._ensemble.predict(symbol, features, ts)

            intent = self._strategy.evaluate(
                prediction=prediction,
                current_price=bar.close,
                account_equity=Decimal(str(self._cash)),
                has_open_position=has_position,
            )

            if intent:
                daily_pnl = sum(t.pnl for t in self._closed_trades)
                context = RiskContext(
                    daily_realized_pnl=Decimal(str(daily_pnl)),
                    open_position_count=len(self._positions),
                    entry_price=bar.close,
                    last_data_time=ts,
                )
                decision = self._risk.evaluate(intent, context)

                if decision.approved:
                    fill_price = self._slippage.apply(bar.close, intent.side, intent.qty)
                    pos = SimulatedPosition(
                        symbol=symbol,
                        side=intent.side,
                        qty=intent.qty,
                        entry_price=fill_price,
                        entry_time=ts,
                        stop_loss=intent.stop_loss,
                        take_profit=intent.take_profit,
                        max_hold_minutes=intent.max_hold_minutes,
                    )
                    self._positions.append(pos)
                    self._cash -= float(fill_price) * intent.qty

            unrealized = sum(
                float(bar.close - p.entry_price) * p.qty if p.side == "buy"
                else float(p.entry_price - bar.close) * p.qty
                for p in self._positions
                if p.symbol == symbol
            )
            equity = self._cash + sum(
                float(p.entry_price) * p.qty for p in self._positions
            ) + unrealized
            self._equity_curve.append(equity)

        return self.get_results()

    def get_results(self) -> dict:
        """Get backtest results summary."""
        from trader.backtest.metrics import compute_metrics

        trade_pnls = [t.pnl for t in self._closed_trades]
        hold_mins = [t.hold_minutes for t in self._closed_trades]
        metrics = compute_metrics(trade_pnls, hold_mins, self._equity_curve, self._initial_capital)
        by_symbol: dict[str, dict] = {}
        for symbol in self._symbols:
            symbol_trades = [t for t in self._closed_trades if t.symbol == symbol]
            symbol_metrics = compute_metrics(
                [t.pnl for t in symbol_trades],
                [t.hold_minutes for t in symbol_trades],
                self._equity_curve,
                self._initial_capital,
            )
            by_symbol[symbol] = {
                "total_trades": symbol_metrics.total_trades,
                "total_pnl": symbol_metrics.total_pnl,
                "win_rate": symbol_metrics.win_rate,
                "profit_factor": symbol_metrics.profit_factor,
            }

        return {
            "total_trades": metrics.total_trades,
            "win_count": metrics.win_count,
            "loss_count": metrics.loss_count,
            "total_pnl": metrics.total_pnl,
            "win_rate": metrics.win_rate,
            "expectancy": metrics.expectancy,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "profit_factor": metrics.profit_factor,
            "avg_hold_minutes": metrics.avg_hold_minutes,
            "turnover": metrics.turnover,
            "equity_curve_length": len(self._equity_curve),
            "by_symbol": by_symbol,
        }
