"""Backtest replay engine for historical bar data."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from trader.backtest.simulator import BacktestSimulator
from trader.backtest.slippage import CommissionModel, SlippageModel
from trader.db.models.backtest_run import BacktestRun
from trader.services.model_loader import ChampionModelLoader

logger = structlog.get_logger(__name__)


class BacktestEngine:
    """High-level backtest engine that manages runs and persistence."""

    def __init__(self, session: AsyncSession | None = None) -> None:
        self._session = session

    async def run(
        self,
        name: str,
        symbols: list[str],
        bars_by_symbol: dict[str, list[dict]],
        start_date: str,
        end_date: str,
        strategy_config: dict | None = None,
        risk_config: dict | None = None,
        slippage_config: dict | None = None,
        initial_capital: float = 100000.0,
        use_champion_models: bool = True,
    ) -> dict:
        """Execute a backtest and optionally persist results.

        Args:
            name: Human-readable name for this backtest run.
            symbols: List of symbols to trade.
            bars_by_symbol: Historical bar data keyed by symbol.
            start_date: Backtest start date (YYYY-MM-DD).
            end_date: Backtest end date (YYYY-MM-DD).
            strategy_config: Optional strategy parameter overrides.
            risk_config: Optional risk parameter overrides.
            slippage_config: Optional slippage/commission parameters.
            initial_capital: Starting capital.

        Returns:
            Dict of backtest metrics.
        """
        sc = slippage_config or {}
        slippage = SlippageModel(
            fixed_bps=sc.get("fixed_bps", 5.0),
            volume_impact_bps=sc.get("volume_impact_bps", 0.0),
        )
        commission = CommissionModel(
            per_share=sc.get("per_share", 0.0),
            per_order=sc.get("per_order", 0.0),
        )

        now = datetime.now(timezone.utc)

        bt_run: BacktestRun | None = None
        if self._session:
            bt_run = BacktestRun(
                name=name,
                status="running",
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
                strategy_config=strategy_config or {},
                risk_config=risk_config or {},
                slippage_config=slippage_config or {},
                started_at=now,
            )
            self._session.add(bt_run)
            await self._session.flush()

        logger.info(
            "backtest_started",
            name=name,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

        try:
            ensemble = None
            if use_champion_models and self._session is not None:
                ensemble = await ChampionModelLoader(require_champions=False).load_ensemble(
                    session=self._session
                )
            simulator = BacktestSimulator(
                symbols=symbols,
                strategy_config=strategy_config,
                risk_config=risk_config,
                slippage=slippage,
                commission=commission,
                initial_capital=initial_capital,
                ensemble=ensemble,
            )
            results = simulator.run(bars_by_symbol)

            if bt_run and self._session:
                bt_run.status = "completed"
                bt_run.completed_at = datetime.now(timezone.utc)
                bt_run.total_trades = results["total_trades"]
                bt_run.win_count = results["win_count"]
                bt_run.loss_count = results["loss_count"]
                bt_run.total_pnl = results["total_pnl"]
                bt_run.win_rate = results["win_rate"]
                bt_run.expectancy = results["expectancy"]
                bt_run.sharpe_ratio = results["sharpe_ratio"]
                bt_run.max_drawdown = results["max_drawdown"]
                bt_run.profit_factor = results["profit_factor"]
                bt_run.avg_hold_minutes = results["avg_hold_minutes"]
                bt_run.turnover = results["turnover"]
                bt_run.detailed_metrics = results
                await self._session.flush()

            logger.info("backtest_completed", name=name, **results)
            return results

        except Exception as e:
            if bt_run and self._session:
                bt_run.status = "failed"
                bt_run.completed_at = datetime.now(timezone.utc)
                bt_run.notes = str(e)
                await self._session.flush()
            logger.error("backtest_failed", name=name, error=str(e))
            raise
