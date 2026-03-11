"""Shared orchestration for live/paper trading, reconciliation, and PnL."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from trader.config import Settings
from trader.core.events import ModelPrediction
from trader.core.exceptions import DuplicateOrderError, KillSwitchActiveError, OrderSubmissionError
from trader.core.time_utils import now_utc
from trader.db.models.model_prediction import ModelPredictionRecord
from trader.db.models.risk_decision import RiskDecisionRecord
from trader.db.models.trade_intent import TradeIntent
from trader.db.repositories.feature_snapshots import FeatureSnapshotRepository
from trader.db.repositories.market_bars import MarketBarRepository
from trader.db.repositories.model_predictions import ModelPredictionRepository
from trader.db.repositories.pnl_snapshots import PnlSnapshotRepository
from trader.db.repositories.positions import PositionRepository
from trader.db.repositories.risk_decisions import RiskDecisionRepository
from trader.db.repositories.trade_intents import TradeIntentRepository
from trader.execution.engine import ExecutionEngine
from trader.execution.reconciliation import PositionReconciler
from trader.features.engine import FeatureEngine
from trader.providers.broker.base import BrokerProvider
from trader.risk.engine import RiskContext, RiskEngine
from trader.strategy.engine import StrategyEngine, TradeIntentParams
from trader.strategy.sizing import PositionSizer
from trader.strategy.universe import SymbolUniverse

from trader.services.model_loader import ChampionModelLoader
from trader.services.system_state import SystemStateStore

logger = structlog.get_logger(__name__)


@dataclass
class SymbolCycleResult:
    symbol: str
    status: str
    reason: str = ""
    trade_intent_id: int | None = None
    order_id: int | None = None


class TradingCycleService:
    """Run the full feature -> prediction -> strategy -> risk -> execution flow."""

    def __init__(
        self,
        settings: Settings,
        session: AsyncSession,
        broker: BrokerProvider,
        model_loader: ChampionModelLoader | None = None,
        state_store: SystemStateStore | None = None,
    ) -> None:
        self._settings = settings
        self._session = session
        self._broker = broker
        self._model_loader = model_loader or ChampionModelLoader()
        self._state_store = state_store or SystemStateStore()

        self._market_bars = MarketBarRepository(session)
        self._feature_snapshots = FeatureSnapshotRepository(session)
        self._predictions = ModelPredictionRepository(session)
        self._trade_intents = TradeIntentRepository(session)
        self._risk_decisions = RiskDecisionRepository(session)
        self._positions = PositionRepository(session)
        self._pnl = PnlSnapshotRepository(session)

    async def run_cycle(self, symbols: list[str] | None = None) -> dict:
        symbols = symbols or self._settings.symbol_universe
        results = {
            "symbols_processed": 0,
            "predictions_persisted": 0,
            "intents_generated": 0,
            "orders_submitted": 0,
            "orders_rejected": 0,
            "skipped_no_bars": 0,
            "kill_switch_active": False,
            "symbols": [],
        }

        if await self._state_store.is_kill_switch_active():
            results["kill_switch_active"] = True
            logger.warning("trading_cycle_halted", reason="kill_switch_active")
            return results

        account = await self._broker.get_account()
        pipeline = await self._model_loader.load_ensemble(session=self._session)

        for symbol in symbols:
            try:
                outcome = await self._process_symbol(symbol, account_equity=account.equity, pipeline=pipeline)
            except Exception as exc:
                logger.error("trading_cycle_symbol_failed", symbol=symbol, error=str(exc))
                outcome = SymbolCycleResult(symbol=symbol, status="error", reason=str(exc))
            results["symbols"].append(outcome.__dict__)
            if outcome.status == "no_bars":
                results["skipped_no_bars"] += 1
                continue

            results["symbols_processed"] += 1
            if outcome.status in {"predicted", "approved", "rejected", "executed", "duplicate", "error"}:
                results["predictions_persisted"] += 1
            if outcome.trade_intent_id:
                results["intents_generated"] += 1
            if outcome.status == "executed":
                results["orders_submitted"] += 1
            if outcome.status == "rejected":
                results["orders_rejected"] += 1

        await self._session.commit()
        return results

    async def snapshot_pnl(self) -> dict:
        account = await self._broker.get_account()
        open_positions = await self._positions.get_open_positions()
        all_positions = await self._positions.get_all(limit=5000)

        realized = sum((p.realized_pnl for p in all_positions if p.status == "closed"), Decimal("0"))
        unrealized = account.equity - account.cash - realized
        total_exposure = sum((p.market_value or Decimal("0")) for p in open_positions)
        win_count = sum(1 for p in all_positions if p.status == "closed" and p.realized_pnl > 0)
        loss_count = sum(1 for p in all_positions if p.status == "closed" and p.realized_pnl <= 0)
        now = now_utc()

        snapshot = await self._pnl.create(
            timestamp=now,
            date_str=now.date().isoformat(),
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            total_pnl=realized + unrealized,
            open_positions=len(open_positions),
            total_exposure=total_exposure,
            win_count=win_count,
            loss_count=loss_count,
            trade_count=win_count + loss_count,
        )
        await self._session.commit()
        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "realized_pnl": float(snapshot.realized_pnl),
            "unrealized_pnl": float(snapshot.unrealized_pnl),
            "total_pnl": float(snapshot.total_pnl),
            "open_positions": snapshot.open_positions,
        }

    async def reconcile_positions(self, auto_fix: bool = True, halt_on_discrepancy: bool = True) -> dict:
        reconciler = PositionReconciler(self._broker, self._session)
        result = await reconciler.reconcile(auto_fix=auto_fix)
        material_discrepancy = bool(result.mismatched or result.broker_only or result.local_only)
        if halt_on_discrepancy and material_discrepancy:
            await self._state_store.set_kill_switch(True)
        await self._session.commit()
        return {
            "matched": len(result.matched),
            "mismatched": len(result.mismatched),
            "broker_only": result.broker_only,
            "local_only": result.local_only,
            "kill_switch_activated": halt_on_discrepancy and material_discrepancy,
        }

    async def _process_symbol(
        self,
        symbol: str,
        *,
        account_equity: Decimal,
        pipeline,
    ) -> SymbolCycleResult:
        bars = await self._market_bars.get_recent(symbol, limit=400)
        if len(bars) < 20:
            return SymbolCycleResult(symbol=symbol, status="no_bars", reason="insufficient_history")

        feature_engine = FeatureEngine(max_bars=400)
        feature_engine.add_bars_bulk(symbol, [self._bar_to_dict(bar) for bar in bars])

        latest_bar = bars[-1]
        timestamp = latest_bar.timestamp
        features = feature_engine.compute_features(symbol, timestamp)
        feature_snapshot = await self._feature_snapshots.create(
            symbol=symbol,
            timestamp=timestamp,
            features=features,
            feature_version=feature_engine.feature_version,
            bar_count=len(bars),
        )

        prediction = pipeline.predict(symbol, features, timestamp).model_copy(
            update={"feature_snapshot_id": feature_snapshot.id}
        )
        prediction_record = await self._persist_prediction(prediction)

        strategy = StrategyEngine(
            universe=SymbolUniverse(self._settings.symbol_universe),
            sizer=PositionSizer(),
            min_confidence=self._settings.min_confidence,
            min_expected_move_bps=self._settings.min_expected_move_bps,
        )

        has_broker_position = await self._broker.get_position(symbol) is not None
        intent = strategy.evaluate(
            prediction=prediction,
            current_price=latest_bar.close,
            account_equity=account_equity,
            has_open_position=has_broker_position,
        )
        if intent is None:
            return SymbolCycleResult(symbol=symbol, status="predicted", reason="strategy_filtered")

        intent.model_prediction_id = prediction_record.id
        intent_record = await self._persist_trade_intent(intent, timestamp=timestamp)

        risk_decision = await self._evaluate_and_persist_risk(intent, intent_record.id, latest_bar.close, timestamp)
        if not risk_decision.approved:
            await self._trade_intents.update_by_id(intent_record.id, status="rejected")
            return SymbolCycleResult(
                symbol=symbol,
                status="rejected",
                reason="; ".join(risk_decision.reasons),
                trade_intent_id=intent_record.id,
            )

        await self._trade_intents.update_by_id(intent_record.id, status="approved")
        try:
            execution = ExecutionEngine(
                broker=self._broker,
                session=self._session,
                state_store=self._state_store,
            )
            order = await execution.execute(intent, trade_intent_id=intent_record.id)
            await self._trade_intents.update_by_id(intent_record.id, status="executed")
            return SymbolCycleResult(
                symbol=symbol,
                status="executed",
                trade_intent_id=intent_record.id,
                order_id=order.id,
            )
        except DuplicateOrderError:
            await self._trade_intents.update_by_id(intent_record.id, status="cancelled")
            return SymbolCycleResult(
                symbol=symbol,
                status="duplicate",
                reason="duplicate_order",
                trade_intent_id=intent_record.id,
            )
        except (KillSwitchActiveError, OrderSubmissionError) as exc:
            await self._trade_intents.update_by_id(intent_record.id, status="cancelled")
            return SymbolCycleResult(
                symbol=symbol,
                status="rejected",
                reason=str(exc),
                trade_intent_id=intent_record.id,
            )

    async def _persist_prediction(self, prediction: ModelPrediction) -> ModelPredictionRecord:
        return await self._predictions.create(
            symbol=prediction.symbol,
            timestamp=prediction.timestamp,
            direction=prediction.direction,
            confidence=prediction.confidence,
            expected_move_bps=prediction.expected_move_bps,
            expected_holding_minutes=prediction.expected_holding_minutes,
            no_trade_score=prediction.no_trade_score,
            regime=prediction.regime,
            feature_snapshot_id=prediction.feature_snapshot_id,
            model_versions_used=prediction.model_versions,
        )

    async def _persist_trade_intent(self, intent: TradeIntentParams, *, timestamp: datetime) -> TradeIntent:
        return await self._trade_intents.create(
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            entry_order_type=intent.entry_order_type,
            limit_price=intent.limit_price,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            max_hold_minutes=intent.max_hold_minutes,
            strategy_tag=intent.strategy_tag,
            status="pending",
            model_prediction_id=intent.model_prediction_id,
            confidence=intent.confidence,
            expected_move_bps=intent.expected_move_bps,
            rationale=intent.rationale,
            timestamp=timestamp,
        )

    async def _evaluate_and_persist_risk(
        self,
        intent: TradeIntentParams,
        trade_intent_id: int,
        entry_price: Decimal,
        timestamp: datetime,
    ):
        if await self._state_store.is_kill_switch_active():
            decision = _InlineDecision(approved=False, reasons=["[kill_switch] Kill switch is active"], rule_results={"circuit_breaker": False})
            await self._persist_risk_decision(trade_intent_id, decision, timestamp)
            return decision

        latest_snapshot = await self._pnl.get_latest()
        open_positions = await self._positions.get_open_positions()
        symbol_position = await self._positions.get_open_by_symbol(intent.symbol)
        spread_bps = await self._state_store.get_spread_bps(intent.symbol)
        last_bar_time = await self._state_store.get_last_bar_timestamp(intent.symbol)

        risk = RiskEngine(
            max_daily_loss=self._settings.max_daily_loss_usd,
            max_loss_per_trade=self._settings.max_loss_per_trade_usd,
            max_notional=self._settings.max_notional_exposure_usd,
            max_positions=self._settings.max_concurrent_positions,
            max_per_symbol=self._settings.max_exposure_per_symbol_usd,
            cooldown_losses=self._settings.cooldown_after_losses,
            max_spread_bps=self._settings.spread_threshold_bps,
        )
        context = RiskContext(
            daily_realized_pnl=latest_snapshot.realized_pnl if latest_snapshot else Decimal("0"),
            current_exposure=sum((p.market_value or Decimal("0")) for p in open_positions),
            symbol_exposure=(symbol_position.market_value or Decimal("0")) if symbol_position else Decimal("0"),
            open_position_count=len(open_positions),
            consecutive_losses=self._count_consecutive_losses(),
            spread_bps=spread_bps or 0.0,
            last_data_time=last_bar_time,
            entry_price=entry_price,
        )
        decision = risk.evaluate(intent, context)
        await self._persist_risk_decision(trade_intent_id, decision, timestamp)
        return decision

    async def _persist_risk_decision(self, trade_intent_id: int, decision, timestamp: datetime) -> RiskDecisionRecord:
        return await self._risk_decisions.create(
            trade_intent_id=trade_intent_id,
            decision="approved" if decision.approved else "rejected",
            reasons=decision.reasons,
            rule_results=decision.rule_results,
            timestamp=timestamp,
        )

    def _count_consecutive_losses(self) -> int:
        return 0

    @staticmethod
    def _bar_to_dict(bar) -> dict:
        return {
            "symbol": bar.symbol,
            "timestamp": bar.timestamp,
            "interval": bar.interval,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "vwap": bar.vwap,
            "trade_count": bar.trade_count,
        }


class _InlineDecision:
    def __init__(self, *, approved: bool, reasons: list[str], rule_results: dict[str, bool]) -> None:
        self.approved = approved
        self.reasons = reasons
        self.rule_results = rule_results
