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
from trader.core.time_utils import now_utc, to_et
from trader.db.models.model_prediction import ModelPredictionRecord
from trader.db.models.risk_decision import RiskDecisionRecord
from trader.db.models.trade_intent import TradeIntent
from trader.db.repositories.feature_snapshots import FeatureSnapshotRepository
from trader.db.repositories.orders import OrderRepository
from trader.db.repositories.quote_snapshots import QuoteSnapshotRepository
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


@dataclass
class PreparedSignal:
    symbol: str
    latest_bar: object
    features: dict[str, float]
    prediction: ModelPrediction
    prediction_record: ModelPredictionRecord
    intent: TradeIntentParams
    intent_record: TradeIntent
    score: float


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
        self._quote_snapshots = QuoteSnapshotRepository(session)
        self._predictions = ModelPredictionRepository(session)
        self._trade_intents = TradeIntentRepository(session)
        self._risk_decisions = RiskDecisionRepository(session)
        self._positions = PositionRepository(session)
        self._orders = OrderRepository(session)
        self._pnl = PnlSnapshotRepository(session)

    async def run_cycle(self, symbols: list[str] | None = None) -> dict:
        symbols = symbols or self._settings.symbol_universe
        results = {
            "symbols_processed": 0,
            "predictions_persisted": 0,
            "intents_generated": 0,
            "orders_submitted": 0,
            "orders_sent": 0,
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
        candidates: list[PreparedSignal] = []

        for symbol in symbols:
            try:
                prepared = await self._prepare_symbol(symbol, account_equity=account.equity, pipeline=pipeline)
            except Exception as exc:
                logger.error("trading_cycle_symbol_failed", symbol=symbol, error=str(exc))
                self._record_result(results, SymbolCycleResult(symbol=symbol, status="error", reason=str(exc)))
                continue

            if isinstance(prepared, SymbolCycleResult):
                self._record_result(results, prepared)
            else:
                candidates.append(prepared)

        selected = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[
            : max(1, self._settings.max_signals_per_cycle)
        ]
        selected_ids = {candidate.intent_record.id for candidate in selected}

        open_positions = await self._positions.get_open_positions()
        open_orders = await self._orders.get_open_orders()
        outstanding_open_order_exposure = sum(
            self._order_notional(order)
            for order in open_orders
            if order.filled_qty < order.qty
        )
        projected_total_exposure = (
            sum((position.market_value or Decimal("0")) for position in open_positions)
            + outstanding_open_order_exposure
        )
        projected_open_count = len(open_positions) + self._open_order_slots(open_orders)
        projected_symbol_exposure = {
            position.symbol: (position.market_value or Decimal("0")) for position in open_positions
        }
        for order in open_orders:
            if order.filled_qty >= order.qty:
                continue
            projected_symbol_exposure[order.symbol] = projected_symbol_exposure.get(
                order.symbol, Decimal("0")
            ) + self._order_notional(order)

        for candidate in candidates:
            if candidate.intent_record.id not in selected_ids:
                await self._trade_intents.update_by_id(candidate.intent_record.id, status="cancelled")
                self._record_result(
                    results,
                    SymbolCycleResult(
                        symbol=candidate.symbol,
                        status="predicted",
                        reason="ranked_out",
                        trade_intent_id=candidate.intent_record.id,
                    ),
                )

        for candidate in selected:
            try:
                resized_intent = self._apply_projected_capacity(
                    intent=candidate.intent,
                    price=candidate.latest_bar.close,
                    projected_total_exposure=projected_total_exposure,
                    projected_symbol_exposure=projected_symbol_exposure.get(
                        candidate.symbol, Decimal("0")
                    ),
                )
                if resized_intent is None:
                    await self._trade_intents.update_by_id(candidate.intent_record.id, status="cancelled")
                    self._record_result(
                        results,
                        SymbolCycleResult(
                            symbol=candidate.symbol,
                            status="predicted",
                            reason="capacity_filtered",
                            trade_intent_id=candidate.intent_record.id,
                        ),
                    )
                    continue
                candidate.intent = resized_intent
                await self._trade_intents.update_by_id(candidate.intent_record.id, qty=resized_intent.qty)
                outcome = await self._execute_candidate(
                    candidate,
                    projected_total_exposure=projected_total_exposure,
                    projected_open_count=projected_open_count,
                    projected_symbol_exposure=projected_symbol_exposure.get(
                        candidate.symbol, Decimal("0")
                    ),
                )
            except Exception as exc:
                logger.error("trading_cycle_execution_failed", symbol=candidate.symbol, error=str(exc))
                outcome = SymbolCycleResult(
                    symbol=candidate.symbol,
                    status="error",
                    reason=str(exc),
                    trade_intent_id=candidate.intent_record.id,
                )

            if outcome.status in {"executed", "approved"}:
                projected_notional = candidate.latest_bar.close * candidate.intent.qty
                projected_total_exposure += projected_notional
                projected_open_count += 1
                projected_symbol_exposure[candidate.symbol] = (
                    projected_symbol_exposure.get(candidate.symbol, Decimal("0")) + projected_notional
                )

            self._record_result(results, outcome)

        await self._session.commit()
        return results

    async def snapshot_pnl(self) -> dict:
        account = await self._broker.get_account()
        open_positions = await self._positions.get_open_positions()
        all_positions = await self._positions.get_all(limit=5000)
        broker_positions = {position.symbol: position for position in await self._broker.get_positions()}

        realized = sum((p.realized_pnl for p in all_positions if p.status == "closed"), Decimal("0"))
        unrealized = sum(
            (
                broker_positions.get(position.symbol).unrealized_pnl
                if broker_positions.get(position.symbol) is not None
                else position.unrealized_pnl
            )
            for position in open_positions
        )
        total_exposure = sum(
            (
                broker_positions.get(position.symbol).market_value
                if broker_positions.get(position.symbol) is not None
                else (position.market_value or Decimal("0"))
            )
            for position in open_positions
        )
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

    async def reconcile_positions(self, auto_fix: bool = False, halt_on_discrepancy: bool = True) -> dict:
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

    async def _prepare_symbol(
        self,
        symbol: str,
        *,
        account_equity: Decimal,
        pipeline,
    ) -> SymbolCycleResult | PreparedSignal:
        bars = await self._market_bars.get_recent(symbol, limit=400)
        if len(bars) < 20:
            return SymbolCycleResult(symbol=symbol, status="no_bars", reason="insufficient_history")

        feature_engine = FeatureEngine(max_bars=400)
        feature_engine.add_bars_bulk(symbol, [self._bar_to_dict(bar) for bar in bars])

        latest_bar = bars[-1]
        timestamp = latest_bar.timestamp
        spread_bps = await self._state_store.get_spread_bps(symbol)
        quote = await self._state_store.get_last_quote(symbol)
        features = feature_engine.compute_features(symbol, timestamp, spread_bps=spread_bps)
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
            min_relative_volume=self._settings.min_relative_volume,
            max_spread_bps=self._settings.spread_threshold_bps,
            limit_entry_buffer_bps=self._settings.limit_entry_buffer_bps,
            use_marketable_limits=getattr(self._settings, "use_marketable_limits", True),
            marketable_limit_buffer_bps=getattr(self._settings, "marketable_limit_buffer_bps", 5.0),
        )

        has_broker_position = await self._broker.get_position(symbol) is not None
        intent = strategy.evaluate(
            prediction=prediction,
            current_price=latest_bar.close,
            account_equity=account_equity,
            has_open_position=has_broker_position,
            features=features,
            current_spread_bps=spread_bps,
            quote_at_decision=quote,
        )
        if intent is None:
            return SymbolCycleResult(symbol=symbol, status="predicted", reason="strategy_filtered")

        quote_at_decision = None
        decision_snapshot_id = None
        if quote:
            quote_at_decision = {
                "bid": quote.get("bid"),
                "ask": quote.get("ask"),
                "mid": quote.get("mid"),
                "spread_bps": quote.get("spread_bps"),
                "timestamp": quote.get("timestamp"),
                "last_bar_ts": latest_bar.timestamp.isoformat(),
            }
        intent.model_prediction_id = prediction_record.id
        intent.rationale = {
            **intent.rationale,
            "reference_price": str(latest_bar.close),
            "reference_timestamp": latest_bar.timestamp.isoformat(),
            "quote_at_decision": quote_at_decision,
        }
        intent_record = await self._persist_trade_intent(intent, timestamp=timestamp)
        if quote and quote.get("bid") is not None and quote.get("ask") is not None:
            bid = Decimal(str(quote["bid"]))
            ask = Decimal(str(quote["ask"]))
            mid = (bid + ask) / 2
            spread_bps = Decimal(str(quote.get("spread_bps", 0) or 0))
            snapshot = await self._quote_snapshots.create_snapshot(
                snapshot_type="decision",
                symbol=symbol,
                bid=bid,
                ask=ask,
                mid=mid,
                timestamp=timestamp,
                spread_bps=spread_bps,
                trade_intent_id=intent_record.id,
            )
            decision_snapshot_id = snapshot.id
            intent.rationale["decision_quote_snapshot_id"] = decision_snapshot_id
        score = self._score_candidate(prediction=prediction, features=features)
        return PreparedSignal(
            symbol=symbol,
            latest_bar=latest_bar,
            features=features,
            prediction=prediction,
            prediction_record=prediction_record,
            intent=intent,
            intent_record=intent_record,
            score=score,
        )

    async def _execute_candidate(
        self,
        candidate: PreparedSignal,
        *,
        projected_total_exposure: Decimal,
        projected_open_count: int,
        projected_symbol_exposure: Decimal,
    ) -> SymbolCycleResult:
        risk_decision = await self._evaluate_and_persist_risk(
            candidate.intent,
            candidate.intent_record.id,
            candidate.latest_bar.close,
            candidate.latest_bar.timestamp,
            projected_total_exposure=projected_total_exposure,
            projected_open_count=projected_open_count,
            projected_symbol_exposure=projected_symbol_exposure,
        )
        if not risk_decision.approved:
            await self._trade_intents.update_by_id(candidate.intent_record.id, status="rejected")
            return SymbolCycleResult(
                symbol=candidate.symbol,
                status="rejected",
                reason="; ".join(risk_decision.reasons),
                trade_intent_id=candidate.intent_record.id,
            )

        await self._trade_intents.update_by_id(candidate.intent_record.id, status="approved")
        try:
            execution = ExecutionEngine(
                broker=self._broker,
                session=self._session,
                state_store=self._state_store,
            )
            order = await execution.execute(candidate.intent, trade_intent_id=candidate.intent_record.id)
            intent_status = "executed" if order.status in {"partially_filled", "filled"} else "approved"
            await self._trade_intents.update_by_id(candidate.intent_record.id, status=intent_status)
            return SymbolCycleResult(
                symbol=candidate.symbol,
                status="executed" if intent_status == "executed" else "approved",
                trade_intent_id=candidate.intent_record.id,
                order_id=order.id,
            )
        except DuplicateOrderError:
            await self._trade_intents.update_by_id(candidate.intent_record.id, status="cancelled")
            return SymbolCycleResult(
                symbol=candidate.symbol,
                status="duplicate",
                reason="duplicate_order",
                trade_intent_id=candidate.intent_record.id,
            )
        except (KillSwitchActiveError, OrderSubmissionError) as exc:
            await self._trade_intents.update_by_id(candidate.intent_record.id, status="cancelled")
            return SymbolCycleResult(
                symbol=candidate.symbol,
                status="rejected",
                reason=str(exc),
                trade_intent_id=candidate.intent_record.id,
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
        *,
        projected_total_exposure: Decimal | None = None,
        projected_open_count: int | None = None,
        projected_symbol_exposure: Decimal | None = None,
    ):
        if await self._state_store.is_kill_switch_active():
            decision = _InlineDecision(
                approved=False,
                reasons=["[kill_switch] Kill switch is active"],
                rule_results={"circuit_breaker": False},
            )
            await self._persist_risk_decision(trade_intent_id, decision, timestamp)
            return decision

        all_positions = await self._positions.get_all(limit=5000)
        open_positions = [position for position in all_positions if position.status == "open"]
        open_orders = await self._orders.get_open_orders()
        last_bar_time = await self._state_store.get_last_bar_timestamp(intent.symbol)
        spread_bps = intent.rationale.get("spread_bps")
        if spread_bps is None:
            decision = _InlineDecision(
                approved=False,
                reasons=["[spread] No recent quote spread available"],
                rule_results={"spread": False},
            )
            await self._persist_risk_decision(trade_intent_id, decision, timestamp)
            return decision

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
            daily_realized_pnl=self._daily_realized_pnl(all_positions),
            current_exposure=projected_total_exposure
            if projected_total_exposure is not None
            else sum((p.market_value or Decimal("0")) for p in open_positions)
            + sum(self._order_notional(order) for order in open_orders if order.filled_qty < order.qty),
            symbol_exposure=projected_symbol_exposure
            if projected_symbol_exposure is not None
            else sum(
                (position.market_value or Decimal("0"))
                for position in open_positions
                if position.symbol == intent.symbol
            ),
            open_position_count=projected_open_count
            if projected_open_count is not None
            else len(open_positions) + self._open_order_slots(open_orders),
            consecutive_losses=self._count_consecutive_losses(all_positions),
            spread_bps=float(spread_bps),
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

    def _count_consecutive_losses(self, positions: list | None = None) -> int:
        all_positions = positions or []
        closed_positions = sorted(
            [position for position in all_positions if position.status == "closed" and position.closed_at],
            key=lambda position: position.closed_at or now_utc(),
            reverse=True,
        )
        losses = 0
        for position in closed_positions:
            if position.realized_pnl < 0:
                losses += 1
                continue
            break
        return losses

    def _daily_realized_pnl(self, positions: list) -> Decimal:
        session_date = to_et(now_utc()).date()
        realized = Decimal("0")
        for position in positions:
            if position.status != "closed" or position.closed_at is None:
                continue
            if to_et(position.closed_at).date() == session_date:
                realized += position.realized_pnl
        return realized

    def _score_candidate(self, *, prediction: ModelPrediction, features: dict[str, float]) -> float:
        relative_volume = max(0.1, float(features.get("relative_volume", 1.0)))
        spread_bps = max(0.0, float(features.get("spread_bps", 0.0)))
        momentum_bonus = abs(float(features.get("momentum_5m", 0.0))) + abs(
            float(features.get("momentum_15m", 0.0))
        )
        regime_multiplier = {
            "trending_up": 1.15,
            "trending_down": 1.15,
            "high_volatility": 1.05,
            "mean_reverting": 1.0,
            "low_volatility": 0.95,
        }.get(prediction.regime, 1.0)
        spread_penalty = 1.0 / (1.0 + spread_bps / 10.0)
        return (
            prediction.confidence
            * prediction.expected_move_bps
            * max(0.1, 1.0 - prediction.no_trade_score)
            * regime_multiplier
            * max(0.5, relative_volume)
            * max(1.0, momentum_bonus * 100.0)
            * spread_penalty
        )

    def _record_result(self, results: dict, outcome: SymbolCycleResult) -> None:
        results["symbols"].append(outcome.__dict__)
        if outcome.status == "no_bars":
            results["skipped_no_bars"] += 1
            return

        results["symbols_processed"] += 1
        if outcome.status in {"predicted", "approved", "rejected", "executed", "duplicate", "error"}:
            results["predictions_persisted"] += 1
        if outcome.trade_intent_id:
            results["intents_generated"] += 1
        if outcome.status == "executed":
            results["orders_submitted"] += 1
        if outcome.status in {"executed", "approved"}:
            results["orders_sent"] += 1
        if outcome.status == "rejected":
            results["orders_rejected"] += 1

    def _apply_projected_capacity(
        self,
        *,
        intent: TradeIntentParams,
        price: Decimal,
        projected_total_exposure: Decimal,
        projected_symbol_exposure: Decimal,
    ) -> TradeIntentParams | None:
        if price <= 0:
            return None
        total_remaining = max(
            Decimal("0"),
            Decimal(str(self._settings.max_notional_exposure_usd)) - projected_total_exposure,
        )
        symbol_remaining = max(
            Decimal("0"),
            Decimal(str(self._settings.max_exposure_per_symbol_usd)) - projected_symbol_exposure,
        )
        max_affordable_qty = int(min(total_remaining, symbol_remaining) / price)
        resized_qty = min(intent.qty, max_affordable_qty)
        if resized_qty <= 0:
            return None
        if resized_qty == intent.qty:
            return intent

        resized_rationale = {**intent.rationale, "resized_from_qty": intent.qty, "resized_to_qty": resized_qty}
        return TradeIntentParams(
            symbol=intent.symbol,
            side=intent.side,
            qty=resized_qty,
            entry_order_type=intent.entry_order_type,
            limit_price=intent.limit_price,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            max_hold_minutes=intent.max_hold_minutes,
            strategy_tag=intent.strategy_tag,
            model_prediction_id=intent.model_prediction_id,
            confidence=intent.confidence,
            expected_move_bps=intent.expected_move_bps,
            rationale=resized_rationale,
        )

    @staticmethod
    def _order_notional(order) -> Decimal:
        remaining_qty = max(0, int(order.qty) - int(order.filled_qty))
        if remaining_qty <= 0:
            return Decimal("0")
        rationale = getattr(order, "rationale", {}) or {}
        broker_metadata = getattr(order, "broker_metadata", {}) or {}
        raw_price = (
            order.limit_price
            or order.filled_avg_price
            or TradingCycleService._safe_decimal(rationale.get("reference_price"))
            or TradingCycleService._safe_decimal(broker_metadata.get("reference_price"))
            or Decimal("0")
        )
        reference_price = raw_price
        return reference_price * remaining_qty

    @staticmethod
    def _open_order_slots(open_orders: list) -> int:
        symbols = {order.symbol for order in open_orders if order.filled_qty < order.qty}
        return len(symbols)

    @staticmethod
    def _safe_decimal(value: object) -> Decimal | None:
        if value in (None, ""):
            return None
        try:
            return Decimal(str(value))
        except Exception:
            return None

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
