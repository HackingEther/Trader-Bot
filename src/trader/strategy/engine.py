"""Strategy engine converting model predictions to trade intents."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import structlog

from trader.core.events import ModelPrediction
from trader.strategy.sizing import PositionSizer
from trader.strategy.universe import SymbolUniverse

logger = structlog.get_logger(__name__)


class TradeIntentParams:
    """Parameters for a proposed trade intent."""

    def __init__(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_order_type: str = "market",
        limit_price: Decimal | None = None,
        stop_loss: Decimal | None = None,
        take_profit: Decimal | None = None,
        max_hold_minutes: int = 60,
        strategy_tag: str = "default",
        model_prediction_id: int | None = None,
        confidence: float = 0.0,
        expected_move_bps: float = 0.0,
        rationale: dict | None = None,
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.entry_order_type = entry_order_type
        self.limit_price = limit_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold_minutes = max_hold_minutes
        self.strategy_tag = strategy_tag
        self.model_prediction_id = model_prediction_id
        self.confidence = confidence
        self.expected_move_bps = expected_move_bps
        self.rationale = rationale or {}


class StrategyEngine:
    """Converts model predictions into trade intents."""

    def __init__(
        self,
        universe: SymbolUniverse,
        sizer: PositionSizer,
        min_confidence: float = 0.6,
        min_expected_move_bps: float = 15.0,
        max_no_trade_score: float = 0.5,
        default_max_hold_minutes: int = 60,
    ) -> None:
        self._universe = universe
        self._sizer = sizer
        self._min_confidence = min_confidence
        self._min_expected_move_bps = min_expected_move_bps
        self._max_no_trade_score = max_no_trade_score
        self._default_max_hold = default_max_hold_minutes

    def evaluate(
        self,
        prediction: ModelPrediction,
        current_price: Decimal,
        account_equity: Decimal = Decimal("100000"),
        has_open_position: bool = False,
    ) -> TradeIntentParams | None:
        """Evaluate a prediction and generate a trade intent if warranted.

        Returns None if no trade should be taken.
        """
        symbol = prediction.symbol

        if not self._universe.is_enabled(symbol):
            logger.debug("strategy_skip_disabled", symbol=symbol)
            return None

        if prediction.direction == "no_trade":
            logger.debug("strategy_skip_no_trade", symbol=symbol)
            return None

        if prediction.confidence < self._min_confidence:
            logger.debug(
                "strategy_skip_low_confidence",
                symbol=symbol,
                confidence=prediction.confidence,
                threshold=self._min_confidence,
            )
            return None

        if prediction.expected_move_bps < self._min_expected_move_bps:
            logger.debug("strategy_skip_low_move", symbol=symbol, move_bps=prediction.expected_move_bps)
            return None

        if prediction.no_trade_score > self._max_no_trade_score:
            logger.debug("strategy_skip_filter", symbol=symbol, no_trade_score=prediction.no_trade_score)
            return None

        if has_open_position:
            logger.debug("strategy_skip_existing_position", symbol=symbol)
            return None

        side = "buy" if prediction.direction == "long" else "sell"
        vol_estimate = max(0.005, prediction.expected_move_bps / 10000.0)
        qty = self._sizer.compute_qty(current_price, stop_distance_pct=vol_estimate, account_equity=account_equity)
        stop_loss = self._sizer.compute_stop_loss(current_price, side, volatility=vol_estimate)
        take_profit = self._sizer.compute_take_profit(current_price, side, volatility=vol_estimate)
        max_hold = int(prediction.expected_holding_minutes) if prediction.expected_holding_minutes > 0 else self._default_max_hold

        rationale = {
            "regime": prediction.regime,
            "direction": prediction.direction,
            "confidence": prediction.confidence,
            "expected_move_bps": prediction.expected_move_bps,
            "no_trade_score": prediction.no_trade_score,
            "model_versions": prediction.model_versions,
        }

        intent = TradeIntentParams(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_order_type="market",
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold_minutes=max_hold,
            strategy_tag="ensemble_v1",
            confidence=prediction.confidence,
            expected_move_bps=prediction.expected_move_bps,
            rationale=rationale,
        )

        logger.info(
            "trade_intent_generated",
            symbol=symbol,
            side=side,
            qty=qty,
            stop_loss=str(stop_loss),
            take_profit=str(take_profit),
        )

        return intent
