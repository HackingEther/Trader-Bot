"""Strategy engine converting model predictions to trade intents."""

from __future__ import annotations

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
        min_relative_volume: float = 0.8,
        max_spread_bps: float | None = None,
        limit_entry_buffer_bps: float = 5.0,
        max_no_trade_score: float = 0.5,
        default_max_hold_minutes: int = 60,
        track_block_reasons: bool = False,
        use_marketable_limits: bool = True,
        marketable_limit_buffer_bps: float = 5.0,
    ) -> None:
        self._universe = universe
        self._sizer = sizer
        self._min_confidence = min_confidence
        self._min_expected_move_bps = min_expected_move_bps
        self._min_relative_volume = min_relative_volume
        self._max_spread_bps = max_spread_bps
        self._limit_entry_buffer_bps = limit_entry_buffer_bps
        self._use_marketable_limits = use_marketable_limits
        self._marketable_limit_buffer_bps = marketable_limit_buffer_bps
        self._max_no_trade_score = max_no_trade_score
        self._default_max_hold = default_max_hold_minutes
        self._track_block_reasons = track_block_reasons
        self._block_counts: dict[str, int] = {}

    def _record_block(self, reason: str) -> None:
        if self._track_block_reasons:
            self._block_counts[reason] = self._block_counts.get(reason, 0) + 1

    def get_block_stats(self) -> dict[str, int]:
        """Return block reason counts when track_block_reasons was enabled."""
        return dict(self._block_counts)

    def reset_block_stats(self) -> None:
        """Clear block counts for a fresh run."""
        self._block_counts.clear()

    def evaluate(
        self,
        prediction: ModelPrediction,
        current_price: Decimal,
        account_equity: Decimal = Decimal("100000"),
        has_open_position: bool = False,
        features: dict[str, float] | None = None,
        current_spread_bps: float | None = None,
        quote_at_decision: dict | None = None,
    ) -> TradeIntentParams | None:
        """Evaluate a prediction and generate a trade intent if warranted.

        Returns None if no trade should be taken.
        """
        symbol = prediction.symbol
        features = features or {}
        relative_volume = float(features.get("relative_volume", 1.0))
        spread_bps = float(current_spread_bps) if current_spread_bps is not None else None
        minutes_since_open = float(features.get("minutes_since_open", 390.0))

        if not self._universe.is_enabled(symbol):
            logger.debug("strategy_skip_disabled", symbol=symbol)
            self._record_block("disabled")
            return None

        if relative_volume < self._min_relative_volume:
            logger.debug(
                "strategy_skip_low_relative_volume",
                symbol=symbol,
                relative_volume=relative_volume,
                threshold=self._min_relative_volume,
            )
            self._record_block("low_relative_volume")
            return None

        if self._max_spread_bps is not None and spread_bps is not None and spread_bps > self._max_spread_bps:
            logger.debug(
                "strategy_skip_wide_spread",
                symbol=symbol,
                spread_bps=spread_bps,
                threshold=self._max_spread_bps,
            )
            self._record_block("wide_spread")
            return None

        if prediction.direction == "no_trade":
            logger.debug("strategy_skip_no_trade", symbol=symbol)
            self._record_block("no_trade")
            return None

        min_confidence = self._dynamic_min_confidence(prediction.regime, minutes_since_open)
        if prediction.confidence < min_confidence:
            logger.debug(
                "strategy_skip_low_confidence",
                symbol=symbol,
                confidence=prediction.confidence,
                threshold=min_confidence,
            )
            self._record_block("low_confidence")
            return None

        min_expected_move_bps = self._dynamic_min_expected_move(prediction.regime, relative_volume, spread_bps)
        move_magnitude = abs(prediction.expected_move_bps)
        if move_magnitude < min_expected_move_bps:
            logger.debug(
                "strategy_skip_low_move",
                symbol=symbol,
                move_bps=prediction.expected_move_bps,
                threshold=min_expected_move_bps,
            )
            self._record_block("low_move")
            return None

        if prediction.no_trade_score > self._max_no_trade_score:
            logger.debug("strategy_skip_filter", symbol=symbol, no_trade_score=prediction.no_trade_score)
            self._record_block("filter")
            return None

        if has_open_position:
            logger.debug("strategy_skip_existing_position", symbol=symbol)
            self._record_block("existing_position")
            return None

        playbook = self._select_playbook(prediction=prediction, features=features, spread_bps=spread_bps)
        if playbook is None:
            logger.debug("strategy_skip_no_playbook", symbol=symbol, regime=prediction.regime)
            self._record_block("no_playbook")
            return None

        side = "buy" if prediction.direction == "long" else "sell"
        playbook_name = playbook["name"]
        volatility_multiplier = playbook["volatility_multiplier"]
        reward_ratio = playbook["reward_ratio"]
        hold_multiplier = playbook["hold_multiplier"]
        use_limit_entry = playbook["use_limit_entry"]

        vol_estimate = max(0.005, abs(prediction.expected_move_bps) / 10000.0) * volatility_multiplier
        qty = self._sizer.compute_qty(current_price, stop_distance_pct=vol_estimate, account_equity=account_equity)
        qty = max(1, int(qty * self._position_adjustment(relative_volume, spread_bps)))

        stop_loss = self._sizer.compute_stop_loss(current_price, side, volatility=vol_estimate)
        take_profit = self._sizer.compute_take_profit(
            current_price,
            side,
            volatility=vol_estimate,
            rr_ratio=reward_ratio,
        )
        predicted_hold = (
            int(prediction.expected_holding_minutes)
            if prediction.expected_holding_minutes > 0
            else self._default_max_hold
        )
        max_hold = max(10, int(predicted_hold * hold_multiplier))

        entry_order_type = "market"
        limit_price: Decimal | None = None
        if use_limit_entry:
            limit_price = self._build_limit_price_from_quote(
                quote_at_decision, side, spread_bps
            ) or self._build_limit_price(current_price, side, spread_bps)
            if limit_price is not None:
                entry_order_type = "limit"

        rationale = {
            "playbook": playbook_name,
            "regime": prediction.regime,
            "direction": prediction.direction,
            "confidence": prediction.confidence,
            "expected_move_bps": prediction.expected_move_bps,
            "no_trade_score": prediction.no_trade_score,
            "relative_volume": relative_volume,
            "spread_bps": spread_bps,
            "minutes_since_open": minutes_since_open,
            "dynamic_min_confidence": min_confidence,
            "dynamic_min_expected_move_bps": min_expected_move_bps,
            "model_versions": prediction.model_versions,
        }

        intent = TradeIntentParams(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_order_type=entry_order_type,
            limit_price=limit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold_minutes=max_hold,
            strategy_tag=playbook_name,
            confidence=prediction.confidence,
            expected_move_bps=prediction.expected_move_bps,
            rationale=rationale,
        )

        logger.info(
            "trade_intent_generated",
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=entry_order_type,
            stop_loss=str(stop_loss),
            take_profit=str(take_profit),
        )

        return intent

    def _dynamic_min_confidence(self, regime: str, minutes_since_open: float) -> float:
        threshold = self._min_confidence
        if regime in {"high_volatility", "mean_reverting"}:
            threshold += 0.03
        if minutes_since_open < 15 or minutes_since_open > 330:
            threshold += 0.03
        return min(threshold, 0.95)

    def _dynamic_min_expected_move(
        self,
        regime: str,
        relative_volume: float,
        spread_bps: float | None,
    ) -> float:
        threshold = self._min_expected_move_bps
        if regime == "high_volatility":
            threshold += 8.0
        elif regime.startswith("trending"):
            threshold += 2.0
        if relative_volume >= 1.5:
            threshold -= 2.0
        if spread_bps is not None:
            threshold = max(threshold, spread_bps * 1.5)
        return max(5.0, threshold)

    def _position_adjustment(self, relative_volume: float, spread_bps: float | None) -> float:
        spread_penalty = 1.0
        if spread_bps is not None and self._max_spread_bps:
            spread_penalty = max(0.4, 1.0 - min(spread_bps / max(self._max_spread_bps, 1.0), 0.6))
        volume_boost = min(1.25, max(0.75, relative_volume))
        return max(0.5, min(1.25, spread_penalty * volume_boost))

    def _build_limit_price_from_quote(
        self, quote: dict | None, side: str, spread_bps: float | None
    ) -> Decimal | None:
        """Build limit price from bid/ask when quote is available."""
        if not quote:
            return None
        bid = quote.get("bid")
        ask = quote.get("ask")
        if bid is None or ask is None or float(bid) <= 0 or float(ask) <= 0:
            return None
        if self._use_marketable_limits:
            buffer_bps = self._marketable_limit_buffer_bps
        else:
            buffer_bps = self._limit_entry_buffer_bps
            if spread_bps is not None:
                buffer_bps = max(buffer_bps, spread_bps * 0.75)
        buffer_pct = Decimal(str(buffer_bps)) / Decimal("10000")
        if side == "buy":
            price = Decimal(str(ask)) * (Decimal("1") + buffer_pct)
        else:
            price = Decimal(str(bid)) * (Decimal("1") - buffer_pct)
        return price.quantize(Decimal("0.01"))

    def _build_limit_price(self, current_price: Decimal, side: str, spread_bps: float | None) -> Decimal | None:
        if current_price <= 0:
            return None
        buffer_bps = self._limit_entry_buffer_bps
        if spread_bps is not None:
            buffer_bps = max(buffer_bps, spread_bps * 0.75)
        buffer_pct = Decimal(str(buffer_bps)) / Decimal("10000")
        if side == "buy":
            price = current_price * (Decimal("1") + buffer_pct)
        else:
            price = current_price * (Decimal("1") - buffer_pct)
        return price.quantize(Decimal("0.01"))

    def _select_playbook(
        self,
        *,
        prediction: ModelPrediction,
        features: dict[str, float],
        spread_bps: float | None,
    ) -> dict[str, float | bool | str] | None:
        minutes_since_open = float(features.get("minutes_since_open", 390.0))
        momentum_5m = float(features.get("momentum_5m", 0.0))
        momentum_15m = float(features.get("momentum_15m", 0.0))
        distance_from_vwap = float(features.get("distance_from_vwap", 0.0))
        relative_volume = float(features.get("relative_volume", 1.0))
        zscore_close = float(features.get("zscore_close_20", 0.0))
        orb_breakout_up = float(features.get("orb_breakout_up", 0.0))
        orb_breakout_down = float(features.get("orb_breakout_down", 0.0))

        if spread_bps is not None and self._max_spread_bps is not None and spread_bps > self._max_spread_bps:
            return None

        if (
            minutes_since_open <= 60
            and relative_volume >= max(1.2, self._min_relative_volume)
            and (
                (prediction.direction == "long" and orb_breakout_up > 0 and momentum_5m > 0 and distance_from_vwap >= 0)
                or (prediction.direction == "short" and orb_breakout_down > 0 and momentum_5m < 0 and distance_from_vwap <= 0)
            )
        ):
            return {
                "name": "orb_continuation",
                "volatility_multiplier": 1.2,
                "reward_ratio": 2.25,
                "hold_multiplier": 0.75,
                "use_limit_entry": True,
            }

        if (
            prediction.regime in {"trending_up", "trending_down", "high_volatility"}
            and relative_volume >= max(0.9, self._min_relative_volume)
            and (
                (prediction.direction == "long" and momentum_5m > 0 and momentum_15m > 0 and distance_from_vwap >= 0)
                or (prediction.direction == "short" and momentum_5m < 0 and momentum_15m < 0 and distance_from_vwap <= 0)
            )
        ):
            return {
                "name": "vwap_continuation",
                "volatility_multiplier": 1.0,
                "reward_ratio": 2.0,
                "hold_multiplier": 1.0,
                "use_limit_entry": True,
            }

        if (
            prediction.regime in {"mean_reverting", "low_volatility"}
            and minutes_since_open >= 30
            and relative_volume >= max(0.6, self._min_relative_volume * 0.75)
            and (
                (prediction.direction == "long" and zscore_close <= -1.5 and distance_from_vwap < 0)
                or (prediction.direction == "short" and zscore_close >= 1.5 and distance_from_vwap > 0)
            )
        ):
            return {
                "name": "vwap_reversion",
                "volatility_multiplier": 0.75,
                "reward_ratio": 1.5,
                "hold_multiplier": 0.5,
                "use_limit_entry": True,
            }

        return None
