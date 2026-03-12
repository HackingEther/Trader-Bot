"""Risk engine - evaluate trade intents against all risk rules."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import structlog

from trader.risk.rules.circuit_breaker import CircuitBreakerRule
from trader.risk.rules.concurrent_positions import ConcurrentPositionsRule
from trader.risk.rules.cooldown import CooldownRule
from trader.risk.rules.daily_loss import DailyLossRule
from trader.risk.rules.market_hours import MarketHoursRule
from trader.risk.rules.notional_exposure import NotionalExposureRule
from trader.risk.rules.per_trade_loss import PerTradeLossRule
from trader.risk.rules.spread import SpreadRule
from trader.risk.rules.stale_data import StaleDataRule
from trader.risk.rules.symbol_exposure import SymbolExposureRule
from trader.strategy.engine import TradeIntentParams

logger = structlog.get_logger(__name__)


class RiskContext:
    """Contextual data needed by risk rules for evaluation."""

    def __init__(
        self,
        daily_realized_pnl: Decimal = Decimal("0"),
        current_exposure: Decimal = Decimal("0"),
        symbol_exposure: Decimal = Decimal("0"),
        open_position_count: int = 0,
        consecutive_losses: int = 0,
        spread_bps: float = 0.0,
        last_data_time: datetime | None = None,
        entry_price: Decimal = Decimal("0"),
    ) -> None:
        self.daily_realized_pnl = daily_realized_pnl
        self.current_exposure = current_exposure
        self.symbol_exposure = symbol_exposure
        self.open_position_count = open_position_count
        self.consecutive_losses = consecutive_losses
        self.spread_bps = spread_bps
        self.last_data_time = last_data_time
        self.entry_price = entry_price


class RiskDecision:
    """Result of risk evaluation."""

    def __init__(self, approved: bool, reasons: list[str], rule_results: dict[str, bool]) -> None:
        self.approved = approved
        self.reasons = reasons
        self.rule_results = rule_results
        self.timestamp = datetime.now(timezone.utc)

    def first_failed_rule(self) -> str | None:
        """Return the first rule name that failed, for funnel tracking."""
        for name, passed in self.rule_results.items():
            if not passed:
                return name
        return None


class RiskEngine:
    """Evaluates trade intents against configurable risk rules.

    Every trade intent must pass through the risk engine before execution.
    The engine runs all rules and returns an explicit approve/reject decision.
    """

    def __init__(
        self,
        max_daily_loss: float = 1000.0,
        max_loss_per_trade: float = 200.0,
        max_notional: float = 50000.0,
        max_positions: int = 10,
        max_per_symbol: float = 10000.0,
        cooldown_losses: int = 3,
        max_spread_bps: float = 50.0,
        max_data_age: float = 30.0,
        enforce_stale_data: bool = True,
        enforce_market_hours: bool = True,
    ) -> None:
        self._circuit_breaker = CircuitBreakerRule()
        self._rules = {
            "circuit_breaker": self._circuit_breaker,
            "daily_loss": DailyLossRule(max_daily_loss),
            "per_trade_loss": PerTradeLossRule(max_loss_per_trade),
            "notional_exposure": NotionalExposureRule(max_notional),
            "concurrent_positions": ConcurrentPositionsRule(max_positions),
            "symbol_exposure": SymbolExposureRule(max_per_symbol),
            "cooldown": CooldownRule(cooldown_losses),
            "spread": SpreadRule(max_spread_bps),
        }
        if enforce_stale_data:
            self._rules["stale_data"] = StaleDataRule(max_data_age)
        if enforce_market_hours:
            self._rules["market_hours"] = MarketHoursRule()

    @property
    def kill_switch(self) -> CircuitBreakerRule:
        return self._circuit_breaker

    def evaluate(self, intent: TradeIntentParams, context: RiskContext) -> RiskDecision:
        """Evaluate a trade intent against all risk rules.

        Returns a RiskDecision with approve/reject and reasons.
        """
        reasons: list[str] = []
        rule_results: dict[str, bool] = {}
        new_notional = context.entry_price * intent.qty

        for name, rule in self._rules.items():
            try:
                if name == "circuit_breaker":
                    passed, reason = rule.check()
                elif name == "daily_loss":
                    passed, reason = rule.check(daily_realized_pnl=context.daily_realized_pnl)
                elif name == "per_trade_loss":
                    passed, reason = rule.check(
                        qty=intent.qty,
                        entry_price=context.entry_price,
                        stop_loss=intent.stop_loss,
                        side=intent.side,
                    )
                elif name == "notional_exposure":
                    passed, reason = rule.check(
                        current_exposure=context.current_exposure,
                        new_notional=new_notional,
                    )
                elif name == "concurrent_positions":
                    passed, reason = rule.check(open_position_count=context.open_position_count)
                elif name == "symbol_exposure":
                    passed, reason = rule.check(
                        symbol_exposure=context.symbol_exposure,
                        new_notional=new_notional,
                    )
                elif name == "cooldown":
                    passed, reason = rule.check(consecutive_losses=context.consecutive_losses)
                elif name == "spread":
                    passed, reason = rule.check(spread_bps=context.spread_bps)
                elif name == "stale_data":
                    passed, reason = rule.check(last_data_time=context.last_data_time)
                elif name == "market_hours":
                    passed, reason = rule.check()
                else:
                    passed, reason = True, ""

                rule_results[name] = passed
                if not passed:
                    reasons.append(f"[{name}] {reason}")
            except Exception as e:
                rule_results[name] = False
                reasons.append(f"[{name}] Rule error: {e}")
                logger.error("risk_rule_error", rule=name, error=str(e))

        approved = len(reasons) == 0
        decision = RiskDecision(approved=approved, reasons=reasons, rule_results=rule_results)

        if approved:
            logger.info("risk_approved", symbol=intent.symbol, side=intent.side, qty=intent.qty)
        else:
            logger.warning(
                "risk_rejected",
                symbol=intent.symbol,
                side=intent.side,
                reasons=reasons,
            )

        return decision
