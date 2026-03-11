"""Custom exception hierarchy for the trading platform."""


class TraderError(Exception):
    """Base exception for all trader errors."""


class ConfigurationError(TraderError):
    """Raised when configuration is invalid or missing."""


class BrokerError(TraderError):
    """Raised when broker operations fail."""


class BrokerConnectionError(BrokerError):
    """Raised when broker connection fails."""


class OrderSubmissionError(BrokerError):
    """Raised when order submission fails."""


class OrderCancellationError(BrokerError):
    """Raised when order cancellation fails."""


class DuplicateOrderError(BrokerError):
    """Raised when a duplicate order is detected via idempotency check."""


class MarketDataError(TraderError):
    """Raised when market data operations fail."""


class MarketDataConnectionError(MarketDataError):
    """Raised when market data connection fails."""


class StaleDataError(MarketDataError):
    """Raised when market data is stale beyond threshold."""


class FeatureComputationError(TraderError):
    """Raised when feature computation fails."""


class ModelError(TraderError):
    """Raised when model inference or loading fails."""


class ModelNotFoundError(ModelError):
    """Raised when a required model is not found in registry."""


class StrategyError(TraderError):
    """Raised when strategy engine encounters an error."""


class RiskError(TraderError):
    """Raised when risk engine encounters an error."""


class RiskRejectionError(RiskError):
    """Raised when risk engine rejects a trade intent."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__(f"Trade rejected: {'; '.join(reasons)}")


class ExecutionError(TraderError):
    """Raised when execution engine encounters an error."""


class ReconciliationError(TraderError):
    """Raised when position reconciliation finds discrepancies."""


class KillSwitchActiveError(TraderError):
    """Raised when kill switch is active and trading is halted."""


class BacktestError(TraderError):
    """Raised when backtesting encounters an error."""
