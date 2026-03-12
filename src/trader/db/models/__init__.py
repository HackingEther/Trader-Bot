"""ORM model registry - import all models so Alembic can discover them."""

from trader.db.models.backtest_run import BacktestRun
from trader.db.models.config_version import ConfigVersion
from trader.db.models.execution_attribution import ExecutionAttribution
from trader.db.models.feature_snapshot import FeatureSnapshot
from trader.db.models.fill import Fill
from trader.db.models.market_bar import MarketBar
from trader.db.models.model_prediction import ModelPredictionRecord
from trader.db.models.model_version import ModelVersion
from trader.db.models.order import Order
from trader.db.models.pnl_snapshot import PnlSnapshot
from trader.db.models.quote_snapshot import QuoteSnapshot
from trader.db.models.position import Position
from trader.db.models.risk_decision import RiskDecisionRecord
from trader.db.models.symbol import Symbol
from trader.db.models.system_event import SystemEvent
from trader.db.models.trade_intent import TradeIntent
from trader.db.models.training_run import TrainingRun

__all__ = [
    "BacktestRun",
    "ConfigVersion",
    "FeatureSnapshot",
    "Fill",
    "MarketBar",
    "ModelPredictionRecord",
    "ModelVersion",
    "Order",
    "PnlSnapshot",
    "Position",
    "QuoteSnapshot",
    "RiskDecisionRecord",
    "Symbol",
    "SystemEvent",
    "TradeIntent",
    "TrainingRun",
]
