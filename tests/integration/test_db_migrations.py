"""Integration tests for database migrations (schema validation only)."""

from __future__ import annotations

import pytest

from trader.db.base import Base
from trader.db.models import (
    BacktestRun,
    ConfigVersion,
    FeatureSnapshot,
    Fill,
    MarketBar,
    ModelPredictionRecord,
    ModelVersion,
    Order,
    PnlSnapshot,
    Position,
    RiskDecisionRecord,
    Symbol,
    SystemEvent,
    TradeIntent,
    TrainingRun,
)


class TestModelsExist:
    """Verify all ORM models are importable and have correct table names."""

    def test_all_models_importable(self) -> None:
        models = [
            BacktestRun, ConfigVersion, FeatureSnapshot, Fill, MarketBar,
            ModelPredictionRecord, ModelVersion, Order, PnlSnapshot, Position,
            RiskDecisionRecord, Symbol, SystemEvent, TradeIntent, TrainingRun,
        ]
        assert len(models) == 15

    def test_table_names(self) -> None:
        assert Symbol.__tablename__ == "symbols"
        assert MarketBar.__tablename__ == "market_bars"
        assert FeatureSnapshot.__tablename__ == "feature_snapshots"
        assert ModelVersion.__tablename__ == "model_versions"
        assert ModelPredictionRecord.__tablename__ == "model_predictions"
        assert TradeIntent.__tablename__ == "trade_intents"
        assert RiskDecisionRecord.__tablename__ == "risk_decisions"
        assert Order.__tablename__ == "orders"
        assert Fill.__tablename__ == "fills"
        assert Position.__tablename__ == "positions"
        assert PnlSnapshot.__tablename__ == "pnl_snapshots"
        assert SystemEvent.__tablename__ == "system_events"
        assert ConfigVersion.__tablename__ == "config_versions"
        assert TrainingRun.__tablename__ == "training_runs"
        assert BacktestRun.__tablename__ == "backtest_runs"

    def test_base_metadata_has_tables(self) -> None:
        table_names = set(Base.metadata.tables.keys())
        expected = {
            "symbols", "market_bars", "feature_snapshots", "model_versions",
            "model_predictions", "trade_intents", "risk_decisions", "orders",
            "fills", "positions", "pnl_snapshots", "system_events",
            "config_versions", "training_runs", "backtest_runs",
        }
        assert expected.issubset(table_names)
