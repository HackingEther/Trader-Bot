"""Initial schema with all tables.

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: str | None = None
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "symbols",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("name", sa.String(255), nullable=False, server_default=""),
        sa.Column("exchange", sa.String(20), nullable=False, server_default=""),
        sa.Column("asset_class", sa.String(20), nullable=False, server_default="us_equity"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("is_tradable", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("is_shortable", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticker"),
    )
    op.create_index("ix_symbols_ticker", "symbols", ["ticker"])

    op.create_table(
        "market_bars",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("interval", sa.String(10), nullable=False, server_default="1m"),
        sa.Column("open", sa.Numeric(12, 4), nullable=False),
        sa.Column("high", sa.Numeric(12, 4), nullable=False),
        sa.Column("low", sa.Numeric(12, 4), nullable=False),
        sa.Column("close", sa.Numeric(12, 4), nullable=False),
        sa.Column("volume", sa.Integer(), nullable=False),
        sa.Column("vwap", sa.Numeric(12, 4), nullable=True),
        sa.Column("trade_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_market_bars_symbol_ts", "market_bars", ["symbol", "timestamp"])
    op.create_index("ix_market_bars_symbol_interval_ts", "market_bars", ["symbol", "interval", "timestamp"])

    op.create_table(
        "feature_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("features", postgresql.JSONB(), nullable=False),
        sa.Column("feature_version", sa.String(20), nullable=False, server_default="v1"),
        sa.Column("bar_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feature_snapshots_symbol_ts", "feature_snapshots", ["symbol", "timestamp"])

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_type", sa.String(50), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("algorithm", sa.String(50), nullable=False, server_default="lightgbm"),
        sa.Column("hyperparameters", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("metrics", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("training_run_id", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_champion", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "model_predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("direction", sa.String(20), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("expected_move_bps", sa.Float(), nullable=False),
        sa.Column("expected_holding_minutes", sa.Float(), nullable=False),
        sa.Column("no_trade_score", sa.Float(), nullable=False),
        sa.Column("regime", sa.String(30), nullable=False),
        sa.Column("feature_snapshot_id", sa.Integer(), nullable=True),
        sa.Column("model_versions_used", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "trade_intents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("qty", sa.Integer(), nullable=False),
        sa.Column("entry_order_type", sa.String(20), nullable=False, server_default="market"),
        sa.Column("limit_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("stop_loss", sa.Numeric(12, 4), nullable=True),
        sa.Column("take_profit", sa.Numeric(12, 4), nullable=True),
        sa.Column("max_hold_minutes", sa.Integer(), nullable=False, server_default=sa.text("60")),
        sa.Column("strategy_tag", sa.String(50), nullable=False, server_default="default"),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("model_prediction_id", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("expected_move_bps", sa.Float(), nullable=True),
        sa.Column("rationale", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trade_intents_symbol_ts", "trade_intents", ["symbol", "timestamp"])

    op.create_table(
        "risk_decisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trade_intent_id", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("reasons", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("rule_results", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_risk_decisions_intent_ts", "risk_decisions", ["trade_intent_id", "timestamp"])

    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("trade_intent_id", sa.Integer(), nullable=True),
        sa.Column("broker_order_id", sa.String(255), nullable=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("order_type", sa.String(20), nullable=False),
        sa.Column("order_class", sa.String(20), nullable=False, server_default="simple"),
        sa.Column("time_in_force", sa.String(10), nullable=False, server_default="day"),
        sa.Column("qty", sa.Integer(), nullable=False),
        sa.Column("limit_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("stop_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("filled_qty", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("filled_avg_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("status", sa.String(30), nullable=False, server_default="pending"),
        sa.Column("strategy_tag", sa.String(50), nullable=False, server_default="default"),
        sa.Column("rationale", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("broker_metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("filled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key"),
    )
    op.create_index("ix_orders_symbol_status", "orders", ["symbol", "status"])

    op.create_table(
        "fills",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("order_id", sa.Integer(), nullable=False),
        sa.Column("broker_order_id", sa.String(255), nullable=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("qty", sa.Integer(), nullable=False),
        sa.Column("price", sa.Numeric(12, 4), nullable=False),
        sa.Column("commission", sa.Numeric(10, 4), nullable=False, server_default=sa.text("0")),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "positions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("qty", sa.Integer(), nullable=False),
        sa.Column("avg_entry_price", sa.Numeric(12, 4), nullable=False),
        sa.Column("current_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("market_value", sa.Numeric(14, 4), nullable=True),
        sa.Column("unrealized_pnl", sa.Numeric(14, 4), nullable=False, server_default=sa.text("0")),
        sa.Column("realized_pnl", sa.Numeric(14, 4), nullable=False, server_default=sa.text("0")),
        sa.Column("status", sa.String(20), nullable=False, server_default="open"),
        sa.Column("strategy_tag", sa.String(50), nullable=False, server_default="default"),
        sa.Column("trade_intent_id", sa.Integer(), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_positions_symbol_status", "positions", ["symbol", "status"])

    op.create_table(
        "pnl_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("date_str", sa.String(10), nullable=False),
        sa.Column("realized_pnl", sa.Numeric(14, 4), nullable=False, server_default=sa.text("0")),
        sa.Column("unrealized_pnl", sa.Numeric(14, 4), nullable=False, server_default=sa.text("0")),
        sa.Column("total_pnl", sa.Numeric(14, 4), nullable=False, server_default=sa.text("0")),
        sa.Column("open_positions", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("total_exposure", sa.Numeric(14, 4), nullable=False, server_default=sa.text("0")),
        sa.Column("win_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("loss_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("trade_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "system_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("level", sa.String(20), nullable=False),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "config_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("config_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("changed_by", sa.String(100), nullable=False, server_default="system"),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("hyperparameters", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("metrics", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("training_samples", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("validation_samples", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("best_score", sa.Float(), nullable=True),
        sa.Column("artifact_path", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("start_date", sa.String(10), nullable=False),
        sa.Column("end_date", sa.String(10), nullable=False),
        sa.Column("symbols", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("strategy_config", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("risk_config", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("slippage_config", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("total_trades", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("win_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("loss_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("total_pnl", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("win_rate", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("expectancy", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("sharpe_ratio", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("max_drawdown", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("profit_factor", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("avg_hold_minutes", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("turnover", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("detailed_metrics", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("backtest_runs")
    op.drop_table("training_runs")
    op.drop_table("config_versions")
    op.drop_table("system_events")
    op.drop_table("pnl_snapshots")
    op.drop_table("positions")
    op.drop_table("fills")
    op.drop_table("orders")
    op.drop_table("risk_decisions")
    op.drop_table("trade_intents")
    op.drop_table("model_predictions")
    op.drop_table("model_versions")
    op.drop_table("feature_snapshots")
    op.drop_table("market_bars")
    op.drop_table("symbols")
