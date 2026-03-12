"""Add quote_snapshots and execution_attribution tables."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "004_quote_snapshots_execution_attribution"
down_revision = "003_fill_execution_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quote_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trade_intent_id", sa.Integer(), nullable=True),
        sa.Column("order_id", sa.Integer(), nullable=True),
        sa.Column("snapshot_type", sa.String(20), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("bid", sa.Numeric(12, 4), nullable=False),
        sa.Column("ask", sa.Numeric(12, 4), nullable=False),
        sa.Column("mid", sa.Numeric(12, 4), nullable=False),
        sa.Column("spread_bps", sa.Numeric(10, 4), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["trade_intent_id"], ["trade_intents.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_quote_snapshots_trade_intent", "quote_snapshots", ["trade_intent_id"])
    op.create_index("ix_quote_snapshots_order", "quote_snapshots", ["order_id"])

    op.create_table(
        "execution_attribution",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trade_intent_id", sa.Integer(), nullable=True),
        sa.Column("order_id", sa.Integer(), nullable=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("filled_qty", sa.Integer(), nullable=False),
        sa.Column("filled_avg_price", sa.Numeric(12, 4), nullable=False),
        sa.Column("decision_quote_snapshot_id", sa.Integer(), nullable=True),
        sa.Column("submit_quote_snapshot_id", sa.Integer(), nullable=True),
        sa.Column("realized_spread_bps", sa.Numeric(10, 4), nullable=True),
        sa.Column("slippage_bps", sa.Numeric(10, 4), nullable=True),
        sa.Column("time_to_fill_seconds", sa.Numeric(12, 2), nullable=True),
        sa.Column("first_fill_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_fill_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["trade_intent_id"], ["trade_intents.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["decision_quote_snapshot_id"], ["quote_snapshots.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["submit_quote_snapshot_id"], ["quote_snapshots.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_execution_attribution_order", "execution_attribution", ["order_id"])
    op.create_index("ix_execution_attribution_trade_intent", "execution_attribution", ["trade_intent_id"])


def downgrade() -> None:
    op.drop_table("execution_attribution")
    op.drop_table("quote_snapshots")
