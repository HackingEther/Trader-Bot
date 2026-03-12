"""Add reprice tracking columns to orders."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "005_reprice_tracking"
down_revision = "004_quote_snapshots_execution_attribution"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("orders", sa.Column("reprice_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("orders", sa.Column("last_reprice_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("orders", "last_reprice_at")
    op.drop_column("orders", "reprice_count")
