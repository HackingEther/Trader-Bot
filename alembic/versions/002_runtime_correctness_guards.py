"""Add market bar uniqueness guard.

Revision ID: 002
Revises: 001
Create Date: 2026-03-10 00:00:00.000000
"""

from __future__ import annotations

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None


def upgrade() -> None:
    with op.batch_alter_table("market_bars") as batch_op:
        batch_op.create_unique_constraint(
            "uq_market_bars_symbol_interval_ts",
            ["symbol", "interval", "timestamp"],
        )


def downgrade() -> None:
    with op.batch_alter_table("market_bars") as batch_op:
        batch_op.drop_constraint("uq_market_bars_symbol_interval_ts", type_="unique")
