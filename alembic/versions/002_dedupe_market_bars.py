"""Deduplicate market bars and enforce unique bar identity.

Revision ID: 002
Revises: 001
Create Date: 2026-03-11 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: str | None = "001"
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            DELETE FROM market_bars
            WHERE id IN (
                SELECT id
                FROM (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               PARTITION BY symbol, interval, timestamp
                               ORDER BY id
                           ) AS row_num
                    FROM market_bars
                ) duplicates
                WHERE duplicates.row_num > 1
            )
            """
        )
    )
    op.create_unique_constraint(
        "uq_market_bars_symbol_interval_ts",
        "market_bars",
        ["symbol", "interval", "timestamp"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_market_bars_symbol_interval_ts", "market_bars", type_="unique")
