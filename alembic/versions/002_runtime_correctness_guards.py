"""Add market bar uniqueness guard.

Revision ID: 002_runtime_correctness_guards
Revises: 002_dedupe_market_bars
Create Date: 2026-03-10 00:00:00.000000

Note: 002_dedupe_market_bars already adds the unique constraint. This migration
is kept for compatibility with deployments that ran it before 002_dedupe was
introduced. On fresh DBs it is a no-op.
"""

from __future__ import annotations

from alembic import op

revision: str = "002_runtime_correctness_guards"
down_revision: str | None = "002_dedupe_market_bars"
branch_labels: tuple[str, ...] | None = None
depends_on: str | None = None


def upgrade() -> None:
    # Constraint already added by 002_dedupe_market_bars. No-op.
    pass


def downgrade() -> None:
    pass
