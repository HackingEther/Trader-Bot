"""Add execution identity columns to fills."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "003_fill_execution_identity"
down_revision = "002_runtime_correctness_guards"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("fills") as batch_op:
        batch_op.add_column(sa.Column("execution_key", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("broker_execution_timestamp", sa.DateTime(timezone=True), nullable=True))
        batch_op.create_index("ix_fills_order_timestamp", ["order_id", "timestamp"], unique=False)
        batch_op.create_index(batch_op.f("ix_fills_execution_key"), ["execution_key"], unique=True)


def downgrade() -> None:
    with op.batch_alter_table("fills") as batch_op:
        batch_op.drop_index(batch_op.f("ix_fills_execution_key"))
        batch_op.drop_index("ix_fills_order_timestamp")
        batch_op.drop_column("broker_execution_timestamp")
        batch_op.drop_column("execution_key")
