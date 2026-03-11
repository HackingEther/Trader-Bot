"""Database type helpers that stay portable across backends."""

from __future__ import annotations

from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB

JSONVariant = JSON().with_variant(JSONB, "postgresql")
