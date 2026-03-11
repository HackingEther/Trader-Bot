"""Abstract base class for model artifact storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ArtifactStore(ABC):
    """Abstract interface for storing and retrieving ML model artifacts."""

    @abstractmethod
    async def save(self, key: str, data: bytes) -> str:
        """Save artifact bytes and return the storage path/URI."""

    @abstractmethod
    async def load(self, key: str) -> bytes:
        """Load artifact bytes by key."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if an artifact exists."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete an artifact. Returns True if deleted."""

    @abstractmethod
    async def list_keys(self, prefix: str = "") -> list[str]:
        """List artifact keys with optional prefix filter."""
