"""Local filesystem artifact storage implementation."""

from __future__ import annotations

from pathlib import Path

import structlog

from trader.config import get_settings
from trader.providers.artifacts.base import ArtifactStore

logger = structlog.get_logger(__name__)


class LocalArtifactStore(ArtifactStore):
    """Store model artifacts on the local filesystem."""

    def __init__(self, base_dir: str | None = None) -> None:
        resolved_base = base_dir or get_settings().artifacts_dir
        self._base = Path(resolved_base)
        self._base.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe_key = key.replace("..", "").lstrip("/\\")
        return self._base / safe_key

    async def save(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.info("artifact_saved", key=key, size=len(data))
        return str(path)

    async def load(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {key}")
        return path.read_bytes()

    async def exists(self, key: str) -> bool:
        return self._path(key).exists()

    async def delete(self, key: str) -> bool:
        path = self._path(key)
        if path.exists():
            path.unlink()
            logger.info("artifact_deleted", key=key)
            return True
        return False

    async def list_keys(self, prefix: str = "") -> list[str]:
        search_path = self._base / prefix if prefix else self._base
        if not search_path.exists():
            return []
        base_len = len(self._base.parts)
        return [
            "/".join(p.parts[base_len:])
            for p in search_path.rglob("*")
            if p.is_file()
        ]
