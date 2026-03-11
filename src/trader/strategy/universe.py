"""Symbol universe management."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


class SymbolUniverse:
    """Manages the set of tradable symbols."""

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = list(set(symbols))
        self._enabled: set[str] = set(self._symbols)

    @property
    def symbols(self) -> list[str]:
        return sorted(self._enabled)

    def enable(self, symbol: str) -> None:
        if symbol in self._symbols:
            self._enabled.add(symbol)

    def disable(self, symbol: str) -> None:
        self._enabled.discard(symbol)

    def is_enabled(self, symbol: str) -> bool:
        return symbol in self._enabled

    def set_symbols(self, symbols: list[str]) -> None:
        self._symbols = list(set(symbols))
        self._enabled = set(self._symbols)
