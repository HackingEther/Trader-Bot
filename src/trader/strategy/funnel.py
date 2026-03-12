"""Decision funnel tracking for audit and diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FunnelEvent:
    """Single decision event in the funnel."""

    framework: str
    symbol: str
    side: str
    block_reason: str
    playbook_candidate: str | None = None
    confidence: float | None = None
    expected_move_bps: float | None = None
    no_trade_score: float | None = None
    best_playbook_fit: float | None = None


class DecisionFunnelTracker:
    """Records decision funnel events for audit: where trades are blocked by framework, symbol, side."""

    def __init__(self, framework: str = "default") -> None:
        self._framework = framework
        self._events: list[FunnelEvent] = []
        self._counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def set_framework(self, framework: str) -> None:
        """Set framework tag for subsequent events."""
        self._framework = framework

    def record(
        self,
        symbol: str,
        side: str,
        block_reason: str,
        *,
        playbook_candidate: str | None = None,
        confidence: float | None = None,
        expected_move_bps: float | None = None,
        no_trade_score: float | None = None,
        best_playbook_fit: float | None = None,
    ) -> None:
        """Record a block event."""
        key = (self._framework, symbol, side)
        self._counts[key][block_reason] += 1
        self._events.append(
            FunnelEvent(
                framework=self._framework,
                symbol=symbol,
                side=side,
                block_reason=block_reason,
                playbook_candidate=playbook_candidate,
                confidence=confidence,
                expected_move_bps=expected_move_bps,
                no_trade_score=no_trade_score,
                best_playbook_fit=best_playbook_fit,
            )
        )

    def get_summary(self) -> dict[str, Any]:
        """Return aggregated counts by (framework, symbol, side) and block_reason."""
        by_key: dict[str, dict[str, int]] = {}
        for (fw, sym, side), reasons in self._counts.items():
            k = f"{fw}|{sym}|{side}"
            by_key[k] = dict(reasons)

        total_by_reason: dict[str, int] = defaultdict(int)
        for reasons in self._counts.values():
            for reason, count in reasons.items():
                total_by_reason[reason] += count

        return {
            "by_framework_symbol_side": by_key,
            "total_by_reason": dict(total_by_reason),
            "event_count": len(self._events),
        }

    def get_events(self) -> list[FunnelEvent]:
        """Return raw events (for detailed analysis)."""
        return list(self._events)

    def reset(self) -> None:
        """Clear all recorded events."""
        self._events.clear()
        self._counts.clear()
