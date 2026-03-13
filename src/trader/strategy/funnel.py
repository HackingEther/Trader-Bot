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

    def get_distribution_summary(self) -> dict[str, Any]:
        """Compute percentile distributions for blocked events by (framework, symbol, side) and block_reason.

        Returns dict with block_distributions: for each key (fw|sym|side) and reason,
        percentiles (p10, p50, p90) for confidence, expected_move_bps, best_playbook_fit when available.
        """
        by_key_reason: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        percentiles = (10, 50, 90)

        for event in self._events:
            key = f"{event.framework}|{event.symbol}|{event.side}"
            reason = event.block_reason
            if key not in by_key_reason:
                by_key_reason[key] = {}
            if reason not in by_key_reason[key]:
                by_key_reason[key][reason] = {
                    "confidence": [],
                    "expected_move_bps": [],
                    "best_playbook_fit": [],
                }
            if event.confidence is not None:
                by_key_reason[key][reason]["confidence"].append(event.confidence)
            if event.expected_move_bps is not None:
                by_key_reason[key][reason]["expected_move_bps"].append(event.expected_move_bps)
            if event.best_playbook_fit is not None:
                by_key_reason[key][reason]["best_playbook_fit"].append(event.best_playbook_fit)

        def _percentiles(vals: list[float]) -> dict[str, float]:
            if not vals:
                return {}
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            return {
                f"p{p}": float(sorted_vals[min(int(n * p / 100), n - 1)])
                for p in percentiles
            }

        block_distributions: dict[str, dict[str, dict[str, Any]]] = {}
        for key, reasons in by_key_reason.items():
            block_distributions[key] = {}
            for reason, vecs in reasons.items():
                dist: dict[str, Any] = {}
                if vecs["confidence"]:
                    dist["confidence"] = _percentiles(vecs["confidence"])
                if vecs["expected_move_bps"]:
                    dist["expected_move_bps"] = _percentiles(vecs["expected_move_bps"])
                if vecs["best_playbook_fit"]:
                    dist["best_playbook_fit"] = _percentiles(vecs["best_playbook_fit"])
                if dist:
                    block_distributions[key][reason] = dist

        return {"block_distributions": block_distributions}

    def reset(self) -> None:
        """Clear all recorded events."""
        self._events.clear()
        self._counts.clear()
