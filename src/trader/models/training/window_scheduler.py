"""Rolling train/test window generation for confidence-testing backtests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator


def generate_windows(
    start: datetime,
    end: datetime,
    train_weeks: int,
    test_weeks: int,
    step_weeks: int = 1,
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """Generate rolling train/test windows.

    Each window has: train on preceding train_weeks, test on following test_weeks.
    Windows step forward by step_weeks. Test windows do not overlap.

    Args:
        start: Start of first train window.
        end: End of last test window must be <= end.
        train_weeks: Length of training period in weeks.
        test_weeks: Length of test period in weeks.
        step_weeks: Step size between windows in weeks.

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    windows: list[tuple[datetime, datetime, datetime, datetime]] = []
    offset_weeks = 0

    while True:
        train_start = start + timedelta(weeks=offset_weeks)
        train_end = train_start + timedelta(weeks=train_weeks) - timedelta(days=1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(weeks=test_weeks) - timedelta(days=1)

        if test_end > end:
            break

        windows.append((train_start, train_end, test_start, test_end))
        offset_weeks += step_weeks

    return windows
