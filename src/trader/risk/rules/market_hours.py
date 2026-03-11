"""Market hours restriction rule."""

from __future__ import annotations

from trader.core.time_utils import is_market_open


class MarketHoursRule:
    """Rejects trades outside regular market hours."""

    def check(self, **kwargs: object) -> tuple[bool, str]:
        if not is_market_open():
            return False, "Market is not in regular session"
        return True, ""
