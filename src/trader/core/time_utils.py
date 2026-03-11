"""Market hours and timezone utilities."""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
PRE_MARKET_OPEN = time(4, 0)
AFTER_HOURS_CLOSE = time(20, 0)


def now_et() -> datetime:
    """Current time in US Eastern."""
    return datetime.now(ET)


def now_utc() -> datetime:
    """Current time in UTC."""
    return datetime.now(UTC)


def to_et(dt: datetime) -> datetime:
    """Convert any datetime to Eastern."""
    return dt.astimezone(ET)


def is_market_open(dt: datetime | None = None) -> bool:
    """Check if regular market session is open."""
    dt = to_et(dt) if dt else now_et()
    if dt.weekday() >= 5:
        return False
    return MARKET_OPEN <= dt.time() < MARKET_CLOSE


def is_pre_market(dt: datetime | None = None) -> bool:
    """Check if pre-market session is active."""
    dt = to_et(dt) if dt else now_et()
    if dt.weekday() >= 5:
        return False
    return PRE_MARKET_OPEN <= dt.time() < MARKET_OPEN


def is_after_hours(dt: datetime | None = None) -> bool:
    """Check if after-hours session is active."""
    dt = to_et(dt) if dt else now_et()
    if dt.weekday() >= 5:
        return False
    return MARKET_CLOSE <= dt.time() < AFTER_HOURS_CLOSE


def get_market_session(dt: datetime | None = None) -> str:
    """Return current market session name."""
    from trader.core.enums import MarketSession

    if is_market_open(dt):
        return MarketSession.REGULAR
    if is_pre_market(dt):
        return MarketSession.PRE_MARKET
    if is_after_hours(dt):
        return MarketSession.AFTER_HOURS
    return MarketSession.CLOSED


def market_open_today(d: date | None = None) -> datetime:
    """Return today's market open as ET datetime."""
    d = d or now_et().date()
    return datetime.combine(d, MARKET_OPEN, tzinfo=ET)


def market_close_today(d: date | None = None) -> datetime:
    """Return today's market close as ET datetime."""
    d = d or now_et().date()
    return datetime.combine(d, MARKET_CLOSE, tzinfo=ET)


def minutes_since_open(dt: datetime | None = None) -> float:
    """Minutes elapsed since market open."""
    dt = to_et(dt) if dt else now_et()
    open_dt = market_open_today(dt.date())
    delta = dt - open_dt
    return max(0.0, delta.total_seconds() / 60.0)


def minutes_until_close(dt: datetime | None = None) -> float:
    """Minutes remaining until market close."""
    dt = to_et(dt) if dt else now_et()
    close_dt = market_close_today(dt.date())
    delta = close_dt - dt
    return max(0.0, delta.total_seconds() / 60.0)


def session_fraction(dt: datetime | None = None) -> float:
    """Fraction of regular session elapsed (0.0 to 1.0)."""
    total = 6.5 * 60  # 390 minutes
    elapsed = minutes_since_open(dt)
    return min(1.0, max(0.0, elapsed / total))
