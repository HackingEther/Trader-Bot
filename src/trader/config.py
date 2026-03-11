"""Application configuration with environment-based settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Core ──
    app_env: str = Field(default="paper", description="local | paper | live")
    log_level: str = Field(default="INFO")
    secret_key: str = Field(default="change-me-to-random-string")

    # ── Database ──
    database_url: str = Field(
        default="postgresql+asyncpg://trader:trader@localhost:5432/trader"
    )
    database_sync_url: str = Field(
        default="postgresql://trader:trader@localhost:5432/trader"
    )

    # ── Redis ──
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ── Broker: Alpaca ──
    alpaca_api_key: str = Field(default="")
    alpaca_api_secret: str = Field(default="")
    alpaca_base_url: str = Field(default="https://paper-api.alpaca.markets")
    alpaca_paper: bool = Field(default=True)
    paper_initial_cash: float = Field(default=100000.0)

    # ── Market Data: Databento ──
    market_data_provider: str = Field(default="databento")
    market_data_autostart: bool = Field(default=False)
    market_data_subscribe_bars: bool = Field(default=True)
    market_data_subscribe_quotes: bool = Field(default=True)
    market_data_subscribe_trades: bool = Field(default=False)
    market_data_staleness_threshold: float = Field(default=30.0)
    databento_api_key: str = Field(default="")
    databento_dataset: str = Field(default="XNAS.ITCH")

    # ── Market Data: Alpaca ──
    alpaca_data_feed: str = Field(default="iex")

    # ── Market Data: Polygon ──
    polygon_api_key: str = Field(default="")

    # ── Live Trading Safety ──
    live_trading: bool = Field(default=False)
    live_trading_confirmed: str = Field(default="")

    # ── Risk Limits ──
    max_daily_loss_usd: float = Field(default=1000.0)
    max_loss_per_trade_usd: float = Field(default=200.0)
    max_notional_exposure_usd: float = Field(default=50000.0)
    max_concurrent_positions: int = Field(default=10)
    max_exposure_per_symbol_usd: float = Field(default=10000.0)
    cooldown_after_losses: int = Field(default=3)
    spread_threshold_bps: float = Field(default=50.0)

    # ── Strategy ──
    symbol_universe: list[str] = Field(
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "SPY", "QQQ", "AMD"]
    )
    min_confidence: float = Field(default=0.6)
    min_expected_move_bps: float = Field(default=15.0)

    # ── Notifications ──
    slack_webhook_url: str = Field(default="")
    smtp_host: str = Field(default="")
    smtp_port: int = Field(default=587)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    notification_email_to: str = Field(default="")

    # ── Observability ──
    sentry_dsn: str = Field(default="")
    prometheus_port: int = Field(default=9090)

    # ── Model Artifacts ──
    artifacts_dir: str = Field(default="artifacts")

    # ── Admin ──
    admin_api_token: str = Field(default="change-me-to-random-string")

    # ── Celery ──
    celery_broker_url: str = Field(default="redis://localhost:6379/1")
    celery_result_backend: str = Field(default="redis://localhost:6379/2")

    @field_validator("symbol_universe", mode="before")
    @classmethod
    def parse_symbol_universe(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return v
        return []

    @field_validator("market_data_provider", mode="before")
    @classmethod
    def parse_market_data_provider(cls, v: object) -> str:
        if isinstance(v, str):
            return v.strip().lower()
        return "databento"

    @field_validator("alpaca_data_feed", mode="before")
    @classmethod
    def parse_alpaca_data_feed(cls, v: object) -> str:
        if isinstance(v, str):
            return v.strip().lower()
        return "iex"

    @property
    def is_live(self) -> bool:
        return (
            self.live_trading
            and self.live_trading_confirmed == "I_CONFIRM_LIVE_TRADING"
            and self.app_env == "live"
        )

    @property
    def is_paper(self) -> bool:
        return not self.is_live


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
