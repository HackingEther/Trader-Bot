# Claude Trader – Implementation Checklist

## Security (do first)

- [ ] **Regenerate Alpaca API keys** – You shared them; create new keys in Alpaca dashboard and update `.env`
- [ ] **Replace SECRET_KEY** – Use a long random string (e.g. `openssl rand -hex 32`)
- [ ] **Replace ADMIN_API_TOKEN** – Same as above
- [ ] **Never commit `.env`** – Ensure `.env` is in `.gitignore` (it should be)

## Deployment (Vultr)

- [ ] **Clear stale ingestion lease** (if bars still not flowing after deploy):
  ```bash
  docker compose exec redis redis-cli DEL "trader:lease:ingestion:AlpacaProvider:AAPL,AMD,AMZN,GOOGL,META,MSFT,NVDA,QQQ,SPY,TSLA"
  docker compose restart app
  ```
- [ ] **Run migrations** (if not already done):
  ```bash
  docker compose exec app alembic upgrade head
  ```
- [ ] **Verify bar ingestion** – After deploy, wait for market hours and check:
  ```bash
  docker compose exec app python -c "
  import asyncio
  from trader.db.session import init_engine, get_session_factory
  from trader.config import get_settings
  from trader.db.repositories.market_bars import MarketBarRepository
  async def check():
      s = get_settings()
      init_engine(s.database_url)
      f = get_session_factory()
      async with f() as session:
          repo = MarketBarRepository(session)
          bars = await repo.get_recent('AAPL', limit=5)
          print('AAPL bars:', len(bars))
  asyncio.run(check())
  "
  ```

## Optional / Future

- [ ] **Train champion models** – Default sklearn models are used if no champions. Run training pipeline to train and register champions for better signals.
- [ ] **Pre/post market trading** – Requires code changes (MarketHoursRule, Alpaca `extended_hours=True`, market data feed).
- [ ] **Notifications** – Slack webhook, SMTP for alerts.
- [ ] **Sentry** – Error tracking.
- [ ] **Run worker as non-root** – `--uid` in Celery worker config.

## Current Fix (this commit)

- **Event loop / Redis lock** – Bar/quote/trade callbacks from Alpaca’s websocket thread now enqueue events to thread-safe queues; consumer tasks in the main loop process them and persist to DB/Redis. This fixes the “Lock bound to different event loop” error.
