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

## Recommended workflow (to get trades)

1. **Backfill historical bars** (before first live day or for training):
   ```bash
   docker compose exec app python scripts/backfill_alpaca_bars.py \
     --symbols "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,SPY,QQQ,AMD" \
     --start "2026-03-01T00:00:00Z" \
     --end "2026-03-11T00:00:00Z" \
     --feed iex
   ```

2. **Train champion models** (so the ensemble stops predicting `no_trade` for everything):
   ```bash
   docker compose exec app python scripts/train_models_from_history.py \
     --symbols "AAPL,MSFT,GOOGL,AMZN,NVDA,SPY,QQQ" \
     --start "2026-03-01T00:00:00Z" \
     --end "2026-03-10T00:00:00Z" \
     --lookahead-bars 15
   ```

3. **Clear bad spread data** (if Redis has 1000+ bps from IEX glitches):
   ```bash
   docker compose exec redis redis-cli KEYS "trader:market:last_spread_bps:*" | xargs -I {} docker compose exec redis redis-cli DEL {}
   ```
   Or: `docker compose restart app` (spreads repopulate from good quotes during RTH).

4. **Run during regular hours** (9:30–16:00 ET) – live ingestion + trading cycle need market hours for good quotes and volume.

## Optional / Future

- [ ] **Train champion models** – Default sklearn models are used if no champions. Run training pipeline to train and register champions for better signals.
- [ ] **Pre/post market trading** – Requires code changes (MarketHoursRule, Alpaca `extended_hours=True`, market data feed).
- [ ] **Notifications** – Slack webhook, SMTP for alerts.
- [ ] **Sentry** – Error tracking.
- [ ] **Run worker as non-root** – `--uid` in Celery worker config.

## Fixes in this codebase

- **Event loop / Redis lock** – Bar/quote/trade callbacks from Alpaca websocket thread now enqueue events to thread-safe queues; consumer tasks in the main loop process them and persist to DB/Redis. Fixes the Lock bound to different event loop error.
- **Spread cap** – IEX can return stale/mismatched bid-ask (1000+ bps). We now only store spread when ≤150 bps, so bad quotes don't overwrite good data and block trading.

