# Intraday Trading System Runbook

Operational notes for the broker-truth trade updates, quote snapshots, execution attribution, and reprice policy.

## Trade Updates Stream

- **Enabled**: `TRADE_UPDATES_STREAM_ENABLED=true` (default)
- **Health**: `GET /health` returns `trade_updates_last_event_age_seconds`; if > 300 during market hours, stream may be stale
- **Restart**: Restart app container; stream reconnects on startup
- **Logs**: `trade_update_fill_applied`, `trade_update_process_failed`, `trade_update_unknown_order`

## Execution Attribution

- **Query**: `SELECT * FROM execution_attribution ORDER BY created_at DESC LIMIT 100`
- **Metrics**: `slippage_bps`, `realized_spread_bps`, `time_to_fill_seconds` for post-trade analysis

## Reprice Policy

- **Config**: `REPRICE_MAX_ATTEMPTS`, `REPRICE_MIN_INTERVAL_SECONDS`, `REPRICE_MAX_DRIFT_BPS`
- **Override**: Set to 0 to disable repricing; increase `reprice_max_drift_bps` for more aggressive repricing
