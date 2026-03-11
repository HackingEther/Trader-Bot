"""Prometheus metrics definitions."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

APP_INFO = Info("trader", "Trading platform information")
APP_INFO.info({"version": "0.1.0"})

ORDERS_SUBMITTED = Counter(
    "trader_orders_submitted_total",
    "Total orders submitted to broker",
    ["symbol", "side", "order_type"],
)

ORDERS_FILLED = Counter(
    "trader_orders_filled_total",
    "Total orders filled",
    ["symbol", "side"],
)

ORDERS_REJECTED = Counter(
    "trader_orders_rejected_total",
    "Total orders rejected by risk engine",
    ["symbol", "rule"],
)

ORDER_LATENCY = Histogram(
    "trader_order_latency_seconds",
    "Order submission latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

OPEN_POSITIONS = Gauge(
    "trader_open_positions",
    "Current number of open positions",
)

TOTAL_EXPOSURE = Gauge(
    "trader_total_exposure_usd",
    "Total notional exposure in USD",
)

DAILY_PNL = Gauge(
    "trader_daily_pnl_usd",
    "Realized daily P&L in USD",
)

PREDICTIONS_GENERATED = Counter(
    "trader_predictions_generated_total",
    "Total model predictions generated",
    ["symbol", "direction"],
)

FEATURE_COMPUTATION_TIME = Histogram(
    "trader_feature_computation_seconds",
    "Feature computation time",
    ["symbol"],
)

DATA_FEED_LATENCY = Histogram(
    "trader_data_feed_latency_seconds",
    "Market data feed latency",
    ["provider"],
)

KILL_SWITCH_ACTIVE = Gauge(
    "trader_kill_switch_active",
    "Whether kill switch is active (1) or not (0)",
)

HEARTBEAT_TIMESTAMP = Gauge(
    "trader_heartbeat_timestamp",
    "Unix timestamp of last heartbeat",
    ["component"],
)
