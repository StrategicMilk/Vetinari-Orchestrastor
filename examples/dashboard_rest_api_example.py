#!/usr/bin/env python
"""
Example: Vetinari Dashboard — Flask REST API Server  (Phase 4)

Shows how to:
  1. Start the dashboard REST API as a standalone server
  2. Optionally attach an alert evaluation loop
  3. Optionally attach the LogAggregator stdlib bridge

Run from the project root:
    python examples/dashboard_rest_api_example.py

Then navigate to:
    http://localhost:5000/dashboard        <- monitoring UI
    http://localhost:5000/api/v1/health    <- health check
"""

import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vetinari.dashboard.rest_api import create_app
from vetinari.dashboard.alerts import (
    get_alert_engine,
    AlertThreshold,
    AlertCondition,
    AlertSeverity,
)
from vetinari.dashboard.log_aggregator import (
    get_log_aggregator,
    AggregatorHandler,
)

# ─── Logging setup ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dashboard_server")


# ─── Optional: attach stdlib logging to the aggregator ───────────────────

def setup_log_aggregator() -> None:
    """Route all Python log output into the in-process aggregator."""
    agg = get_log_aggregator()

    # Also write to a local file for persistence
    agg.configure_backend("file", path="logs/dashboard_server.jsonl")

    handler = AggregatorHandler()
    logging.getLogger().addHandler(handler)
    logger.info("LogAggregator active — logs captured in buffer and file")


# ─── Optional: alert evaluation loop ─────────────────────────────────────

def setup_alerts() -> None:
    """Register example thresholds."""
    engine = get_alert_engine()

    engine.register_threshold(AlertThreshold(
        name="high-adapter-latency",
        metric_key="adapters.average_latency_ms",
        condition=AlertCondition.GREATER_THAN,
        threshold_value=500.0,
        severity=AlertSeverity.HIGH,
        channels=["log"],
    ))

    engine.register_threshold(AlertThreshold(
        name="low-approval-rate",
        metric_key="plan.approval_rate",
        condition=AlertCondition.LESS_THAN,
        threshold_value=60.0,
        severity=AlertSeverity.MEDIUM,
        channels=["log"],
    ))

    engine.register_threshold(AlertThreshold(
        name="high-risk-score",
        metric_key="plan.average_risk_score",
        condition=AlertCondition.GREATER_THAN,
        threshold_value=0.7,
        severity=AlertSeverity.HIGH,
        channels=["log"],
        duration_seconds=30,   # must stay above 0.7 for 30 s before firing
    ))

    logger.info("Registered %d alert thresholds", len(engine.list_thresholds()))


def alert_loop(interval_seconds: float = 30.0, stop_event: threading.Event = None) -> None:
    """Background thread: evaluate alerts every ``interval_seconds``."""
    engine = get_alert_engine()
    while stop_event is None or not stop_event.is_set():
        fired = engine.evaluate_all()
        if fired:
            logger.warning(
                "Alert evaluation: %d alert(s) fired — %s",
                len(fired),
                ", ".join(a.threshold.name for a in fired),
            )
        time.sleep(interval_seconds)


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    host    = "0.0.0.0"
    port    = 5000
    debug   = "--debug" in sys.argv
    no_alerts = "--no-alerts" in sys.argv

    # Set up supporting services
    setup_log_aggregator()
    if not no_alerts:
        setup_alerts()

    # Start background alert loop
    stop_event = threading.Event()
    if not no_alerts:
        t = threading.Thread(
            target=alert_loop,
            kwargs={"interval_seconds": 30.0, "stop_event": stop_event},
            daemon=True,
            name="alert-loop",
        )
        t.start()
        logger.info("Alert evaluation loop started (every 30 s)")

    # Create and start the Flask app
    app = create_app(debug=debug)

    logger.info("Vetinari Dashboard API starting on http://%s:%d", host, port)
    logger.info("Monitoring UI: http://%s:%d/dashboard", host, port)
    logger.info("API health:    http://%s:%d/api/v1/health", host, port)

    try:
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Shutting down …")
    finally:
        stop_event.set()
        # Flush any pending log records before exit
        get_log_aggregator().flush()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
