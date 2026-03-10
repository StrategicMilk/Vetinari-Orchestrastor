"""Flask Blueprint exposing analytics subsystem data via REST API.

Wires the cost tracker, SLA tracker, anomaly detector, and forecaster
into the web dashboard so the UI can display live analytics.

Endpoints
---------
    GET /api/v1/analytics/cost      — Token costs by model / agent / provider
    GET /api/v1/analytics/sla       — SLA compliance metrics
    GET /api/v1/analytics/anomalies — Recent anomaly detections
    GET /api/v1/analytics/forecasts — Capacity-planning forecasts
"""

import logging

from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.route("/api/v1/analytics/cost")
def get_cost_data():
    """Token costs by model / agent / provider."""
    try:
        from vetinari.analytics.cost import get_cost_tracker
        tracker = get_cost_tracker()
        report = tracker.get_report()
        return jsonify({"cost": report.to_dict()})
    except Exception as e:
        logger.warning("Failed to fetch cost data: %s", e)
        return jsonify({"cost": {}, "error": str(e)})


@analytics_bp.route("/api/v1/analytics/sla")
def get_sla_data():
    """SLA compliance metrics."""
    try:
        from vetinari.analytics.sla import get_sla_tracker
        tracker = get_sla_tracker()
        reports = tracker.get_all_reports()
        serialised = [r.to_dict() for r in reports]
        return jsonify({"sla": serialised})
    except Exception as e:
        logger.warning("Failed to fetch SLA data: %s", e)
        return jsonify({"sla": [], "error": str(e)})


@analytics_bp.route("/api/v1/analytics/anomalies")
def get_anomaly_data():
    """Recent anomaly detections."""
    try:
        from vetinari.analytics.anomaly import get_anomaly_detector
        detector = get_anomaly_detector()
        stats = detector.get_stats()
        return jsonify({"anomalies": stats})
    except Exception as e:
        logger.warning("Failed to fetch anomaly data: %s", e)
        return jsonify({"anomalies": {}, "error": str(e)})


@analytics_bp.route("/api/v1/analytics/forecasts")
def get_forecast_data():
    """Capacity-planning forecasts."""
    try:
        from vetinari.analytics.forecasting import get_forecaster
        forecaster = get_forecaster()
        stats = forecaster.get_stats()
        return jsonify({"forecasts": stats})
    except Exception as e:
        logger.warning("Failed to fetch forecast data: %s", e)
        return jsonify({"forecasts": {}, "error": str(e)})
