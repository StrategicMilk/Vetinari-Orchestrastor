"""
Analytics REST API Routes — vetinari.web.analytics_routes

Provides 7 REST endpoints for the analytics dashboard:

    GET /api/analytics/cost        — cost attribution report
    GET /api/analytics/sla         — SLA compliance report
    GET /api/analytics/anomalies   — detected anomalies
    GET /api/analytics/forecast    — capacity forecasts
    GET /api/analytics/models      — model performance stats
    GET /api/analytics/agents      — per-agent metrics
    GET /api/analytics/summary     — dashboard summary (all of the above combined)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

bp = Blueprint("analytics", __name__, url_prefix="/api/analytics")


# ---------------------------------------------------------------------------
# Helper: safe singleton access
# ---------------------------------------------------------------------------

def _cost_tracker():
    from vetinari.analytics.cost import get_cost_tracker
    return get_cost_tracker()


def _sla_tracker():
    from vetinari.analytics.sla import get_sla_tracker
    return get_sla_tracker()


def _anomaly_detector():
    from vetinari.analytics.anomaly import get_anomaly_detector
    return get_anomaly_detector()


def _forecaster():
    from vetinari.analytics.forecasting import get_forecaster
    return get_forecaster()


# ---------------------------------------------------------------------------
# GET /api/analytics/cost
# ---------------------------------------------------------------------------

@bp.route("/cost")
def analytics_cost():
    """Return cost attribution report, optionally filtered by agent or task_id."""
    try:
        agent = request.args.get("agent") or None
        task_id = request.args.get("task_id") or None
        since_hours = request.args.get("since_hours", type=float, default=None)
        since = (time.time() - since_hours * 3600) if since_hours else None

        report = _cost_tracker().get_report(agent=agent, task_id=task_id, since=since)
        return jsonify({"status": "ok", "cost": report.to_dict()})
    except Exception as e:
        logger.error("analytics/cost error: %s", e)
        return jsonify({"status": "error", "error": str(e)}), 500


# ---------------------------------------------------------------------------
# GET /api/analytics/sla
# ---------------------------------------------------------------------------

@bp.route("/sla")
def analytics_sla():
    """Return SLA compliance reports for all registered SLOs."""
    try:
        tracker = _sla_tracker()
        reports = tracker.get_all_reports()
        return jsonify({
            "status": "ok",
            "sla": {
                "slos": [r.to_dict() for r in reports],
                "stats": tracker.get_stats(),
            },
        })
    except Exception as e:
        logger.error("analytics/sla error: %s", e)
        return jsonify({"status": "error", "error": str(e)}), 500


# ---------------------------------------------------------------------------
# GET /api/analytics/anomalies
# ---------------------------------------------------------------------------

@bp.route("/anomalies")
def analytics_anomalies():
    """Return detected anomalies, optionally filtered by metric name."""
    try:
        metric = request.args.get("metric") or None
        detector = _anomaly_detector()
        history = detector.get_history(metric=metric)
        return jsonify({
            "status": "ok",
            "anomalies": {
                "total": len(history),
                "items": [a.to_dict() for a in history[-100:]],  # Most recent 100
                "stats": detector.get_stats(),
            },
        })
    except Exception as e:
        logger.error("analytics/anomalies error: %s", e)
        return jsonify({"status": "error", "error": str(e)}), 500


# ---------------------------------------------------------------------------
# GET /api/analytics/forecast
# ---------------------------------------------------------------------------

@bp.route("/forecast")
def analytics_forecast():
    """Return capacity forecasts for tracked metrics.

    Query params:
        metric   — specific metric (default: all tracked metrics)
        horizon  — number of future steps (default: 10)
        method   — forecasting method: sma | exp_smoothing | linear_trend | seasonal
    """
    try:
        from vetinari.analytics.forecasting import ForecastRequest
        fc = _forecaster()
        horizon = request.args.get("horizon", default=10, type=int)
        method = request.args.get("method", default="linear_trend")
        requested_metric = request.args.get("metric") or None

        metrics = [requested_metric] if requested_metric else fc.list_metrics()
        results: Dict[str, Any] = {}
        for m in metrics[:20]:  # Cap at 20 metrics per request
            try:
                req = ForecastRequest(metric=m, horizon=horizon, method=method)
                result = fc.forecast(req)
                results[m] = result.to_dict()
            except Exception as fe:
                results[m] = {"error": str(fe)}

        return jsonify({
            "status": "ok",
            "forecast": {
                "horizon": horizon,
                "method": method,
                "metrics": results,
                "forecaster_stats": fc.get_stats(),
            },
        })
    except Exception as e:
        logger.error("analytics/forecast error: %s", e)
        return jsonify({"status": "error", "error": str(e)}), 500


# ---------------------------------------------------------------------------
# GET /api/analytics/models
# ---------------------------------------------------------------------------

@bp.route("/models")
def analytics_models():
    """Return per-model performance statistics."""
    try:
        cost_report = _cost_tracker().get_report()
        sla_reports = _sla_tracker().get_all_reports()

        # Build per-model stats from cost tracker
        model_stats: Dict[str, Any] = {}
        for model_key, cost_usd in cost_report.by_model.items():
            model_stats[model_key] = {
                "cost_usd": cost_usd,
                "requests": 0,
            }

        # Enrich with request counts
        for entry in _cost_tracker()._entries:
            key = f"{entry.provider}:{entry.model}"
            if key in model_stats:
                model_stats[key]["requests"] += 1

        # Add SLA data for each model
        for report in sla_reports:
            slo_name = report.slo.name
            for key in model_stats:
                if key in slo_name or slo_name in key:
                    model_stats[key]["sla_compliance_pct"] = report.compliance_pct

        return jsonify({
            "status": "ok",
            "models": {
                "by_model": model_stats,
                "top_cost": _cost_tracker().get_top_models(n=5),
            },
        })
    except Exception as e:
        logger.error("analytics/models error: %s", e)
        return jsonify({"status": "error", "error": str(e)}), 500


# ---------------------------------------------------------------------------
# GET /api/analytics/agents
# ---------------------------------------------------------------------------

@bp.route("/agents")
def analytics_agents():
    """Return per-agent cost and quality metrics."""
    try:
        cost_report = _cost_tracker().get_report()
        top_agents = _cost_tracker().get_top_agents(n=10)

        # Try to get quality scores from feedback loop
        agent_quality: Dict[str, float] = {}
        try:
            from vetinari.learning.feedback_loop import get_feedback_loop
            fl = get_feedback_loop()
            if hasattr(fl, "get_stats"):
                for agent_key, cost in cost_report.by_agent.items():
                    stats = fl.get_stats(agent_key) if hasattr(fl, "get_stats") else {}
                    agent_quality[agent_key] = stats.get("avg_quality", 0.0) if stats else 0.0
        except Exception:
            pass

        return jsonify({
            "status": "ok",
            "agents": {
                "by_agent_cost": cost_report.by_agent,
                "top_agents": top_agents,
                "agent_quality": agent_quality,
                "total_requests": cost_report.total_requests,
            },
        })
    except Exception as e:
        logger.error("analytics/agents error: %s", e)
        return jsonify({"status": "error", "error": str(e)}), 500


# ---------------------------------------------------------------------------
# GET /api/analytics/summary
# ---------------------------------------------------------------------------

@bp.route("/summary")
def analytics_summary():
    """Return a dashboard summary combining all analytics dimensions."""
    try:
        cost_report = _cost_tracker().get_report()
        sla_reports = _sla_tracker().get_all_reports()
        anomaly_stats = _anomaly_detector().get_stats()
        forecaster_stats = _forecaster().get_stats()

        # SLA compliance overview
        compliant_slos = sum(1 for r in sla_reports if r.is_compliant)
        total_slos = len(sla_reports)

        # AutoTuner current config
        tuner_config: Dict[str, Any] = {}
        try:
            from vetinari.learning.auto_tuner import get_auto_tuner
            tuner_config = get_auto_tuner().get_config()
        except Exception:
            pass

        return jsonify({
            "status": "ok",
            "summary": {
                "cost": {
                    "total_usd": cost_report.total_cost_usd,
                    "total_tokens": cost_report.total_tokens,
                    "total_requests": cost_report.total_requests,
                },
                "sla": {
                    "compliant_slos": compliant_slos,
                    "total_slos": total_slos,
                    "compliance_pct": round(compliant_slos / max(total_slos, 1) * 100, 1),
                    "slo_names": [r.slo.name for r in sla_reports],
                },
                "anomalies": {
                    "total_detected": anomaly_stats.get("total_anomalies", 0),
                    "tracked_metrics": anomaly_stats.get("tracked_metrics", 0),
                },
                "forecasting": {
                    "tracked_metrics": forecaster_stats.get("tracked_metrics", 0),
                },
                "tuner_config": tuner_config,
                "generated_at": time.time(),
            },
        })
    except Exception as e:
        logger.error("analytics/summary error: %s", e)
        return jsonify({"status": "error", "error": str(e)}), 500
