"""
Flask REST API for Vetinari Dashboard

Wraps the DashboardAPI to provide HTTP endpoints for:
- Metrics retrieval (latest, time-series)
- Alert management
- Trace exploration
- Dashboard statistics

Run the server:
    from vetinari.dashboard.rest_api import create_app
    
    app = create_app()
    app.run(port=5000)

API Endpoints:
    GET  /api/v1/metrics/latest          - Get latest metrics snapshot
    GET  /api/v1/metrics/timeseries      - Get time-series data for a metric
    GET  /api/v1/traces                  - Search or list traces
    GET  /api/v1/traces/<trace_id>       - Get trace detail
    GET  /api/v1/health                  - Health check
    GET  /api/v1/stats                   - Dashboard statistics
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

import os

try:
    from flask import Flask, request, jsonify, send_from_directory, render_template
except ImportError:
    Flask = None
    request = None
    jsonify = None
    render_template = None

# Resolve UI directories relative to this file's location
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_UI_DIR    = os.path.normpath(os.path.join(_THIS_DIR, '..', '..', 'ui'))
_TEMPLATES = os.path.join(_UI_DIR, 'templates')
_STATIC    = os.path.join(_UI_DIR, 'static')

from vetinari.dashboard.api import get_dashboard_api

logger = logging.getLogger(__name__)


def create_app(debug: bool = False) -> 'Flask':
    """
    Create and configure Flask application for dashboard.
    
    Args:
        debug: Enable debug mode
    
    Returns:
        Flask application instance
    """
    if Flask is None:
        raise ImportError("Flask is required for REST API. Install with: pip install flask")
    
    app = Flask(
        __name__,
        template_folder=_TEMPLATES,
        static_folder=_STATIC,
        static_url_path='/static',
    )
    app.config['JSON_SORT_KEYS'] = False
    
    if debug:
        app.config['DEBUG'] = True
    
    dashboard = get_dashboard_api()
    
    # === Dashboard UI ===

    @app.route('/dashboard', methods=['GET'])
    def dashboard_ui():
        """Serve the monitoring dashboard HTML page."""
        return render_template('dashboard.html')

    # === Health & Status ===
    
    @app.route('/api/v1/health', methods=['GET'])
    def health_check() -> Tuple[Dict[str, Any], int]:
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "vetinari-dashboard"
        }), 200
    
    @app.route('/api/v1/stats', methods=['GET'])
    def get_stats() -> Tuple[Dict[str, Any], int]:
        """Get dashboard statistics."""
        try:
            stats = dashboard.get_stats()
            return jsonify(stats), 200
        except Exception as e:
            logger.error("Error getting stats: %s", e)
            return jsonify({"error": str(e)}), 500
    
    # === Metrics Endpoints ===
    
    @app.route('/api/v1/metrics/latest', methods=['GET'])
    def get_latest_metrics() -> Tuple[Dict[str, Any], int]:
        """Get latest metrics snapshot."""
        try:
            metrics = dashboard.get_latest_metrics()
            return jsonify(metrics.to_dict()), 200
        except Exception as e:
            logger.error("Error getting latest metrics: %s", e)
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/v1/metrics/timeseries', methods=['GET'])
    def get_timeseries() -> Tuple[Dict[str, Any], int]:
        """
        Get time-series data for a metric.
        
        Query Parameters:
            metric (str): Metric name (latency, success_rate, token_usage, memory_latency)
            timerange (str): Time range (1h, 24h, 7d) - currently not filtered
            provider (str): Optional provider filter
        """
        try:
            metric = request.args.get('metric', 'latency')
            timerange = request.args.get('timerange', '24h')
            provider = request.args.get('provider', None)
            
            # Validate metric
            valid_metrics = ['latency', 'success_rate', 'token_usage', 'memory_latency']
            if metric not in valid_metrics:
                return jsonify({
                    "error": f"Invalid metric '{metric}'. Valid metrics: {valid_metrics}"
                }), 400
            
            ts_data = dashboard.get_timeseries_data(metric, timerange, provider)
            
            if ts_data is None:
                return jsonify({"error": f"No data available for metric '{metric}'"}), 404
            
            return jsonify(ts_data.to_dict()), 200
        except Exception as e:
            logger.error("Error getting timeseries: %s", e)
            return jsonify({"error": str(e)}), 500
    
    # === Trace Endpoints ===
    
    @app.route('/api/v1/traces', methods=['GET'])
    def search_traces() -> Tuple[Dict[str, Any], int]:
        """
        Search or list traces.
        
        Query Parameters:
            trace_id (str): Optional specific trace ID to search for
            limit (int): Maximum traces to return (default 100)
        """
        try:
            trace_id = request.args.get('trace_id', None)
            limit = request.args.get('limit', 100, type=int)
            
            if limit < 1 or limit > 1000:
                limit = 100
            
            traces = dashboard.search_traces(trace_id, limit)
            
            return jsonify({
                "count": len(traces),
                "traces": [t.to_dict() for t in traces]
            }), 200
        except Exception as e:
            logger.error("Error searching traces: %s", e)
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/v1/traces/<trace_id>', methods=['GET'])
    def get_trace_detail(trace_id: str) -> Tuple[Dict[str, Any], int]:
        """Get detailed information about a specific trace."""
        try:
            trace = dashboard.get_trace_detail(trace_id)
            
            if trace is None:
                return jsonify({"error": f"Trace '{trace_id}' not found"}), 404
            
            return jsonify(trace.to_dict()), 200
        except Exception as e:
            logger.error("Error getting trace detail: %s", e)
            return jsonify({"error": str(e)}), 500
    
    # === Analytics Endpoints (Phase 5) ===

    @app.route('/api/v1/analytics/cost', methods=['GET'])
    def get_cost_report() -> Tuple[Dict[str, Any], int]:
        """Get cost attribution report, optionally filtered."""
        try:
            from vetinari.analytics.cost import get_cost_tracker
            import time as _time
            agent = request.args.get('agent', None)
            task_id = request.args.get('task_id', None)
            since_str = request.args.get('since', None)
            since = float(since_str) if since_str else None
            report = get_cost_tracker().get_report(agent=agent, task_id=task_id, since=since)
            return jsonify(report.to_dict()), 200
        except Exception as e:
            logger.error("Error getting cost report: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/analytics/cost/top', methods=['GET'])
    def get_cost_top() -> Tuple[Dict[str, Any], int]:
        """Get top agents and models by cost."""
        try:
            from vetinari.analytics.cost import get_cost_tracker
            n = request.args.get('n', 5, type=int)
            tracker = get_cost_tracker()
            return jsonify({
                "top_agents": tracker.get_top_agents(n),
                "top_models": tracker.get_top_models(n),
            }), 200
        except Exception as e:
            logger.error("Error getting top costs: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/analytics/sla', methods=['GET'])
    def get_sla_reports() -> Tuple[Dict[str, Any], int]:
        """Get all SLA compliance reports."""
        try:
            from vetinari.analytics.sla import get_sla_tracker
            reports = get_sla_tracker().get_all_reports()
            return jsonify({
                "count": len(reports),
                "reports": [r.to_dict() for r in reports],
            }), 200
        except Exception as e:
            logger.error("Error getting SLA reports: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/analytics/sla/<name>', methods=['GET'])
    def get_sla_report(name: str) -> Tuple[Dict[str, Any], int]:
        """Get a single SLO compliance report."""
        try:
            from vetinari.analytics.sla import get_sla_tracker
            report = get_sla_tracker().get_report(name)
            if report is None:
                return jsonify({"error": f"SLO '{name}' not found"}), 404
            return jsonify(report.to_dict()), 200
        except Exception as e:
            logger.error("Error getting SLA report: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/analytics/anomalies', methods=['GET'])
    def get_anomalies() -> Tuple[Dict[str, Any], int]:
        """Get recent anomaly detections."""
        try:
            from vetinari.analytics.anomaly import get_anomaly_detector
            metric = request.args.get('metric', None)
            history = get_anomaly_detector().get_history(metric)
            return jsonify({
                "count": len(history),
                "anomalies": [a.to_dict() for a in history],
            }), 200
        except Exception as e:
            logger.error("Error getting anomalies: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/analytics/forecast', methods=['GET'])
    def get_forecast() -> Tuple[Dict[str, Any], int]:
        """Get forecast for a metric."""
        try:
            from vetinari.analytics.forecasting import get_forecaster, ForecastRequest
            metric = request.args.get('metric', 'adapter.latency')
            horizon = request.args.get('horizon', 5, type=int)
            method = request.args.get('method', 'linear_trend')
            req = ForecastRequest(metric=metric, horizon=horizon, method=method)
            result = get_forecaster().forecast(req)
            return jsonify(result.to_dict()), 200
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error("Error getting forecast: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/analytics/autotuner', methods=['GET'])
    def get_autotuner() -> Tuple[Dict[str, Any], int]:
        """Get AutoTuner config and action history."""
        try:
            from vetinari.learning.auto_tuner import get_auto_tuner
            tuner = get_auto_tuner()
            return jsonify({
                "config": tuner.get_config(),
                "history": tuner.get_history(),
            }), 200
        except Exception as e:
            logger.error("Error getting autotuner data: %s", e)
            return jsonify({"error": str(e)}), 500

    # === Inference Config Endpoints (Step 17) ===

    @app.route('/api/v1/config/inference-profiles', methods=['GET'])
    def get_inference_profiles() -> Tuple[Dict[str, Any], int]:
        """Get current inference profiles and stats."""
        try:
            from vetinari.config.inference_config import get_inference_config
            cfg = get_inference_config()
            return jsonify({
                "profiles": cfg.get_all_profiles(),
                "stats": cfg.get_stats(),
            }), 200
        except Exception as e:
            logger.error("Error getting inference profiles: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route('/api/v1/config/inference-profiles/effective', methods=['GET'])
    def get_effective_inference_params() -> Tuple[Dict[str, Any], int]:
        """Get effective params for a task_type + model_id combination."""
        try:
            from vetinari.config.inference_config import get_inference_config
            task_type = request.args.get('task_type', 'general')
            model_id = request.args.get('model_id', '')
            cfg = get_inference_config()
            params = cfg.get_effective_params(task_type, model_id)
            return jsonify({
                "task_type": task_type,
                "model_id": model_id,
                "params": params,
            }), 200
        except Exception as e:
            logger.error("Error getting effective params: %s", e)
            return jsonify({"error": str(e)}), 500

    # === Error Handlers ===
    
    @app.errorhandler(404)
    def not_found(error: Any) -> Tuple[Dict[str, str], int]:
        """Handle 404 errors."""
        return jsonify({"error": "Endpoint not found"}), 404
    
    @app.errorhandler(500)
    def internal_error(error: Any) -> Tuple[Dict[str, str], int]:
        """Handle 500 errors."""
        logger.error("Internal server error: %s", error)
        return jsonify({"error": "Internal server error"}), 500
    
    # === Middleware ===
    
    @app.before_request
    def log_request() -> None:
        """Log incoming requests."""
        logger.debug("%s %s", request.method, request.path)
    
    _DASHBOARD_ALLOWED_ORIGINS = {
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    }

    @app.after_request
    def add_cors_headers(response: Any) -> Any:
        """Add CORS headers restricted to localhost origins (P1.H2)."""
        origin = request.headers.get("Origin", "")
        if origin in _DASHBOARD_ALLOWED_ORIGINS:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Admin-Token'
            response.headers['Vary'] = 'Origin'
        elif 'Access-Control-Allow-Origin' in response.headers:
            del response.headers['Access-Control-Allow-Origin']
        return response
    
    return app


def run_server(host: str = '127.0.0.1', port: int = 5000, debug: bool = False) -> None:
    """
    Run the dashboard REST API server.
    
    Args:
        host: Server host address
        port: Server port
        debug: Enable debug mode
    """
    app = create_app(debug=debug)
    
    logger.info("Starting Vetinari Dashboard API on %s:%s", host, port)
    logger.info("API available at http://%s:%s/api/v1/", host, port)
    logger.info("Health check: http://%s:%s/api/v1/health", host, port)
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    host = sys.argv[1] if len(sys.argv) > 1 else '127.0.0.1'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    debug = '--debug' in sys.argv
    
    run_server(host, port, debug)
