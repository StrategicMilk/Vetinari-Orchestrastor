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
    app.run(debug=True, port=5000)

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
import time
import threading
from collections import defaultdict
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

# Simple in-process rate limiter: max requests per IP per window
_rate_limit_lock = threading.Lock()
_request_counts: Dict[str, list] = defaultdict(list)
_RATE_LIMIT_REQUESTS = int(os.environ.get("VETINARI_RATE_LIMIT_REQUESTS", "60"))
_RATE_LIMIT_WINDOW = int(os.environ.get("VETINARI_RATE_LIMIT_WINDOW", "60"))  # seconds


def _is_rate_limited(ip: str) -> bool:
    """Return True if the IP has exceeded the rate limit."""
    now = time.time()
    cutoff = now - _RATE_LIMIT_WINDOW
    with _rate_limit_lock:
        timestamps = _request_counts[ip]
        # Evict old timestamps
        _request_counts[ip] = [t for t in timestamps if t > cutoff]
        if len(_request_counts[ip]) >= _RATE_LIMIT_REQUESTS:
            return True
        _request_counts[ip].append(now)
        return False


def create_app(debug: bool = False, cors_origin: str = None) -> 'Flask':
    """
    Create and configure Flask application for dashboard.
    
    Args:
        debug: Enable debug mode
        cors_origin: Allowed CORS origin. Defaults to localhost only in production,
                     or VETINARI_CORS_ORIGIN env var. Set to '*' only for development.

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
    app.config['DEBUG'] = debug

    # Determine allowed CORS origin — restrict to localhost by default
    _cors_origin = (
        cors_origin
        or os.environ.get("VETINARI_CORS_ORIGIN")
        or ("*" if debug else "http://localhost:5000")
    )

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
            logger.error(f"Error getting stats: {e}")
            return jsonify({"error": str(e)}), 500
    
    # === Metrics Endpoints ===
    
    @app.route('/api/v1/metrics/latest', methods=['GET'])
    def get_latest_metrics() -> Tuple[Dict[str, Any], int]:
        """Get latest metrics snapshot."""
        try:
            metrics = dashboard.get_latest_metrics()
            return jsonify(metrics.to_dict()), 200
        except Exception as e:
            logger.error(f"Error getting latest metrics: {e}")
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
            logger.error(f"Error getting timeseries: {e}")
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
            logger.error(f"Error searching traces: {e}")
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
            logger.error(f"Error getting trace detail: {e}")
            return jsonify({"error": str(e)}), 500
    
    # === Error Handlers ===
    
    @app.errorhandler(404)
    def not_found(error: Any) -> Tuple[Dict[str, str], int]:
        """Handle 404 errors."""
        return jsonify({"error": "Endpoint not found"}), 404
    
    @app.errorhandler(500)
    def internal_error(error: Any) -> Tuple[Dict[str, str], int]:
        """Handle 500 errors."""
        logger.error(f"Internal server error: {error}")
        return jsonify({"error": "Internal server error"}), 500
    
    # === Middleware ===
    
    @app.before_request
    def log_request():
        """Log incoming requests and enforce rate limiting."""
        logger.debug(f"{request.method} {request.path}")
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
        if _is_rate_limited(client_ip):
            return jsonify({"error": "Rate limit exceeded", "retry_after": _RATE_LIMIT_WINDOW}), 429
    
    @app.after_request
    def add_cors_headers(response: Any) -> Any:
        """Add CORS headers. Origin restricted to localhost by default in production."""
        response.headers['Access-Control-Allow-Origin'] = _cors_origin
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    return app


def run_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False) -> None:
    """
    Run the dashboard REST API server.
    
    Args:
        host: Server host address
        port: Server port
        debug: Enable debug mode
    """
    app = create_app(debug=debug)
    
    logger.info(f"Starting Vetinari Dashboard API on {host}:{port}")
    logger.info(f"API available at http://{host}:{port}/api/v1/")
    logger.info(f"Health check: http://{host}:{port}/api/v1/health")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    host = sys.argv[1] if len(sys.argv) > 1 else '0.0.0.0'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    debug = '--debug' in sys.argv
    
    run_server(host, port, debug)
