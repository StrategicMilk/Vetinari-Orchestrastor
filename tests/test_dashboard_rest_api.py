"""
Tests for Flask REST API endpoints

Tests cover:
- Health check endpoint
- Metrics endpoints (latest, timeseries)
- Trace endpoints (search, detail)
- Error handling and validation
"""

import pytest
import json
from datetime import datetime, timezone

# Check if Flask is available
try:
    from flask import Flask
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

if HAS_FLASK:
    from vetinari.dashboard.rest_api import create_app
    from vetinari.dashboard import reset_dashboard, get_dashboard_api


@pytest.mark.skipif(not HAS_FLASK, reason="Flask not installed")
class TestFlaskRestAPI:
    """Test Flask REST API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create Flask test client."""
        app = create_app(debug=True)
        with app.test_client() as client:
            yield client
    
    # === Health Endpoints ===
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/api/v1/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert data['service'] == 'vetinari-dashboard'
    
    def test_get_stats(self, client):
        """Test dashboard stats endpoint."""
        response = client.get('/api/v1/stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'total_traces_stored' in data
        assert 'trace_list_size' in data
        assert 'timestamp' in data
    
    # === Metrics Endpoints ===
    
    def test_get_latest_metrics(self, client):
        """Test getting latest metrics."""
        response = client.get('/api/v1/metrics/latest')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'timestamp' in data
        assert 'uptime_ms' in data
        assert 'adapters' in data
        assert 'memory' in data
        assert 'plan' in data
    
    def test_latest_metrics_structure(self, client):
        """Test structure of latest metrics response."""
        response = client.get('/api/v1/metrics/latest')
        data = json.loads(response.data)
        
        # Check adapter section
        assert 'total_providers' in data['adapters']
        assert 'total_requests' in data['adapters']
        assert 'providers' in data['adapters']
        
        # Check memory section
        assert 'backends' in data['memory']
        
        # Check plan section
        assert 'total_decisions' in data['plan']
        assert 'approval_rate' in data['plan']
    
    def test_get_timeseries_latency(self, client):
        """Test getting latency time-series."""
        dashboard = get_dashboard_api()
        dashboard.telemetry.record_adapter_latency("openai", "gpt-4", 150.0)
        
        response = client.get('/api/v1/metrics/timeseries?metric=latency')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['metric'] == 'latency'
        assert data['unit'] == 'ms'
        assert 'min' in data
        assert 'max' in data
        assert 'avg' in data
        assert 'points' in data
    
    def test_get_timeseries_success_rate(self, client):
        """Test getting success rate time-series."""
        response = client.get('/api/v1/metrics/timeseries?metric=success_rate')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['metric'] == 'success_rate'
        assert data['unit'] == '%'
    
    def test_get_timeseries_token_usage(self, client):
        """Test getting token usage time-series."""
        response = client.get('/api/v1/metrics/timeseries?metric=token_usage')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['metric'] == 'token_usage'
        assert data['unit'] == 'tokens'
    
    def test_get_timeseries_memory_latency(self, client):
        """Test getting memory latency time-series."""
        response = client.get('/api/v1/metrics/timeseries?metric=memory_latency')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['metric'] == 'memory_latency'
        assert data['unit'] == 'ms'
    
    def test_timeseries_with_provider_filter(self, client):
        """Test time-series with provider filtering."""
        dashboard = get_dashboard_api()
        dashboard.telemetry.record_adapter_latency("openai", "gpt-4", 100.0)
        dashboard.telemetry.record_adapter_latency("anthropic", "claude", 120.0)
        
        response = client.get('/api/v1/metrics/timeseries?metric=latency&provider=openai')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['metric'] == 'latency'
    
    def test_timeseries_invalid_metric(self, client):
        """Test time-series with invalid metric name."""
        response = client.get('/api/v1/metrics/timeseries?metric=invalid')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'invalid' in data['error'].lower()
    
    def test_timeseries_with_timerange(self, client):
        """Test time-series with timerange parameter."""
        response = client.get('/api/v1/metrics/timeseries?metric=latency&timerange=24h')
        
        assert response.status_code == 200
    
    # === Trace Endpoints ===
    
    def test_search_traces_empty(self, client):
        """Test searching when no traces exist."""
        response = client.get('/api/v1/traces')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['count'] == 0
        assert data['traces'] == []
    
    def test_search_traces_list_recent(self, client):
        """Test listing recent traces."""
        dashboard = get_dashboard_api()
        
        from vetinari.dashboard import TraceDetail
        for i in range(5):
            trace = TraceDetail(
                f"trace{i}",
                f"2026-03-03T12:{i:02d}:00Z",
                f"2026-03-03T12:{i:02d}:00.5Z",
                500.0,
                "success",
                []
            )
            dashboard.add_trace(trace)
        
        response = client.get('/api/v1/traces')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['count'] == 5
        assert len(data['traces']) == 5
    
    def test_search_traces_with_limit(self, client):
        """Test searching traces with limit."""
        dashboard = get_dashboard_api()
        
        from vetinari.dashboard import TraceDetail
        for i in range(10):
            trace = TraceDetail(
                f"trace{i}",
                f"2026-03-03T12:{i%6:02d}:00Z",
                f"2026-03-03T12:{i%6:02d}:00.5Z",
                500.0,
                "success",
                []
            )
            dashboard.add_trace(trace)
        
        response = client.get('/api/v1/traces?limit=3')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['count'] <= 3
    
    def test_search_traces_by_id(self, client):
        """Test searching for specific trace by ID."""
        dashboard = get_dashboard_api()
        
        from vetinari.dashboard import TraceDetail
        trace = TraceDetail(
            "specific_trace",
            "2026-03-03T12:00:00Z",
            "2026-03-03T12:00:00.5Z",
            500.0,
            "success",
            []
        )
        dashboard.add_trace(trace)
        
        response = client.get('/api/v1/traces?trace_id=specific_trace')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['count'] == 1
        assert data['traces'][0]['trace_id'] == 'specific_trace'
    
    def test_get_trace_detail(self, client):
        """Test getting trace detail."""
        dashboard = get_dashboard_api()
        
        from vetinari.dashboard import TraceDetail
        trace = TraceDetail(
            "trace123",
            "2026-03-03T12:00:00Z",
            "2026-03-03T12:00:00.5Z",
            500.0,
            "success",
            [{"span_id": "span1", "operation": "test"}]
        )
        dashboard.add_trace(trace)
        
        response = client.get('/api/v1/traces/trace123')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['trace_id'] == 'trace123'
        assert data['duration_ms'] == 500.0
        assert len(data['spans']) == 1
    
    def test_get_nonexistent_trace(self, client):
        """Test getting nonexistent trace."""
        response = client.get('/api/v1/traces/nonexistent')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'not found' in data['error'].lower()
    
    # === Error Handling ===
    
    def test_404_not_found(self, client):
        """Test 404 error for unknown endpoint."""
        response = client.get('/api/v1/unknown')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_invalid_limit_parameter(self, client):
        """Test validation of limit parameter."""
        response = client.get('/api/v1/traces?limit=5000')
        
        # Should clamp to 1000 or use default
        assert response.status_code == 200
    
    def test_timeseries_with_no_data(self, client):
        """Test timeseries when no data is available."""
        
        response = client.get('/api/v1/metrics/timeseries?metric=latency')
        
        # Should still return success with empty data
        assert response.status_code in [200, 404]
    
    # === Response Format ===
    
    def test_response_json_format(self, client):
        """Test that all responses are valid JSON."""
        response = client.get('/api/v1/health')
        
        assert response.content_type == 'application/json'
        # Should not raise JSON decode error
        data = json.loads(response.data)
        assert data is not None
    
    def test_error_response_format(self, client):
        """Test error response format."""
        response = client.get('/api/v1/metrics/timeseries?metric=invalid')
        
        assert response.content_type == 'application/json'
        data = json.loads(response.data)
        assert 'error' in data
    
    # === CORS Headers ===
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present."""
        response = client.get('/api/v1/health')
        
        assert 'Access-Control-Allow-Origin' in response.headers
        assert 'Access-Control-Allow-Methods' in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
