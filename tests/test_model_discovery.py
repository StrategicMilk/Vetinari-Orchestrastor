"""
Tests for model discovery retry logic and resilience.
Tests exponential backoff, fallback to static models, and health tracking.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import requests

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vetinari.model_pool import ModelPool


class TestModelDiscoveryRetry:
    """Test model discovery with retry logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "models": [
                {
                    "id": "static-model-1",
                    "name": "Static Model 1",
                    "capabilities": ["code_gen"],
                    "memory_gb": 4
                }
            ],
            "memory_budget_gb": 48
        }
        self.host = "http://localhost:1234"
    
    def test_successful_discovery_first_attempt(self):
        """Verify successful discovery on first attempt."""
        pool = ModelPool(self.config, host=self.host)
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "qwen-model", "memory_gb": 8}
            ]
        }
        mock_response.raise_for_status.return_value = None
        
        with patch.object(pool.session, 'get', return_value=mock_response):
            pool.discover_models()
        
        assert len(pool.models) >= 1
        assert not pool._discovery_failed
        assert pool._discovery_retry_count == 1
    
    def test_discovery_retry_on_timeout(self):
        """Verify retry happens on timeout."""
        pool = ModelPool(self.config, host=self.host)
        
        # First 2 calls timeout, 3rd succeeds
        mock_response = Mock()
        mock_response.json.return_value = {"data": [{"id": "recovered-model", "memory_gb": 6}]}
        mock_response.raise_for_status.return_value = None
        
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise requests.exceptions.Timeout("Connection timed out")
            return mock_response
        
        with patch.object(pool.session, 'get', side_effect=side_effect):
            pool.discover_models()
        
        # Should retry and eventually succeed
        assert call_count[0] == 3
        assert len(pool.models) >= 1
        assert not pool._discovery_failed
    
    def test_discovery_fallback_after_max_retries(self):
        """Verify fallback to static models after max retries."""
        pool = ModelPool(self.config, host=self.host)
        
        # All attempts fail
        with patch.object(pool.session, 'get', side_effect=requests.exceptions.Timeout("Timeout")):
            pool.discover_models()
        
        # Should have fallen back to static models
        assert pool._discovery_failed
        assert pool._fallback_active
        assert len(pool.models) >= 1
        assert any(m["id"] == "static-model-1" for m in pool.models)
    
    def test_discovery_with_connection_error(self):
        """Verify retry on connection error."""
        pool = ModelPool(self.config, host=self.host)
        
        mock_response = Mock()
        mock_response.json.return_value = {"data": [{"id": "model-after-retry", "memory_gb": 4}]}
        mock_response.raise_for_status.return_value = None
        
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise requests.exceptions.ConnectionError("Connection refused")
            return mock_response
        
        with patch.object(pool.session, 'get', side_effect=side_effect):
            pool.discover_models()
        
        assert call_count[0] == 2  # One failure, one success
        assert len(pool.models) >= 1
        assert not pool._discovery_failed


class TestModelDiscoveryFiltering:
    """Test model discovery filtering by memory budget."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "models": [],
            "memory_budget_gb": 16
        }
        self.host = "http://localhost:1234"
    
    def test_memory_budget_filtering(self):
        """Verify models exceeding memory budget are filtered."""
        pool = ModelPool(self.config, host=self.host, memory_budget_gb=16)
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "small-model", "memory_gb": 8},
                {"id": "large-model", "memory_gb": 32},  # Exceeds budget
                {"id": "medium-model", "memory_gb": 12}
            ]
        }
        mock_response.raise_for_status.return_value = None
        
        with patch.object(pool.session, 'get', return_value=mock_response):
            pool.discover_models()
        
        model_ids = [m["id"] for m in pool.models]
        assert "small-model" in model_ids
        assert "medium-model" in model_ids
        assert "large-model" not in model_ids  # Filtered out
    
    def test_default_memory_when_missing(self):
        """Verify default memory assignment when unknown."""
        pool = ModelPool(self.config, host=self.host)
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "model-no-memory-info"}  # No memory_gb field
            ]
        }
        mock_response.raise_for_status.return_value = None
        
        with patch.object(pool.session, 'get', return_value=mock_response):
            pool.discover_models()
        
        discovered = [m for m in pool.models if m["id"] == "model-no-memory-info"]
        assert len(discovered) == 1
        assert discovered[0]["memory_gb"] == 2  # Default value


class TestModelDiscoveryHealth:
    """Test discovery health tracking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"models": [], "memory_budget_gb": 32}
        self.host = "http://localhost:1234"
    
    def test_get_discovery_health_success(self):
        """Verify health reporting on success."""
        pool = ModelPool(self.config, host=self.host)
        
        mock_response = Mock()
        mock_response.json.return_value = {"data": [{"id": "test-model", "memory_gb": 4}]}
        mock_response.raise_for_status.return_value = None
        
        with patch.object(pool.session, 'get', return_value=mock_response):
            pool.discover_models()
        
        health = pool.get_discovery_health()
        
        assert health["discovery_failed"] is False
        assert health["fallback_active"] is False
        assert health["models_available"] >= 1
        assert health["retry_count"] == 1
        assert health["last_error"] is None
    
    def test_get_discovery_health_failure(self):
        """Verify health reporting on failure."""
        pool = ModelPool(self.config, host=self.host)
        
        with patch.object(pool.session, 'get', side_effect=requests.exceptions.Timeout("Timeout")):
            pool.discover_models()
        
        health = pool.get_discovery_health()
        
        assert health["discovery_failed"] is True
        assert health["fallback_active"] is True
        assert health["last_error"] is not None
        assert "Timeout" in health["last_error"]
        assert health["retry_count"] > 0


class TestModelDiscoveryResponseFormats:
    """Test handling of different response formats."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"models": [], "memory_budget_gb": 32}
        self.host = "http://localhost:1234"
    
    def test_response_with_data_field(self):
        """Test response with 'data' field wrapper."""
        pool = ModelPool(self.config, host=self.host)
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "model-in-data", "memory_gb": 4}]
        }
        mock_response.raise_for_status.return_value = None
        
        with patch.object(pool.session, 'get', return_value=mock_response):
            pool.discover_models()
        
        model_ids = [m["id"] for m in pool.models]
        assert "model-in-data" in model_ids
    
    def test_response_with_models_field(self):
        """Test response with 'models' field wrapper."""
        pool = ModelPool(self.config, host=self.host)
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [{"id": "model-in-models", "memory_gb": 4}]
        }
        mock_response.raise_for_status.return_value = None
        
        with patch.object(pool.session, 'get', return_value=mock_response):
            pool.discover_models()
        
        model_ids = [m["id"] for m in pool.models]
        assert "model-in-models" in model_ids
    
    def test_direct_list_response(self):
        """Test direct list response."""
        pool = ModelPool(self.config, host=self.host)
        
        mock_response = Mock()
        mock_response.json.return_value = [{"id": "direct-list-model", "memory_gb": 4}]
        mock_response.raise_for_status.return_value = None
        
        with patch.object(pool.session, 'get', return_value=mock_response):
            pool.discover_models()
        
        model_ids = [m["id"] for m in pool.models]
        assert "direct-list-model" in model_ids


class TestStaticModelInclusion:
    """Test that static models are always included."""
    
    def test_static_models_included_on_success(self):
        """Verify static models are included with discovered models."""
        config = {
            "models": [{"id": "static-1", "name": "Static 1", "memory_gb": 4}],
            "memory_budget_gb": 32
        }
        pool = ModelPool(config, host="http://localhost:1234")
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "discovered-1", "memory_gb": 4}]
        }
        mock_response.raise_for_status.return_value = None
        
        with patch.object(pool.session, 'get', return_value=mock_response):
            pool.discover_models()
        
        model_ids = [m["id"] for m in pool.models]
        assert "static-1" in model_ids
        assert "discovered-1" in model_ids
    
    def test_static_models_on_discovery_failure(self):
        """Verify static models are available when discovery fails."""
        config = {
            "models": [{"id": "static-fallback", "name": "Fallback", "memory_gb": 4}],
            "memory_budget_gb": 32
        }
        pool = ModelPool(config, host="http://localhost:1234")
        
        with patch.object(pool.session, 'get', side_effect=Exception("Discovery failed")):
            pool.discover_models()
        
        model_ids = [m["id"] for m in pool.models]
        assert "static-fallback" in model_ids
        assert pool._fallback_active


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
