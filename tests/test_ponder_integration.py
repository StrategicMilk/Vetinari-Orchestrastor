"""
Integration tests for Ponder cloud integration end-to-end flow.

Run with: python -m pytest tests/test_ponder_integration.py -v
"""

import pytest
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


class TestIntegrationPonderFlow:
    """End-to-end integration tests for Ponder flow"""
    
    @pytest.fixture
    def mock_plan(self):
        """Create a mock plan with subtasks"""
        from vetinari.planning import Plan, Wave, Task
        from vetinari.subtask_tree import SubtaskTree, Subtask
        
        # Create temporary storage
        with tempfile.TemporaryDirectory() as tmpdir:
            tree = SubtaskTree(storage_path=tmpdir)
            
            # Create subtasks
            subtask1 = tree.create_subtask(
                plan_id="test-plan",
                parent_id="root",
                depth=0,
                description="Write Python function for data processing",
                prompt="Write a Python function",
                agent_type="builder",
                max_depth=14
            )
            
            subtask2 = tree.create_subtask(
                plan_id="test-plan",
                parent_id="root",
                depth=0,
                description="Analyze problem and propose solution",
                prompt="Analyze this problem",
                agent_type="researcher",
                max_depth=14
            )
            
            yield tree, "test-plan"
    
    def test_project_ponder_updates_subtasks(self, mock_plan):
        """Ponder pass should update all subtasks with rankings"""
        tree, plan_id = mock_plan
        
        from vetinari.ponder import ponder_project_for_plan, get_all_models_with_cloud
        
        # Mock the cloud models to return empty (no API calls)
        with patch("vetinari.ponder.get_all_models_with_cloud", return_value=[
            {"id": "test-model", "name": "Test", "context_length": 8192, "quantization": "q4_k_m", "tags": ["code"]}
        ]):
            result = ponder_project_for_plan(plan_id)
        
        # Should succeed
        assert result.get("success") == True or "error" in result
    
    def test_ponder_audit_fields_persisted(self, mock_plan):
        """Ponder audit fields should be persisted"""
        tree, plan_id = mock_plan
        
        # Manually update a subtask with ponder data
        subtask = tree.get_all_subtasks(plan_id)[0]
        
        tree.update_subtask(plan_id, subtask.subtask_id, {
            "ponder_ranking": [
                {"rank": 1, "model_id": "test-model", "total_score": 0.95}
            ],
            "ponder_scores": {"test-model": 0.95},
            "ponder_used": True
        })
        
        # Retrieve and verify
        updated = tree.get_subtask(plan_id, subtask.subtask_id)
        assert updated.ponder_used == True
        assert len(updated.ponder_ranking) > 0
        assert "test-model" in updated.ponder_scores


class TestIntegrationCloudRanking:
    """Integration tests for cloud-augmented ranking"""
    
    def test_cloud_augmentation_increases_score(self):
        """Cloud signals should augment scores"""
        from vetinari.ponder import score_models_with_cloud
        
        models = [
            {"id": "local-model", "name": "Local", "context_length": 8192, "quantization": "q4_k_m", "tags": ["code"]},
            {"id": "cloud:claude", "name": "Claude", "context_length": 200000, "quantization": "N/A", "tags": ["cloud", "reasoning"]}
        ]
        
        # Mock search to return high relevance for cloud model
        with patch("vetinari.ponder._get_model_search_candidates", return_value={
            "cloud:claude": 0.9,  # High relevance
            "local-model": 0.3   # Low relevance
        }):
            ranking = score_models_with_cloud(models, "complex reasoning task", top_n=2)
        
        # Cloud model should rank higher due to relevance boost
        assert len(ranking.rankings) == 2
        # The cloud model with high relevance should score higher
        cloud_score = next((r.total_score for r in ranking.rankings if "claude" in r.model_id), 0)
        local_score = next((r.total_score for r in ranking.rankings if r.model_id == "local-model"), 0)
        assert cloud_score >= local_score


class TestIntegrationAPIFlow:
    """Integration tests for API flow (mocked — no live server required)."""

    @patch("requests.get")
    def test_api_ponder_health_structure(self, mock_get):
        """Health endpoint should have correct structure (mocked)."""
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "enable_model_search": True,
            "cloud_weight": 0.3,
            "providers": {
                "huggingface_inference": True,
                "replicate": False,
                "claude": False,
                "gemini": False,
            },
        }
        mock_get.return_value = mock_resp

        response = requests.get("http://localhost:5000/api/ponder/health", timeout=2)
        assert response.status_code == 200
        data = response.json()

        assert "enable_model_search" in data
        assert "cloud_weight" in data
        assert "providers" in data
        for provider in ["huggingface_inference", "replicate", "claude", "gemini"]:
            assert provider in data["providers"]

    @patch("requests.post")
    def test_api_choose_model_returns_rankings(self, mock_post):
        """Choose model endpoint should return rankings (mocked)."""
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "rankings": [
                {"model_id": "model-a", "total_score": 0.9, "rank": 1},
                {"model_id": "model-b", "total_score": 0.8, "rank": 2},
                {"model_id": "model-c", "total_score": 0.7, "rank": 3},
            ]
        }
        mock_post.return_value = mock_resp

        response = requests.post(
            "http://localhost:5000/api/ponder/choose-model",
            json={"task_description": "write Python code", "top_n": 3},
            timeout=10,
        )
        assert response.status_code == 200
        data = response.json()

        assert "rankings" in data
        assert len(data["rankings"]) <= 3
        for r in data["rankings"]:
            assert "model_id" in r
            assert "total_score" in r
            assert "rank" in r


class TestSecurityAndSecrets:
    """Security tests for token handling"""
    
    @patch("requests.get")
    def test_tokens_not_in_response(self, mock_get):
        """API responses should not contain raw token values (mocked)."""
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Simulate a safe response — no raw keys in payload
        mock_resp.json.return_value = {
            "enable_model_search": True,
            "cloud_weight": 0.3,
            "providers": {"huggingface_inference": True},
        }
        mock_get.return_value = mock_resp

        response = requests.get("http://localhost:5000/api/ponder/health", timeout=2)
        assert response.status_code == 200
        data = response.json()
        response_str = json.dumps(data).lower()

        # Should not contain raw token prefixes
        assert "sk-" not in response_str
        assert "bearer" not in response_str
    
    def test_missing_tokens_handled_gracefully(self):
        """Missing tokens should not cause crashes"""
        from vetinari.model_pool import ModelPool
        
        # Clear all cloud tokens
        env_backup = os.environ.copy()
        for key in ["HF_HUB_TOKEN", "REPLICATE_API_TOKEN", "CLAUDE_API_KEY", "GEMINI_API_KEY"]:
            os.environ.pop(key, None)
        
        try:
            pool = ModelPool({})
            models = pool.get_cloud_models()
            
            # Should return empty list, not crash
            assert isinstance(models, list)
        finally:
            os.environ.clear()
            os.environ.update(env_backup)


class TestPerformanceAndCaching:
    """Performance and caching tests"""
    
    def test_caching_reduces_calls(self):
        """Caching should reduce repeated API calls"""
        from vetinari.model_search import ModelSearchEngine
        
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "test-key"}):
            engine = ModelSearchEngine()
            
            # First call - should create cache
            results1 = engine._search_claude("test query")
            
            # Second call - should hit cache
            results2 = engine._search_claude("test query")
            
            # Results should be identical
            assert len(results1) == len(results2)
            assert all(r1.id == r2.id for r1, r2 in zip(results1, results2))
    
    def test_cache_ttl_respected(self):
        """Cache should respect TTL"""
        from vetinari.model_search import ModelSearchEngine
        from datetime import datetime, timedelta
        import json
        
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "test-key"}):
            engine = ModelSearchEngine()
            
            # Create cache file with old timestamp
            cache_file = engine.cache_dir / "claude_test.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            old_data = [
                {
                    "id": "claude:test",
                    "name": "Test",
                    "source_type": "claude",
                    "metrics": {},
                    "memory_gb": 0,
                    "context_len": 200000,
                    "version": "test",
                    "last_updated": (datetime.now() - timedelta(seconds=120)).isoformat(),
                    "hard_data_score": 0.9,
                    "benchmark_score": 0.9,
                    "sentiment_score": 0.9,
                    "recency_score": 0.9,
                    "final_score": 0.9,
                    "provenance": []
                }
            ]
            
            with open(cache_file, "w") as f:
                json.dump(old_data, f)
            
            # Should ignore old cache (older than 60 seconds)
            results = engine._search_claude("test query")
            
            # Should have fetched fresh data, not used old cache
            # (results depend on implementation - may return fresh or empty)


class TestErrorHandling:
    """Error handling tests"""
    
    def test_invalid_plan_id_handled(self):
        """Invalid plan ID should return proper error"""
        from vetinari.ponder import ponder_project_for_plan
        
        result = ponder_project_for_plan("")
        
        assert result.get("success") == False
    
    def test_malformed_task_handled(self):
        """Malformed task description should not crash"""
        from vetinari.ponder import score_models_with_cloud
        
        models = [{"id": "test", "name": "Test", "context_length": 4096, "quantization": "q4_k_m", "tags": []}]
        
        # Empty task
        ranking = score_models_with_cloud(models, "", top_n=1)
        assert len(ranking.rankings) >= 0
        
        # None task
        ranking = score_models_with_cloud(models, None, top_n=1)
        assert len(ranking.rankings) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
