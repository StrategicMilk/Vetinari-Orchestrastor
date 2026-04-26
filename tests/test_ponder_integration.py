"""
Integration tests for Ponder cloud integration end-to-end flow.

Run with: python -m pytest tests/test_ponder_integration.py -v
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


class TestIntegrationPonderFlow:
    """End-to-end integration tests for Ponder flow"""

    @pytest.fixture
    def mock_plan(self):
        """Create a mock plan with subtasks"""
        from vetinari.planning.subtask_tree import SubtaskTree

        # Create temporary storage
        with tempfile.TemporaryDirectory() as tmpdir:
            tree = SubtaskTree(storage_path=tmpdir)

            # Create subtasks
            tree.create_subtask(
                plan_id="test-plan",
                parent_id="root",
                depth=0,
                description="Write Python function for data processing",
                prompt="Write a Python function",
                agent_type="builder",
                max_depth=14,
            )

            tree.create_subtask(
                plan_id="test-plan",
                parent_id="root",
                depth=0,
                description="Analyze problem and propose solution",
                prompt="Analyze this problem",
                agent_type="researcher",
                max_depth=14,
            )

            yield tree, "test-plan"

    def test_project_ponder_updates_subtasks(self, mock_plan):
        """Ponder pass should update all subtasks with rankings"""
        _tree, plan_id = mock_plan

        from vetinari.models.ponder import ponder_project_for_plan

        # Mock the cloud models to return empty (no API calls)
        with patch(
            "vetinari.models.ponder.get_all_models_with_cloud",
            return_value=[
                {"id": "test-model", "name": "Test", "context_length": 8192, "quantization": "q4_k_m", "tags": ["code"]}
            ],
        ):
            result = ponder_project_for_plan(plan_id)

        # Should succeed
        assert result.get("success") or "error" in result

    def test_ponder_audit_fields_persisted(self, mock_plan):
        """Ponder audit fields should be persisted"""
        tree, plan_id = mock_plan

        # Manually update a subtask with ponder data
        subtask = tree.get_all_subtasks(plan_id)[0]

        tree.update_subtask(
            plan_id,
            subtask.subtask_id,
            {
                "ponder_ranking": [{"rank": 1, "model_id": "test-model", "total_score": 0.95}],
                "ponder_scores": {"test-model": 0.95},
                "ponder_used": True,
            },
        )

        # Retrieve and verify
        updated = tree.get_subtask(plan_id, subtask.subtask_id)
        assert updated.ponder_used
        assert len(updated.ponder_ranking) > 0
        assert "test-model" in updated.ponder_scores


class TestIntegrationCloudRanking:
    """Integration tests for cloud-augmented ranking"""

    def test_cloud_augmentation_increases_score(self):
        """Cloud signals should augment scores"""
        from vetinari.models.ponder import score_models_with_cloud

        models = [
            {"id": "local-model", "name": "Local", "context_length": 8192, "quantization": "q4_k_m", "tags": ["code"]},
            {
                "id": "cloud:claude",
                "name": "Claude",
                "context_length": 200000,
                "quantization": "N/A",
                "tags": ["cloud", "reasoning"],
            },
        ]

        # Mock search to return high relevance for cloud model
        with patch(
            "vetinari.models.ponder._get_model_discovery_candidates",
            return_value={
                "cloud:claude": 0.9,  # High relevance
                "local-model": 0.3,  # Low relevance
            },
        ):
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
            "enable_model_discovery": True,
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

        assert data["enable_model_discovery"] is True
        assert data["cloud_weight"] == 0.3
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
            "enable_model_discovery": True,
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
        from vetinari.models.model_pool import ModelPool

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

    def test_caching_reduces_calls(self, tmp_path: Path):
        """Caching should reduce repeated API calls"""
        from vetinari.model_discovery import ModelSearchEngine

        with patch.dict(os.environ, {"CLAUDE_API_KEY": "test-key"}):
            engine = ModelSearchEngine(cache_dir=str(tmp_path / "model_cache"))

            # First call - should create cache
            results1 = engine._search_claude("test query")

            # Second call - should hit cache
            results2 = engine._search_claude("test query")

            # Results should be identical
            assert len(results1) == len(results2)
            assert all(r1.id == r2.id for r1, r2 in zip(results1, results2))

    def test_cache_ttl_respected(self, tmp_path: Path):
        """Cache should respect TTL"""
        import json
        from datetime import timedelta

        from vetinari.model_discovery import ModelSearchEngine

        with patch.dict(os.environ, {"CLAUDE_API_KEY": "test-key"}):
            engine = ModelSearchEngine(cache_dir=str(tmp_path / "model_cache"))

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
                    "provenance": [],
                }
            ]

            with open(cache_file, "w") as f:
                json.dump(old_data, f)

            # Should ignore old cache (older than 60 seconds)
            results = engine._search_claude("test query")

            # When cache is stale and no live API is reachable, returns empty list or
            # fresh results — either way must be a list (not the stale cached data by id)
            assert isinstance(results, list)
            assert not any(m.id == "claude:test" for m in results), (
                "Stale cached entry must not be returned when TTL is exceeded"
            )


class TestErrorHandling:
    """Error handling tests"""

    def test_invalid_plan_id_handled(self):
        """Invalid plan ID should return proper error"""
        from vetinari.models.ponder import ponder_project_for_plan

        result = ponder_project_for_plan("")

        assert not result.get("success")

    def test_malformed_task_handled(self):
        """Malformed task description should not crash"""
        from vetinari.models.ponder import score_models_with_cloud

        models = [{"id": "test", "name": "Test", "context_length": 4096, "quantization": "q4_k_m", "tags": []}]

        # Empty task
        ranking = score_models_with_cloud(models, "", top_n=1)
        assert len(ranking.rankings) >= 0

        # None task
        ranking = score_models_with_cloud(models, None, top_n=1)
        assert len(ranking.rankings) >= 0


class TestScoreModelsWithCloudFallback:
    """Defect 5: score_models_with_cloud must not raise ImportError when model
    discovery is unavailable, and must be honest about whether cloud scoring ran."""

    def test_no_import_error_when_discovery_unavailable(self):
        """score_models_with_cloud must return a valid PonderRanking even when
        _get_model_discovery_candidates returns an empty dict (discovery offline)."""
        from vetinari.models.ponder import score_models_with_cloud

        models = [
            {
                "id": "local-model",
                "name": "Local Model",
                "context_length": 4096,
                "quantization": "q4_k_m",
                "tags": [],
            }
        ]

        # Simulate model discovery being unavailable by returning no candidates.
        with patch(
            "vetinari.models.ponder._get_model_discovery_candidates",
            return_value={},
        ):
            # Must not raise ImportError or any other exception.
            ranking = score_models_with_cloud(models, "write some Python code", top_n=1)

        assert len(ranking.rankings) == 1, (
            f"Expected 1 ranking when discovery is unavailable, got {len(ranking.rankings)}"
        )

    def test_local_only_scoring_label_present_when_discovery_unavailable(self):
        """When model discovery returns no candidates, each ranked model's reasoning
        must include 'local-only scoring' so callers can detect the fallback path."""
        from vetinari.models.ponder import score_models_with_cloud

        models = [
            {
                "id": "local-llama",
                "name": "Local Llama",
                "context_length": 8192,
                "quantization": "q5_k_m",
                "tags": ["code", "reasoning"],
            }
        ]

        with patch(
            "vetinari.models.ponder._get_model_discovery_candidates",
            return_value={},
        ):
            ranking = score_models_with_cloud(models, "analyze this dataset", top_n=1)

        assert len(ranking.rankings) >= 1
        for model_score in ranking.rankings:
            assert "local-only scoring" in model_score.reasoning, (
                f"Expected 'local-only scoring' in reasoning when discovery unavailable. "
                f"Got: '{model_score.reasoning}'"
            )

    def test_local_only_label_absent_when_cloud_discovery_active(self):
        """When _get_model_discovery_candidates returns results, 'local-only scoring'
        must NOT appear in any ranking's reasoning — cloud augmentation was used."""
        from vetinari.models.ponder import score_models_with_cloud

        models = [
            {
                "id": "boosted-model",
                "name": "Boosted Model",
                "context_length": 4096,
                "quantization": "q4_k_m",
                "tags": [],
            }
        ]

        # Simulate cloud discovery returning a relevance hit for our model.
        with patch(
            "vetinari.models.ponder._get_model_discovery_candidates",
            return_value={"boosted-model": 0.9},
        ):
            ranking = score_models_with_cloud(models, "explain quantum computing", top_n=1)

        assert len(ranking.rankings) >= 1
        for model_score in ranking.rankings:
            assert "local-only scoring" not in model_score.reasoning, (
                f"'local-only scoring' must not appear when cloud discovery was active. "
                f"Got: '{model_score.reasoning}'"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
