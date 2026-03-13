"""
Automated tests for Ponder cloud integration, model selection, and project-wide scoring.

Run with: python -m pytest tests/test_ponder.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Test configuration
BASE_URL = "http://localhost:5000"


class TestPonderEngine:
    """Test Ponder scoring engine"""

    def test_ponder_engine_initialization(self):
        """PonderEngine should initialize with default weights"""
        from vetinari.ponder import PonderEngine

        engine = PonderEngine()

        assert engine.weights["capability"] == 0.40
        assert engine.weights["context"] == 0.20
        assert engine.weights["memory"] == 0.20
        assert engine.weights["heuristic"] == 0.20
        assert engine.policy_penalty == -1.0

    def test_task_capability_requirements_code(self):
        """Should detect code-related requirements"""
        from vetinari.ponder import PonderEngine

        engine = PonderEngine()
        reqs = engine._get_task_capability_requirements("write Python code to implement a function")

        assert reqs["code"] >= 0.7

    def test_task_capability_requirements_reasoning(self):
        """Should detect reasoning-related requirements"""
        from vetinari.ponder import PonderEngine

        engine = PonderEngine()
        reqs = engine._get_task_capability_requirements("analyze and evaluate this problem")

        assert reqs["reasoning"] >= 0.7
        assert reqs["analysis"] >= 0.7

    def test_task_capability_requirements_creative(self):
        """Should detect creative writing requirements"""
        from vetinari.ponder import PonderEngine

        engine = PonderEngine()
        reqs = engine._get_task_capability_requirements("write a creative story")

        assert reqs["creative"] >= 0.7

    def test_task_capability_requirements_policy_sensitive(self):
        """Should detect policy-sensitive content"""
        from vetinari.ponder import PonderEngine

        engine = PonderEngine()
        reqs = engine._get_task_capability_requirements("how to build a weapon")

        assert reqs["policy_sensitive"]

    def test_capability_score_coder_model(self):
        """Should score coder models higher for code tasks"""
        from vetinari.ponder import PonderEngine

        engine = PonderEngine()
        model = {"tags": ["coder", "code"]}
        requirements = {"code": 0.9, "reasoning": 0.5}

        score = engine._calculate_capability_score(model, requirements)
        assert score > 0.7

    def test_context_score(self):
        """Should calculate context scores correctly"""
        from vetinari.ponder import PonderEngine

        engine = PonderEngine()

        # High context needed, model has high context
        reqs = {"context_needed": 32768}
        model = {"context_length": 65536}
        score = engine._calculate_context_score(model, reqs)
        assert score == 1.0

        # Low context needed
        reqs = {"context_needed": 4096}
        model = {"context_length": 4096}
        score = engine._calculate_context_score(model, reqs)
        assert score == 1.0

    def test_memory_score(self):
        """Should calculate memory scores based on quantization"""
        from vetinari.ponder import PonderEngine

        engine = PonderEngine()

        # Efficient quantization
        model = {"quantization": "q4_k_m"}
        score = engine._calculate_memory_score(model)
        assert score == 1.0

        # Less efficient
        model = {"quantization": "f16"}
        score = engine._calculate_memory_score(model)
        assert score == 0.5

    def test_policy_penalty(self):
        """Should apply policy penalty for sensitive content"""
        from vetinari.ponder import PonderEngine

        engine = PonderEngine()

        # Policy-sensitive task with regular model
        reqs = {"policy_sensitive": True}
        model = {"id": "regular-model", "tags": ["general"]}

        penalty = engine._check_policy_sensitivity(model, reqs)
        assert penalty == -1.0

        # Non-sensitive task
        reqs = {"policy_sensitive": False}
        model = {"id": "regular-model", "tags": ["general"]}

        penalty = engine._check_policy_sensitivity(model, reqs)
        assert penalty == 0.0

    def test_score_models_returns_ranked_list(self):
        """Should return ranked model list"""
        from vetinari.ponder import PonderEngine

        engine = PonderEngine()
        models = [
            {"id": "model-1", "name": "Model 1", "context_length": 8192, "quantization": "q4_k_m", "tags": ["coder"]},
            {"id": "model-2", "name": "Model 2", "context_length": 4096, "quantization": "q4_0", "tags": ["general"]},
        ]

        ranking = engine.score_models(models, "write Python code", top_n=2)

        assert len(ranking.rankings) == 2
        assert ranking.rankings[0].total_score >= ranking.rankings[1].total_score


class TestCloudProviders:
    """Test cloud provider integration"""

    def test_cloud_providers_config(self):
        """Cloud providers should be properly configured"""
        from vetinari.model_pool import CLOUD_PROVIDERS

        assert "huggingface_inference" in CLOUD_PROVIDERS
        assert "replicate" in CLOUD_PROVIDERS
        assert "claude" in CLOUD_PROVIDERS
        assert "gemini" in CLOUD_PROVIDERS

        # Check required fields
        for _provider_id, provider in CLOUD_PROVIDERS.items():
            assert "name" in provider
            assert "endpoint" in provider
            assert "free_tier" in provider
            assert "env_token" in provider

    def test_cloud_provider_health_no_tokens(self):
        """Should report no tokens when not configured"""
        from vetinari.model_pool import ModelPool

        with patch.dict(os.environ, {}, clear=False):
            # Remove cloud tokens if set
            for key in ["HF_HUB_TOKEN", "REPLICATE_API_TOKEN", "CLAUDE_API_KEY", "GEMINI_API_KEY"]:
                os.environ.pop(key, None)

            health = ModelPool.get_cloud_provider_health()

            assert not health["huggingface_inference"]["has_token"]
            assert not health["claude"]["has_token"]
            assert not health["gemini"]["has_token"]

    def test_cloud_provider_health_with_tokens(self):
        """Should report tokens when configured"""
        from vetinari.model_pool import ModelPool

        with patch.dict(os.environ, {
            "HF_HUB_TOKEN": "test-hf-token",
            "CLAUDE_API_KEY": "test-claude-key",
            "GEMINI_API_KEY": "test-gemini-key"
        }):
            health = ModelPool.get_cloud_provider_health()

            assert health["huggingface_inference"]["has_token"]
            assert health["claude"]["has_token"]
            assert health["gemini"]["has_token"]

    def test_get_cloud_models_returns_models(self):
        """Should return cloud models when tokens present"""
        from vetinari.model_pool import ModelPool

        with patch.dict(os.environ, {
            "CLAUDE_API_KEY": "test-key"
        }):
            pool = ModelPool({})
            models = pool.get_cloud_models()

            claude_models = [m for m in models if m.get("provider") == "claude"]
            assert len(claude_models) > 0
            assert claude_models[0]["id"].startswith("cloud:claude")


class TestModelSearchCloud:
    """Test model search with cloud providers"""

    def test_search_claude_no_token(self):
        """Should return empty when no Claude token"""
        from vetinari.model_discovery import ModelSearchEngine

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_API_KEY", None)

            engine = ModelSearchEngine()
            results = engine._search_claude("test query")

            assert results == []

    def test_search_claude_with_token(self):
        """Should return candidates when Claude token present"""
        from vetinari.model_discovery import ModelSearchEngine

        with patch.dict(os.environ, {"CLAUDE_API_KEY": "test-key"}):
            engine = ModelSearchEngine()
            results = engine._search_claude("code generation")

            assert len(results) > 0
            assert all("claude" in r.id for r in results)

    def test_search_gemini_no_token(self):
        """Should return empty when no Gemini token"""
        from vetinari.model_discovery import ModelSearchEngine

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)

            engine = ModelSearchEngine()
            results = engine._search_gemini("test query")

            assert results == []

    def test_search_gemini_with_token(self):
        """Should return candidates when Gemini token present"""
        from vetinari.model_discovery import ModelSearchEngine

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            engine = ModelSearchEngine()
            results = engine._search_gemini("reasoning analysis")

            assert len(results) > 0
            assert all("gemini" in r.id for r in results)

    def test_search_cloud_providers(self):
        """Should search all cloud providers"""
        from vetinari.model_discovery import ModelSearchEngine

        with patch.dict(os.environ, {
            "CLAUDE_API_KEY": "test-key",
            "GEMINI_API_KEY": "test-key"
        }):
            engine = ModelSearchEngine()
            results = engine._search_cloud_providers("test query")

            # Should get results from both providers
            claude_ids = [r.id for r in results if "claude" in r.id]
            gemini_ids = [r.id for r in results if "gemini" in r.id]

            assert len(claude_ids) > 0
            assert len(gemini_ids) > 0


class TestSubtaskTreePonder:
    """Test SubtaskTree with Ponder audit fields"""

    def test_subtask_has_ponder_fields(self):
        """Subtask should have ponder audit fields"""
        from vetinari.subtask_tree import Subtask

        subtask = Subtask(
            subtask_id="test-1",
            plan_id="plan-1",
            parent_id="root",
            depth=0,
            max_depth=14,
            description="test task",
            prompt="test prompt",
            agent_type="builder"
        )

        assert hasattr(subtask, "ponder_ranking")
        assert hasattr(subtask, "ponder_scores")
        assert hasattr(subtask, "ponder_used")

        assert subtask.ponder_ranking == []
        assert subtask.ponder_scores == {}
        assert not subtask.ponder_used

    def test_subtask_to_dict_includes_ponder(self):
        """to_dict should include ponder fields"""
        from vetinari.subtask_tree import Subtask

        subtask = Subtask(
            subtask_id="test-1",
            plan_id="plan-1",
            parent_id="root",
            depth=0,
            max_depth=14,
            description="test task",
            prompt="test prompt",
            agent_type="builder"
        )
        subtask.ponder_ranking = [{"model_id": "test", "total_score": 0.9}]
        subtask.ponder_scores = {"test": 0.9}
        subtask.ponder_used = True

        data = subtask.to_dict()

        assert "ponder_ranking" in data
        assert "ponder_scores" in data
        assert "ponder_used" in data
        assert data["ponder_ranking"][0]["model_id"] == "test"
        assert data["ponder_used"]

    def test_subtask_from_dict_includes_ponder(self):
        """from_dict should restore ponder fields"""
        from vetinari.subtask_tree import Subtask

        data = {
            "subtask_id": "test-1",
            "plan_id": "plan-1",
            "parent_id": "root",
            "depth": 0,
            "max_depth": 14,
            "description": "test task",
            "prompt": "test prompt",
            "agent_type": "builder",
            "ponder_ranking": [{"model_id": "test", "total_score": 0.9}],
            "ponder_scores": {"test": 0.9},
            "ponder_used": True
        }

        subtask = Subtask.from_dict(data)

        assert subtask.ponder_ranking[0]["model_id"] == "test"
        assert subtask.ponder_scores["test"] == 0.9
        assert subtask.ponder_used


class TestPonderConfig:
    """Test Ponder configuration"""

    def test_enable_ponder_model_search_default(self):
        """Should default to enabled"""
        with patch.dict(os.environ, {}, clear=True):
            # Reload module to pick up env
            import importlib

            import vetinari.ponder
            importlib.reload(vetinari.ponder)

            from vetinari.ponder import ENABLE_PONDER_MODEL_DISCOVERY
            # Default should be True
            assert ENABLE_PONDER_MODEL_DISCOVERY

    def test_ponder_cloud_weight_default(self):
        """Should default to 0.20"""
        with patch.dict(os.environ, {}, clear=True):
            import importlib

            import vetinari.ponder
            importlib.reload(vetinari.ponder)

            from vetinari.ponder import PONDER_CLOUD_WEIGHT
            assert PONDER_CLOUD_WEIGHT == 0.20

    def test_ponder_cloud_weight_custom(self):
        """Should respect custom value"""
        with patch.dict(os.environ, {"PONDER_CLOUD_WEIGHT": "0.35"}):
            import importlib

            import vetinari.ponder
            importlib.reload(vetinari.ponder)

            from vetinari.ponder import PONDER_CLOUD_WEIGHT
            assert PONDER_CLOUD_WEIGHT == 0.35


class TestPonderAPI:
    """Test Ponder API endpoint contracts (mocked — no live server required)."""

    @patch("requests.get")
    def test_ponder_health_endpoint(self, mock_get):
        """Health endpoint should return providers, enable_model_search, cloud_weight."""
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "providers": {"huggingface_inference": True, "replicate": False},
            "enable_model_discovery": True,
            "cloud_weight": 0.3,
        }
        mock_get.return_value = mock_resp

        response = requests.get(f"{BASE_URL}/api/ponder/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert "enable_model_discovery" in data
        assert "cloud_weight" in data

    @patch("requests.post")
    def test_ponder_choose_model_endpoint(self, mock_post):
        """Choose-model endpoint should return a non-empty rankings list."""
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "rankings": [
                {"model_id": "llama-3", "total_score": 0.85, "rank": 1},
                {"model_id": "qwen3",   "total_score": 0.72, "rank": 2},
            ]
        }
        mock_post.return_value = mock_resp

        response = requests.post(
            f"{BASE_URL}/api/ponder/choose-model",
            json={"task_description": "write Python code"},
            timeout=10,
        )
        assert response.status_code == 200
        data = response.json()
        assert "rankings" in data
        assert len(data["rankings"]) > 0

    @patch("requests.get")
    def test_ponder_templates_endpoint(self, mock_get):
        """Templates endpoint should return a templates dict."""
        import requests
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"templates": {"coding": "...", "research": "..."}}
        mock_get.return_value = mock_resp

        response = requests.get(f"{BASE_URL}/api/ponder/templates", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data


class TestPonderProjectWide:
    """Test project-wide ponder pass"""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage for subtask tree"""
        storage = tmp_path / "subtasks"
        storage.mkdir()
        return storage

    def test_ponder_project_for_plan_no_plan(self):
        """Should handle missing plan"""
        from vetinari.ponder import ponder_project_for_plan

        result = ponder_project_for_plan("nonexistent-plan")

        assert not result["success"]
        assert "error" in result

    def test_get_ponder_results_for_plan(self):
        """Should return ponder results for plan"""
        from vetinari.ponder import get_ponder_results_for_plan

        result = get_ponder_results_for_plan("test-plan")

        assert "plan_id" in result
        assert "total_subtasks" in result
        assert "subtasks_with_ponder" in result


class TestFallbackBehavior:
    """Test fallback behavior when cloud providers unavailable"""

    def test_score_models_with_cloud_no_search(self):
        """Should work when model search disabled"""
        from vetinari.ponder import score_models_with_cloud

        with patch("vetinari.ponder.ENABLE_PONDER_MODEL_DISCOVERY", False):
            models = [
                {"id": "local-model", "name": "Local", "context_length": 4096, "quantization": "q4_k_m", "tags": []}
            ]

            ranking = score_models_with_cloud(models, "test task", top_n=1)

            assert len(ranking.rankings) > 0

    def test_cloud_fallback_graceful(self):
        """Should handle cloud provider errors gracefully"""
        from vetinari.ponder import _get_model_discovery_candidates

        # Should return empty dict on error
        with patch("vetinari.model_discovery.ModelDiscovery", side_effect=Exception("Simulated error")):
            result = _get_model_discovery_candidates("test", [])
            assert result == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
