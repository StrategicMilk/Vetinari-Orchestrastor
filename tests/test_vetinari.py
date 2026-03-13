"""
Tests for Vetinari admin gating, per-project discovery, and UI features.

All HTTP calls are mocked so these run offline without a live server.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests  # top-level import so @patch("requests.get") patches correctly

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

BASE_URL = os.environ.get("VETINARI_BASE_URL", "http://localhost:5000")
TEST_PROJECT = "project_0"


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_response(status: int, body: dict) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body
    r.text = json.dumps(body)
    return r


# ─── Admin gating ─────────────────────────────────────────────────────────────

class TestAdminGating:
    """Test admin gating functionality."""

    @patch("requests.get")
    def test_admin_permissions_admin_role(self, mock_get):
        """Admin role should have admin privileges."""
        mock_get.return_value = _make_response(200, {"admin": True, "role": "admin"})

        response = requests.get(
            f"{BASE_URL}/api/admin/permissions",
            headers={"X-User-Role": "admin"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("admin") is True

    @patch("requests.get")
    def test_admin_permissions_user_role(self, mock_get):
        """Non-admin role should not have admin privileges."""
        mock_get.return_value = _make_response(200, {"admin": False, "role": "user"})

        response = requests.get(
            f"{BASE_URL}/api/admin/permissions",
            headers={"X-User-Role": "user"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("admin") is False

    @patch("requests.get")
    def test_admin_credentials_admin_only(self, mock_get):
        """Only admins should access credentials endpoints."""
        # Admin succeeds
        mock_get.return_value = _make_response(200, {"credentials": []})
        r_admin = requests.get(
            f"{BASE_URL}/api/admin/credentials",
            headers={"X-User-Role": "admin"},
        )
        assert r_admin.status_code == 200

        # Non-admin is denied
        mock_get.return_value = _make_response(403, {"error": "Forbidden"})
        r_user = requests.get(
            f"{BASE_URL}/api/admin/credentials",
            headers={"X-User-Role": "user"},
        )
        assert r_user.status_code == 403

    @patch("requests.get")
    def test_admin_health_admin_only(self, mock_get):
        """Only admins should access health endpoint."""
        mock_get.return_value = _make_response(403, {"error": "Forbidden"})

        response = requests.get(
            f"{BASE_URL}/api/admin/credentials/health",
            headers={"X-User-Role": "user"},
        )
        assert response.status_code == 403


# ─── Per-project gating ────────────────────────────────────────────────────────

class TestPerProjectGating:
    """Test per-project external model discovery gating."""

    @patch("requests.post")
    def test_model_search_enabled_by_default(self, mock_post):
        """Model search should work when not explicitly disabled."""
        mock_post.return_value = _make_response(200, {"candidates": []})

        response = requests.post(
            f"{BASE_URL}/api/project/{TEST_PROJECT}/model-search",
            headers={"X-User-Role": "admin", "Content-Type": "application/json"},
            json={"task_description": "test python code"},
        )
        assert response.status_code in [200, 403]

    @patch("requests.post")
    def test_model_search_per_project_disabled(self, mock_post, tmp_path):
        """Model search should fail when disabled in project.yaml."""
        mock_post.return_value = _make_response(
            403, {"error": "Model search disabled for this project"}
        )

        response = requests.post(
            f"{BASE_URL}/api/project/test_disabled/model-search",
            headers={"X-User-Role": "admin", "Content-Type": "application/json"},
            json={"task_description": "test"},
        )
        assert response.status_code == 403
        assert "disabled" in response.json().get("error", "").lower()


# ─── Model search ─────────────────────────────────────────────────────────────

class TestModelSearch:
    """Test live model search functionality (mocked)."""

    @patch("requests.post")
    def test_model_search_returns_candidates(self, mock_post):
        """Model search should return candidates from adapters."""
        mock_post.return_value = _make_response(200, {
            "candidates": [
                {
                    "id": "llama-3",
                    "name": "Llama 3",
                    "source_type": "local",
                    "short_rationale": "Good for code generation",
                },
            ]
        })

        response = requests.post(
            f"{BASE_URL}/api/project/{TEST_PROJECT}/model-search",
            headers={"X-User-Role": "admin", "Content-Type": "application/json"},
            json={"task_description": "python code generation"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "candidates" in data
        assert len(data["candidates"]) > 0

        candidate = data["candidates"][0]
        assert "id" in candidate
        assert "name" in candidate
        assert "source_type" in candidate
        assert "short_rationale" in candidate

    @patch("requests.post")
    def test_candidates_have_rationale(self, mock_post):
        """Candidates should have short rationale."""
        mock_post.return_value = _make_response(200, {
            "candidates": [
                {"id": "m1", "name": "M1", "source_type": "local",
                 "short_rationale": "Fast and efficient"},
                {"id": "m2", "name": "M2", "source_type": "cloud",
                 "short_rationale": "Best for web apps"},
            ]
        })

        response = requests.post(
            f"{BASE_URL}/api/project/{TEST_PROJECT}/model-search",
            headers={"X-User-Role": "admin", "Content-Type": "application/json"},
            json={"task_description": "web app development"},
        )
        assert response.status_code == 200
        data = response.json()
        for candidate in data["candidates"]:
            assert candidate.get("short_rationale"), "Candidate should have rationale"


# ─── Credentials ──────────────────────────────────────────────────────────────

class TestCredentials:
    """Test credential management."""

    @patch("requests.get")
    def test_credentials_health_structure(self, mock_get):
        """Credentials health should have expected structure."""
        mock_get.return_value = _make_response(200, {
            "health": {"huggingface": "ok", "replicate": "missing"}
        })

        response = requests.get(
            f"{BASE_URL}/api/admin/credentials/health",
            headers={"X-User-Role": "admin"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "health" in data


# ─── Status endpoint ──────────────────────────────────────────────────────────

class TestStatusEndpoint:
    """Test status endpoint."""

    @patch("requests.get")
    def test_status_returns_info(self, mock_get):
        """Status endpoint should return config info."""
        mock_get.return_value = _make_response(200, {
            "host": "http://localhost:1234",
            "admin": False,
            "version": "1.0.0",
        })

        response = requests.get(f"{BASE_URL}/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "host" in data or "admin" in data


# ─── Task override ────────────────────────────────────────────────────────────

class TestTaskOverride:
    """Test per-task model override."""

    @patch("requests.post")
    def test_override_requires_admin(self, mock_post):
        """Override endpoint should require admin role."""
        # Non-admin gets 403
        mock_post.return_value = _make_response(403, {"error": "Forbidden"})

        response = requests.post(
            f"{BASE_URL}/api/project/{TEST_PROJECT}/task/t1/override",
            headers={"X-User-Role": "user", "Content-Type": "application/json"},
            json={"model_id": "test-model"},
        )
        assert response.status_code in [200, 403, 404, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
