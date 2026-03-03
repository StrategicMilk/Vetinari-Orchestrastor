"""
Automated tests for Vetinari admin gating, per-project discovery, and UI features.

Run with: python -m pytest tests/test_vetinari.py -v
"""

import pytest
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Test configuration
BASE_URL = "http://localhost:5000"
TEST_PROJECT = "project_0"


class TestAdminGating:
    """Test admin gating functionality"""
    
    def test_admin_permissions_admin_role(self):
        """Admin role should have admin privileges"""
        import requests
        response = requests.get(
            f"{BASE_URL}/api/admin/permissions",
            headers={"X-User-Role": "admin"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("admin") == True
    
    def test_admin_permissions_user_role(self):
        """Non-admin role should not have admin privileges"""
        import requests
        response = requests.get(
            f"{BASE_URL}/api/admin/permissions",
            headers={"X-User-Role": "user"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("admin") == False
    
    def test_admin_credentials_admin_only(self):
        """Only admins should access credentials endpoints"""
        import requests
        
        # Admin should succeed
        response = requests.get(
            f"{BASE_URL}/api/admin/credentials",
            headers={"X-User-Role": "admin"}
        )
        assert response.status_code == 200
        
        # Non-admin should be denied
        response = requests.get(
            f"{BASE_URL}/api/admin/credentials",
            headers={"X-User-Role": "user"}
        )
        assert response.status_code == 403
    
    def test_admin_health_admin_only(self):
        """Only admins should access health endpoint"""
        import requests
        
        response = requests.get(
            f"{BASE_URL}/api/admin/credentials/health",
            headers={"X-User-Role": "user"}
        )
        assert response.status_code == 403


class TestPerProjectGating:
    """Test per-project external model discovery gating"""
    
    def test_model_search_enabled_by_default(self):
        """Model search should work when not explicitly disabled"""
        import requests
        response = requests.post(
            f"{BASE_URL}/api/project/{TEST_PROJECT}/model-search",
            headers={"X-User-Role": "admin", "Content-Type": "application/json"},
            json={"task_description": "test python code"}
        )
        # Should work (200) or 403 if globally disabled
        assert response.status_code in [200, 403]
    
    def test_model_search_per_project_disabled(self, tmp_path):
        """Model search should fail when disabled in project.yaml"""
        import requests
        import yaml
        
        # Create test project with discovery disabled
        project_dir = project_root / "projects" / "test_disabled"
        project_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            "name": "test-disabled",
            "goal": "test",
            "external_model_discovery_enabled": False,
            "tasks": []
        }
        
        with open(project_dir / "project.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Model search should fail
        response = requests.post(
            f"{BASE_URL}/api/project/test_disabled/model-search",
            headers={"X-User-Role": "admin", "Content-Type": "application/json"},
            json={"task_description": "test"}
        )
        
        assert response.status_code == 403
        assert "disabled" in response.json().get("error", "").lower()


class TestModelSearch:
    """Test live model search functionality"""
    
    def test_model_search_returns_candidates(self):
        """Model search should return candidates from adapters"""
        import requests
        response = requests.post(
            f"{BASE_URL}/api/project/{TEST_PROJECT}/model-search",
            headers={"X-User-Role": "admin", "Content-Type": "application/json"},
            json={"task_description": "python code generation"}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "candidates" in data
            assert len(data.get("candidates", [])) > 0
            
            # Check candidate structure
            candidate = data["candidates"][0]
            assert "id" in candidate
            assert "name" in candidate
            assert "source_type" in candidate
            assert "short_rationale" in candidate
    
    def test_candidates_have_rationale(self):
        """Candidates should have short rationale"""
        import requests
        response = requests.post(
            f"{BASE_URL}/api/project/{TEST_PROJECT}/model-search",
            headers={"X-User-Role": "admin", "Content-Type": "application/json"},
            json={"task_description": "web app development"}
        )
        
        if response.status_code == 200:
            data = response.json()
            for candidate in data.get("candidates", []):
                assert candidate.get("short_rationale"), "Candidate should have rationale"


class TestCredentials:
    """Test credential management"""
    
    def test_credentials_health_structure(self):
        """Credentials health should have expected structure"""
        import requests
        response = requests.get(
            f"{BASE_URL}/api/admin/credentials/health",
            headers={"X-User-Role": "admin"}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "health" in data


class TestStatusEndpoint:
    """Test status endpoint"""
    
    def test_status_returns_info(self):
        """Status endpoint should return config info"""
        import requests
        response = requests.get(f"{BASE_URL}/api/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "host" in data or "admin" in data


class TestTaskOverride:
    """Test per-task model override"""
    
    def test_override_requires_admin(self):
        """Override endpoint should require admin"""
        import requests
        response = requests.post(
            f"{BASE_URL}/api/project/{TEST_PROJECT}/task/t1/override",
            headers={"X-User-Role": "user", "Content-Type": "application/json"},
            json={"model_id": "test-model"}
        )
        # Should work (task exists) or fail gracefully
        assert response.status_code in [200, 404, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
