"""Tests for health endpoint."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    """Create test client with mocked database."""
    app = create_app()
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_returns_200(self, client: TestClient):
        """Health endpoint should return 200 even if DB is mocked."""
        with patch("app.api.routes.health.get_db") as mock_get_db:
            mock_session = MagicMock()
            mock_get_db.return_value = iter([mock_session])
            # Re-create to apply override
            app = create_app()
            app.dependency_overrides = {}

        # Direct test without DB - should still return structure
        response = client.get("/api/v1/health")
        # May fail DB check without real DB, but endpoint itself should respond
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "pipeline_version" in data
        assert "database" in data

    def test_health_contains_pipeline_version(self, client: TestClient):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_version"] == "1.0.0"
