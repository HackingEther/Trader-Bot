"""Integration tests for FastAPI API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from trader.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "environment" in data
        assert "timestamp" in data


class TestKillSwitchEndpoint:
    def test_get_killswitch(self, client: TestClient) -> None:
        response = client.get(
            "/api/killswitch/",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "active" in data

    def test_activate_killswitch(self, client: TestClient) -> None:
        response = client.post(
            "/api/killswitch/activate",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        assert response.json()["active"] is True

    def test_deactivate_killswitch(self, client: TestClient) -> None:
        client.post(
            "/api/killswitch/activate",
            headers={"Authorization": "Bearer test-token"},
        )
        response = client.post(
            "/api/killswitch/deactivate",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        assert response.json()["active"] is False

    def test_killswitch_requires_auth(self, client: TestClient) -> None:
        response = client.get("/api/killswitch/")
        assert response.status_code == 403


class TestConfigEndpoint:
    def test_config_requires_auth(self, client: TestClient) -> None:
        response = client.get("/api/config/")
        assert response.status_code == 403

    def test_config_with_auth(self, client: TestClient) -> None:
        response = client.get(
            "/api/config/",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "app_env" in data
        assert "symbol_universe" in data
