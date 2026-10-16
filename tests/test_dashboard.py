"""Automated integration tests for FastAPI static dashboard serving."""

import pytest
from fastapi.testclient import TestClient

from app.main import FRONTEND_DIR, app


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_dashboard_path_resolution():
    """Verify FRONTEND_DIR is resolved relative to project location and exists."""
    assert FRONTEND_DIR.exists(), f"Frontend directory not found at: {FRONTEND_DIR}"
    assert FRONTEND_DIR.is_dir(), f"Frontend path is not a directory: {FRONTEND_DIR}"
    assert (FRONTEND_DIR / "index.html").is_file()
    assert (FRONTEND_DIR / "css" / "style.css").is_file()
    assert (FRONTEND_DIR / "js" / "api.js").is_file()
    assert (FRONTEND_DIR / "js" / "app.js").is_file()


def test_dashboard_serves_html(client):
    """Verify GET /dashboard/ serves dashboard HTML with 200 OK."""
    response = client.get("/dashboard/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "<title>AI Cybersecurity Assistant — Demo Dashboard</title>" in response.text
    assert "MOCK MODE" in response.text
    assert "id=\"chat-messages\"" in response.text


def test_dashboard_serves_css(client):
    """Verify GET /dashboard/css/style.css serves valid CSS with 200 OK."""
    response = client.get("/dashboard/css/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers.get("content-type", "")
    assert "--bg-canvas:" in response.text
    assert ".chat-workspace" in response.text


def test_dashboard_serves_js(client):
    """Verify GET /dashboard/js/api.js and app.js serve JavaScript with 200 OK."""
    resp_api = client.get("/dashboard/js/api.js")
    assert resp_api.status_code == 200
    assert any(t in resp_api.headers.get("content-type", "") for t in ("javascript", "ecmascript"))
    assert "window.AssistantAPI" in resp_api.text

    resp_app = client.get("/dashboard/js/app.js")
    assert resp_app.status_code == 200
    assert any(t in resp_app.headers.get("content-type", "") for t in ("javascript", "ecmascript"))
    assert "activeConversationId" in resp_app.text


def test_dashboard_missing_file_returns_404_no_exposure(client):
    """Verify non-existent asset returns HTTP 404 without exposing paths or directory listing."""
    response = client.get("/dashboard/nonexistent_secret.env")
    assert response.status_code == 404
    # Ensure raw filesystem paths are not exposed
    assert "D:/" not in response.text
    assert "D:\\" not in response.text


def test_root_info_contains_dashboard_url(client):
    """Verify root / endpoint links to the dashboard endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["dashboard_endpoint"] == "/dashboard/"
    assert data["status"] == "running"
