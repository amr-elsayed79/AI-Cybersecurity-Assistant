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


def test_dashboard_safe_markdown_rendering_and_xss_protection(client):
    """Verify frontend safely renders Markdown using pure DOM APIs without innerHTML or outerHTML."""
    app_js_path = FRONTEND_DIR / "js" / "app.js"
    assert app_js_path.is_file()
    app_js = app_js_path.read_text(encoding="utf-8")

    # Strict XSS prevention: absolute ban on innerHTML / outerHTML
    assert ".innerHTML" not in app_js, "Forbidden .innerHTML detected in app.js!"
    assert ".outerHTML" not in app_js, "Forbidden .outerHTML detected in app.js!"

    # Safe DOM creation and text assignment
    assert "document.createTextNode" in app_js
    assert "document.createElement" in app_js
    assert "textContent" in app_js

    # Markdown structure helpers
    assert "appendFormattedInline" in app_js
    assert "appendSafeContentToBubble" in app_js

    # Markdown element creation
    assert "createElement('strong')" in app_js or "createElement('strong'" in app_js
    assert "createElement('code')" in app_js or "createElement('code'" in app_js
    assert "createElement('ul')" in app_js
    assert "createElement('ol')" in app_js
    assert "createElement('li')" in app_js
    assert "createElement('p')" in app_js

    # CSS styles for markdown elements exist
    style_css_path = FRONTEND_DIR / "css" / "style.css"
    assert style_css_path.is_file()
    style_css = style_css_path.read_text(encoding="utf-8")

    assert ".message-bubble p" in style_css
    assert ".message-bubble .md-heading" in style_css
    assert ".message-bubble ul.md-bullet-list" in style_css
    assert ".message-bubble ol.md-numbered-list" in style_css
    assert ".message-bubble strong" in style_css
    assert ".message-bubble code.inline-code" in style_css
    assert ".message-bubble pre.code-block-wrapper" in style_css


def test_dashboard_language_toggle_and_rtl_support(client):
    """Verify presence and configuration of explicit Arabic/English language buttons and RTL styles."""
    # 1. HTML contains explicit dual language buttons in switch group
    resp_html = client.get("/dashboard/")
    assert resp_html.status_code == 200
    assert 'class="lang-switch-group"' in resp_html.text
    assert 'id="lang-btn-ar"' in resp_html.text
    assert 'id="lang-btn-en"' in resp_html.text
    assert "العربية" in resp_html.text
    assert "English" in resp_html.text

    # 2. app.js contains explicit language button logic and translations
    app_js_path = FRONTEND_DIR / "js" / "app.js"
    app_js = app_js_path.read_text(encoding="utf-8")
    assert "langBtnAr" in app_js
    assert "langBtnEn" in app_js
    assert "setInterfaceLanguage" in app_js
    assert "UI_TRANSLATIONS" in app_js
    assert "document.documentElement.dir = isAr ? 'rtl' : 'ltr'" in app_js or "dir" in app_js
    assert "document.documentElement.lang = currentInterfaceLang" in app_js or "lang" in app_js

    # Verify conversation is preserved during language toggle (activeConversationId is untouched in setInterfaceLanguage)
    assert "setInterfaceLanguage" in app_js
    # Verify strict XSS safety is preserved (zero innerHTML/outerHTML)
    assert ".innerHTML" not in app_js
    assert ".outerHTML" not in app_js

    # 3. style.css contains .lang-switch-group, .lang-btn, and html[dir="rtl"] layout rules
    style_css_path = FRONTEND_DIR / "css" / "style.css"
    style_css = style_css_path.read_text(encoding="utf-8")
    assert ".lang-switch-group" in style_css
    assert ".lang-btn" in style_css
    assert 'html[dir="rtl"]' in style_css
    assert 'html[dir="rtl"] .chat-workspace' in style_css


def test_dashboard_session_id_removed_from_ui(client):
    """Verify Session ID is removed from visible Dashboard UI while in-memory state is preserved."""
    resp_html = client.get("/dashboard/")
    assert resp_html.status_code == 200

    # UI must not display active-session-id or Session ID label in rendered DOM
    assert 'id="active-session-id"' not in resp_html.text
    assert "Session ID:" not in resp_html.text

    # JavaScript retains activeConversationId in runtime memory for conversation continuity
    app_js_path = FRONTEND_DIR / "js" / "app.js"
    app_js = app_js_path.read_text(encoding="utf-8")
    assert "activeConversationId" in app_js
    assert "sendChatMessage(message, activeConversationId)" in app_js


def test_dashboard_evidence_and_health_panel_removed(client):
    """Verify the Knowledge Evidence and System Health panel and badges are removed from UI."""
    resp_html = client.get("/dashboard/")
    assert resp_html.status_code == 200
    html = resp_html.text

    # 1. Panel container and tabs must not exist in HTML
    assert "evidence-panel" not in html
    assert "tab-btn-evidence" not in html
    assert "tab-btn-diagnostics" not in html
    assert "tab-content-evidence" not in html
    assert "tab-content-diagnostics" not in html

    # 2. Header badges (Provider: Groq, Healthy pill) must not exist
    assert "llm-mode-badge" not in html
    assert "system-health-pill" not in html

    # 3. Specific texts must not be in the rendered UI
    assert "Knowledge Evidence System Health" not in html
    assert "Knowledge Evidence" not in html
    assert "Provider: Groq" not in html
    assert "588 KB Chunks" not in html
    assert "588 indexed chunks" not in html

    # 4. Essential chat workspace and language controls must be preserved
    assert 'class="chat-workspace"' in html
    assert 'id="chat-messages"' in html
    assert 'id="chat-input"' in html
    assert 'id="send-btn"' in html
    assert 'id="new-chat-btn"' in html
    assert 'class="lang-switch-group"' in html
    assert 'id="lang-btn-ar"' in html
    assert 'id="lang-btn-en"' in html

    # 5. CSS must no longer allocate 400px side column
    style_css = (FRONTEND_DIR / "css" / "style.css").read_text(encoding="utf-8")
    assert "grid-template-columns: 1fr 400px" not in style_css
    assert "400px" not in style_css
