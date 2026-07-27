from pathlib import Path


def test_chat_markdown_is_sanitized_before_inner_html_assignment():
    dashboard = Path("dashboard.html").read_text(encoding="utf-8")

    assert "dompurify@3.2.6" in dashboard.lower()
    assert dashboard.count('integrity="sha384-') >= 3
    assert 'crossorigin="anonymous"' in dashboard
    assert "DOMPurify.sanitize" in dashboard
    assert "return marked.parse(raw);" not in dashboard


def test_dashboard_sends_api_key_when_loading_spatial_models():
    dashboard = Path("dashboard.html").read_text(encoding="utf-8")

    assert "fetch('/api/spatial-models', { headers: apiHdr() })" in dashboard
