from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import api
import api_state
from api_routers import documents


def test_dashboard_is_public_but_api_routes_require_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret")
    client = TestClient(api.app)

    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert "Content-Security-Policy" in dashboard.headers

    protected = client.get("/api/documents")
    assert protected.status_code == 401


def test_spatial_html_is_served_read_only():
    client = TestClient(api.app)
    model = next(Path("spatial_data").glob("*.html"))

    response = client.get(f"/spatial_data/{model.name}")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_delete_document_reaches_ingest_after_removing_file(tmp_path, monkeypatch):
    target = tmp_path / "report.pdf"
    target.write_bytes(b"%PDF-1.4\n")
    api_state.settings = SimpleNamespace(
        knowledge_dir=tmp_path,
        extra_pdf_dirs=[],
    )
    calls = []
    monkeypatch.setattr(
        documents,
        "delete_document_and_rebuild",
        lambda relative_path: (
            calls.append(relative_path),
            target.unlink(),
        ),
    )

    result = documents.delete_document("report.pdf")

    assert result == {"status": "deleted", "file": "report.pdf"}
    assert target.exists() is False
    assert calls == [Path("report.pdf")]
