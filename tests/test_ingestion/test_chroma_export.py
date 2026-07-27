import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import api_state
from api_routers import _deps
from ingestion.models import IngestionError, IngestionResult


class _Collection:
    def __init__(self):
        self.calls = 0

    def count(self):
        return 2

    def get(self, *, include, limit, offset):
        self.calls += 1
        if offset:
            return {"metadatas": []}
        return {
            "metadatas": [
                {"source": "existing.pdf"},
                {"source": "uploaded.pdf"},
            ]
        }


def _settings(tmp_path):
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    (chroma / "chroma.sqlite3").write_bytes(b"test")
    return SimpleNamespace(
        chroma_dir=chroma,
        collection_name="reports",
        embedding_provider="qwen",
        embed_model="qwen",
        chunk_size=1400,
        chunk_overlap=150,
        knowledge_dir=tmp_path / "knowledge",
        extra_pdf_dirs=[],
    )


def test_export_manifest_describes_sources_in_collection(tmp_path):
    api_state.settings = _settings(tmp_path)
    api_state.vectorstore = SimpleNamespace(_collection=_Collection())

    archive = _deps.archive_chroma()
    try:
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(
                bundle.read("export_manifest.json").decode("utf-8")
            )
    finally:
        archive.unlink(missing_ok=True)

    assert manifest["document_count"] == 2
    assert manifest["documents"] == ["existing.pdf", "uploaded.pdf"]


def test_ingest_export_rejects_failed_ingestion(monkeypatch):
    result = IngestionResult(
        status="failed",
        errors=[
            IngestionError(
                element_id="",
                stage="ingest-pdf",
                message="uploaded.pdf: parse failed",
            )
        ],
    )
    monkeypatch.setattr(_deps, "_run_ingest_unlocked", lambda **kwargs: result)
    monkeypatch.setattr(
        _deps,
        "archive_chroma",
        lambda: pytest.fail("failed ingestion must not be archived"),
    )

    with pytest.raises(_deps.IngestionFailedError, match="parse failed"):
        _deps.run_ingest_and_archive(False, ["uploaded.pdf"])
