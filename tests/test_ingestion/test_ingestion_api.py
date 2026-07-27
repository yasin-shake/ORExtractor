import asyncio
import io
from pathlib import Path
from types import SimpleNamespace

import api_state
from fastapi import HTTPException, UploadFile
from api_routers import _deps, documents
from api_routers._deps import safe_pdf_name, IngestBusyError, try_acquire_ingest_lock, release_ingest_lock
from api_routers.ingestion import _ingest_payload, upload_and_ingest
from ingestion.models import IngestionError, IngestionResult
import pytest


def test_safe_pdf_name_rejects_non_pdf():
    with pytest.raises(HTTPException) as exc:
        safe_pdf_name("notes.txt")
    assert exc.value.status_code == 400


def test_safe_pdf_name_strips_path():
    assert safe_pdf_name(r"C:\Users\x\evil.pdf") == "evil.pdf"
    assert safe_pdf_name("../traverse.pdf") == "traverse.pdf"


def test_ingest_lock_busy():
    try_acquire_ingest_lock()
    try:
        with pytest.raises(IngestBusyError):
            try_acquire_ingest_lock()
    finally:
        release_ingest_lock()


def test_structured_ingest_payload_includes_metrics_and_errors():
    result = IngestionResult(files=["r.pdf"])
    payload = _ingest_payload([], result, "rebuilt")
    assert payload["files"] == ["r.pdf"]
    assert "metrics" in payload
    assert payload["errors"] == []


def test_busy_upload_does_not_mutate_knowledge_directory(tmp_path):
    api_state.settings = SimpleNamespace(
        knowledge_dir=tmp_path,
        extra_pdf_dirs=[],
    )
    upload = UploadFile(
        filename="busy.pdf",
        file=io.BytesIO(b"%PDF-1.4\n"),
    )

    try_acquire_ingest_lock()
    try:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(upload_and_ingest([upload], None, None))
    finally:
        release_ingest_lock()

    assert exc.value.status_code == 409
    assert (tmp_path / "busy.pdf").exists() is False


def test_busy_delete_does_not_remove_document(tmp_path):
    target = tmp_path / "busy.pdf"
    target.write_bytes(b"%PDF-1.4\n")
    api_state.settings = SimpleNamespace(
        knowledge_dir=tmp_path,
        extra_pdf_dirs=[],
    )

    try_acquire_ingest_lock()
    try:
        with pytest.raises(HTTPException) as exc:
            documents.delete_document("busy.pdf")
    finally:
        release_ingest_lock()

    assert exc.value.status_code == 409
    assert target.exists() is True


def test_failed_upload_restores_previous_file(tmp_path, monkeypatch):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    target = knowledge / "report.pdf"
    target.write_bytes(b"old")
    api_state.settings = SimpleNamespace(
        knowledge_dir=knowledge,
        extra_pdf_dirs=[],
    )
    staging = _deps.create_upload_staging_dir()
    (staging / "report.pdf").write_bytes(b"new")
    result = IngestionResult(
        status="failed",
        errors=[
            IngestionError(
                element_id="",
                stage="ingest-pdf",
                message="report.pdf: parse failed",
            )
        ],
    )
    monkeypatch.setattr(
        _deps,
        "_run_ingest_unlocked",
        lambda **kwargs: result,
    )

    returned = _deps.run_staged_ingest(staging, ["report.pdf"])

    assert returned is result
    assert target.read_bytes() == b"old"
    assert staging.exists() is False


def test_failed_delete_restores_quarantined_file(tmp_path, monkeypatch):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    target = knowledge / "report.pdf"
    target.write_bytes(b"%PDF-1.4\n")
    api_state.settings = SimpleNamespace(
        knowledge_dir=knowledge,
        extra_pdf_dirs=[],
    )
    result = IngestionResult(
        status="failed",
        errors=[
            IngestionError(
                element_id="",
                stage="ingest-pdf",
                message="remaining.pdf: parse failed",
            )
        ],
    )
    monkeypatch.setattr(
        _deps,
        "_run_ingest_unlocked",
        lambda **kwargs: result,
    )

    with pytest.raises(_deps.IngestionFailedError, match="parse failed"):
        _deps.delete_document_and_rebuild(Path("report.pdf"))

    assert target.exists() is True
