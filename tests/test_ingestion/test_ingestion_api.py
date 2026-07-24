from api_routers._deps import safe_pdf_name, IngestBusyError, try_acquire_ingest_lock, release_ingest_lock
from api_routers.ingestion import _ingest_payload
from fastapi import HTTPException
from ingestion.models import IngestionResult
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
