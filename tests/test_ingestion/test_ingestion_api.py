from api_routers._deps import safe_pdf_name, IngestBusyError, try_acquire_ingest_lock, release_ingest_lock
from fastapi import HTTPException
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
