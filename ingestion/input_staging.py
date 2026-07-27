"""Validate PDFs and expose them to native parsers through short ASCII paths."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator


@dataclass(frozen=True)
class StagedPdf:
    original_path: Path
    input_path: Path
    transfer: str


def validate_pdf_input(path: Path) -> int:
    """Fail before parser startup when a source is not a readable PDF."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF does not exist: {path}")
    with path.open("rb") as stream:
        if b"%PDF-" not in stream.read(1024):
            raise ValueError(f"Input is not a PDF: {path}")
    try:
        import fitz

        document = fitz.open(str(path))
        try:
            if document.needs_pass:
                raise ValueError(f"PDF is password protected: {path}")
            page_count = int(document.page_count)
        finally:
            document.close()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"PDF cannot be opened: {path}: {exc}") from exc
    if page_count < 1:
        raise ValueError(f"PDF contains no pages: {path}")
    return page_count


@contextmanager
def stage_pdf_input(
    pdf_path: Path,
    work_root: Path,
) -> Iterator[StagedPdf]:
    """Create a short-lived, short-path alias while retaining source metadata."""
    original = Path(pdf_path)
    validate_pdf_input(original)
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix="pdf-", dir=str(root)))
    staged_path = stage_dir / "input.pdf"
    transfer = "hardlink"
    try:
        try:
            os.link(original, staged_path)
        except OSError:
            shutil.copy2(original, staged_path)
            transfer = "copy"
        yield StagedPdf(
            original_path=original,
            input_path=staged_path,
            transfer=transfer,
        )
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

