import os
from pathlib import Path

from rag_app import iter_pdf_paths, pdf_source_id, source_output_path


def test_iter_pdf_paths_discovers_nested_and_uppercase_pdfs(tmp_path):
    knowledge = tmp_path / "knowledge"
    nested = knowledge / "region" / "year"
    nested.mkdir(parents=True)
    root_pdf = knowledge / "root.pdf"
    nested_pdf = nested / "report.PDF"
    ignored = nested / "notes.txt"
    root_pdf.write_bytes(b"%PDF root")
    nested_pdf.write_bytes(b"%PDF nested")
    ignored.write_text("not a PDF", encoding="utf-8")

    paths = list(iter_pdf_paths(knowledge))

    assert paths == [nested_pdf, root_pdf]
    assert [pdf_source_id(path, knowledge) for path in paths] == [
        "region/year/report.PDF",
        "root.pdf",
    ]


def test_iter_pdf_paths_deduplicates_overlapping_roots(tmp_path):
    knowledge = tmp_path / "knowledge"
    nested = knowledge / "nested"
    nested.mkdir(parents=True)
    pdf = nested / "report.pdf"
    pdf.write_bytes(b"%PDF")

    assert list(iter_pdf_paths(knowledge, [nested])) == [pdf]


def test_source_output_path_mirrors_nested_source(tmp_path):
    assert source_output_path(
        tmp_path,
        "region/year/report.pdf",
        ".json",
    ) == tmp_path / "region" / "year" / "report.json"


def test_source_output_path_rejects_parent_traversal(tmp_path):
    try:
        source_output_path(tmp_path, "../outside.pdf", ".json")
    except ValueError as exc:
        assert "Unsafe source path" in str(exc)
    else:
        raise AssertionError("parent traversal should be rejected")


def test_source_output_path_compacts_long_windows_paths(tmp_path):
    if os.name != "nt":
        return
    source_file = f"archive/year/{'x' * 240}.pdf"
    output = source_output_path(tmp_path, source_file, ".json")
    assert "_long_paths" in output.parts
    assert len(str(output.resolve())) < 248
