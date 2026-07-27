from pathlib import Path

import fitz

from ingestion.input_staging import stage_pdf_input


def test_stages_long_unicode_pdf_under_short_ascii_path(tmp_path):
    source_dir = tmp_path / ("nested-" + "x" * 80)
    source_dir.mkdir()
    source = source_dir / ("Corporation Aurifère " + "y" * 80 + ".pdf")
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Valid staged PDF")
    document.save(source)
    document.close()
    work_root = tmp_path / "work"

    with stage_pdf_input(source, work_root) as staged:
        assert staged.original_path == source
        assert staged.input_path.read_bytes() == source.read_bytes()
        assert staged.input_path.suffix == ".pdf"
        assert staged.input_path.name.isascii()
        assert len(str(staged.input_path)) < len(str(source))
        staged_path = staged.input_path

    assert not staged_path.exists()
