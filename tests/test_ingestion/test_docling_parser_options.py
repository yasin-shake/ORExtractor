from types import SimpleNamespace

import fitz
import pytest

from ingestion.models import ElementRecord, ParserQualityReport, ParserResult
from ingestion.parsers.docling_parser import DoclingParser


def _settings(**overrides):
    values = {
        "docling_images_scale": 1.0,
        "docling_ocr_backend": "onnxruntime",
        "docling_ocr_languages": "english",
        "docling_force_full_page_ocr": False,
        "docling_ocr_bitmap_area_threshold": 0.05,
        "docling_ocr_batch_size": 1,
        "docling_layout_batch_size": 1,
        "docling_table_batch_size": 1,
        "docling_queue_max_size": 4,
        "docling_page_batch_size": 1,
        "docling_num_threads": 2,
        "docling_device": "auto",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pipeline_options_apply_low_memory_limits():
    parser = DoclingParser(_settings())

    options = parser._pipeline_options()

    assert options.images_scale == 1.0
    assert options.ocr_batch_size == 1
    assert options.layout_batch_size == 1
    assert options.table_batch_size == 1
    assert options.queue_max_size == 4
    assert options.accelerator_options.num_threads == 2
    assert str(options.accelerator_options.device) == "auto"


def test_nonpositive_concurrency_values_are_clamped():
    parser = DoclingParser(
        _settings(
            docling_ocr_batch_size=0,
            docling_layout_batch_size=-2,
            docling_table_batch_size=0,
            docling_queue_max_size=0,
            docling_num_threads=-1,
        )
    )

    options = parser._pipeline_options()

    assert options.ocr_batch_size == 1
    assert options.layout_batch_size == 1
    assert options.table_batch_size == 1
    assert options.queue_max_size == 1
    assert options.accelerator_options.num_threads == 1


def test_invalid_accelerator_device_is_rejected():
    parser = DoclingParser(_settings(docling_device="invalid"))

    with pytest.raises(ValueError, match="DOCLING_DEVICE"):
        parser._pipeline_options()


def test_gpu_rapidocr_options_are_configurable():
    parser = DoclingParser(
        _settings(
            docling_ocr_backend="torch",
            docling_ocr_languages="english,french",
            docling_force_full_page_ocr=True,
        )
    )

    options = parser._pipeline_options()

    assert options.ocr_options.backend == "torch"
    assert options.ocr_options.lang == ["english", "french"]
    assert options.ocr_options.force_full_page_ocr is True


def test_invalid_ocr_backend_is_rejected():
    parser = DoclingParser(_settings(docling_ocr_backend="invalid"))

    with pytest.raises(ValueError, match="DOCLING_OCR_BACKEND"):
        parser._pipeline_options()


def test_cache_signature_tracks_memory_settings():
    parser = DoclingParser(_settings(docling_queue_max_size=7))

    signature = parser.cache_signature()

    assert signature["queue_max_size"] == 7
    assert signature["page_batch_size"] == 1
    assert signature["num_threads"] == 2


def test_adaptive_ocr_skips_complete_native_text_pdf(tmp_path):
    pdf_path = tmp_path / "native.pdf"
    document = fitz.open()
    for _ in range(2):
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Native technical report text " * 10,
        )
    document.save(pdf_path)
    document.close()

    parser = DoclingParser(
        _settings(
            docling_adaptive_ocr=True,
            docling_native_text_min_chars=80,
            docling_native_text_coverage=0.98,
            docling_native_text_max_empty_pages=0,
        )
    )
    preflight = parser._preflight_pdf(pdf_path)

    assert preflight["effective_do_ocr"] is False
    assert preflight["native_text_coverage"] == 1.0
    assert parser._effective_table_mode(preflight) == "fast"


def test_adaptive_ocr_retains_ocr_for_empty_pages(tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    document = fitz.open()
    document.new_page()
    document.new_page()
    document.save(pdf_path)
    document.close()

    parser = DoclingParser(
        _settings(
            docling_adaptive_ocr=True,
            docling_native_text_max_empty_pages=0,
        )
    )

    assert parser._preflight_pdf(pdf_path)["effective_do_ocr"] is True


def test_native_memory_failure_retries_safe_batch(monkeypatch, tmp_path):
    class _FailingConverter:
        def convert(self, *_args, **_kwargs):
            raise RuntimeError("std::bad_alloc")

    class _SuccessfulConverter:
        def convert(self, *_args, **_kwargs):
            return SimpleNamespace(errors=[])

    parser = DoclingParser(
        _settings(
            docling_batch_fallback_enabled=True,
            docling_safe_batch_size=1,
            docling_page_batch_size=2,
        )
    )
    monkeypatch.setattr(
        parser,
        "_preflight_pdf",
        lambda _path: {
            "effective_do_ocr": True,
            "page_count": 10,
        },
    )
    monkeypatch.setattr(
        parser,
        "_get_converter",
        lambda **_kwargs: next(converters),
    )
    monkeypatch.setattr(
        parser,
        "_release_failed_converter",
        lambda _converter: None,
    )
    converters = iter([_FailingConverter(), _SuccessfulConverter()])

    conversion, runtime = parser._convert_local(tmp_path / "report.pdf")

    assert conversion.errors == []
    assert runtime["safe_batch_fallback_used"] is True
    assert runtime["page_batch_size"] == 1


def test_large_document_is_sent_to_docling_as_one_pdf(
    monkeypatch,
    tmp_path,
):
    pdf_path = tmp_path / "large.pdf"
    document = fitz.open()
    for index in range(5):
        document.new_page().insert_text(
            (72, 72),
            f"Native page {index + 1} " * 20,
        )
    document.save(pdf_path)
    document.close()

    class _Worker:
        def __init__(self):
            self.calls = []

        def parse(
            self,
            _path,
            *,
            source_file,
            artifact_dir,
            page_range=None,
            preflight=None,
        ):
            self.calls.append(page_range)
            first, last = (1, 5)
            elements = [
                ElementRecord(
                    element_id=f"p{page}",
                    source_file=source_file,
                    category="NarrativeText",
                    text=f"Page {page} body",
                    page_number=page,
                    metadata={"docling_ordinal": page},
                )
                for page in range(first, last + 1)
            ]
            return ParserResult(
                source_file=source_file,
                parser="docling",
                parser_version="test",
                elements=elements,
                page_count=len(elements),
                duration_ms=10,
                quality=ParserQualityReport(
                    score=1.0,
                    text_coverage=1.0,
                ),
                metadata={"runtime": {}},
            )

    worker = _Worker()
    parser = DoclingParser(
        _settings(
            docling_process_isolation=True,
            docling_segment_min_pages=3,
            docling_segment_pages=2,
            ingest_work_dir=tmp_path / "work",
            artifact_dir=tmp_path / "artifacts",
        )
    )
    monkeypatch.setattr(parser, "_worker_manager", lambda: worker)

    result = parser.parse(
        pdf_path,
        source_file="nested/large.pdf",
        artifact_dir=tmp_path / "artifacts" / "large",
    )

    assert worker.calls == [None]
    assert result.page_count == 5
    assert [element.page_number for element in result.elements] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert result.metadata["runtime"].get("segmented") is not True
