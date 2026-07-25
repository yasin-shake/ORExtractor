from types import SimpleNamespace

import pytest

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
