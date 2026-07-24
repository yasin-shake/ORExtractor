import json
from pathlib import Path

from ingestion.cache import (
    EnrichmentCache,
    build_manifest_entry,
    should_skip_pdf,
)
from ingestion.models import PIPELINE_VERSION, ElementRecord


class _Settings:
    bedrock_visual_model_id = "haiku-test"
    chunk_size = 1400
    chunk_overlap = 150
    embed_model = "text-embedding-3-small"


def test_enrichment_cache_roundtrip(tmp_path):
    cache = EnrichmentCache(tmp_path / "cache")
    img = tmp_path / "fig.png"
    img.write_bytes(b"fakepng")
    el = ElementRecord(
        element_id="f1",
        source_file="r.pdf",
        category="Image",
        page_number=1,
        image_path=str(img),
    )
    ctx = {"page_number": 1, "caption": "x"}
    assert cache.get_visual(el, ctx, "model", b"fakepng") is None
    cache.put_visual(el, ctx, "model", b"fakepng", {"figure_type": "line_chart", "confidence": 0.9})
    hit = cache.get_visual(el, ctx, "model", b"fakepng")
    assert hit["figure_type"] == "line_chart"


def test_manifest_invalidation_on_pipeline_version(tmp_path):
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    settings = _Settings()
    entry = build_manifest_entry(
        pdf,
        settings,
        element_count=10,
        visual_count=1,
        table_count=2,
        indexed_chunk_count=5,
        failed_element_ids=[],
        partitioner="unstructured-local",
        partition_strategy="hi_res",
    )
    assert entry["pipeline_version"] == PIPELINE_VERSION
    assert should_skip_pdf(entry, pdf, settings) is True

    entry2 = dict(entry)
    entry2["pipeline_version"] = "1"
    assert should_skip_pdf(entry2, pdf, settings) is False

    # Legacy fingerprint string still skips when mtime/size match
    from ingestion.cache import fingerprint_legacy

    assert should_skip_pdf(fingerprint_legacy(pdf), pdf, settings) is True
