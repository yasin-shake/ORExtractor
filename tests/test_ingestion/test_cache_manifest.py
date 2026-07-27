import json
from pathlib import Path

from ingestion.cache import (
    EnrichmentCache,
    build_manifest_entry,
    load_parser_cache,
    save_parser_cache,
    should_skip_pdf,
)
from ingestion.models import (
    PIPELINE_VERSION,
    ElementRecord,
    ParserQualityReport,
    ParserResult,
)


class _Settings:
    bedrock_visual_model_id = "haiku-test"
    chunk_size = 1400
    chunk_overlap = 150
    embed_model = "text-embedding-3-small"
    resolved_embedding_signature = {
        "provider": "qwen",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "dimensions": 1024,
        "normalize": True,
    }
    ingestion_backend = "docling"
    parser_primary = "docling"
    bedrock_visual_confidence_threshold = 0.85
    visual_reconstruct_charts = True
    visual_reconstruct_diagrams = True


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
        parser_result=ParserResult(
            source_file=pdf.name,
            parser="docling",
            parser_version="test",
            quality=ParserQualityReport(score=1.0),
        ),
    )
    assert entry["pipeline_version"] == PIPELINE_VERSION
    assert should_skip_pdf(entry, pdf, settings) is True

    entry2 = dict(entry)
    entry2["pipeline_version"] = "1"
    assert should_skip_pdf(entry2, pdf, settings) is False

    # Legacy fingerprints lack an embedding signature and must be rebuilt.
    from ingestion.cache import fingerprint_legacy

    assert should_skip_pdf(fingerprint_legacy(pdf), pdf, settings) is False

    entry3 = dict(entry)
    entry3["parser_policy"] = {
        **entry3["parser_policy"],
        "primary": "mineru",
    }
    assert should_skip_pdf(entry3, pdf, settings) is False

    entry4 = dict(entry)
    entry4["visual_enrichment_enabled"] = False
    assert should_skip_pdf(entry4, pdf, settings) is False


def test_text_only_manifest_is_resumable_in_text_only_mode(tmp_path):
    pdf = tmp_path / "text-only.pdf"
    pdf.write_bytes(b"%PDF text only")
    settings = _Settings()
    settings.resolved_visual_enrichment_enabled = False
    entry = build_manifest_entry(
        pdf,
        settings,
        element_count=1,
        visual_count=0,
        table_count=0,
        indexed_chunk_count=1,
        failed_element_ids=[],
        visual_enrichment_enabled=False,
        parser_result=ParserResult(
            source_file=pdf.name,
            parser="docling",
            parser_version="test",
            quality=ParserQualityReport(score=1.0),
        ),
    )

    assert should_skip_pdf(entry, pdf, settings) is True
    settings.resolved_visual_enrichment_enabled = True
    assert should_skip_pdf(entry, pdf, settings) is False


def test_accurate_table_manifest_satisfies_fast_text_first_request(tmp_path):
    pdf = tmp_path / "accurate.pdf"
    pdf.write_bytes(b"%PDF accurate")
    settings = _Settings()
    settings.docling_table_mode = "accurate"
    entry = build_manifest_entry(
        pdf,
        settings,
        element_count=1,
        visual_count=0,
        table_count=1,
        indexed_chunk_count=1,
        failed_element_ids=[],
        parser_result=ParserResult(
            source_file=pdf.name,
            parser="docling",
            parser_version="test",
            quality=ParserQualityReport(score=1.0),
        ),
    )

    settings.docling_table_mode = "fast"
    assert should_skip_pdf(entry, pdf, settings) is True


def test_parser_cache_roundtrip_and_source_invalidation(tmp_path):
    class _Parser:
        def cache_signature(self):
            return {"parser": "docling", "version": "test-version"}

    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF first")
    artifact_dir = tmp_path / "artifacts"
    result = ParserResult(
        source_file="r.pdf",
        parser="docling",
        parser_version="test-version",
        elements=[
            ElementRecord(
                element_id="n1",
                source_file="r.pdf",
                category="NarrativeText",
                text="cached",
                parser="docling",
            )
        ],
        quality=ParserQualityReport(score=1.0),
    )
    save_parser_cache(artifact_dir, pdf, _Settings(), _Parser(), result)
    cached = load_parser_cache(artifact_dir, pdf, _Settings(), _Parser())
    assert cached and cached.elements[0].text == "cached"

    pdf.write_bytes(b"%PDF changed")
    assert (
        load_parser_cache(artifact_dir, pdf, _Settings(), _Parser())
        is None
    )


def test_degraded_parser_result_is_not_persisted_or_resumable(tmp_path):
    class _Parser:
        def cache_signature(self):
            return {"parser": "docling", "version": "test-version"}

    pdf = tmp_path / "partial.pdf"
    pdf.write_bytes(b"%PDF partial")
    artifact_dir = tmp_path / "artifacts"
    settings = _Settings()
    degraded = ParserResult(
        source_file=pdf.name,
        parser="docling",
        parser_version="test-version",
        status="degraded",
        elements=[
            ElementRecord(
                element_id="partial",
                source_file=pdf.name,
                category="NarrativeText",
                text="Only a small part of the report",
            )
        ],
        quality=ParserQualityReport(
            score=0.53,
            text_coverage=0.32,
            reasons=["low_text_page_coverage"],
        ),
    )

    assert save_parser_cache(
        artifact_dir,
        pdf,
        settings,
        _Parser(),
        degraded,
    ) is False
    assert load_parser_cache(
        artifact_dir,
        pdf,
        settings,
        _Parser(),
    ) is None

    entry = build_manifest_entry(
        pdf,
        settings,
        element_count=1,
        visual_count=0,
        table_count=0,
        indexed_chunk_count=1,
        failed_element_ids=[],
        parser_result=degraded,
    )
    assert entry["ingestion_acceptance"]["accepted"] is False
    assert entry["ingestion_acceptance"]["retryable"] is True
    assert should_skip_pdf(entry, pdf, settings) is False
