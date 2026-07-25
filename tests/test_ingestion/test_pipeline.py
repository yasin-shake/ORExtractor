"""Pipeline wiring tests with a mocked parser router and Bedrock."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from ingestion.chunking import elements_to_documents
from ingestion.models import (
    ElementRecord,
    ParserQualityReport,
    ParserResult,
    ReportIngestStats,
)
from ingestion.pipeline import (
    IngestionPipeline,
    _delete_document_ids,
    _source_document_ids,
)


class _Settings:
    knowledge_dir = Path(".")
    extra_pdf_dirs = []
    chroma_dir = Path(".chroma_db_test")
    collection_name = "test"
    chunk_size = 1400
    chunk_overlap = 150
    upsert_batch_size = 24
    artifact_dir = Path("ingestion_artifacts_test")
    extracted_dir = Path("extracted_data_test")
    bedrock_visual_model_id = "test-model"
    bedrock_visual_concurrency = 2
    bedrock_visual_confidence_threshold = 0.85
    visual_min_width = 250
    visual_min_height = 150
    visual_max_width = 4096
    visual_max_height = 4096
    visual_max_calls_per_report = 100
    visual_token_budget_per_report = 350000
    bedrock_visual_max_tokens = 3500
    visual_reconstruct_charts = True
    visual_reconstruct_diagrams = True
    visual_enrichment_enabled = False
    ingestion_backend = "docling"
    parser_primary = "docling"
    langsmith_tracing = False
    langsmith_project = "test"
    langsmith_trace_content = False
    aws_region = "us-east-2"
    embed_model = "text-embedding-3-small"
    openai_api_key = "sk-test"
    openai_base_url = None
    embed_batch_size = 64


def test_elements_to_documents_pipeline_boundary():
    els = [
        ElementRecord(
            element_id="n1",
            source_file="r.pdf",
            category="Title",
            text="Item 14 — Mineral Resource Estimates",
            page_number=1,
            ni_item=14,
            section_title="Mineral Resource Estimates",
        ),
        ElementRecord(
            element_id="n2",
            source_file="r.pdf",
            category="NarrativeText",
            text="The mineral resources are estimated using ordinary kriging.",
            page_number=1,
            ni_item=14,
            section_title="Mineral Resource Estimates",
        ),
    ]
    docs = elements_to_documents(els)
    assert docs
    assert all("source" in d.metadata for d in docs)
    assert all("ni_item" in d.metadata for d in docs)


def test_inspect_elements_uses_parser_router(tmp_path):
    settings = _Settings()
    settings.artifact_dir = tmp_path / "artifacts"
    pipeline = IngestionPipeline(settings, enable_visuals=False, partition_only=True)

    fake_elements = [
        ElementRecord(
            element_id="t1",
            source_file="r.pdf",
            category="Title",
            text="Item 1 — Summary",
            page_number=1,
        ),
        ElementRecord(
            element_id="n1",
            source_file="r.pdf",
            category="NarrativeText",
            text="Summary text",
            page_number=1,
        ),
    ]
    pipeline.parser_router = MagicMock()
    pipeline.parser_router.parse.return_value = ParserResult(
        source_file="r.pdf",
        parser="docling",
        parser_version="test",
        elements=fake_elements,
        quality=ParserQualityReport(score=1.0),
    )

    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF")
    info = pipeline.inspect_elements(pdf)
    assert info["figures"] == 0
    assert info["element_category_counts"]["Title"] == 1


def test_partition_only_reuses_parser_cache(tmp_path):
    settings = _Settings()
    settings.artifact_dir = tmp_path / "artifacts"
    settings.extracted_dir = tmp_path / "extracted"
    pipeline = IngestionPipeline(settings, enable_visuals=False, partition_only=True)
    pipeline.parser_router = MagicMock()
    pipeline.parser_router.cache_signature.return_value = {
        "parser": "docling",
        "version": "test-version",
    }
    pipeline.parser_router.parse.return_value = ParserResult(
        source_file="r.pdf",
        parser="docling",
        parser_version="test-version",
        elements=[
            ElementRecord(
                element_id="n1",
                source_file="r.pdf",
                category="NarrativeText",
                text="Body",
                page_number=1,
                parser="docling",
            )
        ],
        page_count=1,
        quality=ParserQualityReport(score=1.0),
    )
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF cache test")

    first = pipeline.ingest_pdf(pdf)
    second = pipeline.ingest_pdf(pdf)
    assert first.partition_cache_hits == 0
    assert second.partition_cache_hits == 1
    assert pipeline.parser_router.parse.call_count == 1


def test_stale_source_ids_are_discoverable_and_deletable():
    vectorstore = MagicMock()
    vectorstore._collection.get.return_value = {"ids": ["old-1", "keep"]}
    ids = _source_document_ids(vectorstore, "r.pdf")
    assert ids == {"old-1", "keep"}
    vectorstore._collection.get.assert_called_once_with(
        where={"source": "r.pdf"},
        include=[],
    )

    _delete_document_ids(vectorstore, {"old-1"})
    vectorstore._collection.delete.assert_called_once_with(ids=["old-1"])


def test_ingest_all_discovers_nested_pdfs_with_relative_source_ids(tmp_path):
    settings = _Settings()
    settings.knowledge_dir = tmp_path / "knowledge"
    settings.extra_pdf_dirs = []
    settings.artifact_dir = tmp_path / "artifacts"
    nested = settings.knowledge_dir / "region" / "year"
    nested.mkdir(parents=True)
    (settings.knowledge_dir / "root.pdf").write_bytes(b"%PDF root")
    (nested / "report.pdf").write_bytes(b"%PDF nested")

    pipeline = IngestionPipeline(
        settings,
        enable_visuals=False,
        partition_only=True,
    )
    pipeline.ingest_pdf = MagicMock(
        side_effect=lambda _path, source_file, **_kwargs: ReportIngestStats(
            filename=source_file
        )
    )

    result = pipeline.ingest_all(rebuild=True)

    assert result.files == ["region/year/report.pdf", "root.pdf"]
    assert [
        call.kwargs["source_file"]
        for call in pipeline.ingest_pdf.call_args_list
    ] == ["region/year/report.pdf", "root.pdf"]
    nested_call = pipeline.ingest_pdf.call_args_list[0]
    assert nested_call.kwargs["artifact_dir"] == (
        settings.artifact_dir / "region" / "year" / "report"
    )
