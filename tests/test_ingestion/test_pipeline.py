"""Pipeline wiring tests with mocked partitioner / Bedrock."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from ingestion.chunking import elements_to_documents
from ingestion.models import ElementRecord, IngestionResult
from ingestion.pipeline import IngestionPipeline


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
    visual_reconstruct_charts = True
    visual_reconstruct_diagrams = True
    visual_enrichment_enabled = False
    unstructured_provider = "local"
    unstructured_strategy = "hi_res"
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


def test_inspect_elements_uses_partitioner(tmp_path):
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
    pipeline.partitioner = MagicMock()
    pipeline.partitioner.partition.return_value = fake_elements

    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF")
    info = pipeline.inspect_elements(pdf)
    assert info["figures"] == 0
    assert info["element_category_counts"]["Title"] == 1
