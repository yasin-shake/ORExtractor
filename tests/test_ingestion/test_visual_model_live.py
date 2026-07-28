"""Opt-in live Qwen3-VL acceptance test using an isolated Chroma database."""

from __future__ import annotations

import os

import pytest
from langchain_core.embeddings import Embeddings

from ingestion.models import (
    ElementRecord,
    ParserQualityReport,
    ParserResult,
)
from ingestion.pipeline import IngestionPipeline
from ingestion.visual_benchmark import build_synthetic_visual_cases
from local_embeddings import embedding_signature
from rag_app import get_vectorstore, load_settings


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LOCAL_VLM_INTEGRATION") != "1",
    reason="set RUN_LOCAL_VLM_INTEGRATION=1 to call the local Qwen3-VL model",
)


class _DeterministicEmbeddings(Embeddings):
    def __init__(self):
        self.orextractor_embedding_signature = embedding_signature(
            provider="test",
            model="deterministic-3d",
            dimensions=3,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def test_qwen_visual_enrichment_indexes_into_isolated_chroma(tmp_path):
    synthetic_dir = tmp_path / "synthetic"
    build_synthetic_visual_cases(synthetic_dir)
    chart_path = synthetic_dir / "production-bar-chart.png"
    pdf_path = tmp_path / "live-qwen.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% isolated live acceptance fixture\n")

    settings = load_settings()
    settings.artifact_dir = tmp_path / "artifacts"
    settings.extracted_dir = tmp_path / "extracted"
    settings.chroma_dir = tmp_path / "chroma"
    settings.collection_name = "qwen_live_acceptance"
    settings.visual_model_provider = "ollama"
    settings.ollama_visual_model = "qwen3-vl:8b-instruct-q8_0"
    settings.visual_model_concurrency = 1
    settings.visual_max_calls_per_report = 1
    settings.visual_max_table_calls_per_report = 0
    settings.visual_max_figure_calls_per_report = 1
    settings.parser_fallback_enabled = False
    settings.parser_require_fallback_ready = False

    parser_result = ParserResult(
        source_file=pdf_path.name,
        parser="acceptance-fixture",
        parser_version="1",
        page_count=1,
        quality=ParserQualityReport(
            score=1.0,
            observed_page_count=1,
            expected_page_count=1,
            page_count_agreement=1.0,
        ),
        elements=[
            ElementRecord(
                element_id="title-1",
                source_file=pdf_path.name,
                category="Title",
                text="Item 14 - Mineral Resource Estimates",
                page_number=1,
            ),
            ElementRecord(
                element_id="figure-1",
                source_file=pdf_path.name,
                category="Figure",
                page_number=1,
                image_path=str(chart_path),
                image_width=1000,
                image_height=600,
            ),
            ElementRecord(
                element_id="caption-1",
                source_file=pdf_path.name,
                category="Caption",
                text="Figure 14-1 Annual Production",
                page_number=1,
            ),
        ],
    )

    class _FixtureRouter:
        @staticmethod
        def cache_signature():
            return {"parser": "acceptance-fixture", "version": "1"}

        @staticmethod
        def parse(*_args, **_kwargs):
            return parser_result

    pipeline = IngestionPipeline(settings, enable_visuals=True)
    pipeline.parser_router = _FixtureRouter()
    vectorstore = get_vectorstore(
        settings,
        _DeterministicEmbeddings(),
    )

    stats = pipeline.ingest_pdf(
        pdf_path,
        source_file=pdf_path.name,
        artifact_dir=settings.artifact_dir / "live-qwen",
        vectorstore=vectorstore,
    )
    hits = vectorstore.similarity_search(
        "annual copper production",
        k=stats.indexed_chunks,
    )

    assert stats.visual_model_provider == "ollama"
    assert stats.visual_model_calls == 1
    assert stats.failed_elements == []
    assert stats.indexed_chunks > 0
    assert vectorstore._collection.count() == stats.indexed_chunks
    assert any("Annual Production" in hit.page_content for hit in hits)
