from pathlib import Path
from unittest.mock import MagicMock

import fitz

from ingestion.cache import build_manifest_entry, save_ingest_manifest
from ingestion.corpus import CorpusIngestion
from ingestion.models import (
    ElementRecord,
    ParserQualityReport,
    ParserResult,
    VisualAnalysis,
)
from ingestion.pipeline import IngestionPipeline
from ingestion.runtime import IngestionRuntime
from ingestion.visual_model import VisualResponse


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
    visual_model_provider = "ollama"
    ollama_visual_model = "qwen3-vl:test"
    bedrock_visual_model_id = "test-model"
    bedrock_visual_concurrency = 1
    visual_model_concurrency = 1
    bedrock_visual_confidence_threshold = 0.85
    visual_min_width = 250
    visual_min_height = 150
    visual_max_width = 4096
    visual_max_height = 4096
    visual_max_calls_per_report = 100
    visual_max_table_calls_per_report = 100
    visual_max_figure_calls_per_report = 100
    visual_token_budget_per_report = 350000
    bedrock_visual_max_tokens = 3500
    visual_reconstruct_charts = True
    visual_reconstruct_diagrams = True
    visual_enrichment_enabled = True
    ingestion_backend = "docling"
    parser_primary = "docling"
    parser_fallback = "mineru"
    parser_fallback_enabled = True
    langsmith_tracing = False
    embed_model = "fake-embedding"
    resolved_embedding_signature = {
        "provider": "test",
        "model": "fake",
        "dimensions": 3,
    }


def _write_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=400, height=400)
    page.draw_rect(fitz.Rect(50, 50, 350, 350), color=(0, 0, 0))
    document.save(path)
    document.close()


def test_visual_backfill_uses_canonical_artifact_without_parsing_and_resumes(
    monkeypatch,
    tmp_path,
):
    settings = _Settings()
    settings.knowledge_dir = tmp_path / "knowledge"
    settings.extra_pdf_dirs = []
    settings.chroma_dir = tmp_path / "chroma"
    settings.artifact_dir = tmp_path / "artifacts"
    settings.extracted_dir = tmp_path / "extracted"
    settings.visual_enrichment_enabled = True
    settings.resolved_visual_enrichment_enabled = True
    settings.resolved_embedding_provider = "test"
    settings.resolved_embedding_signature = {
        "provider": "test",
        "model": "fake",
        "dimensions": 3,
    }
    settings.knowledge_dir.mkdir()
    pdf = settings.knowledge_dir / "report.pdf"
    _write_pdf(pdf)

    parser_result = ParserResult(
        source_file=pdf.name,
        parser="docling",
        parser_version="test",
        page_count=1,
        elements=[
            ElementRecord(
                element_id="figure-1",
                source_file=pdf.name,
                category="Image",
                page_number=1,
                coordinates={
                    "l": 50,
                    "t": 350,
                    "r": 350,
                    "b": 50,
                    "coord_origin": "BOTTOMLEFT",
                },
                parser="docling",
                parser_version="test",
                metadata={"page_width": 400, "page_height": 400},
            )
        ],
        quality=ParserQualityReport(
            score=1.0,
            expected_page_count=1,
            observed_page_count=1,
            page_count_agreement=1.0,
        ),
    )
    artifact_dir = settings.artifact_dir / "report"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "parser_result.json").write_text(
        parser_result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "parser_cache.json").write_text(
        '{"source_sha256": "'
        + __import__("ingestion.cache", fromlist=["file_sha256"])
        .file_sha256(pdf)
        + '"}',
        encoding="utf-8",
    )
    save_ingest_manifest(
        settings.chroma_dir,
        {
            pdf.name: build_manifest_entry(
                pdf,
                settings,
                element_count=1,
                visual_count=1,
                table_count=0,
                indexed_chunk_count=1,
                failed_element_ids=[],
                visual_enrichment_enabled=False,
                parser_result=parser_result,
            )
        },
    )

    class CountingVisualModel:
        provider = "ollama"
        model_id = "qwen3-vl:test"
        cache_id = "ollama:qwen3-vl:test"

        def __init__(self):
            self.calls = 0

        def analyze(self, _request):
            self.calls += 1
            return VisualResponse(
                value=VisualAnalysis(
                    figure_type="technical_drawing",
                    description="A bounded source-page crop.",
                    confidence=0.95,
                )
            )

    model = CountingVisualModel()
    monkeypatch.setattr(
        "ingestion.enrichment.create_visual_model",
        lambda _settings: model,
    )
    vectorstore = MagicMock()
    vectorstore._collection.get.return_value = {"ids": []}
    runtime_calls = {"embedder": 0, "vectorstore": 0}

    def get_embedder(_settings):
        runtime_calls["embedder"] += 1
        return object()

    def get_vectorstore(_settings, _embedder):
        runtime_calls["vectorstore"] += 1
        return vectorstore

    runtime = IngestionRuntime(
        get_embedder=get_embedder,
        get_vectorstore=get_vectorstore,
    )
    pipeline = IngestionPipeline(
        settings,
        enable_visuals=True,
        runtime=runtime,
    )
    pipeline.parser_router = MagicMock()
    corpus = CorpusIngestion(
        settings,
        runtime=runtime,
        pipeline=pipeline,
    )

    first = corpus.resume_visuals(only_file=pdf.name)
    second = corpus.resume_visuals(only_file=pdf.name)

    assert first.status == "completed"
    assert first.reports[0].visual_model_calls == 1
    assert second.status == "completed"
    assert second.reports == []
    assert model.calls == 1
    assert runtime_calls == {"embedder": 1, "vectorstore": 1}
    pipeline.parser_router.parse.assert_not_called()
    updated = ParserResult.model_validate_json(
        (artifact_dir / "parser_result.json").read_text(encoding="utf-8")
    )
    crop = Path(updated.elements[0].image_path)
    assert crop.exists()
    assert updated.elements[0].image_width >= 250
    assert updated.elements[0].image_height >= 150


def test_visual_only_corpus_does_not_initialize_a_parser_router(
    monkeypatch,
    tmp_path,
):
    settings = _Settings()
    settings.knowledge_dir = tmp_path / "knowledge"
    settings.knowledge_dir.mkdir()
    settings.artifact_dir = tmp_path / "artifacts"
    settings.chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr(
        "ingestion.pipeline.get_parser_router",
        lambda _settings: (_ for _ in ()).throw(
            AssertionError("visual-only intent initialized a parser")
        ),
    )

    corpus = CorpusIngestion(settings)

    assert corpus.pipeline.parser_router is None
