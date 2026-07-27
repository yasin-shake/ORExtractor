"""Pipeline wiring tests with a mocked parser router and Bedrock."""

from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import ingestion.indexing as indexing
import ingestion.pipeline as pipeline_module
from ingestion.chunking import elements_to_documents
from ingestion.models import (
    ElementRecord,
    ParserQualityReport,
    ParserResult,
    ReportIngestStats,
)
from ingestion.pipeline import (
    IngestionPipeline,
    _ReportWork,
    _delete_document_ids,
    _promote_rebuilt_collection,
    _source_document_ids,
)
from ingestion.runtime import IngestionRuntime


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


def test_text_first_run_uses_fast_tables_without_mutating_base_settings():
    settings = _Settings()
    settings.docling_table_mode = "accurate"
    settings.docling_text_first_table_mode = "fast"
    settings.docling_generate_page_images = True
    settings.docling_generate_picture_images = True

    pipeline = IngestionPipeline(
        settings,
        enable_visuals=False,
        partition_only=False,
    )

    assert pipeline.settings.docling_table_mode == "fast"
    assert pipeline.settings.docling_generate_page_images is False
    assert pipeline.settings.docling_generate_picture_images is False
    assert settings.docling_table_mode == "accurate"
    assert settings.docling_generate_page_images is True


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


def test_failed_multi_batch_index_removes_new_ids(monkeypatch, tmp_path):
    class RetryableError(Exception):
        pass

    monkeypatch.setattr(indexing, "APIConnectionError", RetryableError)
    monkeypatch.setattr(indexing, "RateLimitError", RetryableError)
    monkeypatch.setattr(indexing, "APIError", RetryableError)
    monkeypatch.setattr(indexing.time, "sleep", lambda _: None)

    class Collection:
        def __init__(self):
            self.ids = set()

        def get(self, **kwargs):
            return {"ids": []}

        def delete(self, ids):
            self.ids.difference_update(ids)

    class Store:
        def __init__(self):
            self._collection = Collection()
            self.calls = 0

        def add_documents(self, *, documents, ids):
            self.calls += 1
            if self.calls == 1:
                self._collection.ids.update(ids)
                return
            raise RetryableError("simulated second-batch failure")

    settings = _Settings()
    settings.upsert_batch_size = 2
    pipeline = IngestionPipeline(settings, enable_visuals=False)
    store = Store()
    work = _ReportWork(
        pdf_path=tmp_path / "r.pdf",
        source_file="r.pdf",
        artifact_dir=tmp_path / "artifacts",
        documents=[
            Document(
                page_content=f"chunk-{index}",
                metadata={"source": "r.pdf", "page": 1, "chunk": index},
            )
            for index in range(3)
        ],
    )

    with pytest.raises(RuntimeError, match="failed for chunks"):
        pipeline._index_report(work, vectorstore=store, tracing=False)

    assert store._collection.ids == set()


def test_failed_rebuild_leaves_active_collection_untouched(monkeypatch, tmp_path):
    settings = _Settings()
    settings.knowledge_dir = tmp_path / "knowledge"
    settings.knowledge_dir.mkdir()
    settings.chroma_dir = tmp_path / "chroma"
    (settings.knowledge_dir / "r.pdf").write_bytes(b"%PDF-1.4\n")

    class Collection:
        def __init__(self, name):
            self.name = name

        def count(self):
            return 0

        @property
        def metadata(self):
            return {}

        def modify(self, *, name=None, metadata=None):
            if name:
                self.name = name

    class Client:
        def __init__(self):
            self.deleted = []

        def delete_collection(self, name):
            self.deleted.append(name)

    client = Client()
    monkeypatch.setattr(
        pipeline_module.chromadb,
        "PersistentClient",
        lambda **kwargs: client,
    )
    runtime = IngestionRuntime(
        get_embedder=lambda settings: object(),
        get_vectorstore=lambda settings, embedder: SimpleNamespace(
            _collection=Collection(settings.collection_name)
        ),
    )
    pipeline = IngestionPipeline(
        settings,
        enable_visuals=False,
        runtime=runtime,
    )
    pipeline.parser_router = MagicMock()
    pipeline.parser_router.parse.side_effect = RuntimeError("parse failed")

    result = pipeline.ingest_all(rebuild=True)

    assert result.status == "failed"
    assert settings.collection_name not in client.deleted


def test_empty_rebuild_clears_active_collection(monkeypatch, tmp_path):
    settings = _Settings()
    settings.knowledge_dir = tmp_path / "knowledge"
    settings.knowledge_dir.mkdir()
    settings.chroma_dir = tmp_path / "chroma"
    client = MagicMock()
    monkeypatch.setattr(
        pipeline_module.chromadb,
        "PersistentClient",
        lambda **kwargs: client,
    )
    runtime = IngestionRuntime(
        get_embedder=lambda settings: object(),
        get_vectorstore=lambda settings, embedder: object(),
    )
    pipeline = IngestionPipeline(
        settings,
        enable_visuals=False,
        runtime=runtime,
    )

    result = pipeline.ingest_all(rebuild=True)

    assert result.status == "completed"
    client.delete_collection.assert_called_once_with(
        name=settings.collection_name
    )


def test_rebuild_promotion_replaces_active_collection():
    class Collection:
        def __init__(self, client, name, *, fail_rename=False):
            self.client = client
            self.name = name
            self.fail_rename = fail_rename

        def modify(self, *, name=None, metadata=None):
            if name is None:
                return
            if self.fail_rename:
                raise RuntimeError("rename failed")
            self.client.collections.pop(self.name, None)
            self.name = name
            self.client.collections[name] = self

    class Client:
        def __init__(self):
            self.collections = {}
            self.deleted = []

        def get_collection(self, name):
            return self.collections[name]

        def delete_collection(self, name):
            self.deleted.append(name)
            self.collections.pop(name)

    client = Client()
    active = Collection(client, "reports")
    rebuilt = Collection(client, "reports__rebuild")
    client.collections = {
        active.name: active,
        rebuilt.name: rebuilt,
    }

    _promote_rebuilt_collection(
        client,
        rebuilt,
        active_name="reports",
    )

    assert client.collections["reports"] is rebuilt
    assert active.name.startswith("reports__backup_")
    assert active.name in client.deleted


def test_rebuild_promotion_restores_active_name_when_swap_fails():
    class Collection:
        def __init__(self, name, *, fail_rename=False):
            self.name = name
            self.fail_rename = fail_rename

        def modify(self, *, name=None, metadata=None):
            if self.fail_rename:
                raise RuntimeError("rename failed")
            if name:
                self.name = name

    active = Collection("reports")
    rebuilt = Collection("reports__rebuild", fail_rename=True)
    client = SimpleNamespace(
        get_collection=lambda name: active,
        delete_collection=lambda name: None,
    )

    with pytest.raises(RuntimeError, match="rename failed"):
        _promote_rebuilt_collection(
            client,
            rebuilt,
            active_name="reports",
        )

    assert active.name == "reports"


def test_rebuild_promotion_switches_real_chroma_collection(tmp_path):
    client = pipeline_module.chromadb.PersistentClient(
        path=str(tmp_path / "chroma")
    )
    active = client.get_or_create_collection("reports")
    active.add(
        ids=["old"],
        embeddings=[[1.0, 0.0]],
        documents=["old document"],
    )
    rebuilt = client.get_or_create_collection("reports__rebuild")
    rebuilt.add(
        ids=["new"],
        embeddings=[[0.0, 1.0]],
        documents=["new document"],
    )

    _promote_rebuilt_collection(
        client,
        rebuilt,
        active_name="reports",
    )

    promoted = client.get_collection("reports")
    assert promoted.get(include=[])["ids"] == ["new"]
    assert sorted(collection.name for collection in client.list_collections()) == [
        "reports"
    ]


def test_successful_rebuild_promotes_only_after_pipeline_validation(tmp_path):
    settings = _Settings()
    settings.knowledge_dir = tmp_path / "knowledge"
    settings.knowledge_dir.mkdir()
    settings.chroma_dir = tmp_path / "chroma"
    settings.artifact_dir = tmp_path / "artifacts"
    settings.resolved_embedding_signature = {"provider": "test"}
    pdf = settings.knowledge_dir / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    client = pipeline_module.chromadb.PersistentClient(
        path=str(settings.chroma_dir)
    )
    old = client.get_or_create_collection(settings.collection_name)
    old.add(
        ids=["old"],
        embeddings=[[1.0, 0.0]],
        documents=["stale"],
    )

    runtime = IngestionRuntime(
        get_embedder=lambda _settings: object(),
        get_vectorstore=lambda vector_settings, _embedder: SimpleNamespace(
            _collection=client.get_or_create_collection(
                vector_settings.collection_name
            )
        ),
    )
    pipeline = IngestionPipeline(
        settings,
        enable_visuals=False,
        runtime=runtime,
    )
    parser_result = ParserResult(
        source_file="report.pdf",
        parser="docling",
        parser_version="test",
        elements=[
            ElementRecord(
                element_id="n1",
                source_file="report.pdf",
                category="NarrativeText",
                text="Body",
            )
        ],
        quality=ParserQualityReport(score=1.0),
    )

    def fake_ingest_pdf(path, *, source_file, artifact_dir, **kwargs):
        pipeline._last_report_work = _ReportWork(
            pdf_path=path,
            source_file=source_file,
            artifact_dir=artifact_dir,
            parser_result=parser_result,
        )
        return ReportIngestStats(filename=source_file)

    pipeline.ingest_pdf = fake_ingest_pdf

    result = pipeline.ingest_all(rebuild=True)

    assert result.status == "completed"
    assert result.files == ["report.pdf"]
    assert client.get_collection(settings.collection_name).count() == 0
    assert sorted(collection.name for collection in client.list_collections()) == [
        settings.collection_name
    ]


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


def test_bounded_pipeline_uses_separate_enrich_and_index_workers(tmp_path):
    settings = _Settings()
    settings.knowledge_dir = tmp_path / "knowledge"
    settings.extra_pdf_dirs = []
    settings.chroma_dir = tmp_path / "chroma"
    settings.artifact_dir = tmp_path / "artifacts"
    settings.extracted_dir = tmp_path / "extracted"
    settings.ingestion_pipeline_enabled = True
    settings.ingestion_pipeline_queue_size = 2
    settings.resolved_embedding_provider = "qwen"
    settings.resolved_embedding_signature = {
        "provider": "qwen",
        "model": "test",
        "dimensions": 3,
    }
    settings.knowledge_dir.mkdir()
    for index in range(3):
        (settings.knowledge_dir / f"r{index}.pdf").write_bytes(
            b"%PDF staged"
        )

    pipeline = IngestionPipeline(
        settings,
        enable_visuals=False,
        runtime=IngestionRuntime(
            get_embedder=lambda _settings: MagicMock(),
            get_vectorstore=lambda _settings, _embedder: MagicMock(),
        ),
    )
    stage_threads = {"parse": [], "enrich": [], "index": []}

    def fake_parse(path, *, source_file, artifact_dir, tracing):
        stage_threads["parse"].append(threading.current_thread().name)
        return _ReportWork(
            pdf_path=path,
            source_file=source_file,
            artifact_dir=artifact_dir,
            parser_result=ParserResult(
                source_file=source_file,
                parser="docling",
                parser_version="test",
                elements=[
                    ElementRecord(
                        element_id=f"{source_file}-1",
                        source_file=source_file,
                        category="NarrativeText",
                        text="Body",
                    )
                ],
                quality=ParserQualityReport(score=1.0),
            ),
            elements=[
                ElementRecord(
                    element_id=f"{source_file}-1",
                    source_file=source_file,
                    category="NarrativeText",
                    text="Body",
                )
            ],
        )

    def fake_prepare(work, **_kwargs):
        stage_threads["enrich"].append(threading.current_thread().name)
        return work

    def fake_index(future, **_kwargs):
        stage_threads["index"].append(threading.current_thread().name)
        work = future.result()
        return (
            ReportIngestStats(filename=work.source_file),
            work,
        )

    pipeline._parse_report = fake_parse
    pipeline._prepare_report = fake_prepare
    pipeline._index_after_prepare = fake_index

    result = pipeline.ingest_all()

    assert result.files == ["r0.pdf", "r1.pdf", "r2.pdf"]
    assert all(name == "MainThread" for name in stage_threads["parse"])
    assert all(
        name.startswith("orextractor-enrich")
        for name in stage_threads["enrich"]
    )
    assert all(
        name.startswith("orextractor-index")
        for name in stage_threads["index"]
    )
