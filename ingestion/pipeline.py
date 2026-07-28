"""Deterministic Docling/MinerU + Bedrock visual ingestion pipeline."""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import chromadb
from tqdm import tqdm

from ingestion.cache import (
    EnrichmentCache,
    build_manifest_entry,
    file_sha256,
    load_ingest_manifest,
    load_parser_cache,
    manifest_entry_accepted,
    save_ingest_manifest,
    save_parser_cache,
    should_skip_pdf,
    parser_result_accepted,
)
from ingestion.chunking import elements_to_documents
from ingestion.config import DoclingExecutionConfig
from ingestion.context import annotate_hierarchy
from ingestion.visual_filtering import filter_visual_artifacts
from ingestion.enrichment import enrich_elements
from ingestion.indexing import add_documents_with_retry, build_doc_id
from ingestion.models import (
    PIPELINE_VERSION,
    IngestionError,
    IngestionMetrics,
    IngestionResult,
    ReportIngestStats,
)
from ingestion.parsers.router import get_parser_router
from ingestion.runtime import IngestionRuntime
from ingestion.sources import (
    filesystem_path,
    iter_pdf_paths,
    pdf_source_id,
    source_output_path,
)
from ingestion.telemetry import configure_langsmith, stage_span
from ingestion.visuals import reconstruct_visuals

# Re-export for callers
__all__ = ["IngestionPipeline", "PIPELINE_VERSION"]


def _source_document_ids(vectorstore, source_file: str) -> set[str]:
    """Read existing IDs so changed chunk layouts do not leave stale records."""
    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        return set()
    try:
        result = collection.get(where={"source": source_file}, include=[])
        return set(result.get("ids") or [])
    except Exception:
        return set()


def _delete_document_ids(vectorstore, document_ids: set[str]) -> None:
    if not document_ids:
        return
    collection = getattr(vectorstore, "_collection", None)
    if collection is not None:
        collection.delete(ids=sorted(document_ids))
        return
    vectorstore.delete(ids=sorted(document_ids))


def _delete_collection(
    client,
    collection_name: str,
    *,
    missing_ok: bool,
) -> None:
    try:
        client.delete_collection(name=collection_name)
    except chromadb.errors.NotFoundError:
        if not missing_ok:
            raise


def _promote_rebuilt_collection(
    client,
    rebuilt_collection,
    *,
    active_name: str,
) -> None:
    """Replace the active collection while retaining a rollback copy."""
    backup_name = f"{active_name}__backup_{uuid.uuid4().hex}"
    active_collection = None
    try:
        active_collection = client.get_collection(name=active_name)
    except chromadb.errors.NotFoundError:
        pass

    if active_collection is not None:
        active_collection.modify(name=backup_name)

    try:
        rebuilt_collection.modify(name=active_name)
    except Exception:
        if active_collection is not None:
            active_collection.modify(name=active_name)
        raise

    if active_collection is not None:
        _delete_collection(client, backup_name, missing_ok=False)


@dataclass
class _ReportWork:
    pdf_path: Path
    source_file: str
    artifact_dir: Path
    started_at: float = field(default_factory=time.perf_counter)
    metrics: IngestionMetrics = field(default_factory=IngestionMetrics)
    parser_result: Any = None
    elements: list[Any] = field(default_factory=list)
    partition_cache_hit: bool = False
    pages: int = 0
    text_elements: int = 0
    tables: int = 0
    figures: int = 0
    analyses: dict[str, Any] = field(default_factory=dict)
    validations: dict[str, Any] = field(default_factory=dict)
    reconstructions: dict[str, Any] = field(default_factory=dict)
    errors: list[IngestionError] = field(default_factory=list)
    enrichment_stats: dict[str, Any] = field(default_factory=dict)
    reconstruction_warning_count: int = 0
    documents: list[Any] = field(default_factory=list)


def _merge_metrics(target: IngestionMetrics, source: IngestionMetrics) -> None:
    for name in type(target).model_fields:
        if name == "total_ms":
            continue
        current = getattr(target, name)
        increment = getattr(source, name)
        if isinstance(current, (int, float)) and isinstance(
            increment, (int, float)
        ):
            setattr(target, name, current + increment)


class IngestionPipeline:
    def __init__(
        self,
        settings,
        *,
        enable_visuals: bool = True,
        partition_only: bool = False,
        runtime: IngestionRuntime | None = None,
    ):
        self.settings = copy(settings)
        self.runtime = runtime
        self.enable_visuals = enable_visuals and getattr(
            self.settings,
            "visual_enrichment_enabled",
            True,
        )
        self.partition_only = partition_only
        self.settings.resolved_visual_enrichment_enabled = self.enable_visuals
        if (
            not self.enable_visuals
            and not partition_only
            and not getattr(
                self.settings,
                "parser_require_picture_crops",
                False,
            )
        ):
            # Text-only ingestion does not consume Docling page/picture crops.
            # Disabling them reduces rendering and retained-image overhead.
            self.settings.docling_generate_page_images = False
            self.settings.docling_generate_picture_images = False
            text_first_mode = DoclingExecutionConfig.from_settings(
                self.settings
            ).text_first_table_mode
            if text_first_mode not in {
                "fast",
                "accurate",
                "configured",
            }:
                raise ValueError(
                    "DOCLING_TEXT_FIRST_TABLE_MODE must be 'fast', "
                    f"'accurate', or 'configured', got {text_first_mode!r}"
                )
            if text_first_mode != "configured":
                self.settings.docling_table_mode = text_first_mode
        self.parser_router = get_parser_router(self.settings)
        configure_langsmith(self.settings)

    def ingest_all(
        self,
        rebuild: bool = False,
        only_file: Optional[str] = None,
        reprocess_visuals: bool = False,
    ) -> IngestionResult:
        t_total = time.perf_counter()
        pdf_paths = list(iter_pdf_paths(self.settings.knowledge_dir, self.settings.extra_pdf_dirs))
        discovered = [
            (
                path,
                pdf_source_id(
                    path,
                    self.settings.knowledge_dir,
                    self.settings.extra_pdf_dirs,
                ),
            )
            for path in pdf_paths
        ]
        source_ids: dict[str, Path] = {}
        for path, source_file in discovered:
            key = source_file.casefold()
            previous = source_ids.get(key)
            if previous is not None and previous.resolve() != path.resolve():
                raise ValueError(
                    f"Duplicate PDF source path {source_file!r}: "
                    f"{previous} and {path}. Configure non-overlapping PDF roots."
                )
            source_ids[key] = path

        if only_file:
            requested = str(only_file).replace("\\", "/").casefold()
            matches = [
                item
                for item in discovered
                if item[1].casefold() == requested
            ]
            if not matches:
                matches = [
                    item
                    for item in discovered
                    if item[0].name.casefold() == Path(only_file).name.casefold()
                    or item[0].stem.casefold() == Path(only_file).stem.casefold()
                ]
            if len(matches) > 1:
                choices = ", ".join(source_file for _, source_file in matches)
                raise ValueError(
                    f"PDF name {only_file!r} is ambiguous. Use one of these relative "
                    f"paths with --file: {choices}"
                )
            discovered = matches

        if not discovered:
            dirs = [self.settings.knowledge_dir] + list(self.settings.extra_pdf_dirs)
            print(f"No PDFs found in {', '.join(str(d) for d in dirs)}")
            if rebuild and not self.partition_only:
                chroma_client = chromadb.PersistentClient(
                    path=str(self.settings.chroma_dir)
                )
                _delete_collection(
                    chroma_client,
                    self.settings.collection_name,
                    missing_ok=True,
                )
                save_ingest_manifest(self.settings.chroma_dir, {})
            return IngestionResult(status="completed", files=[])

        vectorstore = None
        chroma_client = None
        rebuild_collection_name = None
        if not self.partition_only:
            if self.runtime is None:
                raise RuntimeError(
                    "IngestionRuntime is required for vector ingestion."
                )
            # Resolve and health-check the complete embedding backend before a
            # requested rebuild deletes the existing collection.
            embedder = self.runtime.get_embedder(self.settings)
            chroma_client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
            if rebuild:
                rebuild_collection_name = (
                    f"{self.settings.collection_name}__rebuild_{uuid.uuid4().hex}"
                )
                vectorstore_settings = copy(self.settings)
                vectorstore_settings.collection_name = rebuild_collection_name
            else:
                vectorstore_settings = self.settings
            vectorstore = self.runtime.get_vectorstore(
                vectorstore_settings,
                embedder,
            )

        manifest = {} if rebuild else load_ingest_manifest(self.settings.chroma_dir)
        result = IngestionResult(status="completed", files=[])
        metrics = IngestionMetrics()
        tracing = bool(getattr(self.settings, "langsmith_tracing", False))

        pdf_bar = tqdm(
            total=len(discovered),
            desc="Ingesting PDFs (bounded pipeline)",
            unit="pdf",
            dynamic_ncols=True,
        )

        def record_success(
            report_stats: ReportIngestStats,
            work: _ReportWork,
        ) -> None:
            result.reports.append(report_stats)
            result.files.append(work.source_file)
            result.errors.extend(work.errors)
            _merge_metrics(metrics, work.metrics)
            if not self.partition_only:
                manifest[work.source_file] = build_manifest_entry(
                    work.pdf_path,
                    self.settings,
                    element_count=report_stats.elements,
                    visual_count=report_stats.figures,
                    table_count=report_stats.tables,
                    indexed_chunk_count=report_stats.indexed_chunks,
                    failed_element_ids=report_stats.failed_elements,
                    visual_enrichment_enabled=self.enable_visuals,
                    parser_result=work.parser_result,
                )
                if not rebuild:
                    save_ingest_manifest(self.settings.chroma_dir, manifest)

        def record_failure(source_file: str, exc: Exception) -> None:
            existing_entry = manifest.get(source_file)
            if (
                vectorstore is not None
                and isinstance(existing_entry, dict)
                and not manifest_entry_accepted(
                    existing_entry,
                    self.settings,
                )
            ):
                stale_ids = _source_document_ids(
                    vectorstore,
                    source_file,
                )
                _delete_document_ids(vectorstore, stale_ids)
            result.errors.append(
                IngestionError(
                    element_id="",
                    stage="ingest-pdf",
                    message=f"{source_file}: {exc}",
                )
            )
            tqdm.write(f"\nFailed {source_file}: {exc}")

        def finish_future(
            source_file: str,
            future: Future,
        ) -> None:
            try:
                report_stats, work = future.result()
                record_success(report_stats, work)
            except Exception as exc:
                record_failure(source_file, exc)
            finally:
                pdf_bar.update(1)

        pipeline_enabled = (
            not self.partition_only
            and bool(
                getattr(
                    self.settings,
                    "ingestion_pipeline_enabled",
                    True,
                )
            )
            and len(discovered) > 1
        )

        if pipeline_enabled:
            max_inflight = max(
                3,
                int(
                    getattr(
                        self.settings,
                        "ingestion_pipeline_queue_size",
                        2,
                    )
                )
                + 1,
            )
            pending: deque[tuple[str, Future]] = deque()
            enrich_pool = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="orextractor-enrich",
            )
            index_pool = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="orextractor-index",
            )
            try:
                for pdf_path, source_file in discovered:
                    pdf_bar.set_postfix(
                        parse=source_file[-32:],
                        inflight=len(pending),
                    )
                    input_path = filesystem_path(pdf_path)
                    entry = manifest.get(source_file)
                    if (
                        not rebuild
                        and not reprocess_visuals
                        and should_skip_pdf(entry, input_path, self.settings)
                    ):
                        tqdm.write(
                            f"\nSkipping {source_file} "
                            "(unchanged since last ingest)"
                        )
                        pdf_bar.update(1)
                        continue
                    try:
                        work = self._parse_report(
                            input_path,
                            source_file=source_file,
                            artifact_dir=source_output_path(
                                self.settings.artifact_dir,
                                source_file,
                                "",
                            ),
                            tracing=tracing,
                        )
                        enrich_future = enrich_pool.submit(
                            self._prepare_report,
                            work,
                            reprocess_visuals=reprocess_visuals,
                            tracing=tracing,
                        )
                        index_future = index_pool.submit(
                            self._index_after_prepare,
                            enrich_future,
                            vectorstore=vectorstore,
                            tracing=tracing,
                        )
                        pending.append((source_file, index_future))
                    except Exception as exc:
                        record_failure(source_file, exc)
                        pdf_bar.update(1)
                        continue

                    while len(pending) >= max_inflight:
                        completed_source, completed_future = pending.popleft()
                        finish_future(completed_source, completed_future)

                while pending:
                    completed_source, completed_future = pending.popleft()
                    finish_future(completed_source, completed_future)
            finally:
                enrich_pool.shutdown(wait=True, cancel_futures=True)
                index_pool.shutdown(wait=True, cancel_futures=True)
        else:
            for pdf_path, source_file in discovered:
                pdf_bar.set_postfix(file=source_file[-40:])
                input_path = filesystem_path(pdf_path)
                entry = manifest.get(source_file)
                if (
                    not self.partition_only
                    and not rebuild
                    and not reprocess_visuals
                    and should_skip_pdf(entry, input_path, self.settings)
                ):
                    tqdm.write(
                        f"\nSkipping {source_file} "
                        "(unchanged since last ingest)"
                    )
                    pdf_bar.update(1)
                    continue
                try:
                    report_stats = self.ingest_pdf(
                        input_path,
                        source_file=source_file,
                        artifact_dir=source_output_path(
                            self.settings.artifact_dir,
                            source_file,
                            "",
                        ),
                        vectorstore=vectorstore,
                        reprocess_visuals=reprocess_visuals,
                        metrics=None,
                        tracing=tracing,
                    )
                    work = getattr(
                        self,
                        "_last_report_work",
                        _ReportWork(
                            pdf_path=input_path,
                            source_file=source_file,
                            artifact_dir=source_output_path(
                                self.settings.artifact_dir,
                                source_file,
                                "",
                            ),
                        ),
                    )
                    record_success(report_stats, work)
                except Exception as exc:
                    record_failure(source_file, exc)
                finally:
                    pdf_bar.update(1)
        pdf_bar.close()

        metrics.total_ms = (time.perf_counter() - t_total) * 1000
        result.metrics = metrics
        if result.errors:
            result.status = (
                "completed_with_errors" if result.reports else "failed"
            )
        if rebuild and not self.partition_only:
            assert chroma_client is not None
            assert rebuild_collection_name is not None
            if result.errors:
                _delete_collection(
                    chroma_client,
                    rebuild_collection_name,
                    missing_ok=True,
                )
            else:
                rebuilt_collection = getattr(
                    vectorstore,
                    "_collection",
                    None,
                )
                if rebuilt_collection is None:
                    _delete_collection(
                        chroma_client,
                        rebuild_collection_name,
                        missing_ok=True,
                    )
                    raise RuntimeError(
                        "Cannot promote rebuilt index: vector store does not "
                        "expose its Chroma collection."
                    )
                expected_sources = {
                    source_file.casefold()
                    for _, source_file in discovered
                }
                accepted_sources = {
                    source_file.casefold()
                    for source_file in result.files
                }
                manifest_sources = {
                    source_file.casefold()
                    for source_file in manifest
                }
                expected_vectors = sum(
                    report.indexed_chunks
                    for report in result.reports
                )
                actual_vectors = int(rebuilt_collection.count())
                if (
                    accepted_sources != expected_sources
                    or manifest_sources != expected_sources
                    or actual_vectors != expected_vectors
                ):
                    _delete_collection(
                        chroma_client,
                        rebuild_collection_name,
                        missing_ok=True,
                    )
                    raise RuntimeError(
                        "Rebuilt index validation failed before promotion: "
                        f"sources={len(accepted_sources)}/"
                        f"{len(expected_sources)}, "
                        f"manifest={len(manifest_sources)}/"
                        f"{len(expected_sources)}, "
                        f"vectors={actual_vectors}/{expected_vectors}."
                    )
                _promote_rebuilt_collection(
                    chroma_client,
                    rebuilt_collection,
                    active_name=self.settings.collection_name,
                )
                save_ingest_manifest(self.settings.chroma_dir, manifest)
        print(
            f"\nDocument ingestion complete. "
            f"{sum(r.indexed_chunks for r in result.reports)} chunks across "
            f"{len(result.reports)} report(s) in '{self.settings.collection_name}' "
            f"using {getattr(self.settings, 'resolved_embedding_provider', 'unknown')}."
        )
        close_router = getattr(self.parser_router, "close", None)
        if callable(close_router):
            close_router()
        return result

    def _parse_report(
        self,
        pdf_path: Path,
        *,
        source_file: str,
        artifact_dir: Path,
        tracing: bool,
    ) -> _ReportWork:
        work = _ReportWork(
            pdf_path=pdf_path,
            source_file=source_file,
            artifact_dir=artifact_dir,
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with stage_span(
            "parse-document",
            {"source": source_file},
            enabled=tracing,
        ):
            parser_result = load_parser_cache(
                artifact_dir,
                pdf_path,
                self.settings,
                self.parser_router,
            )
            work.partition_cache_hit = parser_result is not None
            if parser_result is None:
                parser_result = self.parser_router.parse(
                    pdf_path,
                    source_file=source_file,
                    artifact_dir=artifact_dir,
                )
                work.partition_cache_hit = bool(
                    parser_result.metadata.get("parser_cache_hit")
                )
                parser_result.elements = filter_visual_artifacts(
                    parser_result.elements
                )
                if parser_result.elements:
                    save_parser_cache(
                        artifact_dir,
                        pdf_path,
                        self.settings,
                        self.parser_router,
                        parser_result,
                    )
            else:
                parser_result.elements = filter_visual_artifacts(
                    parser_result.elements
                )
            if work.partition_cache_hit:
                work.metrics.partition_cache_hits += 1
            if not parser_result.elements:
                details = (
                    "; ".join(parser_result.errors)
                    or "no canonical elements"
                )
                raise RuntimeError(
                    f"{parser_result.parser} produced no usable elements: "
                    f"{details}"
                )
            if (
                not parser_result_accepted(
                    parser_result,
                    self.settings,
                )
                and bool(
                    getattr(
                        self.settings,
                        "parser_fallback_enabled",
                        True,
                    )
                )
            ):
                details = "; ".join(
                    [
                        *parser_result.quality.reasons,
                        *parser_result.errors,
                    ]
                ) or "quality policy rejected the parser result"
                raise RuntimeError(
                    f"{parser_result.parser} produced retryable degraded "
                    f"output: {details}"
                )
            work.parser_result = parser_result
            work.elements = parser_result.elements

        work.metrics.partition_ms += (
            time.perf_counter() - t0
        ) * 1000
        work.metrics.primary_parse_ms += float(
            parser_result.metadata.get(
                "primary_duration_ms",
                parser_result.duration_ms,
            )
        )
        work.metrics.fallback_parse_ms += float(
            parser_result.metadata.get("fallback_duration_ms", 0.0)
        )
        if parser_result.fallback.attempted:
            work.metrics.fallback_attempts += 1
        if parser_result.fallback.used:
            work.metrics.fallback_uses += 1

        source_sha256 = file_sha256(pdf_path)
        for element in work.elements:
            element.metadata["source_sha256"] = source_sha256

        t0 = time.perf_counter()
        with stage_span(
            "normalize-elements",
            {"source": source_file},
            enabled=tracing,
        ):
            work.elements = annotate_hierarchy(work.elements)
        work.metrics.normalize_ms += (
            time.perf_counter() - t0
        ) * 1000

        work.pages = max(
            (element.page_number for element in work.elements),
            default=0,
        )
        work.text_elements = sum(
            1
            for element in work.elements
            if element.category
            in {"NarrativeText", "ListItem", "Title"}
        )
        work.tables = sum(
            1
            for element in work.elements
            if element.category == "Table"
        )
        work.figures = sum(
            1
            for element in work.elements
            if element.category in {"Image", "Figure"}
        )
        return work

    def _prepare_report(
        self,
        work: _ReportWork,
        *,
        reprocess_visuals: bool,
        tracing: bool,
    ) -> _ReportWork:
        cache = EnrichmentCache(work.artifact_dir / "cache")
        default_stats = {
            "bedrock_calls": 0,
            "visual_model_calls": 0,
            "visual_model_provider": str(
                getattr(self.settings, "visual_model_provider", "bedrock")
            ),
            "cache_hits": 0,
            "warnings": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.0,
            "retry_count": 0,
        }
        t0 = time.perf_counter()
        with stage_span(
            "enrich-visuals",
            {"source": work.source_file},
            enabled=tracing,
        ):
            (
                work.analyses,
                work.validations,
                work.errors,
                work.enrichment_stats,
            ) = enrich_elements(
                work.elements,
                self.settings,
                cache=cache,
                enable_visuals=self.enable_visuals,
                bypass_cache=reprocess_visuals,
            )
        work.enrichment_stats = {
            **default_stats,
            **work.enrichment_stats,
        }
        work.metrics.enrich_ms += (
            time.perf_counter() - t0
        ) * 1000
        work.metrics.bedrock_calls += int(
            work.enrichment_stats.get("bedrock_calls", 0)
        )
        work.metrics.visual_model_calls += int(
            work.enrichment_stats.get("visual_model_calls", 0)
        )
        work.metrics.cache_hits += int(
            work.enrichment_stats.get("cache_hits", 0)
        )
        work.metrics.input_tokens += int(
            work.enrichment_stats.get("input_tokens", 0)
        )
        work.metrics.output_tokens += int(
            work.enrichment_stats.get("output_tokens", 0)
        )
        visual_latency_ms = float(
            work.enrichment_stats.get("latency_ms", 0.0)
        )
        work.metrics.visual_model_latency_ms += visual_latency_ms
        if (
            work.enrichment_stats.get("visual_model_provider")
            == "bedrock"
        ):
            work.metrics.bedrock_latency_ms += visual_latency_ms
        work.metrics.retry_count += int(
            work.enrichment_stats.get("retry_count", 0)
        )

        enrichments_dir = work.artifact_dir / "enrichments"
        enrichments_dir.mkdir(parents=True, exist_ok=True)
        for element_id, validation in work.validations.items():
            (enrichments_dir / f"{element_id}-table.json").write_text(
                validation.model_dump_json(indent=2),
                encoding="utf-8",
            )

        t0 = time.perf_counter()
        with stage_span(
            "reconstruct-visuals",
            {"source": work.source_file},
            enabled=tracing,
        ):
            work.reconstructions, _ = reconstruct_visuals(
                work.elements,
                work.analyses,
                self.settings,
                work.artifact_dir,
            )
            work.reconstruction_warning_count = sum(
                max(
                    0,
                    len(info.get("warnings") or [])
                    - len(
                        work.analyses[element_id].warnings
                        if element_id in work.analyses
                        else []
                    ),
                )
                for element_id, info in work.reconstructions.items()
            )
        work.metrics.reconstruct_ms += (
            time.perf_counter() - t0
        ) * 1000

        t0 = time.perf_counter()
        with stage_span(
            "create-chunks",
            {"source": work.source_file},
            enabled=tracing,
        ):
            work.documents = elements_to_documents(
                work.elements,
                analyses=work.analyses,
                validations=work.validations,
                reconstructions=work.reconstructions,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                table_confidence_threshold=(
                    self.settings.bedrock_visual_confidence_threshold
                ),
            )
            (work.artifact_dir / "chunks.json").write_text(
                json.dumps(
                    [
                        {
                            "page_content": document.page_content,
                            "metadata": document.metadata,
                        }
                        for document in work.documents
                    ],
                    indent=2,
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
        work.metrics.chunk_ms += (
            time.perf_counter() - t0
        ) * 1000

        from chapter_index import (
            build_chapter_index_from_documents,
            save_chapter_index,
            tag_documents_with_items,
        )

        chapters = build_chapter_index_from_documents(work.documents)
        work.documents = tag_documents_with_items(
            work.documents,
            chapters,
        )
        save_chapter_index(
            self.settings.extracted_dir,
            work.source_file,
            chapters,
        )
        return work

    def _index_after_prepare(
        self,
        prepared_future: Future,
        *,
        vectorstore,
        tracing: bool,
    ) -> tuple[ReportIngestStats, _ReportWork]:
        work = prepared_future.result()
        return self._index_report(
            work,
            vectorstore=vectorstore,
            tracing=tracing,
        )

    def _index_report(
        self,
        work: _ReportWork,
        *,
        vectorstore,
        tracing: bool,
    ) -> tuple[ReportIngestStats, _ReportWork]:
        t0 = time.perf_counter()
        ids = [
            build_doc_id(
                work.source_file,
                int(document.metadata.get("page", -1) or -1),
                int(document.metadata.get("chunk", 0) or 0),
                document.page_content,
            )
            for document in work.documents
        ]
        with stage_span(
            "upsert-chroma",
            {"source": work.source_file},
            enabled=tracing,
        ):
            if work.documents:
                existing_ids = _source_document_ids(
                    vectorstore,
                    work.source_file,
                )
                try:
                    add_documents_with_retry(
                        vectorstore=vectorstore,
                        docs=work.documents,
                        ids=ids,
                        batch_size=self.settings.upsert_batch_size,
                    )
                except Exception:
                    new_ids = set(ids).difference(existing_ids)
                    _delete_document_ids(vectorstore, new_ids)
                    raise
                stale_ids = existing_ids.difference(ids)
                if stale_ids:
                    _delete_document_ids(vectorstore, stale_ids)
        work.metrics.embed_ms += (
            time.perf_counter() - t0
        ) * 1000
        work.metrics.total_ms = (
            time.perf_counter() - work.started_at
        ) * 1000
        stats = self._report_stats(
            work,
            indexed=len(work.documents),
        )
        self._print_report_stats(stats)
        return stats, work

    def _report_stats(
        self,
        work: _ReportWork,
        *,
        indexed: int,
    ) -> ReportIngestStats:
        reconstructed_charts = sum(
            1
            for value in work.reconstructions.values()
            if value.get("reconstruction_allowed")
            and value.get("reason") == "chart"
        )
        reconstructed_diagrams = sum(
            1
            for value in work.reconstructions.values()
            if value.get("reconstruction_allowed")
            and value.get("reason") == "diagram"
        )
        failed = [
            error.element_id
            for error in work.errors
            if error.element_id
        ]
        runtime = (
            work.parser_result.metadata.get("runtime", {})
            if work.parser_result is not None
            else {}
        )
        enrich_stats = work.enrichment_stats or {}
        return ReportIngestStats(
            filename=work.source_file,
            pages=work.pages,
            elements=len(work.elements),
            text_elements=work.text_elements,
            tables=work.tables,
            figures=work.figures,
            visual_model_provider=str(
                enrich_stats.get("visual_model_provider", "")
            ),
            visual_model_calls=int(
                enrich_stats.get("visual_model_calls", 0)
            ),
            visual_model_latency_ms=float(
                enrich_stats.get("latency_ms", 0.0)
            ),
            bedrock_calls=int(enrich_stats.get("bedrock_calls", 0)),
            cache_hits=int(enrich_stats.get("cache_hits", 0)),
            partition_cache_hits=int(work.partition_cache_hit),
            input_tokens=int(enrich_stats.get("input_tokens", 0)),
            output_tokens=int(enrich_stats.get("output_tokens", 0)),
            bedrock_latency_ms=(
                float(enrich_stats.get("latency_ms", 0.0))
                if enrich_stats.get("visual_model_provider") == "bedrock"
                else 0.0
            ),
            retry_count=int(enrich_stats.get("retry_count", 0)),
            reconstructed_charts=reconstructed_charts,
            reconstructed_diagrams=reconstructed_diagrams,
            warnings=(
                int(enrich_stats.get("warnings", 0))
                + work.reconstruction_warning_count
            ),
            failed_elements=failed,
            indexed_chunks=indexed,
            primary_parser=str(
                work.parser_result.metadata.get(
                    "primary_parser",
                    work.parser_result.parser,
                )
            ),
            selected_parser=work.parser_result.parser,
            parser_version=work.parser_result.parser_version,
            parser_quality_score=work.parser_result.quality.score,
            fallback_attempted=(
                work.parser_result.fallback.attempted
            ),
            fallback_used=work.parser_result.fallback.used,
            fallback_reasons=work.parser_result.fallback.reasons,
            parse_ms=work.metrics.partition_ms,
            enrich_ms=work.metrics.enrich_ms,
            embed_ms=work.metrics.embed_ms,
            total_ms=work.metrics.total_ms,
            effective_do_ocr=runtime.get("effective_do_ocr"),
            safe_batch_fallback_used=bool(
                runtime.get("safe_batch_fallback_used", False)
            ),
        )

    @staticmethod
    def _print_report_stats(stats: ReportIngestStats) -> None:
        tqdm.write(
            f"  {stats.filename}: {stats.pages} pages, "
            f"{stats.elements} elements -> "
            f"{stats.indexed_chunks} chunks "
            f"(parse={stats.parse_ms / 1000:.1f}s, "
            f"enrich={stats.enrich_ms / 1000:.1f}s, "
            f"embed={stats.embed_ms / 1000:.1f}s, "
            f"visual={stats.visual_model_provider or 'disabled'}:"
            f"{stats.visual_model_calls}, "
            f"ocr={stats.effective_do_ocr}, "
            f"parser={stats.selected_parser}, "
            f"quality={stats.parser_quality_score:.3f}, "
            f"fallback={stats.fallback_used})"
        )

    def ingest_pdf(
        self,
        pdf_path: Path,
        *,
        source_file: Optional[str] = None,
        artifact_dir: Optional[Path] = None,
        vectorstore=None,
        reprocess_visuals: bool = False,
        metrics: Optional[IngestionMetrics] = None,
        tracing: bool = False,
        rebuild: bool = False,
    ) -> ReportIngestStats:
        source_file = source_file or pdf_path.name
        artifact_dir = (
            artifact_dir
            or Path(self.settings.artifact_dir) / pdf_path.stem
        )
        work = self._parse_report(
            pdf_path,
            source_file=source_file,
            artifact_dir=artifact_dir,
            tracing=tracing,
        )
        if self.partition_only:
            work.metrics.total_ms = (
                time.perf_counter() - work.started_at
            ) * 1000
            stats = self._report_stats(work, indexed=0)
            self._print_report_stats(stats)
        else:
            work = self._prepare_report(
                work,
                reprocess_visuals=reprocess_visuals,
                tracing=tracing,
            )
            if vectorstore is None:
                if self.runtime is None:
                    raise RuntimeError(
                        "IngestionRuntime is required for vector ingestion."
                    )
                embedder = self.runtime.get_embedder(self.settings)
                vectorstore = self.runtime.get_vectorstore(
                    self.settings,
                    embedder,
                )
            stats, work = self._index_report(
                work,
                vectorstore=vectorstore,
                tracing=tracing,
            )
        self._last_report_work = work
        self._last_parser_result = work.parser_result
        self._last_report_errors = work.errors
        if metrics is not None:
            _merge_metrics(metrics, work.metrics)
        return stats

    def inspect_elements(
        self,
        pdf_path: Path,
        *,
        source_file: Optional[str] = None,
        artifact_dir: Optional[Path] = None,
    ) -> dict:
        source_file = source_file or pdf_path.name
        parser_result = self.parser_router.parse(
            pdf_path,
            source_file=source_file,
            artifact_dir=artifact_dir,
        )
        elements = filter_visual_artifacts(parser_result.elements)
        elements = annotate_hierarchy(elements)
        from collections import Counter

        from ingestion.enrichment import should_enrich_figure

        cats = Counter(el.category for el in elements)
        ni_counts = Counter(el.ni_item for el in elements)
        figures = [el for el in elements if el.category in {"Image", "Figure"}]
        selected = []
        for el in figures:
            ok, reason = should_enrich_figure(el, self.settings)
            if ok:
                selected.append(el.element_id)
        return {
            "file": source_file,
            "element_category_counts": dict(cats),
            "ni_item_counts": {str(k): v for k, v in sorted(ni_counts.items())},
            "tables": sum(1 for el in elements if el.category == "Table"),
            "figures": len(figures),
            "figures_selected_for_bedrock": selected,
            "section_titles": sorted({el.section_title for el in elements if el.section_title}),
            "primary_parser": parser_result.metadata.get(
                "primary_parser", parser_result.parser
            ),
            "selected_parser": parser_result.parser,
            "parser_version": parser_result.parser_version,
            "quality": parser_result.quality.model_dump(mode="json"),
            "fallback": parser_result.fallback.model_dump(mode="json"),
            "parser_errors": parser_result.errors,
        }
