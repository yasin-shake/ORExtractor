"""Deterministic Unstructured + Bedrock visual ingestion pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import chromadb
from tqdm import tqdm

from ingestion.cache import (
    EnrichmentCache,
    build_manifest_entry,
    file_sha256,
    load_ingest_manifest,
    load_partition_cache,
    save_ingest_manifest,
    save_partition_cache,
    should_skip_pdf,
)
from ingestion.chunking import elements_to_documents
from ingestion.context import annotate_hierarchy
from ingestion.enrichment import enrich_elements
from ingestion.models import (
    PIPELINE_VERSION,
    IngestionError,
    IngestionMetrics,
    IngestionResult,
    ReportIngestStats,
)
from ingestion.partitioners import get_partitioner
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


class IngestionPipeline:
    def __init__(self, settings, *, enable_visuals: bool = True, partition_only: bool = False):
        self.settings = settings
        self.enable_visuals = enable_visuals and getattr(settings, "visual_enrichment_enabled", True)
        self.partition_only = partition_only
        self.partitioner = get_partitioner(settings)
        configure_langsmith(settings)

    def ingest_all(
        self,
        rebuild: bool = False,
        only_file: Optional[str] = None,
        reprocess_visuals: bool = False,
    ) -> IngestionResult:
        from rag_app import (
            get_embedder,
            get_vectorstore,
            iter_pdf_paths,
        )

        t_total = time.perf_counter()
        vectorstore = None
        if not self.partition_only:
            chroma_client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
            if rebuild:
                try:
                    chroma_client.delete_collection(name=self.settings.collection_name)
                    print("Rebuilt vector index (old records removed).")
                except Exception as exc:
                    print(f"Warning: could not delete collection '{self.settings.collection_name}': {exc}")

            embedder = get_embedder(self.settings)
            vectorstore = get_vectorstore(self.settings, embedder)

        pdf_paths = list(iter_pdf_paths(self.settings.knowledge_dir, self.settings.extra_pdf_dirs))
        if only_file:
            pdf_paths = [p for p in pdf_paths if p.name == only_file]
            if not pdf_paths:
                # Also allow bare stem match
                pdf_paths = [
                    p
                    for p in iter_pdf_paths(self.settings.knowledge_dir, self.settings.extra_pdf_dirs)
                    if p.name == only_file or p.stem == Path(only_file).stem
                ]
        if not pdf_paths:
            dirs = [self.settings.knowledge_dir] + list(self.settings.extra_pdf_dirs)
            print(f"No PDFs found in {', '.join(str(d) for d in dirs)}")
            return IngestionResult(status="completed", files=[])

        manifest = {} if rebuild else load_ingest_manifest(self.settings.chroma_dir)
        result = IngestionResult(status="completed", files=[])
        metrics = IngestionMetrics()
        tracing = bool(getattr(self.settings, "langsmith_tracing", False))

        pdf_bar = tqdm(pdf_paths, desc="Ingesting PDFs (unstructured)", unit="pdf", dynamic_ncols=True)
        for pdf_path in pdf_bar:
            pdf_bar.set_postfix(file=pdf_path.name[:40])
            entry = manifest.get(pdf_path.name)
            if not rebuild and not reprocess_visuals and should_skip_pdf(entry, pdf_path, self.settings):
                tqdm.write(f"\nSkipping {pdf_path.name} (unchanged since last ingest)")
                continue

            try:
                report_stats = self.ingest_pdf(
                    pdf_path,
                    vectorstore=vectorstore,
                    reprocess_visuals=reprocess_visuals,
                    metrics=metrics,
                    tracing=tracing,
                )
                result.reports.append(report_stats)
                result.files.append(pdf_path.name)
                result.errors.extend(getattr(self, "_last_report_errors", []))
                if not self.partition_only:
                    manifest[pdf_path.name] = build_manifest_entry(
                        pdf_path,
                        self.settings,
                        element_count=report_stats.elements,
                        visual_count=report_stats.figures,
                        table_count=report_stats.tables,
                        indexed_chunk_count=report_stats.indexed_chunks,
                        failed_element_ids=report_stats.failed_elements,
                        partitioner=getattr(self.partitioner, "provider_name", "unstructured-local"),
                        partition_strategy=getattr(self.settings, "unstructured_strategy", "hi_res"),
                        visual_enrichment_enabled=self.enable_visuals,
                    )
                    save_ingest_manifest(self.settings.chroma_dir, manifest)
            except Exception as exc:
                result.errors.append(
                    IngestionError(element_id="", stage="ingest-pdf", message=f"{pdf_path.name}: {exc}")
                )
                tqdm.write(f"\nFailed {pdf_path.name}: {exc}")

        metrics.total_ms = (time.perf_counter() - t_total) * 1000
        result.metrics = metrics
        if result.errors:
            result.status = (
                "completed_with_errors" if result.reports else "failed"
            )
        print(
            f"\nUnstructured ingestion complete. "
            f"{sum(r.indexed_chunks for r in result.reports)} chunks across "
            f"{len(result.reports)} report(s) in '{self.settings.collection_name}'."
        )
        return result

    def ingest_pdf(
        self,
        pdf_path: Path,
        *,
        vectorstore=None,
        reprocess_visuals: bool = False,
        metrics: Optional[IngestionMetrics] = None,
        tracing: bool = False,
        rebuild: bool = False,
    ) -> ReportIngestStats:
        from rag_app import _add_documents_with_retry, build_doc_id, get_embedder, get_vectorstore

        metrics = metrics or IngestionMetrics()
        artifact_dir = Path(self.settings.artifact_dir) / pdf_path.stem
        artifact_dir.mkdir(parents=True, exist_ok=True)
        cache = EnrichmentCache(artifact_dir / "cache")
        self._last_report_errors: List[IngestionError] = []

        with stage_span("ingest-report", {"source": pdf_path.name}, enabled=tracing):
            t0 = time.perf_counter()
            with stage_span("partition-pdf", {"source": pdf_path.name}, enabled=tracing):
                elements = load_partition_cache(
                    artifact_dir,
                    pdf_path,
                    self.settings,
                    self.partitioner,
                )
                partition_cache_hit = elements is not None
                if elements is None:
                    elements = self.partitioner.partition(pdf_path)
                    save_partition_cache(
                        artifact_dir,
                        pdf_path,
                        self.settings,
                        self.partitioner,
                        elements,
                    )
                else:
                    metrics.partition_cache_hits += 1
            metrics.partition_ms += (time.perf_counter() - t0) * 1000
            source_sha256 = file_sha256(pdf_path)
            for element in elements:
                element.metadata["source_sha256"] = source_sha256

            t0 = time.perf_counter()
            with stage_span("normalize-elements", enabled=tracing):
                # Normalization already done inside partitioner; hierarchy next
                elements = annotate_hierarchy(elements)
            metrics.normalize_ms += (time.perf_counter() - t0) * 1000

            pages = max((el.page_number for el in elements), default=0)
            text_n = sum(1 for el in elements if el.category in {"NarrativeText", "ListItem", "Title"})
            table_n = sum(1 for el in elements if el.category == "Table")
            figure_n = sum(1 for el in elements if el.category in {"Image", "Figure"})

            analyses = {}
            validations = {}
            errors: List[IngestionError] = []
            recon = {}
            reconstruction_warning_count = 0
            enrich_stats = {
                "bedrock_calls": 0,
                "cache_hits": 0,
                "warnings": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0.0,
                "retry_count": 0,
            }

            if not self.partition_only:
                t0 = time.perf_counter()
                with stage_span("enrich-visuals", enabled=tracing):
                    analyses, validations, errors, enrich_stats = enrich_elements(
                        elements,
                        self.settings,
                        cache=cache,
                        enable_visuals=self.enable_visuals,
                        bypass_cache=reprocess_visuals,
                    )
                metrics.enrich_ms += (time.perf_counter() - t0) * 1000
                metrics.bedrock_calls += enrich_stats.get("bedrock_calls", 0)
                metrics.cache_hits += enrich_stats.get("cache_hits", 0)
                metrics.input_tokens += enrich_stats.get("input_tokens", 0)
                metrics.output_tokens += enrich_stats.get("output_tokens", 0)
                metrics.bedrock_latency_ms += enrich_stats.get("latency_ms", 0.0)
                metrics.retry_count += enrich_stats.get("retry_count", 0)

                enrichments_dir = artifact_dir / "enrichments"
                enrichments_dir.mkdir(parents=True, exist_ok=True)
                for element_id, validation in validations.items():
                    (enrichments_dir / f"{element_id}-table.json").write_text(
                        validation.model_dump_json(indent=2),
                        encoding="utf-8",
                    )

                t0 = time.perf_counter()
                with stage_span("reconstruct-visuals", enabled=tracing):
                    recon, _ = reconstruct_visuals(elements, analyses, self.settings, artifact_dir)
                    reconstruction_warning_count = sum(
                        max(
                            0,
                            len(info.get("warnings") or [])
                            - len(
                                analyses[element_id].warnings
                                if element_id in analyses
                                else []
                            ),
                        )
                        for element_id, info in recon.items()
                    )
                metrics.reconstruct_ms += (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                with stage_span("create-chunks", enabled=tracing):
                    docs = elements_to_documents(
                        elements,
                        analyses=analyses,
                        validations=validations,
                        reconstructions=recon,
                        chunk_size=self.settings.chunk_size,
                        chunk_overlap=self.settings.chunk_overlap,
                        table_confidence_threshold=self.settings.bedrock_visual_confidence_threshold,
                    )
                    (artifact_dir / "chunks.json").write_text(
                        json.dumps(
                            [
                                {
                                    "page_content": document.page_content,
                                    "metadata": document.metadata,
                                }
                                for document in docs
                            ],
                            indent=2,
                            ensure_ascii=True,
                        ),
                        encoding="utf-8",
                    )
                metrics.chunk_ms += (time.perf_counter() - t0) * 1000

                # Chapter consistency pass
                from chapter_index import (
                    build_chapter_index_from_documents,
                    save_chapter_index,
                    tag_documents_with_items,
                )

                chapters = build_chapter_index_from_documents(docs)
                docs = tag_documents_with_items(docs, chapters)
                save_chapter_index(self.settings.extracted_dir, pdf_path.name, chapters)

                if vectorstore is None:
                    embedder = get_embedder(self.settings)
                    vectorstore = get_vectorstore(self.settings, embedder)

                t0 = time.perf_counter()
                with stage_span("embed-chunks", enabled=tracing):
                    ids = [
                        build_doc_id(
                            pdf_path.name,
                            int(d.metadata.get("page", -1) or -1),
                            int(d.metadata.get("chunk", 0) or 0),
                            d.page_content,
                        )
                        for d in docs
                    ]
                with stage_span("upsert-chroma", enabled=tracing):
                    if docs:
                        existing_ids = _source_document_ids(
                            vectorstore, pdf_path.name
                        )
                        _add_documents_with_retry(
                            vectorstore=vectorstore,
                            docs=docs,
                            ids=ids,
                            batch_size=self.settings.upsert_batch_size,
                        )
                        stale_ids = existing_ids.difference(ids)
                        if stale_ids:
                            _delete_document_ids(vectorstore, stale_ids)
                metrics.embed_ms += (time.perf_counter() - t0) * 1000
                indexed = len(docs)
            else:
                indexed = 0

        reconstructed_charts = sum(
            1 for v in recon.values() if v.get("reconstruction_allowed") and v.get("reason") == "chart"
        )
        reconstructed_diagrams = sum(
            1 for v in recon.values() if v.get("reconstruction_allowed") and v.get("reason") == "diagram"
        )
        failed = [e.element_id for e in errors if e.element_id]
        self._last_report_errors = errors

        tqdm.write(
            f"  {pdf_path.name}: {len(elements)} elements → {indexed} chunks "
            f"(bedrock={enrich_stats.get('bedrock_calls', 0)}, cache={enrich_stats.get('cache_hits', 0)})"
        )

        return ReportIngestStats(
            filename=pdf_path.name,
            pages=pages,
            elements=len(elements),
            text_elements=text_n,
            tables=table_n,
            figures=figure_n,
            bedrock_calls=enrich_stats.get("bedrock_calls", 0),
            cache_hits=enrich_stats.get("cache_hits", 0),
            partition_cache_hits=int(partition_cache_hit),
            input_tokens=enrich_stats.get("input_tokens", 0),
            output_tokens=enrich_stats.get("output_tokens", 0),
            bedrock_latency_ms=enrich_stats.get("latency_ms", 0.0),
            retry_count=enrich_stats.get("retry_count", 0),
            reconstructed_charts=reconstructed_charts,
            reconstructed_diagrams=reconstructed_diagrams,
            warnings=(
                enrich_stats.get("warnings", 0)
                + reconstruction_warning_count
            ),
            failed_elements=failed,
            indexed_chunks=indexed,
        )

    def inspect_elements(self, pdf_path: Path) -> dict:
        elements = self.partitioner.partition(pdf_path)
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
            "file": pdf_path.name,
            "element_category_counts": dict(cats),
            "ni_item_counts": {str(k): v for k, v in sorted(ni_counts.items())},
            "tables": sum(1 for el in elements if el.category == "Table"),
            "figures": len(figures),
            "figures_selected_for_bedrock": selected,
            "section_titles": sorted({el.section_title for el in elements if el.section_title}),
        }
