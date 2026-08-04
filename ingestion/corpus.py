"""Intent-oriented corpus operations built on canonical ingestion artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from ingestion.cache import (
    build_manifest_entry,
    file_sha256,
    load_ingest_manifest,
    parser_result_accepted,
    save_ingest_manifest,
    visual_model_signature,
)
from ingestion.context import annotate_hierarchy
from ingestion.crops import CropMaterializer
from ingestion.models import (
    PIPELINE_VERSION,
    VISUAL_PROMPT_VERSION,
    VISUAL_SCHEMA_VERSION,
    IngestionError,
    IngestionResult,
    ParserResult,
    VisualBackfillFileStatus,
    VisualBackfillStatus,
)
from ingestion.pipeline import (
    IngestionPipeline,
    _ReportWork,
    _merge_metrics,
)
from ingestion.runtime import IngestionRuntime
from ingestion.sources import (
    filesystem_path,
    iter_pdf_paths,
    pdf_source_id,
    source_output_path,
)
from ingestion.visual_filtering import filter_visual_artifacts


class VisualPrerequisiteMissing(RuntimeError):
    """Visual backfill cannot proceed from the accepted parser artifact."""


def _visual_policy(settings) -> dict:
    return {
        "max_calls": int(
            getattr(settings, "visual_max_calls_per_report", 30)
        ),
        "max_table_calls": int(
            getattr(settings, "visual_max_table_calls_per_report", 20)
        ),
        "max_figure_calls": int(
            getattr(settings, "visual_max_figure_calls_per_report", 10)
        ),
        "token_budget": int(
            getattr(
                settings,
                "visual_token_budget_per_report",
                350000,
            )
        ),
    }


def _configured_embedding_signature(settings) -> dict | None:
    resolved = getattr(
        settings,
        "resolved_embedding_signature",
        None,
    )
    if isinstance(resolved, dict):
        return dict(resolved)
    from local_embeddings import embedding_signature

    provider = str(
        getattr(settings, "embedding_provider", "")
    ).strip().lower()
    if provider == "qwen":
        return embedding_signature(
            provider="qwen",
            model=str(settings.local_embed_model),
            dimensions=int(settings.local_embed_dimensions),
            query_instruction=str(
                settings.local_embed_query_instruction
            ),
            max_length=int(settings.local_embed_max_length),
            normalize=True,
        )
    if provider == "openai":
        return embedding_signature(
            provider="openai",
            model=str(settings.embed_model),
            dimensions=int(settings.openai_embed_dimensions),
            normalize=True,
        )
    return None


def visual_backfill_reasons(
    entry,
    pdf_path: Path,
    settings,
) -> list[str]:
    """Return exact reasons the visual/index stages are not current."""
    reasons: list[str] = []
    if not isinstance(entry, dict):
        return ["manifest_missing"]
    if entry.get("source_sha256") != file_sha256(pdf_path):
        reasons.append("source_changed")
    if entry.get("pipeline_version") != PIPELINE_VERSION:
        reasons.append("pipeline_version_changed")
    if entry.get("visual_prompt_version") != VISUAL_PROMPT_VERSION:
        reasons.append("visual_prompt_changed")
    if entry.get("visual_schema_version") != VISUAL_SCHEMA_VERSION:
        reasons.append("visual_schema_changed")
    if entry.get("visual_enrichment_enabled") is not True:
        reasons.append("visuals_not_enabled")
    if entry.get("visual_model") != visual_model_signature(settings):
        reasons.append("visual_model_changed")
    if entry.get("failed_element_ids"):
        reasons.append("failed_elements")
    if entry.get("pending_element_ids"):
        reasons.append("pending_elements")
    if entry.get("visual_policy") != _visual_policy(settings):
        reasons.append("visual_policy_changed")
    if entry.get("visual_confidence_threshold") != getattr(
        settings,
        "bedrock_visual_confidence_threshold",
        0.85,
    ):
        reasons.append("visual_confidence_changed")
    if entry.get("visual_reconstruct_charts") != getattr(
        settings,
        "visual_reconstruct_charts",
        True,
    ):
        reasons.append("chart_policy_changed")
    if entry.get("visual_reconstruct_diagrams") != getattr(
        settings,
        "visual_reconstruct_diagrams",
        True,
    ):
        reasons.append("diagram_policy_changed")
    if entry.get("chunk_size") != settings.chunk_size:
        reasons.append("chunk_size_changed")
    if entry.get("chunk_overlap") != settings.chunk_overlap:
        reasons.append("chunk_overlap_changed")
    configured_embedding = _configured_embedding_signature(settings)
    if (
        configured_embedding is None
        or entry.get("embedding_signature")
        != configured_embedding
    ):
        reasons.append("embedding_space_changed")
    return reasons


def visual_backfill_current(entry, pdf_path: Path, settings) -> bool:
    """Return whether the visual/index stages are complete for one source."""
    return not visual_backfill_reasons(
        entry,
        pdf_path,
        settings,
    )


def _load_visual_prerequisite(
    artifact_dir: Path,
    pdf_path: Path,
    settings,
) -> ParserResult:
    metadata_path = artifact_dir / "parser_cache.json"
    result_path = artifact_dir / "parser_result.json"
    if not metadata_path.exists() or not result_path.exists():
        raise VisualPrerequisiteMissing(
            f"accepted parser artifact is missing under {artifact_dir}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_sha256") != file_sha256(pdf_path):
            raise VisualPrerequisiteMissing(
                "parser artifact belongs to a different PDF revision"
            )
        result = ParserResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
    except VisualPrerequisiteMissing:
        raise
    except Exception as exc:
        raise VisualPrerequisiteMissing(
            f"parser artifact is unreadable: {exc}"
        ) from exc
    if not parser_result_accepted(result, settings):
        raise VisualPrerequisiteMissing(
            "parser artifact does not satisfy the current quality gate"
        )
    return result


class CorpusIngestion:
    """High-level corpus intents that hide parser and publication mechanics."""

    def __init__(
        self,
        settings,
        *,
        runtime: IngestionRuntime | None = None,
        pipeline: IngestionPipeline | None = None,
    ):
        self.settings = settings
        self.runtime = runtime
        self.pipeline = pipeline or IngestionPipeline(
            settings,
            enable_visuals=True,
            runtime=runtime,
            initialize_parser=False,
        )
        self.crop_materializer = CropMaterializer(
            scale=float(
                getattr(settings, "visual_crop_render_scale", 2.0)
            )
        )

    def _discover(
        self,
        only_file: Optional[str],
    ) -> list[tuple[Path, str]]:
        discovered = [
            (
                path,
                pdf_source_id(
                    path,
                    self.settings.knowledge_dir,
                    self.settings.extra_pdf_dirs,
                ),
            )
            for path in iter_pdf_paths(
                self.settings.knowledge_dir,
                self.settings.extra_pdf_dirs,
            )
        ]
        if not only_file:
            return discovered
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
                if item[0].name.casefold()
                == Path(only_file).name.casefold()
                or item[0].stem.casefold()
                == Path(only_file).stem.casefold()
            ]
        if len(matches) > 1:
            choices = ", ".join(source for _, source in matches)
            raise ValueError(
                f"PDF name {only_file!r} is ambiguous. Use one of: "
                f"{choices}"
            )
        if not matches:
            raise FileNotFoundError(
                f"No corpus PDF matches {only_file!r}"
            )
        return matches

    def _work_from_artifact(
        self,
        pdf_path: Path,
        source_file: str,
        artifact_dir: Path,
    ) -> _ReportWork:
        parser_result = _load_visual_prerequisite(
            artifact_dir,
            pdf_path,
            self.pipeline.settings,
        )
        parser_result.elements = filter_visual_artifacts(
            parser_result.elements
        )
        materialized = self.crop_materializer.materialize(
            pdf_path,
            parser_result.elements,
            artifact_dir,
        )
        if materialized:
            parser_result.elements = filter_visual_artifacts(
                parser_result.elements
            )
            (artifact_dir / "parser_result.json").write_text(
                parser_result.model_dump_json(indent=2),
                encoding="utf-8",
            )

        source_sha256 = file_sha256(pdf_path)
        for element in parser_result.elements:
            element.metadata["source_sha256"] = source_sha256
        elements = annotate_hierarchy(parser_result.elements)
        return _ReportWork(
            pdf_path=pdf_path,
            source_file=source_file,
            artifact_dir=artifact_dir,
            parser_result=parser_result,
            elements=elements,
            partition_cache_hit=True,
            pages=max(
                (element.page_number for element in elements),
                default=parser_result.page_count,
            ),
            text_elements=sum(
                element.category
                in {"NarrativeText", "ListItem", "Title"}
                for element in elements
            ),
            tables=sum(
                element.category == "Table"
                for element in elements
            ),
            figures=sum(
                element.category in {"Image", "Figure"}
                for element in elements
            ),
        )

    def resume_visuals(
        self,
        *,
        only_file: Optional[str] = None,
        refresh: bool = False,
    ) -> IngestionResult:
        """Resume missing/stale visual work without invoking a PDF parser."""
        if self.runtime is None:
            raise RuntimeError(
                "IngestionRuntime is required to publish visual backfill"
            )
        started = time.perf_counter()
        manifest = load_ingest_manifest(self.settings.chroma_dir)
        result = IngestionResult(status="completed")
        discovered = self._discover(only_file)
        pending = [
            (path, source_file)
            for path, source_file in discovered
            if refresh
            or not visual_backfill_current(
                manifest.get(source_file),
                filesystem_path(path),
                self.pipeline.settings,
            )
        ]
        if not pending:
            result.metrics.total_ms = (
                time.perf_counter() - started
            ) * 1000
            return result
        embedder = self.runtime.get_embedder(self.pipeline.settings)
        vectorstore = self.runtime.get_vectorstore(
            self.pipeline.settings,
            embedder,
        )

        for discovered_path, source_file in pending:
            pdf_path = filesystem_path(discovered_path)
            existing_entry = manifest.get(source_file)
            artifact_dir = source_output_path(
                self.settings.artifact_dir,
                source_file,
                "",
            )
            try:
                work = self._work_from_artifact(
                    pdf_path,
                    source_file,
                    artifact_dir,
                )
                work = self.pipeline._prepare_report(
                    work,
                    reprocess_visuals=refresh,
                    tracing=bool(
                        getattr(
                            self.settings,
                            "langsmith_tracing",
                            False,
                        )
                    ),
                )
                report, work = self.pipeline._index_report(
                    work,
                    vectorstore=vectorstore,
                    tracing=bool(
                        getattr(
                            self.settings,
                            "langsmith_tracing",
                            False,
                        )
                    ),
                )
                result.reports.append(report)
                result.files.append(source_file)
                result.errors.extend(work.errors)
                _merge_metrics(result.metrics, work.metrics)
                updated_entry = build_manifest_entry(
                    pdf_path,
                    self.pipeline.settings,
                    element_count=report.elements,
                    visual_count=report.figures,
                    table_count=report.tables,
                    indexed_chunk_count=report.indexed_chunks,
                    failed_element_ids=report.failed_elements,
                    pending_element_ids=report.pending_elements,
                    visual_enrichment_enabled=True,
                    parser_result=work.parser_result,
                )
                if isinstance(existing_entry, dict):
                    updated_entry["parser_policy"] = existing_entry.get(
                        "parser_policy",
                        updated_entry["parser_policy"],
                    )
                manifest[source_file] = updated_entry
                save_ingest_manifest(
                    self.settings.chroma_dir,
                    manifest,
                )
            except Exception as exc:
                result.errors.append(
                    IngestionError(
                        stage="visual-backfill",
                        message=f"{source_file}: {exc}",
                    )
                )

        result.metrics.total_ms = (
            time.perf_counter() - started
        ) * 1000
        if result.errors:
            result.status = (
                "completed_with_errors"
                if result.reports
                else "failed"
            )
        return result

    def visual_status(
        self,
        *,
        only_file: Optional[str] = None,
    ) -> VisualBackfillStatus:
        """Inspect visual-backfill readiness without parsing or publishing."""
        manifest = load_ingest_manifest(self.settings.chroma_dir)
        files: list[VisualBackfillFileStatus] = []
        for discovered_path, source_file in self._discover(only_file):
            pdf_path = filesystem_path(discovered_path)
            if visual_backfill_current(
                manifest.get(source_file),
                pdf_path,
                self.pipeline.settings,
            ):
                files.append(
                    VisualBackfillFileStatus(
                        source_file=source_file,
                        state="current",
                    )
                )
                continue
            artifact_dir = source_output_path(
                self.settings.artifact_dir,
                source_file,
                "",
            )
            try:
                _load_visual_prerequisite(
                    artifact_dir,
                    pdf_path,
                    self.pipeline.settings,
                )
                files.append(
                    VisualBackfillFileStatus(
                        source_file=source_file,
                        state="pending",
                        detail=", ".join(
                            visual_backfill_reasons(
                                manifest.get(source_file),
                                pdf_path,
                                self.pipeline.settings,
                            )
                        ),
                    )
                )
            except VisualPrerequisiteMissing as exc:
                files.append(
                    VisualBackfillFileStatus(
                        source_file=source_file,
                        state="blocked",
                        detail=str(exc),
                    )
                )
        return VisualBackfillStatus(files=files)
