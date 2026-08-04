"""Selective visual enrichment with routing, retries, and caching."""

from __future__ import annotations

import base64
import re
from io import BytesIO
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from ingestion.cache import EnrichmentCache
from ingestion.context import build_visual_context, needs_table_validation
from ingestion.models import (
    ElementRecord,
    IngestionError,
    TableValidation,
    VisualAnalysis,
)
from ingestion.visual_model import VisualRequest, create_visual_model


FIGURE_ANALYSIS_INSTRUCTIONS = (
    "You are analysing a figure from an NI 43-101 mineral project report.\n"
    "Use the supplied caption, preceding_text, following_text, and "
    "figure_references together to understand what the figure represents. "
    "Treat that narrative as report context, not as permission to manufacture "
    "visual labels, geometry, or chart points that are not visible.\n"
    "Apply the prohibited-geometry safety rule before chart or process-diagram "
    "rules. A geological cross-section with labeled features or arrows is "
    "still a geological cross-section, never a process diagram.\n"
    "Return a complete VisualAnalysis object and make an explicit decision for "
    "every field. Never leave classification, quantitative-data, "
    "reconstruction, method, confidence, or description fields at a schema "
    "default merely because a nested chart or diagram was extracted.\n"
    "Confidence is a calibrated decimal from 0.0 to 1.0, never a percentage.\n"
    "Use figure_type=bar_chart, line_chart, scatter_chart, or pie_chart for "
    "those chart types, and set chart.chart_type to bar, line, scatter, or pie. "
    "Use figure_type=process_diagram for a readable process flow and set "
    "diagram.diagram_type=process.\n"
    "For a clearly readable chart, set contains_quantitative_data=true and "
    "reconstruction_supported=true only when its data is sufficient for a "
    "faithful Plotly reconstruction. For a readable process diagram, if all "
    "labeled boxes and directed connections are visible and captured in "
    "complete node and edge lists, set reconstruction_supported=true and "
    "reconstruction_method=graphviz. Do not reject it merely because it is "
    "non-quantitative.\n"
    "Only extract chart or diagram data when values and relationships are "
    "clearly readable. Use only numbers printed in the image: never infer "
    "convenient axis bounds or series counts. Set unprinted x/y bounds to null. "
    "For charts set diagram=null; for diagrams set chart=null. If values are "
    "estimated, set values_are_estimated=true.\n"
    "Never invent geological geometry. Geological maps, cross-sections, mine "
    "plans, drill-hole maps, pit shells, contour maps, and 3D geological views "
    "must use the exact schema taxonomy (for example mine_plan), have "
    "reconstruction_supported=false and reconstruction_method=none, and set "
    "both chart=null and diagram=null.\n"
)

TABLE_VALIDATION_INSTRUCTIONS = (
    "Validate this table extracted from an NI 43-101 report. Return a complete "
    "TableValidation object, explicitly setting validity, description, "
    "issues, normalized_markdown, confidence, and warnings. Flag malformed "
    "HTML, inconsistent columns, merged headers, and missing numeric content. "
    "If reliable, provide normalized markdown. Confidence is a calibrated "
    "decimal from 0.0 to 1.0, never a percentage. Do not invent numbers.\n"
)

_TABLE_CONTEXT_CHAR_LIMIT = 8000
_TEXT_ONLY_TABLE_MODEL_CHAR_LIMIT = 4096


def should_enrich_figure(el: ElementRecord, settings) -> Tuple[bool, str]:
    if el.category not in {"Image", "Figure"}:
        return False, "not_figure"
    if el.is_duplicate or el.skip_reason:
        return False, el.skip_reason or "duplicate"
    if not el.image_path or not Path(el.image_path).exists():
        return False, "missing_image"
    w = el.image_width or 0
    h = el.image_height or 0
    if w and h and (w < settings.visual_min_width or h < settings.visual_min_height):
        return False, "below_min_dimensions"
    # Try to open and confirm size when metadata missing
    if not w or not h:
        try:
            from PIL import Image

            with Image.open(el.image_path) as img:
                w, h = img.size
                el.image_width, el.image_height = w, h
            if w < settings.visual_min_width or h < settings.visual_min_height:
                return False, "below_min_dimensions"
        except Exception:
            return False, "unreadable_image"
    return True, ""


def requires_deterministic_table_fallback(el: ElementRecord) -> bool:
    """Whether the model cannot receive a complete table or a source image."""

    table_source = el.text_as_html or el.text
    has_image = bool(
        el.image_path and Path(el.image_path).exists()
    )
    return (
        len(table_source) > _TEXT_ONLY_TABLE_MODEL_CHAR_LIMIT
        and not has_image
    )


def _image_payload(path: str, settings) -> Tuple[str, str]:
    """Return a bounded image payload without modifying the authoritative crop."""
    data = Path(path).read_bytes()
    max_width = max(1, int(getattr(settings, "visual_max_width", 4096)))
    max_height = max(1, int(getattr(settings, "visual_max_height", 4096)))
    try:
        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            if image.width > max_width or image.height > max_height:
                image.thumbnail((max_width, max_height))
                output = BytesIO()
                image.convert("RGB").save(output, format="PNG", optimize=True)
                data = output.getvalue()
            else:
                media_type = Image.MIME.get(image.format or "", "image/png")
                return base64.b64encode(data).decode("ascii"), media_type
    except Exception:
        return base64.b64encode(data).decode("ascii"), "image/png"
    return base64.b64encode(data).decode("ascii"), "image/png"


def _figure_message(el: ElementRecord, context_payload: dict, settings) -> VisualRequest:
    import json

    text = (
        FIGURE_ANALYSIS_INSTRUCTIONS
        + "\n"
        + f"Context:\n{json.dumps(context_payload, ensure_ascii=True)}\n"
    )
    image_b64, media_type = _image_payload(el.image_path, settings)
    return VisualRequest(
        task="figure",
        prompt=text,
        image_base64=image_b64,
        media_type=media_type,
    )


def _table_message(el: ElementRecord, context_payload: dict, settings) -> VisualRequest:
    import json

    table_source = el.text_as_html or el.text
    truncated = len(table_source) > _TABLE_CONTEXT_CHAR_LIMIT
    truncation_instruction = (
        "\nThe supplied table HTML is truncated. You cannot safely normalize "
        "the complete table. MUST leave normalized_markdown empty, add "
        "input_truncated to issues, and describe only structural problems "
        "visible in the supplied prefix.\n"
        if truncated
        else ""
    )
    text = (
        TABLE_VALIDATION_INSTRUCTIONS
        + truncation_instruction
        + "\n"
        + f"Context:\n{json.dumps(context_payload, ensure_ascii=True)}\n\n"
        + f"Table HTML:\n{table_source[:_TABLE_CONTEXT_CHAR_LIMIT]}"
    )
    image_b64 = ""
    media_type = "image/png"
    if el.image_path and Path(el.image_path).exists():
        image_b64, media_type = _image_payload(el.image_path, settings)
    return VisualRequest(
        task="table",
        prompt=text,
        image_base64=image_b64,
        media_type=media_type,
    )


def _invoke(model, request: VisualRequest):
    attempts = 0
    response = None
    for attempt in Retrying(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    ):
        with attempt:
            attempts += 1
            response = model.analyze(request)
    return response.value, {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "retry_count": max(0, attempts - 1),
        "provider_latency_ms": response.latency_ms,
    }


def enrich_elements(
    elements: List[ElementRecord],
    settings,
    cache: Optional[EnrichmentCache] = None,
    enable_visuals: bool = True,
    bypass_cache: bool = False,
    visual_model=None,
) -> Tuple[Dict[str, VisualAnalysis], Dict[str, TableValidation], List[IngestionError], dict]:
    """Enrich selected figures/tables. Returns analyses, validations, errors, stats."""
    analyses: Dict[str, VisualAnalysis] = {}
    validations: Dict[str, TableValidation] = {}
    errors: List[IngestionError] = []
    stats = {
        "bedrock_calls": 0,
        "visual_model_calls": 0,
        "visual_model_provider": str(
            getattr(settings, "visual_model_provider", "bedrock")
        ),
        "cache_hits": 0,
        "warnings": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0.0,
        "retry_count": 0,
        "budget_skips": 0,
        "deferred_element_ids": [],
    }
    elements_by_id = {element.element_id: element for element in elements}

    if not enable_visuals or not getattr(settings, "visual_enrichment_enabled", True):
        return analyses, validations, errors, stats

    figure_jobs: List[ElementRecord] = []
    table_jobs: List[ElementRecord] = []
    for el in elements:
        if el.skip_reason == "visual_budget_limit":
            # A budget deferral is transient. It must be eligible on the next
            # resumable pass after earlier jobs have become cache hits.
            el.skip_reason = None
        ok, reason = should_enrich_figure(el, settings)
        if ok:
            figure_jobs.append(el)
        elif el.category in {"Image", "Figure"}:
            el.skip_reason = el.skip_reason or reason
        if needs_table_validation(el):
            table_jobs.append(el)

    max_calls = max(
        0,
        int(getattr(settings, "visual_max_calls_per_report", 30)),
    )
    token_budget = max(
        0, int(getattr(settings, "visual_token_budget_per_report", 350000))
    )
    max_tokens = max(1, int(getattr(settings, "bedrock_visual_max_tokens", 3500)))
    max_table_calls = max(
        0,
        int(
            getattr(
                settings,
                "visual_max_table_calls_per_report",
                max_calls,
            )
        ),
    )
    max_figure_calls = max(
        0,
        int(
            getattr(
                settings,
                "visual_max_figure_calls_per_report",
                max_calls,
            )
        ),
    )

    def figure_priority(element: ElementRecord) -> tuple[int, int, int]:
        due_diligence_item = int(element.ni_item or 0) in {
            13,
            14,
            15,
            16,
            17,
            18,
            21,
            22,
            25,
            26,
        }
        quantitative = bool(
            re.search(
                r"\b(grade|tonnage|resource|reserve|npv|irr|recovery|"
                r"production|cash flow|capex|opex)\b",
                f"{element.caption} {element.preceding_text}",
                flags=re.IGNORECASE,
            )
        )
        area = int(element.image_width or 0) * int(
            element.image_height or 0
        )
        return (
            0 if due_diligence_item else 1,
            0 if quantitative else 1,
            -area,
        )

    table_jobs = sorted(
        table_jobs,
        key=lambda element: (
            0 if int(element.ni_item or 0) in {14, 15, 16, 21, 22} else 1,
            element.page_number,
        ),
    )
    deterministic_table_jobs = [
        element
        for element in table_jobs
        if requires_deterministic_table_fallback(element)
    ]
    table_jobs = [
        element
        for element in table_jobs
        if not requires_deterministic_table_fallback(element)
    ]
    figure_jobs = sorted(
        figure_jobs,
        key=figure_priority,
    )
    all_jobs = [("table", el) for el in table_jobs] + [
        ("figure", el) for el in figure_jobs
    ]
    if all_jobs and visual_model is None:
        visual_model = create_visual_model(settings)
    model_id = (
        getattr(
            visual_model,
            "cache_id",
            getattr(visual_model, "model_id", "unknown"),
        )
        if visual_model is not None
        else "visual-model-not-invoked"
    )
    provider = str(
        getattr(
            visual_model,
            "provider",
            getattr(settings, "visual_model_provider", "bedrock"),
        )
    )
    stats["visual_model_provider"] = provider

    preloaded: dict[
        tuple[str, str],
        VisualAnalysis | TableValidation,
    ] = {}
    if cache and not bypass_cache:
        for kind, el in all_jobs:
            ctx = build_visual_context(
                el,
                task=(
                    "Validate the attached table extraction."
                    if kind == "table"
                    else "Classify and analyse the attached visual."
                ),
            )
            payload = ctx.model_dump()
            image_bytes = (
                Path(el.image_path).read_bytes()
                if el.image_path and Path(el.image_path).exists()
                else b""
            )
            cached = (
                cache.get_table(el, payload, model_id, image_bytes)
                if kind == "table"
                else cache.get_visual(el, payload, model_id, image_bytes)
            )
            if cached is not None:
                preloaded[(kind, el.element_id)] = (
                    TableValidation.model_validate(cached)
                    if kind == "table"
                    else VisualAnalysis.model_validate(cached)
                )

    selected_jobs = []
    cached_jobs = []
    deferred_jobs = []
    estimated_tokens = 0
    selected_by_kind = {"table": 0, "figure": 0}
    for kind, el in all_jobs:
        if (kind, el.element_id) in preloaded:
            cached_jobs.append((kind, el))
            continue
        text_chars = len(el.preceding_text) + len(el.following_text) + len(el.caption)
        if kind == "table":
            text_chars += len(el.text_as_html or el.text)
        width = min(
            el.image_width or 0,
            int(getattr(settings, "visual_max_width", 4096)),
        )
        height = min(
            el.image_height or 0,
            int(getattr(settings, "visual_max_height", 4096)),
        )
        # Conservative preflight estimate: text plus Anthropic-style image token area.
        job_estimate = max_tokens + (text_chars // 4) + ((width * height) // 750)
        if (
            len(selected_jobs) >= max_calls
            or (
                kind == "table"
                and selected_by_kind["table"] >= max_table_calls
            )
            or (
                kind == "figure"
                and selected_by_kind["figure"] >= max_figure_calls
            )
            or estimated_tokens + job_estimate > token_budget
        ):
            deferred_jobs.append((kind, el))
            continue
        selected_jobs.append((kind, el))
        selected_by_kind[kind] += 1
        estimated_tokens += job_estimate
    runnable_jobs = cached_jobs + selected_jobs
    figure_jobs = [
        el for kind, el in runnable_jobs if kind == "figure"
    ]
    table_jobs = [
        el for kind, el in runnable_jobs if kind == "table"
    ] + deterministic_table_jobs
    for kind, el in deferred_jobs:
        if kind == "figure":
            el.skip_reason = "visual_budget_limit"
    stats["budget_skips"] = len(deferred_jobs)
    stats["deferred_element_ids"] = [
        el.element_id for _, el in deferred_jobs
    ]
    stats["estimated_token_budget_used"] = estimated_tokens
    stats["warnings"] += len(deferred_jobs)

    def _analyse_figure(el: ElementRecord):
        ctx = build_visual_context(el)
        payload = ctx.model_dump()
        image_bytes = Path(el.image_path).read_bytes() if el.image_path else b""
        cached = preloaded.get(("figure", el.element_id))
        if cached is not None:
            return el.element_id, cached, None, True, {}, 0.0
        try:
            t0 = time.perf_counter()
            result, usage = _invoke(
                visual_model, _figure_message(el, payload, settings)
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            if not isinstance(result, VisualAnalysis):
                result = VisualAnalysis.model_validate(result)
            if cache:
                cache.put_visual(el, payload, model_id, image_bytes, result.model_dump())
            return el.element_id, result, None, False, usage, latency_ms
        except Exception as exc:
            return (
                el.element_id,
                None,
                IngestionError(element_id=el.element_id, stage="enrich-figure", message=str(exc)),
                False,
                {},
                0.0,
            )

    def _validate_table(el: ElementRecord):
        if requires_deterministic_table_fallback(el):
            return (
                el.element_id,
                TableValidation(
                    is_valid=False,
                    description=(
                        "Table exceeds the safe text-only validation size and "
                        "has no source image; original parser output was "
                        "preserved without model normalization."
                    ),
                    issues=["input_truncated"],
                    normalized_markdown="",
                    confidence=1.0,
                    warnings=["visual_model_not_called"],
                ),
                None,
                False,
                {"model_invoked": False},
                0.0,
            )
        ctx = build_visual_context(el, task="Validate the attached table extraction.")
        payload = ctx.model_dump()
        image_bytes = Path(el.image_path).read_bytes() if el.image_path and Path(el.image_path).exists() else b""
        cached = preloaded.get(("table", el.element_id))
        if cached is not None:
            return el.element_id, cached, None, True, {}, 0.0
        try:
            t0 = time.perf_counter()
            result, usage = _invoke(
                visual_model, _table_message(el, payload, settings)
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            if not isinstance(result, TableValidation):
                result = TableValidation.model_validate(result)
            if cache:
                cache.put_table(el, payload, model_id, image_bytes, result.model_dump())
            return el.element_id, result, None, False, usage, latency_ms
        except Exception as exc:
            return (
                el.element_id,
                None,
                IngestionError(element_id=el.element_id, stage="enrich-table", message=str(exc)),
                False,
                {},
                0.0,
            )

    configured_concurrency = int(
        getattr(
            settings,
            "visual_model_concurrency",
            getattr(settings, "bedrock_visual_concurrency", 8),
        )
    )
    concurrency = max(1, configured_concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_analyse_figure, el) for el in figure_jobs]
        futures += [pool.submit(_validate_table, el) for el in table_jobs]
        for fut in as_completed(futures):
            element_id, result, err, cache_hit, usage, latency_ms = fut.result()
            if cache_hit:
                stats["cache_hits"] += 1
            elif usage.get("model_invoked", True):
                stats["visual_model_calls"] += 1
                if provider == "bedrock":
                    stats["bedrock_calls"] += 1
                stats["input_tokens"] += usage.get("input_tokens", 0)
                stats["output_tokens"] += usage.get("output_tokens", 0)
                stats["retry_count"] += usage.get("retry_count", 0)
                stats["latency_ms"] += latency_ms
            if err:
                errors.append(err)
                failed_element = elements_by_id.get(element_id)
                if failed_element is not None:
                    failed_element.skip_reason = "enrichment_failed"
                continue
            if isinstance(result, VisualAnalysis):
                analyses[element_id] = result
                stats["warnings"] += len(result.warnings)
            elif isinstance(result, TableValidation):
                validations[element_id] = result
                stats["warnings"] += len(result.warnings) + len(result.issues)

    # Persist enrichment artifacts
    return analyses, validations, errors, stats
