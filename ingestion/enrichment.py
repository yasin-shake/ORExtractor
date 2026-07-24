"""Selective Bedrock visual enrichment with routing, retries, and caching."""

from __future__ import annotations

import base64
from io import BytesIO
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from ingestion.bedrock import get_table_validation_model, get_visual_analysis_model
from ingestion.cache import EnrichmentCache
from ingestion.context import build_visual_context, needs_table_validation
from ingestion.models import (
    ElementRecord,
    IngestionError,
    TableValidation,
    VisualAnalysis,
)


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


def _figure_message(el: ElementRecord, context_payload: dict, settings) -> HumanMessage:
    import json

    text = (
        "You are analysing a figure from an NI 43-101 mineral project report.\n"
        "Classify the figure, describe it using the surrounding context, and decide whether "
        "deterministic reconstruction is supported.\n"
        "Only extract chart/diagram data when values are clearly readable. "
        "If values are estimated, set values_are_estimated=true.\n"
        "Never invent geological geometry. Geological maps, cross-sections, mine plans, "
        "drill-hole maps, pit shells, contour maps, and 3D geological views must have "
        "reconstruction_supported=false.\n\n"
        f"Context:\n{json.dumps(context_payload, ensure_ascii=True)}\n"
    )
    image_b64, media_type = _image_payload(el.image_path, settings)
    return HumanMessage(
        content=[
            {"type": "text", "text": text},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_b64,
                },
            },
        ]
    )


def _table_message(el: ElementRecord, context_payload: dict, settings) -> HumanMessage:
    import json

    text = (
        "Validate this table extracted from an NI 43-101 report. "
        "Flag malformed HTML, inconsistent columns, merged headers, and missing numeric content. "
        "If reliable, provide normalized markdown. Do not invent numbers.\n\n"
        f"Context:\n{json.dumps(context_payload, ensure_ascii=True)}\n\n"
        f"Table HTML:\n{(el.text_as_html or el.text)[:8000]}"
    )
    content: List[dict] = [{"type": "text", "text": text}]
    if el.image_path and Path(el.image_path).exists():
        image_b64, media_type = _image_payload(el.image_path, settings)
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_b64,
                },
            }
        )
    return HumanMessage(content=content)


def _invoke(model, message: HumanMessage):
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
            response = model.invoke([message])
    raw = None
    parsed = response
    if isinstance(response, dict) and "parsed" in response:
        parsing_error = response.get("parsing_error")
        if parsing_error:
            if isinstance(parsing_error, BaseException):
                raise parsing_error
            raise RuntimeError(str(parsing_error))
        parsed = response.get("parsed")
        raw = response.get("raw")
    usage = getattr(raw, "usage_metadata", None) or {}
    response_metadata = getattr(raw, "response_metadata", None) or {}
    if not usage and isinstance(response_metadata, dict):
        usage = response_metadata.get("usage", {}) or response_metadata.get(
            "usage_metadata", {}
        )
    return parsed, {
        "input_tokens": int(
            usage.get("input_tokens", usage.get("inputTokens", 0)) or 0
        ),
        "output_tokens": int(
            usage.get("output_tokens", usage.get("outputTokens", 0)) or 0
        ),
        "retry_count": max(0, attempts - 1),
    }


def enrich_elements(
    elements: List[ElementRecord],
    settings,
    cache: Optional[EnrichmentCache] = None,
    enable_visuals: bool = True,
    bypass_cache: bool = False,
) -> Tuple[Dict[str, VisualAnalysis], Dict[str, TableValidation], List[IngestionError], dict]:
    """Enrich selected figures/tables. Returns analyses, validations, errors, stats."""
    analyses: Dict[str, VisualAnalysis] = {}
    validations: Dict[str, TableValidation] = {}
    errors: List[IngestionError] = []
    stats = {
        "bedrock_calls": 0,
        "cache_hits": 0,
        "warnings": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 0.0,
        "retry_count": 0,
        "budget_skips": 0,
    }
    elements_by_id = {element.element_id: element for element in elements}

    if not enable_visuals or not getattr(settings, "visual_enrichment_enabled", True):
        return analyses, validations, errors, stats

    figure_jobs: List[ElementRecord] = []
    table_jobs: List[ElementRecord] = []
    for el in elements:
        ok, reason = should_enrich_figure(el, settings)
        if ok:
            figure_jobs.append(el)
        elif el.category in {"Image", "Figure"}:
            el.skip_reason = el.skip_reason or reason
        if needs_table_validation(el):
            table_jobs.append(el)

    max_calls = max(0, int(getattr(settings, "visual_max_calls_per_report", 100)))
    token_budget = max(
        0, int(getattr(settings, "visual_token_budget_per_report", 350000))
    )
    max_tokens = max(1, int(getattr(settings, "bedrock_visual_max_tokens", 3500)))
    all_jobs = [("figure", el) for el in figure_jobs] + [
        ("table", el) for el in table_jobs
    ]
    selected_jobs = []
    deferred_jobs = []
    estimated_tokens = 0
    for kind, el in all_jobs:
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
            or estimated_tokens + job_estimate > token_budget
        ):
            deferred_jobs.append((kind, el))
            continue
        selected_jobs.append((kind, el))
        estimated_tokens += job_estimate
    figure_jobs = [el for kind, el in selected_jobs if kind == "figure"]
    table_jobs = [el for kind, el in selected_jobs if kind == "table"]
    for kind, el in deferred_jobs:
        if kind == "figure":
            el.skip_reason = "visual_budget_limit"
    stats["budget_skips"] = len(deferred_jobs)
    stats["estimated_token_budget_used"] = estimated_tokens
    stats["warnings"] += len(deferred_jobs)

    model_id = settings.bedrock_visual_model_id
    visual_model = get_visual_analysis_model(settings) if figure_jobs else None
    table_model = get_table_validation_model(settings) if table_jobs else None

    def _analyse_figure(el: ElementRecord):
        ctx = build_visual_context(el)
        payload = ctx.model_dump()
        image_bytes = Path(el.image_path).read_bytes() if el.image_path else b""
        if cache and not bypass_cache:
            cached = cache.get_visual(el, payload, model_id, image_bytes)
            if cached is not None:
                return el.element_id, VisualAnalysis.model_validate(cached), None, True, {}, 0.0
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
        ctx = build_visual_context(el, task="Validate the attached table extraction.")
        payload = ctx.model_dump()
        image_bytes = Path(el.image_path).read_bytes() if el.image_path and Path(el.image_path).exists() else b""
        if cache and not bypass_cache:
            cached = cache.get_table(el, payload, model_id, image_bytes)
            if cached is not None:
                return el.element_id, TableValidation.model_validate(cached), None, True, {}, 0.0
        try:
            t0 = time.perf_counter()
            result, usage = _invoke(
                table_model, _table_message(el, payload, settings)
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

    concurrency = max(1, int(getattr(settings, "bedrock_visual_concurrency", 6)))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_analyse_figure, el) for el in figure_jobs]
        futures += [pool.submit(_validate_table, el) for el in table_jobs]
        for fut in as_completed(futures):
            element_id, result, err, cache_hit, usage, latency_ms = fut.result()
            if cache_hit:
                stats["cache_hits"] += 1
            else:
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
