"""Selective Bedrock visual enrichment with routing, retries, and caching."""

from __future__ import annotations

import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

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


def _image_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _figure_message(el: ElementRecord, context_payload: dict) -> HumanMessage:
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
    media_type = "image/png"
    if el.image_path and el.image_path.lower().endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    return HumanMessage(
        content=[
            {"type": "text", "text": text},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": _image_b64(el.image_path),
                },
            },
        ]
    )


def _table_message(el: ElementRecord, context_payload: dict) -> HumanMessage:
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
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _image_b64(el.image_path),
                },
            }
        )
    return HumanMessage(content=content)


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential_jitter(initial=1, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _invoke(model, message: HumanMessage):
    return model.invoke([message])


def enrich_elements(
    elements: List[ElementRecord],
    settings,
    cache: Optional[EnrichmentCache] = None,
    enable_visuals: bool = True,
) -> Tuple[Dict[str, VisualAnalysis], Dict[str, TableValidation], List[IngestionError], dict]:
    """Enrich selected figures/tables. Returns analyses, validations, errors, stats."""
    analyses: Dict[str, VisualAnalysis] = {}
    validations: Dict[str, TableValidation] = {}
    errors: List[IngestionError] = []
    stats = {"bedrock_calls": 0, "cache_hits": 0, "warnings": 0}

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

    model_id = settings.bedrock_visual_model_id
    visual_model = None
    table_model = None

    def _analyse_figure(el: ElementRecord) -> Tuple[str, Optional[VisualAnalysis], Optional[IngestionError], bool]:
        nonlocal visual_model
        ctx = build_visual_context(el)
        payload = ctx.model_dump()
        image_bytes = Path(el.image_path).read_bytes() if el.image_path else b""
        if cache:
            cached = cache.get_visual(el, payload, model_id, image_bytes)
            if cached is not None:
                return el.element_id, VisualAnalysis.model_validate(cached), None, True
        if visual_model is None:
            visual_model = get_visual_analysis_model(settings)
        try:
            t0 = time.perf_counter()
            result = _invoke(visual_model, _figure_message(el, payload))
            _ = time.perf_counter() - t0
            if not isinstance(result, VisualAnalysis):
                result = VisualAnalysis.model_validate(result)
            if cache:
                cache.put_visual(el, payload, model_id, image_bytes, result.model_dump())
            return el.element_id, result, None, False
        except Exception as exc:
            return (
                el.element_id,
                None,
                IngestionError(element_id=el.element_id, stage="enrich-figure", message=str(exc)),
                False,
            )

    def _validate_table(el: ElementRecord) -> Tuple[str, Optional[TableValidation], Optional[IngestionError], bool]:
        nonlocal table_model
        ctx = build_visual_context(el, task="Validate the attached table extraction.")
        payload = ctx.model_dump()
        image_bytes = Path(el.image_path).read_bytes() if el.image_path and Path(el.image_path).exists() else b""
        if cache:
            cached = cache.get_table(el, payload, model_id, image_bytes)
            if cached is not None:
                return el.element_id, TableValidation.model_validate(cached), None, True
        if table_model is None:
            table_model = get_table_validation_model(settings)
        try:
            result = _invoke(table_model, _table_message(el, payload))
            if not isinstance(result, TableValidation):
                result = TableValidation.model_validate(result)
            if cache:
                cache.put_table(el, payload, model_id, image_bytes, result.model_dump())
            return el.element_id, result, None, False
        except Exception as exc:
            return (
                el.element_id,
                None,
                IngestionError(element_id=el.element_id, stage="enrich-table", message=str(exc)),
                False,
            )

    concurrency = max(1, int(getattr(settings, "bedrock_visual_concurrency", 6)))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_analyse_figure, el) for el in figure_jobs]
        futures += [pool.submit(_validate_table, el) for el in table_jobs]
        for fut in as_completed(futures):
            element_id, result, err, cache_hit = fut.result()
            if cache_hit:
                stats["cache_hits"] += 1
            else:
                stats["bedrock_calls"] += 1
            if err:
                errors.append(err)
                continue
            if isinstance(result, VisualAnalysis):
                analyses[element_id] = result
                stats["warnings"] += len(result.warnings)
            elif isinstance(result, TableValidation):
                validations[element_id] = result
                stats["warnings"] += len(result.warnings)

    # Persist enrichment artifacts
    return analyses, validations, errors, stats
