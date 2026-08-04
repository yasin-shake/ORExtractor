"""Opt-in page routing between native PDF extraction and Docling."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Protocol

import fitz

from ingestion.cache import runtime_package_version
from ingestion.config import ParserQualityPolicy
from ingestion.models import (
    ElementRecord,
    ParserQualityReport,
    ParserResult,
)
from ingestion.quality import assess_parser_quality


class PageWindowError(RuntimeError):
    """A page extractor violated its requested page boundary."""


@dataclass(frozen=True)
class PageRouteWindow:
    route: str
    first_page: int
    last_page: int
    reasons: list[str]

    @property
    def page_range(self) -> tuple[int, int]:
        return self.first_page, self.last_page

    def as_dict(self) -> dict:
        return {
            "route": self.route,
            "first_page": self.first_page,
            "last_page": self.last_page,
            "reasons": list(self.reasons),
        }


class PageExtractor(Protocol):
    parser_name: str
    parser_version: str

    def extract(
        self,
        pdf_path: Path,
        *,
        source_file: str,
        artifact_dir: Path,
        page_range: tuple[int, int],
    ) -> ParserResult:
        ...


def _table_candidate(text: str) -> bool:
    candidate_lines = sum(
        len(re.findall(r"[-+]?\d[\d,.]*", line)) >= 2
        and (
            "\t" in line
            or bool(re.search(r"\S+\s{2,}\S+", line))
        )
        for line in text.splitlines()
    )
    return candidate_lines >= 2


def _multi_column(blocks: list[tuple], page_width: float) -> bool:
    candidates = [
        block
        for block in blocks
        if len(block) >= 5
        and int(block[6] if len(block) > 6 else 0) == 0
        and len(re.sub(r"\s+", "", str(block[4] or ""))) >= 80
        and (float(block[2]) - float(block[0])) <= page_width * 0.68
    ]
    for index, first in enumerate(candidates):
        first_center = (float(first[0]) + float(first[2])) / 2
        for second in candidates[index + 1 :]:
            second_center = (
                float(second[0]) + float(second[2])
            ) / 2
            if abs(first_center - second_center) < page_width * 0.28:
                continue
            overlap = min(float(first[3]), float(second[3])) - max(
                float(first[1]),
                float(second[1]),
            )
            minimum_height = min(
                float(first[3]) - float(first[1]),
                float(second[3]) - float(second[1]),
            )
            if minimum_height > 0 and overlap / minimum_height >= 0.35:
                return True
    return False


class BornDigitalPageRoutingPolicy:
    """Conservative, cheap page classifier; uncertainty routes to Docling."""

    def __init__(self, settings):
        self.minimum_characters = max(
            1,
            int(getattr(settings, "hybrid_native_min_chars", 300)),
        )
        self.max_drawings = max(
            0,
            int(getattr(settings, "hybrid_native_max_drawings", 20)),
        )
        self.max_image_area_ratio = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        settings,
                        "hybrid_native_max_image_area_ratio",
                        0.03,
                    )
                ),
            ),
        )
        self.max_window_pages = max(
            1,
            int(getattr(settings, "hybrid_max_window_pages", 50)),
        )
        self.minimum_native_window_pages = max(
            1,
            int(
                getattr(
                    settings,
                    "hybrid_native_min_window_pages",
                    3,
                )
            ),
        )

    def _page_route(self, page) -> tuple[str, list[str]]:
        text = page.get_text("text", sort=True) or ""
        character_count = len(re.sub(r"\s+", "", text))
        reasons: list[str] = []
        if character_count < self.minimum_characters:
            reasons.append("insufficient_native_text")
        if _table_candidate(text):
            reasons.append("table_candidate")
        blocks = page.get_text("blocks", sort=True) or []
        if _multi_column(blocks, float(page.rect.width)):
            reasons.append("multi_column")
        if int(page.rotation or 0) % 360:
            reasons.append("rotated_page")
        if len(page.get_drawings()) > self.max_drawings:
            reasons.append("vector_drawing_complexity")
        page_area = max(1.0, float(page.rect.width * page.rect.height))
        for image in page.get_images(full=True):
            xref = int(image[0])
            for rectangle in page.get_image_rects(xref):
                margin_image = (
                    rectangle.y1 <= page.rect.height * 0.12
                    or rectangle.y0 >= page.rect.height * 0.88
                )
                if margin_image:
                    continue
                if float(rectangle.width * rectangle.height) / page_area > (
                    self.max_image_area_ratio
                ):
                    reasons.append("significant_raster_image")
                    break
            if "significant_raster_image" in reasons:
                break
        if reasons:
            return "docling", list(dict.fromkeys(reasons))
        return "native", ["born_digital_simple"]

    def plan(self, pdf_path: Path) -> list[PageRouteWindow]:
        page_routes: list[tuple[str, int, list[str]]] = []
        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document):
                route, reasons = self._page_route(page)
                page_routes.append((route, page_index + 1, reasons))
        windows: list[PageRouteWindow] = []
        for route, page_number, reasons in page_routes:
            if (
                windows
                and windows[-1].route == route
                and page_number == windows[-1].last_page + 1
                and (
                    windows[-1].last_page
                    - windows[-1].first_page
                    + 1
                )
                < self.max_window_pages
            ):
                previous = windows[-1]
                windows[-1] = PageRouteWindow(
                    route=route,
                    first_page=previous.first_page,
                    last_page=page_number,
                    reasons=list(
                        dict.fromkeys([*previous.reasons, *reasons])
                    ),
                )
            else:
                windows.append(
                    PageRouteWindow(
                        route=route,
                        first_page=page_number,
                        last_page=page_number,
                        reasons=reasons,
                    )
                )
        reclassified = [
            (
                PageRouteWindow(
                    route="docling",
                    first_page=window.first_page,
                    last_page=window.last_page,
                    reasons=["native_run_too_short"],
                )
                if (
                    window.route == "native"
                    and (
                        window.last_page
                        - window.first_page
                        + 1
                    )
                    < self.minimum_native_window_pages
                )
                else window
            )
            for window in windows
        ]
        merged: list[PageRouteWindow] = []
        for window in reclassified:
            if (
                merged
                and merged[-1].route == window.route
                and window.first_page == merged[-1].last_page + 1
                and (
                    window.last_page
                    - merged[-1].first_page
                    + 1
                )
                <= self.max_window_pages
            ):
                previous = merged[-1]
                merged[-1] = PageRouteWindow(
                    route=window.route,
                    first_page=previous.first_page,
                    last_page=window.last_page,
                    reasons=list(
                        dict.fromkeys(
                            [
                                *previous.reasons,
                                *window.reasons,
                            ]
                        )
                    ),
                )
            else:
                merged.append(window)
        return merged


def _element_id(
    source_file: str,
    page_number: int,
    block_number: int,
    text: str,
) -> str:
    seed = (
        f"{source_file}|native|{page_number}|{block_number}|"
        f"{text[:256]}"
    )
    return hashlib.sha256(
        seed.encode("utf-8", errors="replace")
    ).hexdigest()[:24]


class NativePdfPageExtractor:
    """Extract simple born-digital pages using the existing PDF text layer."""

    parser_name = "pymupdf-native"
    parser_version = runtime_package_version("PyMuPDF")

    @staticmethod
    def _page_elements(
        page,
        *,
        source_file: str,
        page_number: int,
    ) -> list[ElementRecord]:
        page_dict = page.get_text("dict", sort=True) or {}
        blocks = [
            block
            for block in page_dict.get("blocks", [])
            if int(block.get("type", 0)) == 0
        ]
        font_sizes = [
            float(span.get("size", 0.0) or 0.0)
            for block in blocks
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if str(span.get("text", "")).strip()
        ]
        body_size = median(font_sizes) if font_sizes else 0.0
        elements: list[ElementRecord] = []
        for block_number, block in enumerate(blocks):
            lines = []
            block_sizes = []
            for line in block.get("lines", []):
                line_text = "".join(
                    str(span.get("text", ""))
                    for span in line.get("spans", [])
                ).strip()
                if line_text:
                    lines.append(line_text)
                block_sizes.extend(
                    float(span.get("size", 0.0) or 0.0)
                    for span in line.get("spans", [])
                    if str(span.get("text", "")).strip()
                )
            text = "\n".join(lines).strip()
            if not text:
                continue
            x0, y0, x1, y1 = (
                float(value)
                for value in block.get("bbox", (0, 0, 0, 0))
            )
            page_height = float(page.rect.height)
            if y1 <= page_height * 0.07:
                category = "Header"
            elif y0 >= page_height * 0.93:
                category = "Footer"
            else:
                largest_size = max(block_sizes, default=body_size)
                heading_by_font = (
                    body_size > 0
                    and largest_size >= body_size * 1.25
                    and len(text) <= 180
                )
                heading_by_label = bool(
                    re.match(
                        r"^\s*(?:item\s+)?\d{1,2}(?:\.\d+)*"
                        r"(?:\s*[-\u2013\u2014:.]|\s+[A-Z])",
                        text,
                        flags=re.IGNORECASE,
                    )
                )
                category = (
                    "Title"
                    if heading_by_font or heading_by_label
                    else "NarrativeText"
                )
            element_id = _element_id(
                source_file,
                page_number,
                block_number,
                text,
            )
            elements.append(
                ElementRecord(
                    element_id=element_id,
                    source_file=source_file,
                    category=category,
                    text=text,
                    page_number=page_number,
                    coordinates={
                        "l": x0,
                        "t": y0,
                        "r": x1,
                        "b": y1,
                        "coord_origin": "TOPLEFT",
                    },
                    parser=NativePdfPageExtractor.parser_name,
                    parser_version=(
                        NativePdfPageExtractor.parser_version
                    ),
                    parser_element_id=f"page-{page_number}-block-{block_number}",
                    metadata={
                        "page_width": float(page.rect.width),
                        "page_height": page_height,
                        "native_block_number": block_number,
                    },
                )
            )
        return elements

    def extract(
        self,
        pdf_path: Path,
        *,
        source_file: str,
        artifact_dir: Path,
        page_range: tuple[int, int],
    ) -> ParserResult:
        started = time.perf_counter()
        first_page, last_page = page_range
        elements: list[ElementRecord] = []
        with fitz.open(pdf_path) as document:
            for page_number in range(first_page, last_page + 1):
                elements.extend(
                    self._page_elements(
                        document.load_page(page_number - 1),
                        source_file=source_file,
                        page_number=page_number,
                    )
                )
        duration_ms = (time.perf_counter() - started) * 1000
        result = ParserResult(
            source_file=source_file,
            parser=self.parser_name,
            parser_version=self.parser_version,
            elements=elements,
            page_count=last_page - first_page + 1,
            duration_ms=duration_ms,
            quality=ParserQualityReport(
                score=1.0 if elements else 0.0,
                element_count=len(elements),
                duration_ms=duration_ms,
                reasons=[] if elements else ["no_elements"],
            ),
            metadata={"page_range": list(page_range)},
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "native_result.json").write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return result


class DoclingPageExtractor:
    parser_name = "docling"

    def __init__(self, parser):
        self.parser = parser
        self.parser_version = parser.parser_version

    def extract(
        self,
        pdf_path: Path,
        *,
        source_file: str,
        artifact_dir: Path,
        page_range: tuple[int, int],
    ) -> ParserResult:
        return self.parser.parse_pages(
            pdf_path,
            source_file=source_file,
            artifact_dir=artifact_dir,
            page_range=page_range,
        )

    def close(self) -> None:
        close = getattr(self.parser, "close", None)
        if callable(close):
            close()

    def cache_signature(self) -> dict:
        signature = getattr(self.parser, "cache_signature", None)
        return (
            signature()
            if callable(signature)
            else {
                "parser": self.parser_name,
                "version": self.parser_version,
            }
        )


class HybridDocumentParser:
    """Deep parser module that hides page routing and window recombination."""

    parser_name = "hybrid"

    def __init__(
        self,
        settings,
        *,
        routing_policy=None,
        native_extractor: PageExtractor | None = None,
        complex_extractor: PageExtractor | None = None,
    ):
        self.settings = settings
        self.quality_policy = ParserQualityPolicy.from_settings(settings)
        self.routing_policy = (
            routing_policy or BornDigitalPageRoutingPolicy(settings)
        )
        self.native_extractor = (
            native_extractor or NativePdfPageExtractor()
        )
        if complex_extractor is None:
            from ingestion.parsers.docling_parser import DoclingParser

            complex_extractor = DoclingPageExtractor(
                DoclingParser(settings)
            )
        self.complex_extractor = complex_extractor
        signature = json.dumps(
            self.cache_signature(),
            sort_keys=True,
        )
        self.parser_version = hashlib.sha256(
            signature.encode("utf-8")
        ).hexdigest()[:16]

    def cache_signature(self) -> dict:
        complex_signature = getattr(
            self.complex_extractor,
            "cache_signature",
            None,
        )
        return {
            "parser": self.parser_name,
            "native_version": self.native_extractor.parser_version,
            "complex_parser": self.complex_extractor.parser_name,
            "complex_version": self.complex_extractor.parser_version,
            "complex_signature": (
                complex_signature()
                if callable(complex_signature)
                else None
            ),
            "routing": {
                "native_min_chars": int(
                    getattr(
                        self.settings,
                        "hybrid_native_min_chars",
                        300,
                    )
                ),
                "native_max_drawings": int(
                    getattr(
                        self.settings,
                        "hybrid_native_max_drawings",
                        20,
                    )
                ),
                "native_max_image_area_ratio": float(
                    getattr(
                        self.settings,
                        "hybrid_native_max_image_area_ratio",
                        0.03,
                    )
                ),
                "max_window_pages": int(
                    getattr(
                        self.settings,
                        "hybrid_max_window_pages",
                        50,
                    )
                ),
                "native_min_window_pages": int(
                    getattr(
                        self.settings,
                        "hybrid_native_min_window_pages",
                        3,
                    )
                ),
            },
        }

    @staticmethod
    def _validate_window(
        result: ParserResult,
        window: PageRouteWindow,
    ) -> None:
        for element in result.elements:
            if not (
                window.first_page
                <= int(element.page_number)
                <= window.last_page
            ):
                raise PageWindowError(
                    f"{element.element_id} page {element.page_number} is "
                    "outside requested window "
                    f"{window.first_page}-{window.last_page}"
                )

    def parse(
        self,
        pdf_path: Path,
        *,
        source_file: str | None = None,
        artifact_dir: Path | None = None,
    ) -> ParserResult:
        started = time.perf_counter()
        source_file = source_file or pdf_path.name
        artifact_dir = artifact_dir or (
            Path(
                getattr(
                    self.settings,
                    "artifact_dir",
                    Path("ingestion_artifacts"),
                )
            )
            / pdf_path.stem
        )
        windows = self.routing_policy.plan(pdf_path)
        elements: list[ElementRecord] = []
        seen_ids: set[str] = set()
        warnings: list[str] = []
        errors: list[str] = []
        artifacts: dict[str, str] = {}
        statuses: list[str] = []
        duration_ms = 0.0
        for window in windows:
            extractor = (
                self.native_extractor
                if window.route == "native"
                else self.complex_extractor
            )
            window_dir = (
                Path(artifact_dir)
                / "hybrid"
                / f"{window.first_page:04d}-{window.last_page:04d}"
                / extractor.parser_name
            )
            result = extractor.extract(
                pdf_path,
                source_file=source_file,
                artifact_dir=window_dir,
                page_range=window.page_range,
            )
            self._validate_window(result, window)
            for element in result.elements:
                if element.element_id in seen_ids:
                    raise PageWindowError(
                        f"duplicate element ID across page windows: "
                        f"{element.element_id}"
                    )
                seen_ids.add(element.element_id)
                elements.append(element)
            duration_ms += result.duration_ms
            statuses.append(result.status)
            warnings.extend(result.warnings)
            errors.extend(result.errors)
            label = (
                f"{window.first_page:04d}-{window.last_page:04d}"
            )
            artifacts.update(
                {
                    f"{window.route}_{label}_{name}": value
                    for name, value in result.artifact_paths.items()
                }
            )
        elements.sort(
            key=lambda element: (
                element.page_number,
                int(
                    element.metadata.get(
                        "docling_ordinal",
                        element.metadata.get(
                            "native_block_number",
                            0,
                        ),
                    )
                ),
            )
        )
        page_count = max(
            (window.last_page for window in windows),
            default=0,
        )
        status = (
            "success"
            if windows
            and all(
                value in {"success", "completed"}
                for value in statuses
            )
            else "partial_success"
        )
        quality = assess_parser_quality(
            elements,
            page_count=page_count,
            conversion_status=status,
            **self.quality_policy.assessment_kwargs(),
        )
        total_duration_ms = (
            time.perf_counter() - started
        ) * 1000
        quality.duration_ms = total_duration_ms
        route_payload = [window.as_dict() for window in windows]
        result = ParserResult(
            source_file=source_file,
            parser=self.parser_name,
            parser_version=self.parser_version,
            status=status,
            elements=elements,
            artifact_paths=artifacts,
            page_count=page_count,
            duration_ms=total_duration_ms,
            warnings=list(dict.fromkeys(warnings)),
            errors=list(dict.fromkeys(errors)),
            quality=quality,
            metadata={
                "page_routes": route_payload,
                "native_page_count": sum(
                    window.last_page - window.first_page + 1
                    for window in windows
                    if window.route == "native"
                ),
                "docling_page_count": sum(
                    window.last_page - window.first_page + 1
                    for window in windows
                    if window.route == "docling"
                ),
                "window_duration_ms": duration_ms,
            },
        )
        report_dir = Path(artifact_dir) / "parsers" / "hybrid"
        report_dir.mkdir(parents=True, exist_ok=True)
        routes_path = report_dir / "page_routes.json"
        routes_path.write_text(
            json.dumps(route_payload, indent=2),
            encoding="utf-8",
        )
        result_path = report_dir / "result.json"
        result_path.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        result.artifact_paths.update(
            {
                "page_routes": str(routes_path),
                "hybrid_result": str(result_path),
            }
        )
        return result

    def close(self) -> None:
        seen: set[int] = set()
        for extractor in (
            self.native_extractor,
            self.complex_extractor,
        ):
            if extractor is None:
                continue
            if id(extractor) in seen:
                continue
            seen.add(id(extractor))
            close = getattr(extractor, "close", None)
            if callable(close):
                close()
