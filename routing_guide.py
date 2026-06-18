"""NI 43-101 chapter routing guide — loaded from data/routing_guide.json (BMRC memo)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

_GUIDE_PATH = Path(__file__).parent / "data" / "routing_guide.json"


class NIItem(BaseModel):
    number: int
    title: str
    gather_fields: List[str] = Field(default_factory=list)
    dd_question: str = ""


class QueryLookupEntry(BaseModel):
    question_pattern: str
    primary_items: List[int] = Field(default_factory=list)
    cross_check_items: List[int] = Field(default_factory=list)


class RoutingMatrix(BaseModel):
    search: str = ""
    cross_check: List[int] = Field(default_factory=list)
    extract: List[str] = Field(default_factory=list)
    compare: List[str] = Field(default_factory=list)
    flag: List[str] = Field(default_factory=list)


class BenchmarkTemplate(BaseModel):
    name: str
    keywords: List[str] = Field(default_factory=list)
    search_criteria: List[str] = Field(default_factory=list)
    extract_fields: List[str] = Field(default_factory=list)
    output_format: str = ""


class RoutingGuide(BaseModel):
    version: str = "1.0"
    source: str = ""
    items: List[NIItem] = Field(default_factory=list)
    query_lookup: List[QueryLookupEntry] = Field(default_factory=list)
    routing_matrices: dict[str, RoutingMatrix] = Field(default_factory=dict)
    benchmark_templates: List[BenchmarkTemplate] = Field(default_factory=list)
    core_dd_questions: List[str] = Field(default_factory=list)
    metadata_schema: dict[str, str] = Field(default_factory=dict)


class RoutingResult(BaseModel):
    primary_items: List[int] = Field(default_factory=list)
    cross_check_items: List[int] = Field(default_factory=list)
    matched_patterns: List[str] = Field(default_factory=list)
    benchmark_template: Optional[str] = None
    needs_peer_benchmark: bool = False


@lru_cache(maxsize=1)
def load_routing_guide() -> RoutingGuide:
    raw = json.loads(_GUIDE_PATH.read_text(encoding="utf-8"))
    matrices = {k: RoutingMatrix(**v) for k, v in raw.get("routing_matrices", {}).items()}
    raw["routing_matrices"] = matrices
    return RoutingGuide(**raw)


def get_item(number: int) -> Optional[NIItem]:
    guide = load_routing_guide()
    for item in guide.items:
        if item.number == number:
            return item
    return None


def get_item_title(number: int) -> str:
    item = get_item(number)
    return item.title if item else f"Item {number}"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


_BENCHMARK_KEYWORDS = (
    "benchmark", "comparable", "peer", "reasonable compared", "typical range",
    "normal range", "vs peer", "against peer", "similar project",
)


def resolve_items_for_question(question: str) -> RoutingResult:
    """Match question text to NI Items via keyword lookup table."""
    guide = load_routing_guide()
    q = _normalize(question)
    primary: set[int] = set()
    cross: set[int] = set()
    matched: List[str] = []

    for entry in guide.query_lookup:
        if entry.question_pattern in q:
            primary.update(entry.primary_items)
            cross.update(entry.cross_check_items)
            matched.append(entry.question_pattern)

    if not primary:
        # Heuristic fallbacks for common NI 43-101 topics
        if any(w in q for w in ("resource", "tonnage", "measured", "indicated", "inferred")):
            primary.update([14, 10])
        if any(w in q for w in ("reserve", "proven", "probable")):
            primary.update([15, 16])
        if any(w in q for w in ("geolog", "deposit", "mineralization", "mineralisation")):
            primary.update([7, 8])
        if any(w in q for w in ("economic", "npv", "irr", "payback")):
            primary.update([22, 21, 19])
        if any(w in q for w in ("environment", "permit", "esg", "closure")):
            primary.update([20])
        if any(w in q for w in ("conclusion", "recommendation", "go", "no-go", "no go")):
            primary.update([25, 26, 1])
        if not primary:
            primary.update([1, 25])

    cross -= primary
    needs_benchmark = any(kw in q for kw in _BENCHMARK_KEYWORDS)
    template = match_benchmark_template(question)

    return RoutingResult(
        primary_items=sorted(primary),
        cross_check_items=sorted(cross),
        matched_patterns=matched,
        benchmark_template=template,
        needs_peer_benchmark=needs_benchmark or template is not None,
    )


def match_benchmark_template(question: str) -> Optional[str]:
    guide = load_routing_guide()
    q = _normalize(question)
    for tmpl in guide.benchmark_templates:
        for kw in tmpl.keywords:
            if kw in q:
                return tmpl.name
    return None


def get_playbook(item_numbers: List[int]) -> str:
    """Return Extract/Compare/Flag checklist for routed Items."""
    guide = load_routing_guide()
    lines: List[str] = []

    def _matrix_for_item(num: int) -> Optional[tuple[str, RoutingMatrix]]:
        for key, matrix in guide.routing_matrices.items():
            parts = key.replace(" ", "").split("-")
            nums = [int(p) for p in parts if p.isdigit()]
            if num in nums:
                return key, matrix
        return None

    seen_keys: set[str] = set()
    for num in item_numbers:
        item = get_item(num)
        if item:
            lines.append(f"## Item {num}: {item.title}")
            lines.append(f"DD question: {item.dd_question}")
            if item.gather_fields:
                lines.append("Gather: " + "; ".join(item.gather_fields))
        found = _matrix_for_item(num)
        if found:
            key, matrix = found
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if matrix.search:
                lines.append(f"Search: {matrix.search}")
            if matrix.cross_check:
                lines.append(f"Cross-check Items: {', '.join(str(i) for i in matrix.cross_check)}")
            if matrix.extract:
                lines.append("Extract: " + "; ".join(matrix.extract))
            if matrix.compare:
                lines.append("Compare: " + "; ".join(matrix.compare))
            if matrix.flag:
                lines.append("Flag: " + "; ".join(matrix.flag))
        lines.append("")

    return "\n".join(lines).strip()


def get_benchmark_template(name: str) -> Optional[BenchmarkTemplate]:
    guide = load_routing_guide()
    for tmpl in guide.benchmark_templates:
        if tmpl.name == name:
            return tmpl
    return None


def all_core_dd_questions() -> List[str]:
    return list(load_routing_guide().core_dd_questions)


def item_titles() -> dict[int, str]:
    """Standard NI 43-101F1 item titles for heading detection."""
    return {i.number: i.title for i in load_routing_guide().items}
