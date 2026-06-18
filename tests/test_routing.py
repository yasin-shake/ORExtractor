"""Unit tests for chapter routing and NI Item index parsing."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chapter_index import (
    build_chapter_index_from_documents,
    detect_item_from_text,
    item_for_page,
    tag_documents_with_items,
    ChapterEntry,
)
from langchain_core.documents import Document
from routing_guide import (
    get_playbook,
    load_routing_guide,
    match_benchmark_template,
    resolve_items_for_question,
)


class TestRoutingGuide:
    def test_guide_loads(self):
        guide = load_routing_guide()
        assert len(guide.items) == 27
        assert len(guide.query_lookup) >= 16
        assert len(guide.benchmark_templates) == 13

    def test_resolve_cutoff_question(self):
        r = resolve_items_for_question("What is the typical cut-off grade for copper?")
        assert 14 in r.primary_items
        assert 15 in r.primary_items or 22 in r.primary_items

    def test_resolve_qaqc_question(self):
        r = resolve_items_for_question("Are the QAQC results acceptable and complete?")
        assert 11 in r.primary_items
        assert 12 in r.primary_items

    def test_resolve_drill_spacing(self):
        r = resolve_items_for_question("Is drill spacing adequate for Measured classification?")
        assert 10 in r.primary_items
        assert 14 in r.primary_items

    def test_benchmark_template_nsr(self):
        name = match_benchmark_template("Compare NSR cut-off values across peers")
        assert name == "NSR Cut-off Benchmark"

    def test_playbook_contains_flags(self):
        pb = get_playbook([14, 10])
        assert "Item 14" in pb
        assert "Flag" in pb or "Extract" in pb


class TestChapterIndex:
    def test_detect_item_heading(self):
        text = "Item 14 — Mineral Resource Estimates\n\nThe following table..."
        detected = detect_item_from_text(text)
        assert detected is not None
        assert detected[0] == 14

    def test_build_index_from_docs(self):
        docs = [
            Document(page_content="Item 10 — Drilling\n\nDrill summary.", metadata={"page": 50}),
            Document(page_content="Item 14 — Mineral Resource Estimates\n\nResources.", metadata={"page": 80}),
        ]
        chapters = build_chapter_index_from_documents(docs)
        assert len(chapters) >= 2
        assert chapters[0].item == 10
        assert chapters[1].item == 14

    def test_item_for_page(self):
        chapters = [
            ChapterEntry(item=10, title="Drilling", page_start=50, page_end=79),
            ChapterEntry(item=14, title="Resources", page_start=80, page_end=None),
        ]
        ch = item_for_page(chapters, 65)
        assert ch is not None
        assert ch.item == 10
        ch2 = item_for_page(chapters, 90)
        assert ch2 is not None
        assert ch2.item == 14

    def test_tag_documents(self):
        docs = [
            Document(page_content="Resource table data", metadata={"source": "a.pdf", "page": 85, "chunk": 0}),
        ]
        chapters = [ChapterEntry(item=14, title="Mineral Resource Estimates", page_start=80, page_end=100)]
        tagged = tag_documents_with_items(docs, chapters)
        assert tagged[0].metadata["ni_item"] == 14
        assert "Resource" in tagged[0].metadata["section_title"]
