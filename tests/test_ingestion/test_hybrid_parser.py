from pathlib import Path

import pytest

from ingestion.models import (
    ElementRecord,
    ParserQualityReport,
    ParserResult,
)
from ingestion.parsers.hybrid_parser import (
    HybridDocumentParser,
    NativePdfPageExtractor,
    PageRouteWindow,
    PageWindowError,
)


class _Settings:
    parser_min_quality_score = 0.0
    parser_min_text_page_coverage = 0.0
    parser_min_page_count_agreement = 0.0
    parser_max_suspicious_page_ratio = 1.0
    parser_max_near_empty_page_ratio = 1.0
    parser_max_replacement_character_ratio = 1.0
    parser_max_duplicate_header_footer_ratio = 1.0
    parser_min_table_valid_ratio = 0.0
    parser_min_table_row_consistency = 0.0
    parser_min_table_column_consistency = 0.0
    parser_min_caption_association_rate = 0.0
    hybrid_native_min_chars = 50
    hybrid_native_max_drawings = 20
    hybrid_native_max_image_area_ratio = 0.03
    hybrid_max_window_pages = 50
    hybrid_native_min_window_pages = 1


class _Plan:
    def plan(self, _pdf_path):
        return [
            PageRouteWindow("native", 1, 2, ["born_digital_simple"]),
            PageRouteWindow("docling", 3, 3, ["table_candidate"]),
        ]


class _Extractor:
    def __init__(self, parser_name):
        self.parser_name = parser_name
        self.parser_version = "test"
        self.calls = []

    def extract(
        self,
        _pdf_path,
        *,
        source_file,
        artifact_dir,
        page_range,
    ):
        self.calls.append(page_range)
        elements = [
            ElementRecord(
                element_id=f"{self.parser_name}-{page}",
                source_file=source_file,
                category="NarrativeText",
                text=f"Page {page} extracted by {self.parser_name}.",
                page_number=page,
                parser=self.parser_name,
                parser_version=self.parser_version,
            )
            for page in range(page_range[0], page_range[1] + 1)
        ]
        return ParserResult(
            source_file=source_file,
            parser=self.parser_name,
            parser_version=self.parser_version,
            elements=elements,
            page_count=len(elements),
            quality=ParserQualityReport(score=1.0),
        )


def test_hybrid_parser_routes_windows_and_preserves_page_provenance(tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF route plan")
    native = _Extractor("pymupdf-native")
    docling = _Extractor("docling")
    parser = HybridDocumentParser(
        _Settings(),
        routing_policy=_Plan(),
        native_extractor=native,
        complex_extractor=docling,
    )

    result = parser.parse(
        pdf,
        source_file="nested/report.pdf",
        artifact_dir=tmp_path / "artifacts",
    )

    assert native.calls == [(1, 2)]
    assert docling.calls == [(3, 3)]
    assert [element.page_number for element in result.elements] == [1, 2, 3]
    assert [element.parser for element in result.elements] == [
        "pymupdf-native",
        "pymupdf-native",
        "docling",
    ]
    assert result.parser == "hybrid"
    assert result.page_count == 3
    assert result.metadata["page_routes"] == [
        {
            "route": "native",
            "first_page": 1,
            "last_page": 2,
            "reasons": ["born_digital_simple"],
        },
        {
            "route": "docling",
            "first_page": 3,
            "last_page": 3,
            "reasons": ["table_candidate"],
        },
    ]


def test_hybrid_parser_rejects_elements_outside_their_window(tmp_path):
    class BadExtractor(_Extractor):
        def extract(self, *args, **kwargs):
            result = super().extract(*args, **kwargs)
            result.elements[0].page_number = 99
            return result

    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF invalid provenance")
    parser = HybridDocumentParser(
        _Settings(),
        routing_policy=_Plan(),
        native_extractor=BadExtractor("pymupdf-native"),
        complex_extractor=_Extractor("docling"),
    )

    with pytest.raises(PageWindowError, match="outside requested window"):
        parser.parse(pdf, artifact_dir=tmp_path / "artifacts")


def test_default_policy_keeps_simple_page_native_and_routes_table_page(
    tmp_path,
):
    import fitz

    pdf = tmp_path / "report.pdf"
    document = fitz.open()
    simple = document.new_page(width=600, height=800)
    simple.insert_textbox(
        fitz.Rect(60, 80, 540, 700),
        (
            "This born-digital technical report paragraph contains a "
            "straightforward single-column narrative text layer. " * 8
        ),
    )
    table = document.new_page(width=600, height=800)
    table.insert_textbox(
        fitz.Rect(60, 80, 540, 700),
        (
            "Resource classification\n"
            "Grade  Tonnes  Contained Metal\n"
            "1.20  1000000  12000\n"
        ),
    )
    for offset in range(25):
        y = 200 + offset * 5
        table.draw_line((60, y), (540, y))
    document.save(pdf)
    document.close()

    complex_extractor = _Extractor("docling")
    parser = HybridDocumentParser(
        _Settings(),
        complex_extractor=complex_extractor,
    )
    result = parser.parse(
        pdf,
        artifact_dir=tmp_path / "artifacts",
    )

    assert complex_extractor.calls == [(2, 2)]
    assert result.metadata["native_page_count"] == 1
    assert result.metadata["docling_page_count"] == 1
    assert {element.parser for element in result.elements} == {
        "pymupdf-native",
        "docling",
    }


def test_native_extractor_recognizes_section_title_with_em_dash(tmp_path):
    import fitz

    pdf = tmp_path / "report.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((60, 100), "1.1 \u2014 Project Overview", fontsize=10)
    page.insert_text(
        (60, 140),
        "This paragraph contains the ordinary report narrative.",
        fontsize=10,
    )
    document.save(pdf)
    document.close()

    result = NativePdfPageExtractor().extract(
        pdf,
        source_file="report.pdf",
        artifact_dir=tmp_path / "artifacts",
        page_range=(1, 1),
    )

    title = next(
        element
        for element in result.elements
        if "Project Overview" in element.text
    )
    assert title.category == "Title"
