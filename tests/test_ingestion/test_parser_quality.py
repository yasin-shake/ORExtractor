from ingestion.models import ElementRecord
from ingestion.quality import assess_parser_quality


def _text(page: int, value: str) -> ElementRecord:
    return ElementRecord(
        element_id=f"p{page}",
        source_file="report.pdf",
        category="NarrativeText",
        text=value,
        page_number=page,
        parser="docling",
    )


def test_quality_passes_complete_text_pages():
    report = assess_parser_quality(
        [_text(1, "A sufficiently long body paragraph on page one."), _text(2, "A sufficiently long body paragraph on page two.")],
        page_count=2,
    )
    assert report.text_coverage == 1.0
    assert report.reasons == []
    assert report.score == 1.0


def test_quality_emits_deterministic_reason_codes():
    report = assess_parser_quality(
        [_text(1, "short \ufffd")],
        page_count=3,
        max_replacement_char_ratio=0.01,
    )
    assert "low_text_page_coverage" in report.reasons
    assert "high_empty_page_ratio" in report.reasons
    assert "high_replacement_character_ratio" in report.reasons

