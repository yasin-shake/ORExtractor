from pathlib import Path

from ingestion.models import ElementRecord, ParserQualityReport, ParserResult
from ingestion.parsers.router import ParserRouter


class _Settings:
    ingestion_backend = "docling"
    parser_primary = "docling"
    parser_fallback = "mineru"
    parser_fallback_enabled = True
    force_parser = ""


class _Parser:
    def __init__(self, name: str, score: float, reasons: list[str]):
        self.parser_name = name
        self.parser_version = "test"
        self.score = score
        self.reasons = reasons
        self.calls = 0

    def parse(
        self,
        pdf_path: Path,
        *,
        source_file: str | None = None,
        artifact_dir: Path | None = None,
    ) -> ParserResult:
        self.calls += 1
        source_file = source_file or pdf_path.name
        return ParserResult(
            source_file=source_file,
            parser=self.parser_name,
            parser_version=self.parser_version,
            elements=[
                ElementRecord(
                    element_id=self.parser_name,
                    source_file=source_file,
                    category="NarrativeText",
                    text="A complete parser result body paragraph.",
                    parser=self.parser_name,
                )
            ],
            page_count=1,
            quality=ParserQualityReport(
                score=self.score,
                element_count=1,
                text_coverage=1.0,
                reasons=self.reasons,
            ),
        )


def test_router_skips_fallback_when_primary_passes(tmp_path):
    settings = _Settings()
    settings.artifact_dir = tmp_path
    primary = _Parser("docling", 0.98, [])
    fallback = _Parser("mineru", 0.99, [])
    result = ParserRouter(settings, primary=primary, fallback=fallback).parse(
        tmp_path / "report.pdf"
    )
    assert result.parser == "docling"
    assert fallback.calls == 0
    assert result.fallback.attempted is False


def test_router_selects_higher_quality_fallback(tmp_path):
    settings = _Settings()
    settings.artifact_dir = tmp_path
    primary = _Parser("docling", 0.40, ["low_text_page_coverage"])
    fallback = _Parser("mineru", 0.90, [])
    result = ParserRouter(settings, primary=primary, fallback=fallback).parse(
        tmp_path / "report.pdf"
    )
    assert result.parser == "mineru"
    assert result.fallback.attempted is True
    assert result.fallback.used is True
    assert result.fallback.reasons == ["low_text_page_coverage"]
    assert (tmp_path / "report" / "parser_selection.json").exists()


def test_router_retains_degraded_primary_when_fallback_fails(tmp_path):
    class _Failing(_Parser):
        def parse(
            self,
            pdf_path: Path,
            *,
            source_file: str | None = None,
            artifact_dir: Path | None = None,
        ) -> ParserResult:
            raise RuntimeError("worker unavailable")

    settings = _Settings()
    settings.artifact_dir = tmp_path
    primary = _Parser("docling", 0.50, ["table_structure_failures"])
    result = ParserRouter(
        settings,
        primary=primary,
        fallback=_Failing("mineru", 0.0, []),
    ).parse(tmp_path / "report.pdf")
    assert result.parser == "docling"
    assert result.status == "degraded"
    assert result.fallback.attempted is True
    assert result.fallback.used is False
    assert "worker unavailable" in result.errors


def test_router_preserves_nested_source_and_artifact_path(tmp_path):
    settings = _Settings()
    settings.artifact_dir = tmp_path / "artifacts"
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF")
    artifact_dir = settings.artifact_dir / "region" / "report"
    primary = _Parser("docling", 0.98, [])

    result = ParserRouter(settings, primary=primary).parse(
        pdf,
        source_file="region/report.pdf",
        artifact_dir=artifact_dir,
    )

    assert result.source_file == "region/report.pdf"
    assert result.elements[0].source_file == "region/report.pdf"
    assert (artifact_dir / "parser_selection.json").exists()


def test_quality_policy_change_reuses_adapter_parse_cache(tmp_path):
    class _Cacheable(_Parser):
        def cache_signature(self):
            return {"parser": self.parser_name, "version": self.parser_version}

    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF cache")
    settings = _Settings()
    settings.artifact_dir = tmp_path
    settings.parser_fallback_enabled = False
    parser = _Cacheable("docling", 0.98, [])

    ParserRouter(settings, primary=parser).parse(pdf)
    settings.parser_min_text_page_coverage = 0.95
    second = ParserRouter(settings, primary=parser).parse(pdf)

    assert parser.calls == 1
    assert second.metadata["parser_cache_hit"] is True
