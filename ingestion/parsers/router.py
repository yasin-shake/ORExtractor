"""Primary/fallback parser routing with deterministic selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ingestion.models import (
    NORMALIZER_VERSION,
    PIPELINE_VERSION,
    FallbackDecision,
    ParserQualityReport,
    ParserResult,
)
from ingestion.parsers.base import DocumentParser
from ingestion.parsers.docling_parser import DoclingParser
from ingestion.parsers.mineru_parser import MinerUParser
from ingestion.quality import assess_parser_quality


class ParserRouter:
    """Run a primary parser and invoke MinerU only on explicit quality failure."""

    parser_name = "router"

    def __init__(
        self,
        settings,
        *,
        primary: DocumentParser | None = None,
        fallback: DocumentParser | None = None,
    ):
        self.settings = settings
        primary_name = str(
            getattr(settings, "force_parser", "")
            or getattr(settings, "parser_primary", "")
            or getattr(settings, "ingestion_backend", "docling")
        ).lower()
        if primary_name == "mineru":
            self.primary = fallback or MinerUParser(settings)
        elif primary_name == "docling":
            self.primary = primary or DoclingParser(settings)
        else:
            raise ValueError(
                f"Primary parser must be 'docling' or 'mineru', got {primary_name!r}"
            )

        fallback_name = str(getattr(settings, "parser_fallback", "mineru") or "").lower()
        self.fallback = fallback
        if (
            self.fallback is None
            and fallback_name == "mineru"
            and self.primary.parser_name != "mineru"
        ):
            self.fallback = MinerUParser(settings)
        self.parser_version = self._signature_hash_material()
        self.last_result: ParserResult | None = None

    def _signature_hash_material(self) -> str:
        primary = (
            self.primary.cache_signature()
            if hasattr(self.primary, "cache_signature")
            else {
                "parser": self.primary.parser_name,
                "version": self.primary.parser_version,
            }
        )
        fallback = (
            self.fallback.cache_signature()
            if self.fallback is not None and hasattr(self.fallback, "cache_signature")
            else None
        )
        return json.dumps(
            {
                "primary": primary,
                "fallback": fallback,
                "pipeline_version": PIPELINE_VERSION,
                "normalizer_version": NORMALIZER_VERSION,
                "thresholds": {
                    "text_coverage": getattr(
                        self.settings, "parser_min_text_page_coverage", 0.90
                    ),
                    "empty_ratio": getattr(
                        self.settings, "parser_max_empty_page_ratio", 0.10
                    ),
                    "replacement_ratio": getattr(
                        self.settings,
                        "parser_max_replacement_char_ratio",
                        0.01,
                    ),
                    "table_valid_ratio": getattr(
                        self.settings, "parser_min_table_valid_ratio", 0.80
                    ),
                },
            },
            sort_keys=True,
        )

    def cache_signature(self) -> dict[str, Any]:
        return json.loads(self._signature_hash_material())

    @staticmethod
    def _failed(
        pdf_path: Path,
        parser: DocumentParser,
        exc: Exception,
        source_file: str | None = None,
    ) -> ParserResult:
        return ParserResult(
            source_file=source_file or pdf_path.name,
            parser=parser.parser_name,
            parser_version=parser.parser_version,
            status="failed",
            quality=ParserQualityReport(
                score=0.0,
                reasons=["conversion_failed", "no_elements"],
            ),
            errors=[str(exc)],
        )

    def _adapter_signature(
        self, parser: DocumentParser, pdf_path: Path
    ) -> dict[str, Any] | None:
        cache_signature = getattr(parser, "cache_signature", None)
        if not pdf_path.exists() or not callable(cache_signature):
            return None
        from ingestion.cache import file_sha256

        return {
            "source_sha256": file_sha256(pdf_path),
            "parser": cache_signature(),
            "normalizer_version": NORMALIZER_VERSION,
        }

    def _refresh_quality(self, result: ParserResult) -> None:
        result.quality = assess_parser_quality(
            result.elements,
            page_count=result.page_count,
            conversion_status=result.status,
            min_text_page_coverage=getattr(
                self.settings, "parser_min_text_page_coverage", 0.90
            ),
            max_empty_page_ratio=getattr(
                self.settings, "parser_max_empty_page_ratio", 0.10
            ),
            max_replacement_char_ratio=getattr(
                self.settings, "parser_max_replacement_char_ratio", 0.01
            ),
            min_table_valid_ratio=getattr(
                self.settings, "parser_min_table_valid_ratio", 0.80
            ),
            require_picture_crops=getattr(
                self.settings, "parser_require_picture_crops", False
            ),
        )

    def _run(
        self,
        parser: DocumentParser,
        pdf_path: Path,
        *,
        source_file: str | None = None,
        artifact_dir: Path | None = None,
    ) -> ParserResult:
        signature = self._adapter_signature(parser, pdf_path)
        report_dir = artifact_dir or (
            Path(getattr(self.settings, "artifact_dir", Path("ingestion_artifacts")))
            / pdf_path.stem
        )
        parser_dir = report_dir / "parsers" / parser.parser_name
        result_path = parser_dir / "adapter_result.json"
        signature_path = parser_dir / "adapter_cache.json"
        if signature is not None and result_path.exists() and signature_path.exists():
            try:
                if (
                    json.loads(signature_path.read_text(encoding="utf-8"))
                    == signature
                ):
                    cached = ParserResult.model_validate_json(
                        result_path.read_text(encoding="utf-8")
                    )
                    if all(
                        not element.image_path or Path(element.image_path).exists()
                        for element in cached.elements
                    ):
                        self._refresh_quality(cached)
                        cached.duration_ms = 0.0
                        cached.quality.duration_ms = 0.0
                        cached.metadata["parser_cache_hit"] = True
                        return cached
            except Exception:
                pass
        try:
            if source_file is None and artifact_dir is None:
                result = parser.parse(pdf_path)
            else:
                result = parser.parse(
                    pdf_path,
                    source_file=source_file,
                    artifact_dir=report_dir,
                )
        except Exception as exc:
            return self._failed(pdf_path, parser, exc, source_file)
        if signature is not None and result.elements:
            parser_dir.mkdir(parents=True, exist_ok=True)
            result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            signature_path.write_text(
                json.dumps(signature, indent=2, sort_keys=True), encoding="utf-8"
            )
        return result

    def _persist_decision(
        self,
        pdf_path: Path,
        primary: ParserResult,
        selected: ParserResult,
        fallback_result: ParserResult | None,
        artifact_dir: Path | None = None,
    ) -> None:
        artifact_dir = artifact_dir or (
            Path(getattr(self.settings, "artifact_dir", Path("ingestion_artifacts")))
            / pdf_path.stem
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "primary_parser": primary.parser,
            "primary_status": primary.status,
            "quality_score": primary.quality.score,
            "fallback_required": bool(primary.quality.reasons),
            "fallback_parser": fallback_result.parser if fallback_result else None,
            "fallback_status": fallback_result.status if fallback_result else None,
            "fallback_quality_score": (
                fallback_result.quality.score if fallback_result else None
            ),
            "reason_codes": selected.fallback.reasons,
            "selected_result": selected.parser,
        }
        (artifact_dir / "parser_selection.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def parse(
        self,
        pdf_path: Path,
        *,
        source_file: str | None = None,
        artifact_dir: Path | None = None,
    ) -> ParserResult:
        forced = str(getattr(self.settings, "force_parser", "") or "").lower()
        primary_result = self._run(
            self.primary,
            pdf_path,
            source_file=source_file,
            artifact_dir=artifact_dir,
        )
        reasons = list(primary_result.quality.reasons)
        fallback_result: ParserResult | None = None
        fallback_allowed = (
            not forced
            and self.fallback is not None
            and bool(getattr(self.settings, "parser_fallback_enabled", True))
        )

        if reasons and fallback_allowed:
            fallback_result = self._run(
                self.fallback,
                pdf_path,
                source_file=source_file,
                artifact_dir=artifact_dir,
            )

        selected = primary_result
        fallback_used = False
        if fallback_result is not None:
            fallback_usable = bool(fallback_result.elements) and (
                fallback_result.status not in {"failed", "failure"}
            )
            if fallback_usable and (
                not primary_result.elements
                or fallback_result.quality.score > primary_result.quality.score
            ):
                selected = fallback_result
                fallback_used = True

        selected.fallback = FallbackDecision(
            attempted=fallback_result is not None,
            used=fallback_used,
            forced=bool(forced),
            reasons=reasons,
            primary_score=primary_result.quality.score,
            fallback_score=(
                fallback_result.quality.score if fallback_result is not None else None
            ),
        )
        selected.metadata.update(
            {
                "primary_parser": primary_result.parser,
                "primary_parser_version": primary_result.parser_version,
                "primary_duration_ms": primary_result.duration_ms,
                "fallback_parser": fallback_result.parser if fallback_result else None,
                "fallback_duration_ms": (
                    fallback_result.duration_ms if fallback_result else 0.0
                ),
                "fallback_errors": fallback_result.errors if fallback_result else [],
            }
        )
        if reasons and not fallback_used and selected.status not in {"failed", "failure"}:
            selected.status = "degraded"
            if fallback_result is not None and fallback_result.errors:
                selected.errors.extend(fallback_result.errors)
            elif not fallback_allowed:
                selected.warnings.append(
                    "Parser quality gates failed and MinerU fallback was disabled."
                )
        self._persist_decision(
            pdf_path,
            primary_result,
            selected,
            fallback_result,
            artifact_dir,
        )
        self.last_result = selected
        return selected

def get_parser_router(settings) -> ParserRouter:
    return ParserRouter(settings)
