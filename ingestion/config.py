"""Grouped immutable policy views over the compatibility Settings object."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParserQualityPolicy:
    min_text_page_coverage: float = 0.90
    max_empty_page_ratio: float = 0.10
    max_replacement_char_ratio: float = 0.01
    min_table_valid_ratio: float = 0.80
    require_picture_crops: bool = False
    min_cache_quality_score: float = 0.90
    min_page_count_agreement: float = 0.90

    @classmethod
    def from_settings(cls, settings: Any) -> "ParserQualityPolicy":
        return cls(
            min_text_page_coverage=float(
                getattr(
                    settings,
                    "parser_min_text_page_coverage",
                    0.90,
                )
            ),
            max_empty_page_ratio=float(
                getattr(
                    settings,
                    "parser_max_empty_page_ratio",
                    0.10,
                )
            ),
            max_replacement_char_ratio=float(
                getattr(
                    settings,
                    "parser_max_replacement_char_ratio",
                    0.01,
                )
            ),
            min_table_valid_ratio=float(
                getattr(
                    settings,
                    "parser_min_table_valid_ratio",
                    0.80,
                )
            ),
            require_picture_crops=bool(
                getattr(
                    settings,
                    "parser_require_picture_crops",
                    False,
                )
            ),
            min_cache_quality_score=float(
                getattr(
                    settings,
                    "parser_min_cache_quality_score",
                    0.90,
                )
            ),
            min_page_count_agreement=float(
                getattr(
                    settings,
                    "parser_min_page_count_agreement",
                    0.90,
                )
            ),
        )

    def assessment_kwargs(self) -> dict[str, Any]:
        return {
            "min_text_page_coverage": self.min_text_page_coverage,
            "max_empty_page_ratio": self.max_empty_page_ratio,
            "max_replacement_char_ratio": (
                self.max_replacement_char_ratio
            ),
            "min_table_valid_ratio": self.min_table_valid_ratio,
            "require_picture_crops": self.require_picture_crops,
        }

    def signature(self) -> dict[str, Any]:
        return {
            **self.assessment_kwargs(),
            "min_cache_quality_score": self.min_cache_quality_score,
            "min_page_count_agreement": self.min_page_count_agreement,
        }


@dataclass(frozen=True)
class DoclingExecutionConfig:
    process_isolation: bool = True
    hard_timeout_seconds: float = 900
    text_first_table_mode: str = "fast"
    work_dir: Path = Path(".ingestion_work")

    @classmethod
    def from_settings(cls, settings: Any) -> "DoclingExecutionConfig":
        return cls(
            process_isolation=bool(
                getattr(settings, "docling_process_isolation", True)
            ),
            hard_timeout_seconds=max(
                0.001,
                float(
                    getattr(
                        settings,
                        "docling_hard_timeout_seconds",
                        getattr(
                            settings,
                            "docling_timeout_seconds",
                            900,
                        ),
                    )
                ),
            ),
            text_first_table_mode=str(
                getattr(
                    settings,
                    "docling_text_first_table_mode",
                    "fast",
                )
            ).lower(),
            work_dir=Path(
                getattr(
                    settings,
                    "ingest_work_dir",
                    Path(".ingestion_work"),
                )
            ),
        )
