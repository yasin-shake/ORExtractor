"""Isolated MinerU fallback adapter (HTTP service or subprocess CLI)."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ingestion.config import ParserQualityPolicy
from ingestion.input_staging import stage_pdf_input
from ingestion.models import MinerUConversionMetadata, ParserResult
from ingestion.normalizers.mineru import normalize_mineru_content
from ingestion.quality import assess_parser_quality


def _package_version() -> str:
    for name in ("mineru", "magic-pdf"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return "external"


class MinerUParser:
    parser_name = "mineru"

    def __init__(self, settings):
        self.settings = settings
        self.parser_version = _package_version()
        self.artifact_dir = Path(
            getattr(settings, "artifact_dir", Path("ingestion_artifacts"))
        )
        self.quality_policy = ParserQualityPolicy.from_settings(settings)

    def cache_signature(self) -> dict[str, Any]:
        return {
            "parser": self.parser_name,
            "version": self.parser_version,
            "execution_mode": getattr(self.settings, "mineru_execution_mode", "service"),
            "backend": getattr(self.settings, "mineru_backend", "pipeline"),
        }

    def readiness_error(self) -> str | None:
        mode = str(
            getattr(self.settings, "mineru_execution_mode", "service")
        ).lower()
        if mode == "service":
            if not str(
                getattr(self.settings, "mineru_api_url", "") or ""
            ).strip():
                return (
                    "MinerU fallback service is not configured. Set "
                    "MINERU_API_URL or use MINERU_EXECUTION_MODE=cli."
                )
            return None
        if mode == "cli":
            command = str(
                getattr(self.settings, "mineru_command", "mineru")
                or "mineru"
            )
            if shutil.which(command) is None:
                return (
                    f"MinerU command {command!r} was not found. Install MinerU "
                    "in its isolated environment or configure MINERU_API_URL."
                )
            return None
        return (
            "MINERU_EXECUTION_MODE must be 'service' or 'cli', "
            f"got {mode!r}."
        )

    def _service(self, pdf_path: Path, output_dir: Path) -> list[dict[str, Any]]:
        endpoint = str(getattr(self.settings, "mineru_api_url", "") or "").rstrip("/")
        if not endpoint:
            raise RuntimeError(
                "MinerU fallback service is not configured. Set MINERU_API_URL "
                "or use MINERU_EXECUTION_MODE=cli."
            )
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests is required for the MinerU service adapter") from exc

        token = str(getattr(self.settings, "mineru_api_token", "") or "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with pdf_path.open("rb") as stream:
            response = requests.post(
                endpoint,
                files={"file": (pdf_path.name, stream, "application/pdf")},
                data={"backend": getattr(self.settings, "mineru_backend", "pipeline")},
                headers=headers,
                timeout=int(getattr(self.settings, "mineru_timeout_seconds", 1800)),
            )
        response.raise_for_status()
        payload = response.json()
        (output_dir / "service_response.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        content = payload.get("content_list", payload.get("content", payload))
        if isinstance(content, dict):
            content = content.get("items", [])
        if not isinstance(content, list):
            raise RuntimeError("MinerU service response did not contain a content list")
        return content

    def _cli(self, pdf_path: Path, output_dir: Path) -> list[dict[str, Any]]:
        command = str(getattr(self.settings, "mineru_command", "mineru") or "mineru")
        args = [
            command,
            "-p",
            str(pdf_path.resolve()),
            "-o",
            str(output_dir.resolve()),
            "-b",
            str(getattr(self.settings, "mineru_backend", "pipeline")),
        ]
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                shell=False,
                check=False,
                timeout=int(getattr(self.settings, "mineru_timeout_seconds", 1800)),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"MinerU command {command!r} was not found. Install MinerU in its "
                "isolated environment or configure MINERU_API_URL."
            ) from exc
        (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(
                f"MinerU exited with code {completed.returncode}: "
                f"{completed.stderr[-500:]}"
            )
        candidates = sorted(output_dir.rglob("*content_list*.json"))
        if not candidates:
            candidates = sorted(output_dir.rglob("*.json"))
        for candidate in candidates:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                continue
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict) and isinstance(payload.get("content_list"), list):
                return payload["content_list"]
        raise RuntimeError("MinerU completed but no content-list JSON artifact was found")

    def parse(
        self,
        pdf_path: Path,
        *,
        source_file: str | None = None,
        artifact_dir: Path | None = None,
    ) -> ParserResult:
        source_file = source_file or pdf_path.name
        started = time.perf_counter()
        mode = str(getattr(self.settings, "mineru_execution_mode", "service")).lower()
        output_dir = (
            artifact_dir or self.artifact_dir / pdf_path.stem
        ) / "parsers" / "mineru"
        output_dir.mkdir(parents=True, exist_ok=True)
        work_root = Path(
            getattr(
                self.settings,
                "ingest_work_dir",
                Path(".ingestion_work"),
            )
        )
        with stage_pdf_input(pdf_path, work_root) as staged:
            if mode == "service":
                content = self._service(staged.input_path, output_dir)
            elif mode == "cli":
                content = self._cli(staged.input_path, output_dir)
            else:
                raise ValueError(
                    "MINERU_EXECUTION_MODE must be 'service' or 'cli', "
                    f"got {mode!r}"
                )

        normalized_path = output_dir / "content_list.json"
        normalized_path.write_text(
            json.dumps(content, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        elements = normalize_mineru_content(
            content,
            source_file=source_file,
            artifact_dir=output_dir,
            parser_version=self.parser_version,
        )
        page_count = max((element.page_number for element in elements), default=0)
        quality = assess_parser_quality(
            elements,
            page_count=page_count,
            conversion_status="success",
            **self.quality_policy.assessment_kwargs(),
        )
        duration_ms = (time.perf_counter() - started) * 1000
        quality.duration_ms = duration_ms
        metadata = MinerUConversionMetadata(
            execution_mode=mode,
            backend=str(getattr(self.settings, "mineru_backend", "pipeline")),
            page_count=page_count,
            endpoint_or_command=(
                str(getattr(self.settings, "mineru_api_url", "") or "")
                if mode == "service"
                else str(getattr(self.settings, "mineru_command", "mineru"))
            ),
            output_files=[
                str(path.relative_to(output_dir))
                for path in output_dir.rglob("*")
                if path.is_file()
            ],
        )
        result = ParserResult(
            source_file=source_file,
            parser=self.parser_name,
            parser_version=self.parser_version,
            elements=elements,
            artifact_paths={"content_list": str(normalized_path)},
            page_count=page_count,
            duration_ms=duration_ms,
            quality=quality,
            metadata=metadata.model_dump(),
        )
        (output_dir / "quality.json").write_text(
            quality.model_dump_json(indent=2), encoding="utf-8"
        )
        return result
