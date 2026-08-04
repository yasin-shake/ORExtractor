"""Docling primary parser adapter."""

from __future__ import annotations

import json
import os
import re
import time
import gc
from collections import OrderedDict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ingestion.config import DoclingExecutionConfig, ParserQualityPolicy
from ingestion.docling_worker import DoclingWorkerManager
from ingestion.input_staging import stage_pdf_input
from ingestion.models import DoclingConversionMetadata, ParserResult
from ingestion.normalizers.docling import normalize_docling_document
from ingestion.quality import assess_parser_quality


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "export_to_dict"):
        return _jsonable(value.export_to_dict())
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"repr": repr(value)}


def _configure_windows_huggingface_cache() -> None:
    """Use copies on Windows hosts where non-admin symlinks are unavailable."""
    if os.name != "nt":
        return
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    try:
        import huggingface_hub
        import huggingface_hub.file_download as file_download
    except ImportError:
        return
    # huggingface_hub >=1 honors HF_HUB_DISABLE_SYMLINKS. Version 0.x can
    # misclassify support under concurrent downloads and leak WinError 1314.
    major = int(str(huggingface_hub.__version__).split(".", 1)[0])
    if major == 0:
        file_download.are_symlinks_supported = lambda *_args, **_kwargs: False


def _positive_setting(settings, name: str, default: int) -> int:
    return max(1, int(getattr(settings, name, default)))


class DoclingParser:
    parser_name = "docling"

    def __init__(self, settings, *, process_isolation: bool | None = None):
        self.settings = settings
        self.parser_version = _package_version("docling")
        self.artifact_dir = Path(
            getattr(settings, "artifact_dir", Path("ingestion_artifacts"))
        )
        self._converters: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._safe_batch_mode = False
        self.execution = DoclingExecutionConfig.from_settings(settings)
        self.quality_policy = ParserQualityPolicy.from_settings(settings)
        self._process_isolation = (
            self.execution.process_isolation
            if process_isolation is None
            else bool(process_isolation)
        )
        self._worker: DoclingWorkerManager | None = None

    def cache_signature(self) -> dict[str, Any]:
        return {
            "parser": self.parser_name,
            "version": self.parser_version,
            "execution_mode": getattr(self.settings, "docling_execution_mode", "local"),
            "do_ocr": getattr(self.settings, "docling_do_ocr", True),
            "ocr_backend": getattr(
                self.settings, "docling_ocr_backend", "onnxruntime"
            ),
            "ocr_languages": getattr(
                self.settings, "docling_ocr_languages", "english"
            ),
            "force_full_page_ocr": getattr(
                self.settings, "docling_force_full_page_ocr", False
            ),
            "ocr_bitmap_area_threshold": getattr(
                self.settings, "docling_ocr_bitmap_area_threshold", 0.05
            ),
            "do_table_structure": getattr(
                self.settings, "docling_do_table_structure", True
            ),
            "table_mode": getattr(self.settings, "docling_table_mode", "accurate"),
            "generate_page_images": getattr(
                self.settings, "docling_generate_page_images", True
            ),
            "generate_picture_images": getattr(
                self.settings, "docling_generate_picture_images", True
            ),
            "images_scale": getattr(self.settings, "docling_images_scale", 1.0),
            "ocr_batch_size": _positive_setting(
                self.settings, "docling_ocr_batch_size", 1
            ),
            "layout_batch_size": _positive_setting(
                self.settings, "docling_layout_batch_size", 1
            ),
            "table_batch_size": _positive_setting(
                self.settings, "docling_table_batch_size", 1
            ),
            "queue_max_size": _positive_setting(
                self.settings, "docling_queue_max_size", 2
            ),
            "page_batch_size": _positive_setting(
                self.settings, "docling_page_batch_size", 1
            ),
            "num_threads": _positive_setting(
                self.settings, "docling_num_threads", 2
            ),
            "device": str(
                getattr(self.settings, "docling_device", "auto")
            ).lower(),
            "adaptive_ocr": bool(
                getattr(self.settings, "docling_adaptive_ocr", True)
            ),
            "adaptive_ocr_preflight_version": "1",
            "pymupdf_version": _package_version("PyMuPDF"),
            "native_text_min_chars": int(
                getattr(self.settings, "docling_native_text_min_chars", 80)
            ),
            "native_text_coverage": float(
                getattr(self.settings, "docling_native_text_coverage", 0.98)
            ),
            "native_text_max_empty_pages": int(
                getattr(
                    self.settings,
                    "docling_native_text_max_empty_pages",
                    2,
                )
            ),
            "batch_fallback_enabled": bool(
                getattr(
                    self.settings,
                    "docling_batch_fallback_enabled",
                    True,
                )
            ),
            "safe_batch_size": _positive_setting(
                self.settings, "docling_safe_batch_size", 1
            ),
            "fast_table_max_pages": max(
                0,
                int(
                    getattr(
                        self.settings,
                        "docling_fast_table_max_pages",
                        20,
                    )
                ),
            ),
            "process_isolation": self._process_isolation,
            "hard_timeout_seconds": float(
                self.execution.hard_timeout_seconds
            ),
        }

    def _pipeline_options(
        self,
        *,
        effective_do_ocr: bool | None = None,
        effective_table_mode: str | None = None,
        safe_batches: bool = False,
    ):
        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except ImportError as exc:
            raise RuntimeError(
                "Docling is required for the primary ingestion parser. "
                "Install the ingestion dependencies with: pip install -r requirements.txt"
            ) from exc

        options = PdfPipelineOptions()
        safe_batch_size = _positive_setting(
            self.settings, "docling_safe_batch_size", 1
        )
        configured_do_ocr = bool(
            getattr(self.settings, "docling_do_ocr", True)
        )
        do_ocr = (
            configured_do_ocr
            if effective_do_ocr is None
            else bool(effective_do_ocr)
        )

        def stage_batch(name: str, default: int) -> int:
            if safe_batches:
                return safe_batch_size
            return _positive_setting(self.settings, name, default)

        for name, value in {
            "do_ocr": do_ocr,
            "do_table_structure": getattr(
                self.settings, "docling_do_table_structure", True
            ),
            "generate_page_images": getattr(
                self.settings, "docling_generate_page_images", True
            ),
            "generate_picture_images": getattr(
                self.settings, "docling_generate_picture_images", True
            ),
            "images_scale": max(
                0.1, float(getattr(self.settings, "docling_images_scale", 1.0))
            ),
            "ocr_batch_size": stage_batch(
                "docling_ocr_batch_size", 1
            ),
            "layout_batch_size": stage_batch(
                "docling_layout_batch_size", 1
            ),
            "table_batch_size": (
                safe_batch_size
                if safe_batches
                else _positive_setting(
                    self.settings, "docling_table_batch_size", 1
                )
            ),
            "queue_max_size": _positive_setting(
                self.settings, "docling_queue_max_size", 2
            ),
            "document_timeout": getattr(
                self.settings, "docling_timeout_seconds", 900
            ),
        }.items():
            if hasattr(options, name):
                setattr(options, name, value)

        if do_ocr:
            try:
                from docling.datamodel.pipeline_options import RapidOcrOptions

                backend = str(
                    getattr(
                        self.settings,
                        "docling_ocr_backend",
                        "onnxruntime",
                    )
                ).strip().lower()
                valid_backends = {"onnxruntime", "openvino", "torch"}
                if backend not in valid_backends:
                    raise ValueError(
                        "DOCLING_OCR_BACKEND must be one of "
                        f"{', '.join(sorted(valid_backends))}, got {backend!r}"
                    )
                languages = [
                    language.strip()
                    for language in str(
                        getattr(
                            self.settings,
                            "docling_ocr_languages",
                            "english",
                        )
                    ).split(",")
                    if language.strip()
                ] or ["english"]
                options.ocr_options = RapidOcrOptions(
                    backend=backend,
                    lang=languages,
                    force_full_page_ocr=bool(
                        getattr(
                            self.settings,
                            "docling_force_full_page_ocr",
                            False,
                        )
                    ),
                    bitmap_area_threshold=max(
                        0.0,
                        float(
                            getattr(
                                self.settings,
                                "docling_ocr_bitmap_area_threshold",
                                0.05,
                            )
                        ),
                    ),
                )
            except ImportError:
                pass

        table_options = getattr(options, "table_structure_options", None)
        if table_options is not None and hasattr(table_options, "mode"):
            mode_name = str(
                effective_table_mode
                or getattr(self.settings, "docling_table_mode", "accurate")
            ).upper()
            try:
                from docling.datamodel.pipeline_options import TableFormerMode

                table_options.mode = getattr(TableFormerMode, mode_name)
            except (ImportError, AttributeError):
                pass

        accelerator_options = getattr(options, "accelerator_options", None)
        if accelerator_options is not None:
            accelerator_options.num_threads = _positive_setting(
                self.settings, "docling_num_threads", 2
            )
            device = str(
                getattr(self.settings, "docling_device", "auto")
            ).strip().lower()
            valid_devices = {"auto", "cpu", "cuda", "mps", "xpu"}
            if device not in valid_devices:
                raise ValueError(
                    "DOCLING_DEVICE must be one of "
                    f"{', '.join(sorted(valid_devices))}, got {device!r}"
                )
            accelerator_options.device = device
        return options

    def _preflight_pdf(self, pdf_path: Path) -> dict[str, Any]:
        """Cheap native-text scan used to avoid unnecessary OCR/model work."""
        configured_ocr = bool(
            getattr(self.settings, "docling_do_ocr", True)
        )
        adaptive = bool(
            getattr(self.settings, "docling_adaptive_ocr", True)
        )
        minimum_chars = max(
            1,
            int(
                getattr(
                    self.settings,
                    "docling_native_text_min_chars",
                    80,
                )
            ),
        )
        required_coverage = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        self.settings,
                        "docling_native_text_coverage",
                        0.98,
                    )
                ),
            ),
        )
        max_empty_pages = max(
            0,
            int(
                getattr(
                    self.settings,
                    "docling_native_text_max_empty_pages",
                    2,
                )
            ),
        )
        page_count = 0
        native_pages = 0
        empty_pages = 0
        table_candidate_pages = 0
        try:
            import fitz

            with fitz.open(pdf_path) as document:
                page_count = len(document)
                for page in document:
                    text = page.get_text("text") or ""
                    character_count = len(re.sub(r"\s+", "", text))
                    if character_count >= minimum_chars:
                        native_pages += 1
                    else:
                        empty_pages += 1
                    table_like = any(
                        (
                            len(re.findall(r"[-+]?\d[\d,.]*", line)) >= 2
                            and (
                                "\t" in line
                                or bool(re.search(r"\S+\s{2,}\S+", line))
                            )
                        )
                        for line in text.splitlines()
                    )
                    if table_like:
                        table_candidate_pages += 1
        except Exception as exc:
            return {
                "page_count": 0,
                "native_text_pages": 0,
                "empty_pages": 0,
                "native_text_coverage": 0.0,
                "table_candidate_pages": 0,
                "effective_do_ocr": configured_ocr,
                "reason": f"preflight_failed:{exc}",
            }

        coverage = native_pages / max(1, page_count)
        can_skip_ocr = (
            adaptive
            and coverage >= required_coverage
            and empty_pages <= max_empty_pages
        )
        return {
            "page_count": page_count,
            "native_text_pages": native_pages,
            "empty_pages": empty_pages,
            "native_text_coverage": round(coverage, 6),
            "table_candidate_pages": table_candidate_pages,
            "effective_do_ocr": configured_ocr and not can_skip_ocr,
            "reason": (
                "native_text_complete"
                if configured_ocr and can_skip_ocr
                else "ocr_required"
                if configured_ocr
                else "ocr_disabled"
            ),
        }

    def _effective_table_mode(self, preflight: dict[str, Any]) -> str:
        configured = str(
            getattr(self.settings, "docling_table_mode", "accurate")
        ).lower()
        fast_max_pages = max(
            0,
            int(
                getattr(
                    self.settings,
                    "docling_fast_table_max_pages",
                    20,
                )
            ),
        )
        page_count = int(preflight.get("page_count", 0) or 0)
        if configured == "accurate" and 0 < page_count <= fast_max_pages:
            return "fast"
        return configured

    def _converter_key(
        self,
        *,
        effective_do_ocr: bool,
        effective_table_mode: str,
        safe_batches: bool,
    ) -> tuple[Any, ...]:
        return (
            effective_do_ocr,
            effective_table_mode,
            safe_batches,
            json.dumps(self.cache_signature(), sort_keys=True),
        )

    def _get_converter(
        self,
        *,
        effective_do_ocr: bool,
        effective_table_mode: str,
        safe_batches: bool,
    ):
        _configure_windows_huggingface_cache()
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise RuntimeError(
                "Docling is not installed. Run: pip install -r requirements.txt"
            ) from exc

        key = self._converter_key(
            effective_do_ocr=effective_do_ocr,
            effective_table_mode=effective_table_mode,
            safe_batches=safe_batches,
        )
        existing = self._converters.get(key)
        if existing is not None:
            self._converters.move_to_end(key)
            return existing
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self._pipeline_options(
                        effective_do_ocr=effective_do_ocr,
                        effective_table_mode=effective_table_mode,
                        safe_batches=safe_batches,
                    )
                )
            },
        )
        self._converters[key] = converter
        cache_size = max(
            1,
            int(
                getattr(
                    self.settings,
                    "docling_converter_cache_size",
                    2,
                )
            ),
        )
        while len(self._converters) > cache_size:
            _, evicted = self._converters.popitem(last=False)
            del evicted
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        return converter

    @staticmethod
    def _conversion_memory_failure(conversion: Any) -> bool:
        markers = (
            "std::bad_alloc",
            "out of memory",
            "page backend was unloaded",
            "assertionerror",
        )
        errors = getattr(conversion, "errors", None) or []
        text = " ".join(str(error).lower() for error in errors)
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_memory_exception(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(
            marker in text
            for marker in (
                "memoryerror",
                "std::bad_alloc",
                "out of memory",
                "native-memory failure",
                "page backend was unloaded",
                "assertionerror",
            )
        )

    def _release_failed_converter(self, converter: Any) -> None:
        for key, value in list(self._converters.items()):
            if value is converter:
                self._converters.pop(key, None)
        del converter
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _convert_local(
        self,
        pdf_path: Path,
        *,
        page_range: tuple[int, int] | None = None,
        preflight: dict[str, Any] | None = None,
    ):
        from docling.datamodel.settings import settings as docling_settings

        preflight = preflight or self._preflight_pdf(pdf_path)
        docling_settings.debug.profile_pipeline_timings = bool(
            getattr(self.settings, "docling_profiling", True)
        )
        effective_do_ocr = bool(preflight["effective_do_ocr"])
        effective_table_mode = self._effective_table_mode(preflight)
        fallback_enabled = bool(
            getattr(
                self.settings,
                "docling_batch_fallback_enabled",
                True,
            )
        )
        attempts = [self._safe_batch_mode]
        if not self._safe_batch_mode and fallback_enabled:
            attempts.append(True)
        last_error: Exception | None = None
        for safe_batches in attempts:
            page_batch_size = (
                _positive_setting(
                    self.settings, "docling_safe_batch_size", 1
                )
                if safe_batches
                else _positive_setting(
                    self.settings, "docling_page_batch_size", 1
                )
            )
            docling_settings.perf.page_batch_size = page_batch_size
            converter_key = self._converter_key(
                effective_do_ocr=effective_do_ocr,
                effective_table_mode=effective_table_mode,
                safe_batches=safe_batches,
            )
            converter_reused = converter_key in self._converters
            converter = self._get_converter(
                effective_do_ocr=effective_do_ocr,
                effective_table_mode=effective_table_mode,
                safe_batches=safe_batches,
            )
            try:
                convert_options = {
                    "raises_on_error": False,
                    "max_num_pages": int(
                        getattr(self.settings, "docling_max_pages", 1000)
                    ),
                    "max_file_size": int(
                        getattr(self.settings, "docling_max_file_mb", 2048)
                        * 1024
                        * 1024
                    ),
                }
                if page_range is not None:
                    convert_options["page_range"] = page_range
                conversion = converter.convert(pdf_path, **convert_options)
                if self._conversion_memory_failure(conversion):
                    raise RuntimeError(
                        "Docling conversion reported a native-memory failure."
                    )
                runtime = {
                    "preflight": preflight,
                    "effective_do_ocr": effective_do_ocr,
                    "effective_table_mode": effective_table_mode,
                    "page_batch_size": page_batch_size,
                    "safe_batch_fallback_used": safe_batches,
                    "converter_reused": converter_reused,
                    "page_range": list(page_range) if page_range else None,
                }
                if safe_batches:
                    self._safe_batch_mode = True
                return conversion, runtime
            except Exception as exc:
                last_error = exc
                self._release_failed_converter(converter)
                if safe_batches or not fallback_enabled or not self._is_memory_exception(
                    exc
                ):
                    raise
                self._safe_batch_mode = True
        assert last_error is not None
        raise last_error

    def _convert_service(self, pdf_path: Path):
        endpoint = str(getattr(self.settings, "docling_serve_url", "") or "").rstrip("/")
        if not endpoint:
            raise RuntimeError(
                "DOCLING_EXECUTION_MODE=serve requires DOCLING_SERVE_URL."
            )
        if not endpoint.endswith("/v1/convert/file"):
            endpoint = f"{endpoint}/v1/convert/file"
        try:
            import requests
            from docling_core.types.doc import DoclingDocument
        except ImportError as exc:
            raise RuntimeError(
                "Docling core and requests are required for Docling Serve."
            ) from exc
        api_key = str(getattr(self.settings, "docling_serve_api_key", "") or "")
        headers = {"X-Api-Key": api_key} if api_key else {}
        data = [
            ("from_formats", "pdf"),
            ("to_formats", "json"),
            ("to_formats", "md"),
            ("do_ocr", str(getattr(self.settings, "docling_do_ocr", True)).lower()),
            (
                "table_mode",
                str(getattr(self.settings, "docling_table_mode", "accurate")),
            ),
            ("image_export_mode", "referenced"),
        ]
        with pdf_path.open("rb") as stream:
            response = requests.post(
                endpoint,
                files={"files": (pdf_path.name, stream, "application/pdf")},
                data=data,
                headers=headers,
                timeout=int(getattr(self.settings, "docling_timeout_seconds", 900)),
            )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload.get("documents"), list):
            payload = payload["documents"][0]
        document_payload = payload.get("document", {})
        json_content = document_payload.get("json_content", document_payload)
        if isinstance(json_content, str):
            json_content = json.loads(json_content)
        document = DoclingDocument.model_validate(json_content)
        return SimpleNamespace(
            document=document,
            status=payload.get("status", "success"),
            errors=payload.get("errors", []),
            processing_time=payload.get("processing_time", 0.0),
        ), {
            "preflight": {},
            "effective_do_ocr": bool(
                getattr(self.settings, "docling_do_ocr", True)
            ),
            "effective_table_mode": str(
                getattr(self.settings, "docling_table_mode", "accurate")
            ),
            "service": True,
        }

    def _parse_document(
        self,
        pdf_path: Path,
        *,
        source_file: str | None = None,
        artifact_dir: Path | None = None,
        page_range: tuple[int, int] | None = None,
        preflight: dict[str, Any] | None = None,
    ) -> ParserResult:
        source_file = source_file or pdf_path.name
        mode = str(getattr(self.settings, "docling_execution_mode", "local")).lower()
        started = time.perf_counter()
        if mode == "local":
            conversion, runtime = self._convert_local(
                pdf_path,
                page_range=page_range,
                preflight=preflight,
            )
        elif mode == "serve":
            conversion, runtime = self._convert_service(pdf_path)
        else:
            raise ValueError(
                f"DOCLING_EXECUTION_MODE must be 'local' or 'serve', got {mode!r}"
            )
        document = getattr(conversion, "document", conversion)
        status = str(getattr(conversion, "status", "success"))
        if "." in status:
            status = status.rsplit(".", 1)[-1]
        status = status.lower()

        report_dir = (
            artifact_dir or self.artifact_dir / pdf_path.stem
        ) / "parsers" / "docling"
        report_dir.mkdir(parents=True, exist_ok=True)
        document_json = report_dir / "document.json"
        document_json.write_text(
            json.dumps(_jsonable(document), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        markdown = ""
        export_markdown = getattr(document, "export_to_markdown", None)
        if callable(export_markdown):
            markdown = str(export_markdown())
            (report_dir / "document.md").write_text(markdown, encoding="utf-8")

        elements = normalize_docling_document(
            document,
            source_file=source_file,
            artifact_dir=report_dir,
            parser_version=self.parser_version,
        )
        pages = getattr(document, "pages", None)
        page_count = len(pages) if pages is not None else max(
            (element.page_number for element in elements), default=0
        )
        quality_elements = elements
        if page_range is not None:
            first_page = page_range[0]
            quality_elements = [
                element.model_copy(
                    update={
                        "page_number": (
                            element.page_number - first_page + 1
                        )
                    }
                )
                for element in elements
            ]
        quality = assess_parser_quality(
            quality_elements,
            page_count=page_count,
            conversion_status=status,
            **self.quality_policy.assessment_kwargs(),
        )
        duration_ms = (time.perf_counter() - started) * 1000
        quality.duration_ms = duration_ms
        conversion_errors = [
            str(error) for error in (getattr(conversion, "errors", None) or [])
        ]
        metadata = DoclingConversionMetadata(
            execution_mode=mode,
            conversion_status=status,
            page_count=page_count,
            model_artifact_revision=str(
                getattr(self.settings, "docling_model_artifact_revision", "")
            ),
            pipeline_options=self.cache_signature(),
        )
        metadata_payload = metadata.model_dump()
        metadata_payload["runtime"] = runtime
        profiling_path = None
        if bool(getattr(self.settings, "docling_profiling", True)):
            timings = _jsonable(getattr(conversion, "timings", {}) or {})
            profiling_path = report_dir / "profiling.json"
            profiling_path.write_text(
                json.dumps(
                    {
                        "duration_ms": duration_ms,
                        "timings": timings,
                        "runtime": runtime,
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )
        result = ParserResult(
            source_file=source_file,
            parser=self.parser_name,
            parser_version=self.parser_version,
            status=status,
            elements=elements,
            artifact_paths={
                "document_json": str(document_json),
                **(
                    {"profiling": str(profiling_path)}
                    if profiling_path is not None
                    else {}
                ),
                **(
                    {"document_markdown": str(report_dir / "document.md")}
                    if markdown
                    else {}
                ),
            },
            page_count=page_count,
            duration_ms=duration_ms,
            errors=conversion_errors,
            quality=quality,
            metadata=metadata_payload,
        )
        (report_dir / "quality.json").write_text(
            result.quality.model_dump_json(indent=2), encoding="utf-8"
        )
        return result

    def _worker_manager(self) -> DoclingWorkerManager:
        if self._worker is None:
            self._worker = DoclingWorkerManager(
                settings=dict(vars(self.settings)),
                timeout_seconds=self.execution.hard_timeout_seconds,
            )
        return self._worker

    def _combine_segments(
        self,
        results: list[tuple[tuple[int, int], ParserResult]],
        *,
        source_file: str,
        artifact_dir: Path,
        page_count: int,
        preflight: dict[str, Any],
    ) -> ParserResult:
        elements = []
        seen_ids: set[str] = set()
        errors: list[str] = []
        warnings: list[str] = []
        artifacts: dict[str, str] = {}
        duration_ms = 0.0
        statuses: list[str] = []
        segment_runtime: list[dict[str, Any]] = []
        for page_range, result in results:
            duration_ms += result.duration_ms
            statuses.append(result.status)
            errors.extend(result.errors)
            warnings.extend(result.warnings)
            for element in result.elements:
                if element.element_id not in seen_ids:
                    elements.append(element)
                    seen_ids.add(element.element_id)
            label = f"{page_range[0]:04d}-{page_range[1]:04d}"
            artifacts.update(
                {
                    f"segment_{label}_{name}": path
                    for name, path in result.artifact_paths.items()
                }
            )
            segment_runtime.append(
                {
                    "page_range": list(page_range),
                    "duration_ms": result.duration_ms,
                    "status": result.status,
                    "runtime": result.metadata.get("runtime", {}),
                }
            )
        elements.sort(
            key=lambda element: (
                element.page_number,
                int(element.metadata.get("docling_ordinal", 0)),
            )
        )
        accepted_statuses = {"success", "completed"}
        status = (
            "success"
            if statuses and all(item in accepted_statuses for item in statuses)
            else "partial_success"
        )
        quality = assess_parser_quality(
            elements,
            page_count=page_count,
            conversion_status=status,
            **self.quality_policy.assessment_kwargs(),
        )
        quality.duration_ms = duration_ms
        runtime = {
            "preflight": preflight,
            "segmented": True,
            "segment_count": len(results),
            "segments": segment_runtime,
        }
        report_dir = Path(artifact_dir) / "parsers" / "docling"
        report_dir.mkdir(parents=True, exist_ok=True)
        profiling_path = report_dir / "profiling.json"
        profiling_path.write_text(
            json.dumps(
                {
                    "duration_ms": duration_ms,
                    "timings": {},
                    "runtime": runtime,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        result = ParserResult(
            source_file=source_file,
            parser=self.parser_name,
            parser_version=self.parser_version,
            status=status,
            elements=elements,
            artifact_paths={
                **artifacts,
                "profiling": str(profiling_path),
            },
            page_count=page_count,
            duration_ms=duration_ms,
            warnings=list(dict.fromkeys(warnings)),
            errors=list(dict.fromkeys(errors)),
            quality=quality,
            metadata={
                **DoclingConversionMetadata(
                    execution_mode="local",
                    conversion_status=status,
                    page_count=page_count,
                    model_artifact_revision=str(
                        getattr(
                            self.settings,
                            "docling_model_artifact_revision",
                            "",
                        )
                    ),
                    pipeline_options=self.cache_signature(),
                ).model_dump(),
                "runtime": runtime,
            },
        )
        (report_dir / "quality.json").write_text(
            quality.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return result

    def parse(
        self,
        pdf_path: Path,
        *,
        source_file: str | None = None,
        artifact_dir: Path | None = None,
    ) -> ParserResult:
        source_file = source_file or pdf_path.name
        mode = str(
            getattr(self.settings, "docling_execution_mode", "local")
        ).lower()
        if mode != "local":
            return self._parse_document(
                pdf_path,
                source_file=source_file,
                artifact_dir=artifact_dir,
            )

        report_dir = artifact_dir or self.artifact_dir / pdf_path.stem
        work_root = self.execution.work_dir
        with stage_pdf_input(pdf_path, work_root) as staged:
            if not self._process_isolation:
                return self._parse_document(
                    staged.input_path,
                    source_file=source_file,
                    artifact_dir=report_dir,
                )

            preflight = self._preflight_pdf(staged.input_path)
            manager = self._worker_manager()
            return manager.parse(
                staged.input_path,
                source_file=source_file,
                artifact_dir=Path(report_dir),
                preflight=preflight,
            )

    def parse_pages(
        self,
        pdf_path: Path,
        *,
        source_file: str,
        artifact_dir: Path,
        page_range: tuple[int, int],
    ) -> ParserResult:
        """Parse one bounded, one-based page window for hybrid routing."""
        mode = str(
            getattr(self.settings, "docling_execution_mode", "local")
        ).lower()
        if mode != "local":
            try:
                import fitz

                with fitz.open(pdf_path) as document:
                    full_range = (1, len(document))
            except Exception:
                full_range = None
            if page_range != full_range:
                raise RuntimeError(
                    "Docling Serve does not support bounded hybrid page "
                    "windows; disable INGESTION_FAST_LANE or use local mode"
                )
            return self.parse(
                pdf_path,
                source_file=source_file,
                artifact_dir=artifact_dir,
            )

        with stage_pdf_input(pdf_path, self.execution.work_dir) as staged:
            preflight = self._preflight_pdf(staged.input_path)
            if not self._process_isolation:
                return self._parse_document(
                    staged.input_path,
                    source_file=source_file,
                    artifact_dir=artifact_dir,
                    page_range=page_range,
                    preflight=preflight,
                )
            return self._worker_manager().parse(
                staged.input_path,
                source_file=source_file,
                artifact_dir=artifact_dir,
                page_range=page_range,
                preflight=preflight,
            )

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        self._converters.clear()
