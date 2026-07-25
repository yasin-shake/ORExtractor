"""Docling primary parser adapter."""

from __future__ import annotations

import json
import os
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
        return value.export_to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
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

    def __init__(self, settings):
        self.settings = settings
        self.parser_version = _package_version("docling")
        self.artifact_dir = Path(
            getattr(settings, "artifact_dir", Path("ingestion_artifacts"))
        )

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
        }

    def _pipeline_options(self):
        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except ImportError as exc:
            raise RuntimeError(
                "Docling is required for the primary ingestion parser. "
                "Install the ingestion dependencies with: pip install -r requirements.txt"
            ) from exc

        options = PdfPipelineOptions()
        for name, value in {
            "do_ocr": getattr(self.settings, "docling_do_ocr", True),
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
            "document_timeout": getattr(
                self.settings, "docling_timeout_seconds", 900
            ),
        }.items():
            if hasattr(options, name):
                setattr(options, name, value)

        if getattr(self.settings, "docling_do_ocr", True):
            try:
                from docling.datamodel.pipeline_options import RapidOcrOptions

                backend = str(
                    getattr(
                        self.settings,
                        "docling_ocr_backend",
                        "onnxruntime",
                    )
                ).strip().lower()
                valid_backends = {"onnxruntime", "openvino", "paddle", "torch"}
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
                getattr(self.settings, "docling_table_mode", "accurate")
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

    def _convert_local(self, pdf_path: Path):
        _configure_windows_huggingface_cache()
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.settings import settings as docling_settings
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise RuntimeError(
                "Docling is not installed. Run: pip install -r requirements.txt"
            ) from exc

        # Docling's process-wide page batch controls how many PDF pages enter the
        # threaded pipeline together. Keep it aligned with the low-memory stage
        # batches so large reports do not accumulate native page buffers.
        docling_settings.perf.page_batch_size = _positive_setting(
            self.settings, "docling_page_batch_size", 1
        )
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self._pipeline_options()
                )
            },
        )
        return converter.convert(
            pdf_path,
            raises_on_error=False,
            max_num_pages=int(getattr(self.settings, "docling_max_pages", 1000)),
            max_file_size=int(
                getattr(self.settings, "docling_max_file_mb", 2048) * 1024 * 1024
            ),
        )

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
        )

    def parse(
        self,
        pdf_path: Path,
        *,
        source_file: str | None = None,
        artifact_dir: Path | None = None,
    ) -> ParserResult:
        source_file = source_file or pdf_path.name
        mode = str(getattr(self.settings, "docling_execution_mode", "local")).lower()
        started = time.perf_counter()
        if mode == "local":
            conversion = self._convert_local(pdf_path)
        elif mode == "serve":
            conversion = self._convert_service(pdf_path)
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
        quality = assess_parser_quality(
            elements,
            page_count=page_count,
            conversion_status=status,
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
        result = ParserResult(
            source_file=source_file,
            parser=self.parser_name,
            parser_version=self.parser_version,
            status=status,
            elements=elements,
            artifact_paths={
                "document_json": str(document_json),
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
            metadata=metadata.model_dump(),
        )
        (report_dir / "quality.json").write_text(
            result.quality.model_dump_json(indent=2), encoding="utf-8"
        )
        return result
