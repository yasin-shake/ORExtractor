"""Document partitioner protocol and Unstructured implementations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Protocol, runtime_checkable

from ingestion.cache import runtime_partitioner_version
from ingestion.models import ElementRecord
from ingestion.normalizer import normalize_elements


@runtime_checkable
class DocumentPartitioner(Protocol):
    def partition(self, pdf_path: Path) -> List[ElementRecord]:
        ...


def _raw_element_dict(element: Any) -> dict:
    """Return an audit-safe JSON representation of an Unstructured element."""
    if isinstance(element, dict):
        return element
    if hasattr(element, "to_dict"):
        try:
            payload = element.to_dict()
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    metadata = getattr(element, "metadata", None)
    if hasattr(metadata, "to_dict"):
        try:
            metadata = metadata.to_dict()
        except Exception:
            metadata = {"repr": str(metadata)}
    elif metadata is not None and not isinstance(metadata, dict):
        metadata = {"repr": str(metadata)}
    return {
        "type": getattr(element, "category", element.__class__.__name__),
        "text": getattr(element, "text", ""),
        "metadata": metadata or {},
    }


def _persist_partition_output(
    raw_elements: List[Any],
    artifact_dir: Path,
    source_file: str,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "partition_raw.json").write_text(
        json.dumps(
            {
                "source_file": source_file,
                "elements": [_raw_element_dict(element) for element in raw_elements],
            },
            indent=2,
            ensure_ascii=True,
            default=str,
        ),
        encoding="utf-8",
    )


class LocalUnstructuredPartitioner:
    """Partition a PDF with the local Unstructured library."""

    def __init__(
        self,
        strategy: str = "hi_res",
        artifact_dir: Path | None = None,
        infer_table_structure: bool = True,
        extract_images: bool = True,
    ):
        self.strategy = strategy
        self.artifact_dir = artifact_dir or Path("ingestion_artifacts")
        self.infer_table_structure = infer_table_structure
        self.extract_images = extract_images
        self.version = runtime_partitioner_version()
        self.provider_name = "unstructured-local"

    def partition(self, pdf_path: Path) -> List[ElementRecord]:
        os.environ.setdefault("DO_NOT_TRACK", "true")
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as exc:
            raise RuntimeError(
                "unstructured[pdf] is required for INGESTION_BACKEND=unstructured. "
                "Install with: pip install 'unstructured[pdf]'"
            ) from exc

        kwargs: dict[str, Any] = {
            "filename": str(pdf_path),
            "strategy": self.strategy,
            "infer_table_structure": self.infer_table_structure,
        }
        if self.extract_images:
            kwargs["extract_images_in_pdf"] = True
            kwargs["extract_image_block_types"] = ["Image", "Table"]
            kwargs["extract_image_block_to_payload"] = True

        try:
            raw_elements = partition_pdf(**kwargs)
        except TypeError:
            # Older Unstructured versions may not accept image extraction kwargs.
            kwargs.pop("extract_images_in_pdf", None)
            kwargs.pop("extract_image_block_types", None)
            kwargs.pop("extract_image_block_to_payload", None)
            raw_elements = partition_pdf(**kwargs)

        report_artifact_dir = self.artifact_dir / pdf_path.stem
        raw_elements = list(raw_elements)
        _persist_partition_output(raw_elements, report_artifact_dir, pdf_path.name)
        return normalize_elements(
            raw_elements,
            source_file=pdf_path.name,
            artifact_dir=report_artifact_dir,
        )


class UnstructuredApiPartitioner:
    """Partition through the hosted or self-hosted Unstructured Partition API."""

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str = "",
        strategy: str = "hi_res",
        artifact_dir: Path | None = None,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.strategy = strategy
        self.artifact_dir = artifact_dir or Path("ingestion_artifacts")
        self.version = runtime_partitioner_version()
        self.provider_name = "unstructured-api"

    def partition(self, pdf_path: Path) -> List[ElementRecord]:
        os.environ.setdefault("DO_NOT_TRACK", "true")
        try:
            from unstructured.partition.api import partition_via_api
        except ImportError as exc:
            raise RuntimeError(
                "unstructured is required for UNSTRUCTURED_PROVIDER=api. "
                "Install with: pip install 'unstructured[pdf]'"
            ) from exc

        kwargs: dict[str, Any] = {
            "filename": str(pdf_path),
            "api_key": self.api_key,
            "strategy": self.strategy,
        }
        if self.api_url:
            kwargs["api_url"] = self.api_url

        try:
            raw_elements = partition_via_api(
                **kwargs,
                infer_table_structure=True,
                extract_image_block_types=["Image", "Table"],
                extract_image_block_to_payload=True,
            )
        except TypeError:
            # Older clients expose fewer optional API fields.
            raw_elements = partition_via_api(**kwargs)

        report_artifact_dir = self.artifact_dir / pdf_path.stem
        raw_elements = list(raw_elements)
        _persist_partition_output(raw_elements, report_artifact_dir, pdf_path.name)
        return normalize_elements(
            raw_elements,
            source_file=pdf_path.name,
            artifact_dir=report_artifact_dir,
        )


def get_partitioner(settings) -> DocumentPartitioner:
    provider = (getattr(settings, "unstructured_provider", "local") or "local").lower()
    if provider == "api":
        return UnstructuredApiPartitioner(
            api_url=getattr(settings, "unstructured_api_url", None),
            api_key=getattr(settings, "unstructured_api_key", ""),
            strategy=getattr(settings, "unstructured_strategy", "hi_res"),
            artifact_dir=getattr(settings, "artifact_dir", Path("ingestion_artifacts")),
        )
    if provider == "local":
        return LocalUnstructuredPartitioner(
            strategy=getattr(settings, "unstructured_strategy", "hi_res"),
            artifact_dir=getattr(settings, "artifact_dir", Path("ingestion_artifacts")),
        )
    raise ValueError(
        f"UNSTRUCTURED_PROVIDER must be 'local' or 'api', got {provider!r}"
    )
