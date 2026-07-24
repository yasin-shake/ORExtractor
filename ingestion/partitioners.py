"""Document partitioner protocol and Unstructured implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Protocol, runtime_checkable

from ingestion.models import ElementRecord, PARTITIONER_VERSION
from ingestion.normalizer import normalize_elements


@runtime_checkable
class DocumentPartitioner(Protocol):
    def partition(self, pdf_path: Path) -> List[ElementRecord]:
        ...


class LocalUnstructuredPartitioner:
    """Partition a PDF with the local unstructured library."""

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
        self.version = PARTITIONER_VERSION
        self.provider_name = "unstructured-local"

    def partition(self, pdf_path: Path) -> List[ElementRecord]:
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
        # Prefer image extraction when the installed unstructured supports it.
        if self.extract_images:
            kwargs["extract_images_in_pdf"] = True
            kwargs["extract_image_block_types"] = ["Image", "Table"]
            kwargs["extract_image_block_to_payload"] = True

        try:
            raw_elements = partition_pdf(**kwargs)
        except TypeError:
            # Older unstructured versions may not accept image extraction kwargs.
            kwargs.pop("extract_images_in_pdf", None)
            kwargs.pop("extract_image_block_types", None)
            kwargs.pop("extract_image_block_to_payload", None)
            raw_elements = partition_pdf(**kwargs)

        report_artifact_dir = self.artifact_dir / pdf_path.stem
        return normalize_elements(
            raw_elements,
            source_file=pdf_path.name,
            artifact_dir=report_artifact_dir,
        )


class UnstructuredApiPartitioner:
    """Stub for hosted Unstructured API — not implemented in Phase A+B."""

    def __init__(self, api_url: str | None = None, api_key: str = ""):
        self.api_url = api_url
        self.api_key = api_key

    def partition(self, pdf_path: Path) -> List[ElementRecord]:
        raise NotImplementedError(
            "Hosted Unstructured API partitioner is deferred. "
            "Set UNSTRUCTURED_PROVIDER=local."
        )


def get_partitioner(settings) -> DocumentPartitioner:
    provider = (getattr(settings, "unstructured_provider", "local") or "local").lower()
    if provider == "api":
        return UnstructuredApiPartitioner(
            api_url=getattr(settings, "unstructured_api_url", None),
            api_key=getattr(settings, "unstructured_api_key", ""),
        )
    return LocalUnstructuredPartitioner(
        strategy=getattr(settings, "unstructured_strategy", "hi_res"),
        artifact_dir=getattr(settings, "artifact_dir", Path("ingestion_artifacts")),
    )
