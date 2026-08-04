"""Regenerate parser visual crops from canonical PDF coordinates."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import fitz
from PIL import Image

from ingestion.models import ElementRecord


class CropMaterializationError(RuntimeError):
    """A canonical visual element cannot be cropped from its source page."""


def _source_rectangle(
    element: ElementRecord,
    *,
    page_width: float,
    page_height: float,
) -> fitz.Rect:
    coordinates = element.coordinates or {}
    try:
        left = float(coordinates["l"])
        right = float(coordinates["r"])
        first_y = float(coordinates["t"])
        second_y = float(coordinates["b"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CropMaterializationError(
            f"{element.element_id} has no usable bounding box"
        ) from exc

    metadata = element.metadata or {}
    try:
        source_width = float(metadata.get("page_width") or page_width)
        source_height = float(metadata.get("page_height") or page_height)
    except (TypeError, ValueError) as exc:
        raise CropMaterializationError(
            f"{element.element_id} has invalid page dimensions"
        ) from exc
    if source_width <= 0 or source_height <= 0:
        raise CropMaterializationError(
            f"{element.element_id} has invalid page dimensions"
        )

    x0, x1 = sorted((left, right))
    origin = str(
        coordinates.get("coord_origin", "BOTTOMLEFT")
    ).upper()
    if "TOPLEFT" in origin:
        y0, y1 = sorted((first_y, second_y))
    else:
        y0, y1 = sorted(
            (
                source_height - first_y,
                source_height - second_y,
            )
        )

    x_scale = page_width / source_width
    y_scale = page_height / source_height
    rectangle = fitz.Rect(
        x0 * x_scale,
        y0 * y_scale,
        x1 * x_scale,
        y1 * y_scale,
    )
    page_rectangle = fitz.Rect(0, 0, page_width, page_height)
    rectangle &= page_rectangle
    if rectangle.is_empty or rectangle.width <= 1 or rectangle.height <= 1:
        raise CropMaterializationError(
            f"{element.element_id} bounding box is outside page "
            f"{element.page_number}"
        )
    return rectangle


class CropMaterializer:
    """Materialize missing image/table crops without invoking a parser."""

    def __init__(self, *, scale: float = 2.0):
        self.scale = max(1.0, float(scale))

    def materialize(
        self,
        pdf_path: Path,
        elements: Iterable[ElementRecord],
        artifact_dir: Path,
    ) -> int:
        candidates = [
            element
            for element in elements
            if element.category in {"Image", "Figure", "Table"}
            and not element.is_duplicate
            and (
                not element.image_path
                or not Path(element.image_path).exists()
            )
        ]
        if not candidates:
            return 0

        target_dir = artifact_dir / "visual_backfill" / "images"
        target_dir.mkdir(parents=True, exist_ok=True)
        materialized = 0
        with fitz.open(pdf_path) as document:
            for element in candidates:
                page_index = int(element.page_number or 1) - 1
                if page_index < 0 or page_index >= document.page_count:
                    raise CropMaterializationError(
                        f"{element.element_id} references missing page "
                        f"{element.page_number}"
                    )
                page = document.load_page(page_index)
                rectangle = _source_rectangle(
                    element,
                    page_width=float(page.rect.width),
                    page_height=float(page.rect.height),
                )
                target = target_dir / f"{element.element_id}.png"
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(self.scale, self.scale),
                    clip=rectangle,
                    alpha=False,
                )
                pixmap.save(target)
                with Image.open(target) as image:
                    element.image_width = int(image.width)
                    element.image_height = int(image.height)
                element.image_path = str(target)
                element.metadata["crop_materialized_from_coordinates"] = True
                materialized += 1
        return materialized
