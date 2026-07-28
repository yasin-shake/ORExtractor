"""Parser-neutral rejection and deduplication of decorative visual crops."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from ingestion.models import ElementRecord


_PAGE_MARGIN_RATIO = 0.10
_VISUAL_CATEGORIES = frozenset({"Image", "Figure"})


def _is_page_margin_visual(element: ElementRecord) -> bool:
    coordinates = element.coordinates or {}
    metadata = element.metadata or {}
    try:
        page_height = float(metadata["page_height"])
        first_y = float(coordinates["t"])
        second_y = float(coordinates["b"])
    except (KeyError, TypeError, ValueError):
        return False
    if page_height <= 0:
        return False

    origin = str(coordinates.get("coord_origin", "BOTTOMLEFT")).upper()
    upper = max(first_y, second_y)
    lower = min(first_y, second_y)
    if origin == "TOPLEFT":
        return (
            upper <= page_height * _PAGE_MARGIN_RATIO
            or lower >= page_height * (1.0 - _PAGE_MARGIN_RATIO)
        )
    return (
        lower >= page_height * (1.0 - _PAGE_MARGIN_RATIO)
        or upper <= page_height * _PAGE_MARGIN_RATIO
    )


def _discard_crop(element: ElementRecord) -> None:
    if not element.image_path:
        return
    try:
        Path(element.image_path).unlink(missing_ok=True)
    except OSError:
        # A failed cleanup must not make parsing fail. The element is still
        # excluded from enrichment and indexing.
        pass


def _crop_sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _visual_fingerprint(
    path: Path,
) -> tuple[int, int, int, bytes] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            grayscale = rgb.convert("L")
            width, height = grayscale.size
            hash_image = grayscale.resize((17, 16))
            flattened = getattr(hash_image, "get_flattened_data", None)
            pixels = list(
                flattened() if callable(flattened) else hash_image.getdata()
            )
            difference_hash = 0
            for row in range(16):
                offset = row * 17
                for column in range(16):
                    difference_hash <<= 1
                    difference_hash |= int(
                        pixels[offset + column]
                        > pixels[offset + column + 1]
                    )
            normalized = rgb.resize((32, 32)).tobytes()
    except (OSError, ValueError):
        return None
    return width, height, difference_hash, normalized


def _visually_identical(
    first: tuple[int, int, int, bytes],
    second: tuple[int, int, int, bytes],
) -> bool:
    first_width, first_height, first_hash, first_pixels = first
    second_width, second_height, second_hash, second_pixels = second
    if min(first_width, first_height, second_width, second_height) <= 0:
        return False
    if max(first_width, second_width) / min(first_width, second_width) > 1.10:
        return False
    if max(first_height, second_height) / min(first_height, second_height) > 1.10:
        return False
    if (first_hash ^ second_hash).bit_count() > 1:
        return False
    mean_difference = sum(
        abs(first_value - second_value)
        for first_value, second_value in zip(
            first_pixels,
            second_pixels,
        )
    ) / len(first_pixels)
    return mean_difference <= 1.0


def _mark_duplicate(
    element: ElementRecord,
    canonical: ElementRecord,
) -> None:
    crop_path = Path(element.image_path or "")
    canonical_path = Path(canonical.image_path or "")
    if crop_path.resolve() != canonical_path.resolve():
        _discard_crop(element)
    element.image_path = canonical.image_path
    element.is_duplicate = True
    element.skip_reason = "duplicate_visual"
    element.metadata["duplicate_of_element_id"] = canonical.element_id


def filter_visual_artifacts(
    elements: Iterable[ElementRecord],
) -> list[ElementRecord]:
    """Return content elements with decorative crops removed and files deduped."""

    filtered: list[ElementRecord] = []
    canonical_by_digest: dict[str, ElementRecord] = {}
    canonical_fingerprints: list[
        tuple[ElementRecord, tuple[int, int, int, bytes]]
    ] = []
    for element in elements:
        if (
            element.category in _VISUAL_CATEGORIES
            and _is_page_margin_visual(element)
        ):
            _discard_crop(element)
            continue
        if element.category in _VISUAL_CATEGORIES and element.image_path:
            crop_path = Path(element.image_path)
            digest = _crop_sha256(crop_path)
            canonical = canonical_by_digest.get(digest or "")
            fingerprint = None
            if canonical is None:
                fingerprint = _visual_fingerprint(crop_path)
                if fingerprint is not None:
                    canonical = next(
                        (
                            candidate
                            for candidate, candidate_fingerprint
                            in canonical_fingerprints
                            if _visually_identical(
                                fingerprint,
                                candidate_fingerprint,
                            )
                        ),
                        None,
                    )
            if canonical is not None:
                _mark_duplicate(element, canonical)
            else:
                if digest:
                    canonical_by_digest[digest] = element
                if fingerprint is not None:
                    canonical_fingerprints.append((element, fingerprint))
        filtered.append(element)
    return filtered
