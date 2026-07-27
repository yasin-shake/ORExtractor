"""PDF discovery and stable source/output path mapping."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Optional


def _pdf_roots(
    knowledge_dir: Path,
    extra_dirs: Optional[List[Path]] = None,
) -> List[Path]:
    return [
        Path(knowledge_dir),
        *(Path(path) for path in (extra_dirs or [])),
    ]


def iter_pdf_paths(
    knowledge_dir: Path,
    extra_dirs: Optional[List[Path]] = None,
) -> Iterable[Path]:
    """Yield every PDF below the configured roots, recursively and once."""
    if not knowledge_dir.exists():
        raise FileNotFoundError(
            f"Knowledge directory does not exist: {knowledge_dir}"
        )
    paths: dict[str, Path] = {}
    for root in _pdf_roots(knowledge_dir, extra_dirs):
        if not root.exists():
            continue
        for directory, _, filenames in os.walk(root):
            for filename in filenames:
                path = Path(directory) / filename
                if path.suffix.casefold() != ".pdf":
                    continue
                resolved = path.resolve()
                paths.setdefault(os.path.normcase(str(resolved)), path)
    return sorted(
        paths.values(),
        key=lambda path: pdf_source_id(
            path,
            knowledge_dir,
            extra_dirs,
        ).casefold(),
    )


def pdf_source_id(
    pdf_path: Path,
    knowledge_dir: Path,
    extra_dirs: Optional[List[Path]] = None,
) -> str:
    """Return the stable Chroma/source identifier for a discovered PDF."""
    resolved_pdf = Path(pdf_path).resolve()
    matching_roots: List[Path] = []
    for root in _pdf_roots(knowledge_dir, extra_dirs):
        try:
            resolved_pdf.relative_to(root.resolve())
            matching_roots.append(root.resolve())
        except ValueError:
            continue
    if not matching_roots:
        return Path(pdf_path).name
    root = max(matching_roots, key=lambda candidate: len(candidate.parts))
    return resolved_pdf.relative_to(root).as_posix()


def filesystem_path(path: Path) -> Path:
    """Return a Windows extended path when the path exceeds legacy MAX_PATH."""
    candidate = Path(path)
    if os.name != "nt":
        return candidate
    resolved = candidate.resolve()
    raw = str(resolved)
    if raw.startswith("\\\\?\\") or len(raw) < 248:
        return candidate
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw.lstrip("\\"))
    return Path("\\\\?\\" + raw)


def source_output_path(
    base_dir: Path,
    source_file: str,
    suffix: str,
) -> Path:
    """Map a safe relative source ID to a mirrored output path."""
    relative = PurePosixPath(str(source_file).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe source path: {source_file!r}")
    source_path = Path(*relative.parts)
    output = (
        Path(base_dir)
        / source_path.parent
        / f"{source_path.stem}{suffix}"
    )
    if os.name == "nt" and len(str(output.resolve())) >= 248:
        digest = hashlib.sha256(
            relative.as_posix().encode("utf-8", errors="replace")
        ).hexdigest()[:16]
        compact_stem = source_path.stem[:80].rstrip(" .") or "report"
        output = (
            Path(base_dir)
            / "_long_paths"
            / f"{compact_stem}-{digest}{suffix}"
        )
    return output

