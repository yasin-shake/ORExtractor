"""Manual cross-section digitizing tool (Phase 3 of the spatial POC).

Usage:
    python digitize_sections.py <report.pdf> [--pages 123,456] [--dpi 150]
                                [--port 8765] [--render-only]

Workflow:
1. Candidate cross-section pages are found by scanning the PDF text directly
   with PyMuPDF for section-like figure captions (marker's chunk page metadata
   does not match physical pages, so the index can't be used for this).
   Override or extend with --pages.
2. Those PDF pages are rendered to PNGs in spatial_data/{stem}_sections/.
3. A localhost-only server hosts digitizer.html: calibrate each section with
   two diagonal anchor points of known (easting, northing, elevation), then
   click contact points. The SERVER converts pixels to world coordinates via
   pixel_to_world() below — the browser math is preview-only — and appends
   DigitizedPoint rows to spatial_data/{stem}.json with pixel coordinates,
   anchors, and datum recorded in each point's `source` for audit.

Digitizing does not flip the extraction's `confirmed` flag: the points are
human-made, but the reviewer decides when the dataset as a whole is trusted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional, Tuple

_HERE = Path(__file__).parent
_EPS = 1e-9

_SECTION_CAPTION_RE = re.compile(
    r"(?i)(cross[\s-]?section|long[\s-]?section|\bsection\s+[A-Z0-9]|looking\s+(north|south|east|west))"
)


# ---------------------------------------------------------------------------
# Pixel -> world transform (pure, unit-tested)
# ---------------------------------------------------------------------------


def pixel_to_world(px: float, py: float, a1: dict, a2: dict) -> Tuple[float, float, float]:
    """Map an image pixel to world (x, y, z) using two calibration anchors.

    Each anchor is {px, py, x, y, z}: a point clicked on the image whose world
    position is known (grid intersections on the section frame). The anchors
    must be diagonal — different pixel column AND row — and must differ in both
    elevation and plan position, so both axes are constrained. Horizontal pixel
    position interpolates linearly along the section line in plan; vertical
    pixel position interpolates elevation. Vertical exaggeration is handled
    naturally because the two axes are scaled independently.
    """
    dpx = a2["px"] - a1["px"]
    dpy = a2["py"] - a1["py"]
    if abs(dpx) < _EPS or abs(dpy) < _EPS:
        raise ValueError("anchors must differ in both pixel axes — pick diagonal grid corners")
    if abs(a2["z"] - a1["z"]) < _EPS:
        raise ValueError("anchors must differ in elevation")
    if abs(a2["x"] - a1["x"]) < _EPS and abs(a2["y"] - a1["y"]) < _EPS:
        raise ValueError("anchors must differ in plan position (easting/northing)")
    t = (px - a1["px"]) / dpx
    v = (py - a1["py"]) / dpy
    x = a1["x"] + t * (a2["x"] - a1["x"])
    y = a1["y"] + t * (a2["y"] - a1["y"])
    z = a1["z"] + v * (a2["z"] - a1["z"])
    return x, y, z


# ---------------------------------------------------------------------------
# Section page discovery & rendering
# ---------------------------------------------------------------------------


def find_section_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Return [(physical_page, caption_snippet)] for section-like figure captions.

    Scans the PDF text directly with PyMuPDF rather than trusting the Chroma
    chunk metadata: marker's block.page attribute does NOT match physical page
    numbers (observed: 'page 1537' in a 177-page PDF), so index-derived pages
    cannot drive rendering.
    """
    import fitz

    hits: dict = {}
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc):
            for line in (page.get_text() or "").splitlines():
                stripped = line.strip()
                if not re.match(r"(?i)^figure\s+[\w.\-]+", stripped):
                    continue
                if _SECTION_CAPTION_RE.search(stripped):
                    hits.setdefault(idx + 1, stripped[:120])
                    break
    return sorted(hits.items())


def render_pages(pdf_path: Path, pages: List[int], out_dir: Path, dpi: int) -> List[Path]:
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: List[Path] = []
    with fitz.open(pdf_path) as doc:
        for page_no in pages:
            if not (1 <= page_no <= len(doc)):
                print(f"  page {page_no} out of range (PDF has {len(doc)} pages), skipped")
                continue
            out = out_dir / f"page_{page_no:04d}.png"
            if not out.exists():
                pix = doc[page_no - 1].get_pixmap(dpi=dpi)
                pix.save(str(out))
            rendered.append(out)
    return rendered


# ---------------------------------------------------------------------------
# Localhost digitizing server
# ---------------------------------------------------------------------------


class _DigitizerState:
    def __init__(self, spatial_path: Path, sections_dir: Path, captions: dict):
        self.spatial_path = spatial_path
        self.sections_dir = sections_dir
        self.captions = captions  # png name -> caption snippet

    def spatial(self) -> dict:
        return json.loads(self.spatial_path.read_text(encoding="utf-8"))

    def unit_names(self, spatial: dict) -> List[str]:
        units: List[str] = []
        for u in spatial.get("stratigraphic_pile") or []:
            if u.get("unit_name"):
                units.append(u["unit_name"])
        for iv in spatial.get("lithology_intervals") or []:
            name = iv.get("unit_name")
            if name and name not in units:
                units.append(name)
        for f in spatial.get("faults") or []:
            if f.get("fault_name"):
                units.append(f["fault_name"])
        return units

    def append_points(self, payload: dict) -> int:
        section_id = (payload.get("section_id") or "").strip()
        datum = (payload.get("datum") or "").strip()
        a1, a2 = payload["anchors"]["a1"], payload["anchors"]["a2"]
        anchor_desc = (
            f"anchors ({a1['x']},{a1['y']},{a1['z']})@px({a1['px']},{a1['py']}) / "
            f"({a2['x']},{a2['y']},{a2['z']})@px({a2['px']},{a2['py']})"
        )
        spatial = self.spatial()
        rows = spatial.setdefault("cross_section_points", [])
        added = 0
        for p in payload.get("points") or []:
            surface = (p.get("surface_name") or "").strip()
            if not surface:
                continue
            x, y, z = pixel_to_world(float(p["px"]), float(p["py"]), a1, a2)
            rows.append(
                {
                    "section_id": section_id,
                    "surface_name": surface,
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "z": round(z, 2),
                    "source": (
                        f"digitized {section_id}"
                        + (f" [{datum}]" if datum else "")
                        + f" px({p['px']},{p['py']}); {anchor_desc}"
                    ),
                }
            )
            added += 1
        if added:
            self.spatial_path.write_text(json.dumps(spatial, indent=2), encoding="utf-8")
        return added


def _make_handler(state: _DigitizerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (_HERE / "digitizer.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                spatial = state.spatial()
                sections = sorted(p.name for p in state.sections_dir.glob("*.png"))
                self._json(
                    {
                        "report": spatial.get("source_file"),
                        "coordinate_system": spatial.get("coordinate_system"),
                        "confirmed": bool(spatial.get("confirmed")),
                        "sections": [
                            {"file": s, "caption": state.captions.get(s, "")} for s in sections
                        ],
                        "units": state.unit_names(spatial),
                        "existing_points": len(spatial.get("cross_section_points") or []),
                    }
                )
            elif self.path.startswith("/section/"):
                name = Path(self.path[len("/section/"):]).name  # no traversal
                target = state.sections_dir / name
                if target.exists() and target.suffix == ".png":
                    self._send(200, target.read_bytes(), "image/png")
                else:
                    self._json({"error": "not found"}, 404)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/api/points":
                self._json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                added = state.append_points(payload)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
                return
            spatial = state.spatial()
            self._json({"added": added, "total": len(spatial.get("cross_section_points") or [])})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Digitize cross-section figures into spatial_data JSON.")
    parser.add_argument("filename", help="Source PDF filename (must exist in the knowledge dirs).")
    parser.add_argument("--pages", default=None, help="Comma-separated PDF page numbers (skips caption discovery).")
    parser.add_argument("--dpi", type=int, default=150, help="Render resolution (default 150).")
    parser.add_argument("--port", type=int, default=8765, help="Localhost port (default 8765).")
    parser.add_argument("--render-only", action="store_true", help="Render section PNGs and exit.")
    args = parser.parse_args()

    import os

    from dotenv import load_dotenv
    load_dotenv()
    from rag_app import load_settings, iter_pdf_paths

    settings = load_settings()
    stem = Path(args.filename).stem
    spatial_path = settings.spatial_dir / f"{stem}.json"
    if not spatial_path.exists():
        print(f"No spatial extraction at {spatial_path} — run: python rag_app.py extract --spatial --file {args.filename}")
        return 1

    pdf_path = next(
        (p for p in iter_pdf_paths(settings.knowledge_dir, settings.extra_pdf_dirs) if p.name == args.filename),
        None,
    )
    if pdf_path is None:
        print(f"PDF {args.filename!r} not found in knowledge directories.")
        return 1

    captions: dict = {}
    if args.pages:
        pages = [int(p) for p in args.pages.split(",") if p.strip().isdigit()]
    else:
        found = find_section_pages(pdf_path)
        pages = [p for p, _ in found]
        for p, cap in found:
            captions[f"page_{p:04d}.png"] = cap
        print(f"Found {len(pages)} candidate section page(s) from figure captions.")
        for p, cap in found:
            print(f"  p.{p}: {cap}")
        if not pages:
            print("No section-like captions found — pass explicit pages with --pages.")
            return 1

    sections_dir = settings.spatial_dir / f"{stem}_sections"
    rendered = render_pages(pdf_path, pages, sections_dir, args.dpi)
    print(f"{len(rendered)} section image(s) in {sections_dir}")
    if args.render_only:
        return 0

    state = _DigitizerState(spatial_path, sections_dir, captions)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _make_handler(state))
    print(f"Digitizer running: http://127.0.0.1:{args.port}  (Ctrl-C to stop)")
    print(f"Points append to {spatial_path} — re-run the model builder afterwards.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
