"""Unit tests for the section digitizer — transform math and JSON write-back.

No server, no PyMuPDF, no LLM.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from digitize_sections import _DigitizerState, pixel_to_world


# Two diagonal grid corners of an E–W section image, 1000 px wide, 500 px tall:
# left-bottom of frame at (E 604000, N 5367000, RL 0) = pixel (100, 450),
# right-top at (E 604400, N 5367000, RL 200) = pixel (900, 50).
A1 = {"px": 100.0, "py": 450.0, "x": 604000.0, "y": 5367000.0, "z": 0.0}
A2 = {"px": 900.0, "py": 50.0, "x": 604400.0, "y": 5367000.0, "z": 200.0}


def test_anchor_pixels_map_to_their_own_world_coords():
    assert pixel_to_world(A1["px"], A1["py"], A1, A2) == pytest.approx((604000.0, 5367000.0, 0.0))
    assert pixel_to_world(A2["px"], A2["py"], A1, A2) == pytest.approx((604400.0, 5367000.0, 200.0))


def test_midpoint_and_extrapolation():
    x, y, z = pixel_to_world(500.0, 250.0, A1, A2)
    assert (x, y, z) == pytest.approx((604200.0, 5367000.0, 100.0))
    # Clicks outside the calibrated frame extrapolate linearly (e.g. below RL 0).
    _, _, z_below = pixel_to_world(500.0, 550.0, A1, A2)
    assert z_below == pytest.approx(-50.0)


def test_vertical_exaggeration_axes_scale_independently():
    # Same world frame but the image is stretched 4x vertically: elevations
    # must be unaffected because the vertical scale comes from the anchors.
    a2_stretched = dict(A2, py=-1550.0)  # 4x the pixel span of the original
    _, _, z = pixel_to_world(500.0, (450.0 + -1550.0) / 2, A1, a2_stretched)
    assert z == pytest.approx(100.0)


def test_oblique_section_line_interpolates_plan_position():
    # NE-striking section: both easting and northing vary along the horizontal axis.
    a1 = {"px": 0.0, "py": 100.0, "x": 1000.0, "y": 2000.0, "z": 100.0}
    a2 = {"px": 100.0, "py": 0.0, "x": 1100.0, "y": 2100.0, "z": 300.0}
    x, y, z = pixel_to_world(50.0, 50.0, a1, a2)
    assert (x, y, z) == pytest.approx((1050.0, 2050.0, 200.0))


def test_degenerate_anchors_rejected():
    with pytest.raises(ValueError, match="pixel axes"):
        pixel_to_world(5, 5, A1, dict(A2, px=A1["px"]))
    with pytest.raises(ValueError, match="pixel axes"):
        pixel_to_world(5, 5, A1, dict(A2, py=A1["py"]))
    with pytest.raises(ValueError, match="elevation"):
        pixel_to_world(5, 5, A1, dict(A2, z=A1["z"]))
    with pytest.raises(ValueError, match="plan position"):
        pixel_to_world(5, 5, A1, dict(A2, x=A1["x"], y=A1["y"]))


def test_append_points_writes_provenance_and_preserves_confirmed(tmp_path):
    spatial = {
        "source_file": "x.pdf",
        "confirmed": True,
        "stratigraphic_pile": [{"unit_name": "Porphyry"}],
        "cross_section_points": [],
    }
    path = tmp_path / "x.json"
    path.write_text(json.dumps(spatial))
    state = _DigitizerState(path, tmp_path, {})

    added = state.append_points(
        {
            "section_id": "page_0100.png",
            "datum": "NAD27 Z21N",
            "anchors": {"a1": A1, "a2": A2},
            "points": [
                {"px": 500.0, "py": 250.0, "surface_name": "Porphyry (top)"},
                {"px": 500.0, "py": 250.0, "surface_name": ""},  # no surface → skipped
            ],
        }
    )
    assert added == 1
    saved = json.loads(path.read_text())
    assert saved["confirmed"] is True  # digitizing never flips the review gate
    pt = saved["cross_section_points"][0]
    assert pt["surface_name"] == "Porphyry (top)"
    assert (pt["x"], pt["y"], pt["z"]) == (604200.0, 5367000.0, 100.0)
    for fragment in ("page_0100.png", "NAD27 Z21N", "px(500.0,250.0)", "anchors"):
        assert fragment in pt["source"]


def test_append_points_bad_anchor_raises(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"source_file": "x.pdf", "cross_section_points": []}))
    state = _DigitizerState(path, tmp_path, {})
    with pytest.raises(ValueError):
        state.append_points(
            {
                "section_id": "s",
                "anchors": {"a1": A1, "a2": dict(A2, z=A1["z"])},
                "points": [{"px": 1.0, "py": 1.0, "surface_name": "X"}],
            }
        )
    # A failed save must not corrupt the file.
    assert json.loads(path.read_text())["cross_section_points"] == []
