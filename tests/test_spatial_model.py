"""Unit tests for spatial_model.py — desurvey math, contacts, plane fit, gate.

No GemPy/Plotly/LLM required.
"""

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spatial_model import (
    _canon_id,
    _lerp_angle_deg,
    assemble_model_inputs,
    build_contact_points,
    build_envelope_points,
    desurvey_hole,
    fit_plane_orientation,
    load_confirmed_spatial,
    min_curvature_step,
)


# ── minimum curvature ────────────────────────────────────────────────────────


def test_vertical_step_goes_straight_down():
    de, dn, dz = min_curvature_step(100.0, 0.0, 90.0, 0.0, 90.0)
    assert de == pytest.approx(0.0, abs=1e-9)
    assert dn == pytest.approx(0.0, abs=1e-9)
    assert dz == pytest.approx(-100.0)


def test_straight_inclined_step_matches_trig():
    # az 090 (east), dip 60: horizontal = cos(60°) = 0.5, vertical = sin(60°)
    de, dn, dz = min_curvature_step(100.0, 90.0, 60.0, 90.0, 60.0)
    assert de == pytest.approx(50.0)
    assert dn == pytest.approx(0.0, abs=1e-9)
    assert dz == pytest.approx(-100.0 * math.sin(math.radians(60)))


def test_dip_sign_is_ignored():
    assert min_curvature_step(100.0, 90.0, -60.0, 90.0, -60.0) == pytest.approx(
        min_curvature_step(100.0, 90.0, 60.0, 90.0, 60.0)
    )


def test_dogleg_displacement_shorter_than_arc():
    de, dn, dz = min_curvature_step(100.0, 0.0, 60.0, 90.0, 45.0)
    assert math.sqrt(de * de + dn * dn + dz * dz) < 100.0


def test_angle_lerp_wraps_shortest_path():
    assert _lerp_angle_deg(350.0, 10.0, 0.5) == pytest.approx(0.0)
    assert _lerp_angle_deg(10.0, 350.0, 0.5) == pytest.approx(0.0)


# ── desurvey ─────────────────────────────────────────────────────────────────


def _hole(**kw):
    base = {"hole_id": "DDH-01", "x": 1000.0, "y": 2000.0, "z_collar": 300.0}
    base.update(kw)
    return base


def test_desurvey_straight_inclined_hole():
    logs = []
    trace = desurvey_hole(
        _hole(azimuth=90.0, dip=-60.0, total_depth_m=100.0), logs
    )
    end = trace.stations[-1]
    assert end.x == pytest.approx(1050.0)
    assert end.y == pytest.approx(2000.0)
    assert end.z == pytest.approx(300.0 - 100.0 * math.sin(math.radians(60)))
    assert logs == []


def test_desurvey_missing_dip_assumes_vertical_and_logs():
    logs = []
    trace = desurvey_hole(_hole(total_depth_m=50.0), logs)
    end = trace.stations[-1]
    assert (end.x, end.y) == (1000.0, 2000.0)
    assert end.z == pytest.approx(250.0)
    assert any("default-assumed vertical" in l for l in logs)


def test_desurvey_missing_collar_coordinates_skips():
    logs = []
    assert desurvey_hole({"hole_id": "DDH-02", "x": None, "y": 5.0}, logs) is None
    assert any("DDH-02" in l for l in logs)


def test_desurvey_uses_survey_points():
    logs = []
    trace = desurvey_hole(
        _hole(
            azimuth=0.0,
            dip=-90.0,
            total_depth_m=200.0,
            survey_points=[
                {"depth_m": 0.0, "azimuth": 0.0, "dip": -90.0},
                {"depth_m": 100.0, "azimuth": 0.0, "dip": -90.0},
            ],
        ),
        logs,
    )
    assert trace.total_md == 200.0
    assert trace.stations[-1].z == pytest.approx(100.0)  # 300 - 200


def test_position_at_interpolates_and_clamps():
    logs = []
    trace = desurvey_hole(_hole(azimuth=90.0, dip=-60.0, total_depth_m=100.0), logs)
    x, y, z = trace.position_at(50.0)
    assert x == pytest.approx(1025.0)
    assert z == pytest.approx(300.0 - 50.0 * math.sin(math.radians(60)))
    beyond = trace.position_at(9999.0)
    assert beyond == pytest.approx((trace.stations[-1].x, trace.stations[-1].y, trace.stations[-1].z))


# ── contact points ───────────────────────────────────────────────────────────


def _spatial_two_units():
    return {
        "boreholes": [_hole(azimuth=0.0, dip=-90.0, total_depth_m=150.0)],
        "lithology_intervals": [
            {"hole_id": "DDH-01", "from_m": 0.0, "to_m": 50.0, "unit_name": "A"},
            {"hole_id": "DDH-01", "from_m": 50.0, "to_m": 150.0, "unit_name": "B"},
        ],
    }


def test_contact_point_at_unit_boundary():
    logs = []
    points, traces = build_contact_points(_spatial_two_units(), logs)
    assert len(points) == 1
    p = points[0]
    assert p["surface"] == "B"  # top of the underlying unit
    assert p["z"] == pytest.approx(250.0)  # 300 - 50
    assert "DDH-01" in p["source"]


def test_no_contact_for_repeated_unit():
    spatial = _spatial_two_units()
    spatial["lithology_intervals"][1]["unit_name"] = "a"  # same unit, case-insensitive
    logs = []
    points, _ = build_contact_points(spatial, logs)
    assert points == []


def test_lithology_for_unknown_hole_is_logged():
    spatial = _spatial_two_units()
    spatial["lithology_intervals"].append(
        {"hole_id": "GHOST-9", "from_m": 0.0, "to_m": 10.0, "unit_name": "A"}
    )
    logs = []
    build_contact_points(spatial, logs)
    assert any("GHOST-9" in l for l in logs)


# ── hole ID canonicalisation & envelope interpretation ───────────────────────


def test_canon_id_joins_report_variants():
    # Real case from the Caracle report: intercept table vs collar table.
    assert _canon_id("HX06-01") == _canon_id("HX-06-1")
    assert _canon_id("hx 06 01") == _canon_id("HX-06-1")
    assert _canon_id("HX-06-1") != _canon_id("HX-06-10")


def test_contacts_join_across_id_conventions():
    spatial = _spatial_two_units()
    spatial["boreholes"][0]["hole_id"] = "DDH-01"
    for iv in spatial["lithology_intervals"]:
        iv["hole_id"] = "DDH01"  # different punctuation than the collar table
    logs = []
    points, _ = build_contact_points(spatial, logs)
    assert len(points) == 1
    assert any("canonical matching" in l for l in logs)


def test_envelope_points_from_isolated_intervals():
    spatial = {
        "boreholes": [
            _hole(hole_id="H1", azimuth=0.0, dip=-90.0, total_depth_m=100.0),
            _hole(hole_id="H2", x=1100.0, azimuth=0.0, dip=-90.0, total_depth_m=100.0),
        ],
        "lithology_intervals": [
            {"hole_id": "H1", "from_m": 10.0, "to_m": 40.0, "unit_name": "Porphyry"},
            {"hole_id": "H2", "from_m": 20.0, "to_m": 55.0, "unit_name": "Porphyry"},
        ],
    }
    logs = []
    _, traces = build_contact_points(spatial, logs)
    points = build_envelope_points(spatial, traces, ["porphyry"], logs)
    assert len(points) == 4
    tops = [p for p in points if p["surface"] == "Porphyry (top)"]
    bases = [p for p in points if p["surface"] == "Porphyry (base)"]
    assert len(tops) == 2 and len(bases) == 2
    assert tops[0]["z"] == pytest.approx(290.0)   # 300 - 10
    assert bases[0]["z"] == pytest.approx(260.0)  # 300 - 40
    assert all("envelope interpretation" in p["source"] for p in points)
    assert any("ENVELOPE INTERPRETATION" in l for l in logs)


def test_assemble_with_envelope_builds_top_above_base():
    spatial = {
        "boreholes": [
            _hole(hole_id=f"H{i}", x=1000.0 + 100.0 * (i % 3), y=2000.0 + 100.0 * (i // 3),
                  azimuth=0.0, dip=-90.0, total_depth_m=100.0)
            for i in range(4)
        ],
        "lithology_intervals": [
            {"hole_id": f"H{i}", "from_m": 10.0 + i, "to_m": 40.0 + i, "unit_name": "Porphyry"}
            for i in range(4)
        ],
    }
    inputs = assemble_model_inputs(spatial, envelope_units=["Porphyry"])
    # Orphan ordering by mean elevation must put the top surface first (youngest).
    assert inputs["series_mapping"] == {"Strat_Series": ["Porphyry (top)", "Porphyry (base)"]}
    assert len(inputs["surface_points"]) == 8


# ── plane fit ────────────────────────────────────────────────────────────────


def test_plane_fit_horizontal():
    fit = fit_plane_orientation([(0, 0, 10), (100, 0, 10), (0, 100, 10), (100, 100, 10)])
    dip, _ = fit
    assert dip == pytest.approx(0.0, abs=1e-6)


def test_plane_fit_45_degrees_east():
    # z = -x → dips 45° toward east (090)
    fit = fit_plane_orientation([(0, 0, 0), (100, 0, -100), (0, 100, 0), (100, 100, -100)])
    dip, dip_dir = fit
    assert dip == pytest.approx(45.0, abs=1e-6)
    assert dip_dir == pytest.approx(90.0, abs=1e-6)


def test_plane_fit_rejects_collinear_and_tiny_inputs():
    assert fit_plane_orientation([(0, 0, 0), (1, 1, 1)]) is None
    assert fit_plane_orientation([(0, 0, 0), (1, 1, 1), (2, 2, 2)]) is None


# ── assembly ─────────────────────────────────────────────────────────────────


def _spatial_three_holes():
    holes = [
        {"hole_id": "H1", "x": 0.0, "y": 0.0, "z_collar": 100.0, "dip": -90.0, "azimuth": 0.0, "total_depth_m": 150.0},
        {"hole_id": "H2", "x": 100.0, "y": 0.0, "z_collar": 100.0, "dip": -90.0, "azimuth": 0.0, "total_depth_m": 150.0},
        {"hole_id": "H3", "x": 0.0, "y": 100.0, "z_collar": 100.0, "dip": -90.0, "azimuth": 0.0, "total_depth_m": 150.0},
    ]
    intervals = []
    for hid, d in (("H1", 50.0), ("H2", 60.0), ("H3", 50.0)):
        intervals += [
            {"hole_id": hid, "from_m": 0.0, "to_m": d, "unit_name": "A"},
            {"hole_id": hid, "from_m": d, "to_m": 150.0, "unit_name": "B"},
        ]
    return {
        "source_file": "x.pdf",
        "boreholes": holes,
        "lithology_intervals": intervals,
        "stratigraphic_pile": [
            {"unit_name": "A", "order_top_down": 1},
            {"unit_name": "B", "order_top_down": 2},
        ],
    }


def test_assemble_full_synthetic_model():
    inputs = assemble_model_inputs(_spatial_three_holes())
    assert len(inputs["surface_points"]) == 3
    assert inputs["series_mapping"] == {"Strat_Series": ["B"]}
    # No extracted orientations → plane fit derived one, and said so.
    assert len(inputs["orientations"]) == 1
    assert "plane fit" in inputs["orientations"][0]["source"]
    assert any("plane-fit" in l for l in inputs["logs"])
    # A is in the pile but has no contact points (it's the topmost unit).
    assert any("'A' has no contact points" in l for l in inputs["logs"])
    xmin, xmax, ymin, ymax, zmin, zmax = inputs["extent"]
    assert xmin < 0 < 100 < xmax and zmin < 40 and zmax > 100


def test_assemble_uses_extracted_orientation_over_plane_fit():
    spatial = _spatial_three_holes()
    spatial["orientations"] = [
        {"surface_name": "b", "x": 50.0, "y": 50.0, "z": 50.0, "dip": 30.0, "dip_direction": 90.0}
    ]
    inputs = assemble_model_inputs(spatial)
    assert len(inputs["orientations"]) == 1
    assert inputs["orientations"][0]["surface"] == "B"
    assert inputs["orientations"][0]["source"] == "extracted"


def test_assemble_orphan_units_ordered_by_elevation():
    spatial = _spatial_three_holes()
    spatial["stratigraphic_pile"] = []
    inputs = assemble_model_inputs(spatial)
    assert inputs["series_mapping"] == {"Strat_Series": ["B"]}
    assert any("order inferred" in l for l in inputs["logs"])


def test_assemble_raises_when_nothing_to_model():
    with pytest.raises(ValueError, match="No surface"):
        assemble_model_inputs({"boreholes": [], "lithology_intervals": []})


def test_assemble_includes_digitized_points():
    spatial = _spatial_three_holes()
    spatial["cross_section_points"] = [
        {"section_id": "Fig 7-3", "surface_name": "B", "x": 50.0, "y": 50.0, "z": 45.0,
         "source": "digitized Fig 7-3"}
    ]
    inputs = assemble_model_inputs(spatial)
    assert len(inputs["surface_points"]) == 4
    assert any(p.get("source") == "digitized Fig 7-3" for p in inputs["surface_points"])


# ── confirmed gate ───────────────────────────────────────────────────────────


def test_gate_refuses_unconfirmed(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"source_file": "x.pdf", "confirmed": False}))
    with pytest.raises(ValueError, match="confirmed=false"):
        load_confirmed_spatial(tmp_path, "x.pdf")
    # Development override works, and confirmed files load normally.
    assert load_confirmed_spatial(tmp_path, "x.pdf", allow_unconfirmed=True)["source_file"] == "x.pdf"
    (tmp_path / "y.json").write_text(json.dumps({"source_file": "y.pdf", "confirmed": True}))
    assert load_confirmed_spatial(tmp_path, "y.pdf")["confirmed"] is True


def test_gate_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_confirmed_spatial(tmp_path, "nope.pdf")
