"""Unit tests for spatial extraction schemas and merge logic (no LLM required)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from extractor import _merge_spatial, _SPATIAL_PASSES
from spatial_schemas import (
    Borehole,
    Fault,
    LithologyInterval,
    Orientation,
    SpatialExtraction,
    StratigraphicUnit,
    SurveyPoint,
)


def test_empty_extraction_defaults():
    ex = SpatialExtraction()
    assert ex.boreholes == []
    assert ex.lithology_intervals == []
    assert ex.stratigraphic_pile == []
    assert ex.orientations == []
    assert ex.faults == []
    assert ex.cross_section_points == []
    assert ex.confirmed is False
    assert ex.coordinate_system is None


def test_list_coercion_from_llm_null_string():
    # LLMs sometimes return "null" or a bare string for list fields.
    ex = SpatialExtraction(boreholes="null", faults=None, orientations="")
    assert ex.boreholes == []
    assert ex.faults == []
    assert ex.orientations == []


def test_borehole_with_survey_points():
    bh = Borehole(
        hole_id="DDH-01",
        x=451200.0,
        y=5631400.0,
        z_collar=312.5,
        azimuth=45.0,
        dip=-60.0,
        total_depth_m=250.0,
        survey_points=[
            SurveyPoint(depth_m=0.0, azimuth=45.0, dip=-60.0),
            SurveyPoint(depth_m=100.0, azimuth=47.0, dip=-58.5),
        ],
        source="Table 10-1, page 87",
    )
    assert bh.dip == -60.0  # sign preserved, never flipped
    assert len(bh.survey_points) == 2
    assert bh.source.startswith("Table")


def test_json_roundtrip():
    ex = SpatialExtraction(
        source_file="report.pdf",
        coordinate_system="UTM Zone 17N NAD83",
        boreholes=[Borehole(hole_id="DDH-01", x=1.0, y=2.0, z_collar=3.0)],
        lithology_intervals=[
            LithologyInterval(hole_id="DDH-01", from_m=0.0, to_m=12.0, unit_name="Overburden")
        ],
        stratigraphic_pile=[StratigraphicUnit(unit_name="Basalt", order_top_down=1)],
        orientations=[Orientation(surface_name="bedding", dip=45.0, dip_direction=135.0)],
        faults=[Fault(fault_name="North Fault", affected_units=["Basalt"])],
    )
    parsed = SpatialExtraction(**json.loads(ex.model_dump_json()))
    assert parsed == ex


def test_merge_concatenates_lists_and_keeps_first_scalar():
    a = SpatialExtraction(
        source_file="x.pdf",
        coordinate_system="UTM Zone 17N",
        boreholes=[Borehole(hole_id="DDH-01")],
        notes="table truncated",
    )
    b = SpatialExtraction(
        source_file="x.pdf",
        coordinate_system="local grid",  # loses to first pass
        boreholes=[Borehole(hole_id="DDH-02")],
        orientations=[Orientation(surface_name="bedding", dip=45.0)],
        notes="strike converted via right-hand rule",
    )
    merged = _merge_spatial("x.pdf", a, b)
    assert merged.source_file == "x.pdf"
    assert merged.coordinate_system == "UTM Zone 17N"
    assert [h.hole_id for h in merged.boreholes] == ["DDH-01", "DDH-02"]
    assert len(merged.orientations) == 1
    assert "table truncated" in merged.notes
    assert "right-hand rule" in merged.notes


def test_merge_always_resets_confirmed():
    # A fresh extraction must invalidate any prior human review.
    a = SpatialExtraction(source_file="x.pdf", confirmed=True)
    merged = _merge_spatial("x.pdf", a)
    assert merged.confirmed is False


def test_merge_empty_partials():
    merged = _merge_spatial("x.pdf")
    assert merged.source_file == "x.pdf"
    assert merged.boreholes == []


def test_spatial_passes_shape():
    # Drilling pass must not be Item-scoped (appendix chunks are tagged ni_item=27
    # and an Item filter would hide them); it must prefer table chunks instead.
    drilling = next(p for p in _SPATIAL_PASSES if p["name"] == "drilling")
    assert drilling["filter_items"] is None
    assert drilling["filter_types"] == ["table"]
    structure = next(p for p in _SPATIAL_PASSES if p["name"] == "structure")
    assert 7 in structure["filter_items"]
    for pass_def in _SPATIAL_PASSES:
        assert pass_def["queries"], f"pass {pass_def['name']} has no queries"
        assert pass_def["focus"]
