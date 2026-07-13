"""Pydantic models for spatial / geological-model data extracted from NI 43-101 reports.

These models feed the 3D geological model reconstruction pipeline (GemPy) and are
deliberately separate from ``NI43101Report``: spatial data is written to its own
``spatial_data/{stem}.json`` file, every record carries a ``source`` provenance
string, and the whole extraction must pass a human review gate (``confirmed``)
before any 3D model is built from it. A wrong contact point produces a
plausible-looking but wrong surface, silently — so unlike the narrative schema,
"never fabricate, return null" is necessary but not sufficient here.

Every field is optional because any individual report may omit a given dataset;
the extractor must degrade gracefully rather than invent coordinates.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from schemas import _coerce_to_list

_SOURCE_DESC = (
    "Where in the report this record came from, e.g. 'Table 10-2, page 87' or "
    "'Section 7.3 text, page 45'. Use 'default-assumed' for values not stated in "
    "the report but filled by convention (e.g. vertical hole where no survey exists)."
)


class SurveyPoint(BaseModel):
    """One downhole survey measurement for a deviated drill hole."""

    depth_m: Optional[float] = Field(None, description="Measured depth down the hole in metres.")
    azimuth: Optional[float] = Field(
        None, description="Hole azimuth at this depth in degrees, exactly as printed."
    )
    dip: Optional[float] = Field(
        None,
        description=(
            "Hole dip/inclination at this depth in degrees, exactly as printed — "
            "do not flip the sign to match a convention."
        ),
    )


class Borehole(BaseModel):
    """One drill hole collar, from a collar coordinate table."""

    hole_id: Optional[str] = Field(None, description="Drill hole identifier exactly as printed.")
    x: Optional[float] = Field(
        None, description="Collar easting / X coordinate in the report's stated grid."
    )
    y: Optional[float] = Field(
        None, description="Collar northing / Y coordinate in the report's stated grid."
    )
    z_collar: Optional[float] = Field(
        None, description="Collar elevation (Z / RL) in metres, as printed."
    )
    azimuth: Optional[float] = Field(
        None, description="Collar azimuth in degrees, exactly as printed."
    )
    dip: Optional[float] = Field(
        None,
        description=(
            "Collar dip in degrees, exactly as printed (e.g. -60 or 60 as the table shows) — "
            "do not flip the sign."
        ),
    )
    total_depth_m: Optional[float] = Field(
        None, description="Total hole depth / length in metres."
    )
    survey_points: List[SurveyPoint] = Field(
        default_factory=list,
        description=(
            "Downhole survey measurements for this hole, if a downhole survey table exists. "
            "Leave empty when the report gives no survey data for the hole."
        ),
    )
    source: Optional[str] = Field(None, description=_SOURCE_DESC)

    @field_validator("survey_points", mode="before")
    @classmethod
    def _coerce_survey_points(cls, v):
        return _coerce_to_list(v)


class LithologyInterval(BaseModel):
    """One lithology log interval within a drill hole."""

    hole_id: Optional[str] = Field(None, description="Drill hole identifier this interval belongs to.")
    from_m: Optional[float] = Field(None, description="Interval start depth in metres.")
    to_m: Optional[float] = Field(None, description="Interval end depth in metres.")
    unit_name: Optional[str] = Field(
        None,
        description=(
            "Lithological / stratigraphic unit name exactly as logged "
            "(e.g. 'Basalt', 'QFP dyke', 'Overburden')."
        ),
    )
    description: Optional[str] = Field(
        None, description="Any additional logged description for the interval."
    )
    source: Optional[str] = Field(None, description=_SOURCE_DESC)


class StratigraphicUnit(BaseModel):
    """One unit in the report's stratigraphic column / pile, in age order."""

    unit_name: Optional[str] = Field(None, description="Formation / unit name exactly as printed.")
    series_name: Optional[str] = Field(
        None,
        description=(
            "Group, series, or supergroup this unit belongs to, if the report names one "
            "(e.g. 'Timiskaming assemblage')."
        ),
    )
    order_top_down: Optional[int] = Field(
        None,
        description=(
            "Position in the stratigraphic pile: 1 = youngest / structurally topmost, "
            "increasing downward/older. Only assign when the report states or clearly "
            "implies the ordering."
        ),
    )
    relative_age: Optional[str] = Field(
        None, description="Geological age as stated (e.g. 'Archean', '2.7 Ga', 'Cretaceous')."
    )
    source: Optional[str] = Field(None, description=_SOURCE_DESC)


class Orientation(BaseModel):
    """One structural orientation measurement (bedding, contact, foliation, vein)."""

    surface_name: Optional[str] = Field(
        None,
        description=(
            "What the measurement orients: a named contact, unit, vein, foliation or "
            "bedding (e.g. 'Main Zone vein', 'bedding', 'contact Basalt/Sediments')."
        ),
    )
    x: Optional[float] = Field(
        None, description="Easting / X of the measurement location, only if explicitly stated."
    )
    y: Optional[float] = Field(
        None, description="Northing / Y of the measurement location, only if explicitly stated."
    )
    z: Optional[float] = Field(
        None, description="Elevation / Z of the measurement location, only if explicitly stated."
    )
    location_description: Optional[str] = Field(
        None,
        description=(
            "Textual location when no coordinates are given "
            "(e.g. 'Main Zone, section 4+50E')."
        ),
    )
    dip: Optional[float] = Field(None, description="Dip in degrees, exactly as printed.")
    dip_direction: Optional[float] = Field(
        None,
        description=(
            "Dip direction (dip azimuth) in degrees. If the report gives strike instead, "
            "convert with the right-hand rule (dip direction = strike + 90°) ONLY when the "
            "dip side is unambiguous, and quote the original strike text in `source`."
        ),
    )
    polarity: Optional[int] = Field(
        None,
        description="1 if the younging/facing direction is stated as normal, -1 if overturned; null if unstated.",
    )
    source: Optional[str] = Field(None, description=_SOURCE_DESC)


class Fault(BaseModel):
    """One named fault or shear zone."""

    fault_name: Optional[str] = Field(None, description="Fault / shear zone name exactly as printed.")
    dip: Optional[float] = Field(None, description="Fault dip in degrees, if stated.")
    dip_direction: Optional[float] = Field(
        None, description="Fault dip direction in degrees, if stated (same strike rule as Orientation)."
    )
    trace_description: Optional[str] = Field(
        None,
        description=(
            "Where the fault runs, as described (trend, length, which zones/sections it "
            "crosses)."
        ),
    )
    offset_sense: Optional[str] = Field(
        None, description="Sense and amount of displacement as stated (e.g. 'dextral, ~200 m')."
    )
    affected_units: List[str] = Field(
        default_factory=list,
        description="Stratigraphic units or zones the report says this fault offsets or bounds.",
    )
    source: Optional[str] = Field(None, description=_SOURCE_DESC)

    @field_validator("affected_units", mode="before")
    @classmethod
    def _coerce_affected_units(cls, v):
        return _coerce_to_list(v)


class DigitizedPoint(BaseModel):
    """One manually digitized point from a cross-section figure.

    Populated by the digitizing tool (Phase 3), never by the LLM extraction pass.
    """

    section_id: Optional[str] = Field(
        None, description="Which cross-section figure the point was digitized from."
    )
    surface_name: Optional[str] = Field(
        None, description="Contact / surface the point lies on."
    )
    x: Optional[float] = Field(None, description="World easting / X after coordinate transform.")
    y: Optional[float] = Field(None, description="World northing / Y after coordinate transform.")
    z: Optional[float] = Field(None, description="World elevation / Z after coordinate transform.")
    source: Optional[str] = Field(None, description=_SOURCE_DESC)


class SpatialExtraction(BaseModel):
    """Top-level spatial dataset for one report — input to the GemPy model builder."""

    source_file: Optional[str] = Field(
        None, description="Filename of the source PDF this data was extracted from."
    )
    coordinate_system: Optional[str] = Field(
        None,
        description=(
            "Coordinate system / grid the collar and orientation coordinates are in, "
            "exactly as stated (e.g. 'UTM Zone 17N NAD83', 'local mine grid')."
        ),
    )
    boreholes: List[Borehole] = Field(
        default_factory=list, description="One entry per drill hole collar table row."
    )
    lithology_intervals: List[LithologyInterval] = Field(
        default_factory=list, description="One entry per lithology log interval row."
    )
    stratigraphic_pile: List[StratigraphicUnit] = Field(
        default_factory=list,
        description="Stratigraphic units in age order (order_top_down: 1 = youngest).",
    )
    orientations: List[Orientation] = Field(
        default_factory=list, description="Structural orientation measurements."
    )
    faults: List[Fault] = Field(default_factory=list, description="Named faults / shear zones.")
    cross_section_points: List[DigitizedPoint] = Field(
        default_factory=list,
        description="Manually digitized cross-section points — never populated by the LLM.",
    )
    notes: Optional[str] = Field(
        None,
        description=(
            "Extraction caveats: truncated tables (state how many rows were omitted), "
            "ambiguous conventions, mixed coordinate grids, etc."
        ),
    )
    confirmed: bool = Field(
        False,
        description=(
            "Human review gate — set to true only by a reviewer after checking the data "
            "against the source PDF. The model builder refuses unconfirmed extractions. "
            "Never set by the LLM."
        ),
    )

    @field_validator(
        "boreholes",
        "lithology_intervals",
        "stratigraphic_pile",
        "orientations",
        "faults",
        "cross_section_points",
        mode="before",
    )
    @classmethod
    def _coerce_spatial_lists(cls, v):
        return _coerce_to_list(v)
